import json, asyncio, time, torch, soundfile as sf, numpy as np, base64, tempfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts
import argparse
import uvicorn
from datetime import datetime
import yaml
import copy

# ============================================================
# ConversationEngine
# ============================================================

class ConversationEngine:
    def __init__(self, websocket: WebSocket = None, prompts: dict = None, delay: dict = None):

        self.SAMPLE_RATE = 16000
        self.WINDOW_SIZE = 256
        self.FRAME_SEC = 256 / 16000

        self.STATE = "LISTEN"
        self.IN_SPEECH = False
        self.BUFFER = []
        self.TURN_IDX = 0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.INTERRUPT_COUNT = 0

        # yaml
        self.prompts = prompts
        self.delay = delay
        self.END_HOLD_FRAMES = float(delay["end_hold_frame"])
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = float(delay["after_continue_time"])

        self.SILENCE_COUNTER = 0
        self.CONTINUE_INFER_TIMES = []
        self.CONTINUE_START_TIME = None

        self.CONTINUE_ARMED = False
        self.FROM_INTERRUPT = False
        self.history = []
        self.interrupt_buf = []
        self.INTERRUPT_START_TIME = 0

        self.output_dir = None

        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)
        self.websocket = websocket

        # prompts
        self.JUDGE_PROMPT = prompts.get("judge", "")
        self.INTERRUPT_PROMPT = prompts.get("interrupt", "")
        self.RESPONSE_PROMPT = prompts.get("response", "")

        self.assistant_history = []
        self.user_history = []


    # 构造 LLM messages
    def build_messages(self, system_prompt, user_history, assistant_history, user_audio):

        messages = [{"role": "system", "content": system_prompt}]
        if len(user_history) == 0 and len(assistant_history) == 0:
            audio_base64 = None
            if user_audio is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True, dir=str(self.output_dir)) as tmp:
                    sf.write(tmp.name, user_audio, self.SAMPLE_RATE)
                    tmp.seek(0)
                    audio_base64 = base64.b64encode(tmp.read()).decode("utf-8")

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "audio_url",
                         "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                    ]
                })
            return messages
        # ---- 有历史 ----
        rounds = min(len(user_history), len(assistant_history))
        for i in range(rounds):
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_history[i]}]
            })
            messages.append({
                "role": "assistant",
                "content": assistant_history[i]
            })

        # ---- 当前用户音频 ----
        if user_audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True, dir=str(self.output_dir)) as tmp:
                sf.write(tmp.name, user_audio, self.SAMPLE_RATE)
                tmp.seek(0)
                audio_base64 = base64.b64encode(tmp.read()).decode("utf-8")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio_url",
                     "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                ]
            })

        return messages


    # ==================================================
    def reset(self):
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.BUFFER.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.IN_SPEECH = False
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.CONTINUE_START_TIME = None
        self.INTERRUPT_COUNT = 0
        self.FROM_INTERRUPT = False
        self.interrupt_buf.clear()
        self.assistant_history.clear()
        self.user_history.clear()


    async def send_control(self, event_type: str, data=None):
        if not self.websocket:
            return
        payload = {"event": event_type, "data": data or {}}
        await self.websocket.send_text(json.dumps(payload))


    # ==================================================
    def detect_vad_frame(self, chunk):
        if not hasattr(self, "_vad_buf"):
            self._vad_buf = np.zeros(0, dtype=np.float32)

        self._vad_buf = np.concatenate([self._vad_buf, chunk])

        if len(self._vad_buf) >= 2 * self.WINDOW_SIZE:
            tensor = torch.from_numpy(self._vad_buf[: 2 * self.WINDOW_SIZE])
            event = self.vad_iterator(tensor, return_seconds=True)
            self._vad_buf = np.zeros(0, dtype=np.float32)
            return event

        return None


    # ==================================================
    async def async_asr(self, user_audio, turn_id):
        tmp = self.output_dir / f"stream_turn{turn_id}_input.wav"
        sf.write(tmp, user_audio, self.SAMPLE_RATE)
        user_text = await asyncio.to_thread(asr, str(tmp))

        await self.send_control("asr_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "state": self.STATE,
            "content": user_text
        })

        self.user_history.append(str(user_text))
        self.BUFFER.clear()
        return user_text


    # ==================================================
    async def async_llm(self, system_prompt, user_audio, turn_id, add_to_history=False):

        messages = self.build_messages(
            system_prompt=system_prompt,
            user_history=self.user_history,
            assistant_history=self.assistant_history,
            user_audio=user_audio
        )

        start_t = time.perf_counter()
        decision = await asyncio.to_thread(llm_qwen3o, messages)
        infer_time = round(time.perf_counter() - start_t, 3)

        # ============ 🔥 在这里过滤消息内容（只用于 send_control）============
        messages_clean = copy.deepcopy(messages)

        for msg in messages_clean:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "audio_url":
                        block["audio_url"]["url"] = "<AUDIO_BASE64_OMITTED>"

        # ===============================================================

        await self.send_control("llm_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "content": decision,
            "prompt": messages_clean,
            "turn": turn_id,
            "state": self.STATE,
        })

        if add_to_history:
            self.assistant_history.append(str(decision))

        self.IN_SPEECH = False
        return decision


    # ==================================================
    async def async_tts(self, text, turn_id):
        tts_path = self.output_dir / f"turn{turn_id}_tts.wav"
        tts_file = await asyncio.to_thread(tts, text, tts_path)

        await self.send_control("tts_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "state": self.STATE
        })

        with open(tts_file, "rb") as f:
            await self.websocket.send_bytes(f.read())

        self.STATE = "SPEAK"


    # ==================================================
    # LISTEN / SPEAK 状态逻辑
    # ==================================================
    async def handle_listen(self, frame, event):
        # ---- 用户开始说话 ----
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE
            })
            self.IN_SPEECH = True
            self.BUFFER = [frame]
            return

        # 未在讲话
        if not self.IN_SPEECH:
            return

        self.BUFFER.append(frame)

        # ---- 检测 end ----
        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            self.INTERRUPT_END_TIME = time.time()
            await self.send_control("vad_done", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE
            })
            return

        # ---- 静音计时 ----
        if self.SILENCE_COUNTER > 0:
            if event and "start" in event:
                self.SILENCE_COUNTER = 0
                return
            else:
                elapsed_silence = time.time() - self.INTERRUPT_END_TIME

                if elapsed_silence >= self.END_HOLD_FRAMES:
                    self.SILENCE_COUNTER = 0

                    await self.send_control("vad_640_done", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": self.TURN_IDX,
                        "state": self.STATE
                    })

                    user_audio = np.concatenate(self.BUFFER)

                    # ---- 阶段一 continue 判定 ----
                    decision = await self.async_llm(self.JUDGE_PROMPT, user_audio, self.TURN_IDX)
                    if "continue" in decision.lower():
                        self.CONTINUE_ARMED = True
                        self.CONTINUE_START_TIME = time.time()
                        self.IN_SPEECH = True
                        return

                    # ---- 完整响应流程 ----
                    asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                    decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                    asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                    return

        # ---- continue 超时逻辑 ----
        if self.CONTINUE_ARMED:
            elapsed = time.time() - self.CONTINUE_START_TIME
            if elapsed >= self.AFTER_CONTINUE_TIMEOUT_FRAMES:
                user_audio = np.concatenate(self.BUFFER)
                asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                asyncio.create_task(self.async_tts(decision, self.TURN_IDX))

                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None
                self.IN_SPEECH = False
                self.BUFFER.clear()
                return

            if event and "start" in event:
                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None


    # ==================================================
    async def handle_speak(self, frame, event):

        # 用户开始打断
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "turn": self.TURN_IDX,
                "state": self.STATE,
                "timestamp": round(time.time() - self.start_wall, 3)
            })
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.INTERRUPT_COUNT = 1
            self.SILENCE_COUNTER = 0
            self.INTERRUPT_START_TIME = time.time()
            return

        # 正在打断
        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            if event and "end" in event:
                self.SILENCE_COUNTER = 1
                await self.send_control("vad_done", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE
                })
                self.INTERRUPT_END_TIME = time.time()
                return

            # 640ms 内打断结束
            if self.SILENCE_COUNTER > 0:
                if event and "start" in event:
                    self.SILENCE_COUNTER = 0
                    return
                else:
                    elapsed_silence = time.time() - self.INTERRUPT_END_TIME

                    if elapsed_silence >= self.END_HOLD_FRAMES:
                        seg_audio = np.concatenate(self.interrupt_buf)
                        intent = await self.async_llm(
                            self.INTERRUPT_PROMPT, seg_audio,
                            self.TURN_IDX, add_to_history=False
                        )

                        if "interrupt" in intent.lower():

                            await self.send_control("shot_interrupt", {
                                "timestamp": round(time.time() - self.start_wall, 3),
                                "turn": self.TURN_IDX,
                                "state": self.STATE
                            })

                            self.BUFFER = self.interrupt_buf.copy()
                            self.TURN_IDX += 1

                            user_audio = np.concatenate(self.interrupt_buf)
                            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                            decision = await self.async_llm(
                                self.RESPONSE_PROMPT, user_audio,
                                self.TURN_IDX, add_to_history=True
                            )
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))

                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

                        else:
                            await self.send_control("no_interrupt", {
                                "timestamp": round(time.time() - self.start_wall, 3),
                                "turn": self.TURN_IDX,
                                "state": self.STATE
                            })
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

            # 长打断：无需 end
            if (self.interrupt_buf and
                self.SILENCE_COUNTER == 0 and
                time.time() - self.INTERRUPT_START_TIME >= 1.5):

                self.TURN_IDX += 1
                self.STATE = "LISTEN"

                await self.send_control("long_interrupt", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE
                })

                self.BUFFER = self.interrupt_buf.copy()
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                self.SILENCE_COUNTER = 0
                return

        return


    # ==================================================
    async def run_realtime(self, websocket: WebSocket):
        print("前端已连接")
        self.start_wall = time.time()

        try:
            while True:
                message = await websocket.receive()
                if "type" in message and message["type"] == "websocket.disconnect":
                    break

                # 文本消息
                if "text" in message and message["text"] is not None:
                    obj = json.loads(message["text"])

                    if obj.get("event") == "end":
                        self.vad_iterator.reset_states()
                        self.reset()
                        continue

                # 音频帧
                if "bytes" in message and message["bytes"]:
                    raw = message["bytes"]
                    frame = np.frombuffer(raw, dtype=np.float32)

                    if frame.size == 0:
                        continue

                    self.FRAME_IDX += 1
                    event = self.detect_vad_frame(frame)

                    if self.STATE == "LISTEN":
                        await self.handle_listen(frame, event)
                    else:
                        await self.handle_speak(frame, event)

        except WebSocketDisconnect:
            print("WebSocket断开")
        except Exception as e:
            print("Realtime 出错：", e)
        finally:
            self.vad_iterator.reset_states()
            self.reset()
            print("结束")


# ============================================================
# FastAPI
# ============================================================

def create_app(prompts, delay) -> FastAPI:
    app = FastAPI()

    @app.websocket("/realtime")
    async def realtime_ws(websocket: WebSocket):
        await websocket.accept()
        msg = await websocket.receive_json()

        data = msg.get("data", {})
        exp = data.get("exp", {})
        lang = data.get("lang", {})

        engine = ConversationEngine(websocket=websocket, prompts=prompts, delay=delay)

        engine.output_dir = Path("exp") / exp / f"realtimeout_{lang}"
        engine.output_dir.mkdir(parents=True, exist_ok=True)

        await engine.run_realtime(websocket)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts_cfg = cfg.get("prompts", {})
    delay_cfg = cfg.get("time", {})
    server_cfg = cfg.get("server", {})

    host = server_cfg.get("host", {})
    port = server_cfg.get("port", {})

    app = create_app(prompts_cfg, delay_cfg)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
