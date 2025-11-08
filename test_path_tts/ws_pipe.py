import os, json, asyncio, wave, time, torch, soundfile as sf, numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts
import argparse
import uvicorn

# ============================================================
# ConversationEngine （与你的逻辑一致，只加 websocket 通知能力）
# ============================================================

class ConversationEngine:
    def __init__(self, sample_rate=16000, window_size=256, websocket: WebSocket = None):
        self.SAMPLE_RATE = sample_rate
        self.WINDOW_SIZE = window_size
        self.FRAME_SEC = window_size / sample_rate
        self.INTERRUPT_LIMIT = int(1.5 / self.FRAME_SEC)    #speak打断硬时间，可以改

        self.STATE = "LISTEN"
        self.IN_SPEECH = False
        self.BUFFER = []
        self.TURN_IDX = 0
        # self.MEDIA_TIME = 0.0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.INTERRUPT_COUNT = 0
        self.FILE_NAME = "stream"
        self.END_HOLD_FRAMES = int(0.64 / self.FRAME_SEC)
        self.SILENCE_COUNTER = 0
        self.CONTINUE_INFER_TIMES = []
        self.CONTINUE_START_TIME = None
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = max(1, int(round(2.0 / self.FRAME_SEC)))
        self.CONTINUE_ARMED = False
        self.FROM_INTERRUPT = False
        self.history = []
        self.interrupt_buf = []
        self.INTERRUPT_START_TIME = 0

        self.output_dir = Path("realtime_out1")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)
        self.websocket = websocket  # ✅ 保存 websocket 引用用于事件推送

        #prompt
        self.RES_PROMPT = (
            "你是一个自然聊天的语音助手，要像朋友一样回答用户的问题。"
            "必须根据你听到的语言回答对应的语言，只有中文和英文"
            "不要反问，也不要解释，不要输出任何格式说明。"
            f"如果用户要求重复询问，请你参考历史信息，{self.history}进行正确的重复回答。"
            "以下是用户刚才说的话，请你进行回应："
        )
        self.JUDGE_PROMPT = (
            "你需要从日常对话的角度,而不是语法的角度去判断这句话是否说完了。"
            "首先，如果你认为用户这句话明显没有说完，请只输出字符串'continue'。"
            "如果你认为用户已经说完，请只输出字符串'end'。不要输出其他内容。"
        )
        self.INTERRUPT_PROMPT = (
            "你现在处于 SPEAK 状态，用户刚才在你说话时发出了一段语音。"
            "请根据语义判断他是否真的想打断你。"
            "如果是明确的反驳、否定、提出问题、要求停止、要求更正等，返回 'interrupt'；"
            "如果只是附和、回应、赞同或鼓励（例如“好的”“知道了”“说得好”“嗯嗯”“行”），"
            "请返回 'continue'。"
            "你只能返回这两个单词之一。不要解释、不要输出其它内容。\n\n"

            "以下是一些示例：\n"
            "用户：知道了。\n助手：continue\n"
            "用户：好得很。\n助手：continue\n"
            "用户：你说得真棒。\n助手：continue\n"
            "用户：嗯嗯，对。\n助手：continue\n"
            "用户：我不同意你说的。\n助手：interrupt\n"
            "用户：不是这样的。\n助手：interrupt\n"
            "用户：你别说了。\n助手：interrupt\n"
            "用户：等一下。\n助手：interrupt\n\n"

            "现在请判断当前用户这段语音的类型，只返回 'interrupt' 或 'continue'："
        )

    def reset(self):
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        # self.MEDIA_TIME = 0.0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.BUFFER.clear()
        self.history.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.IN_SPEECH = False
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.CONTINUE_START_TIME = None
        self.INTERRUPT_COUNT = 0
        self.FROM_INTERRUPT = False
        self.interrupt_buf.clear()

    def build_res_prompt(self):
        """动态构造RES_PROMPT，包含最新history"""
        return (
            "你是一个自然聊天的语音助手，要像朋友一样回答用户的问题\n"
            "不要反问，也不要解释，不要输出任何格式说明，必须根据你听到的语言回答对应的语言，只有中文or英文\n"
            f"只有用户提到“重复”“重新”，才需要参考历史信息{self.history}进行重复回答，否则不要关注历史信息\n"
            "如果用户提到“如果”“怎么办”“不行”等假设性或否定性内容，请回答当下问题，不要参考历史信息。"
            "以下是用户刚才说的话，请你进行回应："
        )


    async def send_control(self, event_type: str, data=None):
        """发送控制事件给前端"""
        if not self.websocket:
            return
        payload = {"event": event_type, "data": data or {}}
        await self.websocket.send_text(json.dumps(payload))  

    # ============= 与原逻辑一致的核心处理函数 =============
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

    async def async_asr(self, user_audio, turn_id):
        """异步ASR任务：仅做文本转写与历史更新，不参与当前逻辑"""
        tmp = self.output_dir / f"{self.FILE_NAME}_turn{turn_id}_input.wav"
        sf.write(tmp, user_audio, self.SAMPLE_RATE)
        user_text = await asyncio.to_thread(asr, str(tmp))
        await self.send_control("asr_done", {
            "turn": turn_id,
            "text": user_text,
            "timestamp": round(time.time() - self.start_wall, 3)
        })
        self.history.append({"role": "user", "content": user_text})
        # await self.send_control("context", {"turn":self.history,"timestamp": round(time.time() - self.start_wall, 3)})
        self.BUFFER.clear()
        return user_text

    async def async_llm(self, prompt, user_audio, turn_id, add_to_history: bool = False):
        start_t = time.perf_counter()
        decision = await asyncio.to_thread(llm_qwen3o, prompt, user_audio)
        infer_time = round(time.perf_counter() - start_t, 3)

        await self.send_control("llm_done", {
            "turn": turn_id,
            "content": decision,
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time
        })
        if add_to_history:
            self.history.append({"role": "assistant", "content": decision})
        self.IN_SPEECH = False
        return decision

    async def async_tts(self, text, turn_id):
        tts_path = self.output_dir / f"turn{turn_id}_tts.wav"
        tts_file = await asyncio.to_thread(tts, text, tts_path)
        await self.send_control("tts_done", {
            "turn": turn_id,
            "timestamp": round(time.time() - self.start_wall, 3)
        })
        with open(tts_file, "rb") as f:
            await self.websocket.send_bytes(f.read())

        self.STATE = "SPEAK"


    # -------------------------------------------------------
    async def handle_listen(self, frame, event):
        """LISTEN 状态：检测用户语音、判断是否说完、决定是否进入 SPEAK"""
        # --- 1. 用户开始说话 ---
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
            self.IN_SPEECH = True
            self.BUFFER = [frame]
            return
        # --- 2. 用户正在说话 ---
        if not self.IN_SPEECH:
            return  # 用户未发言，直接跳过本帧

        self.BUFFER.append(frame)

        # --- 3. 检测语音结束 ---
        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            await self.send_control("vad_done", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
            return

        if self.SILENCE_COUNTER > 0:
        # 如果在静音期间出现新的 start，则继续接上 buffer
            if event and "start" in event:
                self.SILENCE_COUNTER = 0
                return
            else:
                self.SILENCE_COUNTER += 1
                # 达到 640 ms（END_HOLD_FRAMES）后，确认结束
                if self.SILENCE_COUNTER >= self.END_HOLD_FRAMES:
                    self.SILENCE_COUNTER = 0

                    # === 新增阶段一判断 ===
                    await self.send_control("vad_640_done", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
                    user_audio = np.concatenate(self.BUFFER)
                    decision = await self.async_llm(self.JUDGE_PROMPT, user_audio, self.TURN_IDX, add_to_history=False)
                    print(f"用户语音完整性判定: {decision}")
                    if "continue" in decision.lower():
                        self.CONTINUE_ARMED = True
                        self.CONTINUE_START_TIME = time.time()
                        print("🔁 用户未说完，继续累积帧")
                        self.IN_SPEECH = True
                        return

                    # === 说完了，进入完整流程 ===
                    asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                    prompt_res_1 = self.build_res_prompt()
                    await self.send_control("prompt_lis", {"p": prompt_res_1,"timestamp": round(time.time() - self.start_wall, 3)})
                    decision = await self.async_llm(prompt_res_1, user_audio, self.TURN_IDX, add_to_history=True)
                    asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                    
                    return

        # --- 仅在上一次判定为 continue 且已武装时才计数 ---
        if self.CONTINUE_ARMED:
            # 检查当前是否超时
            elapsed = time.time() - self.CONTINUE_START_TIME
            if elapsed >= 2.0:  # 超过2秒没新语音就强制处理
                print(f"⚠️ continue 超时 {elapsed:.2f}s，强制进入完整处理")
                # 调用完整响应逻辑
                user_audio = np.concatenate(self.BUFFER)
                asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                prompt_res_2 = self.build_res_prompt()
                await self.send_control("prompt_lis", {"p": prompt_res_2,"timestamp": round(time.time() - self.start_wall, 3)})
                decision = await self.async_llm(prompt_res_2, user_audio, self.TURN_IDX, add_to_history=True)
                asyncio.create_task(self.async_tts(decision, self.TURN_IDX))

                # 清理状态
                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None
                self.IN_SPEECH = False
                self.BUFFER.clear()
                return

            # 如果出现新语音事件，则重置
            if event and "start" in event:
                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None


    async def handle_speak(self, frame, event):
        """SPEAK 状态：检测短打断或长打断"""
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.INTERRUPT_COUNT = 1
            self.SILENCE_COUNTER = 0
            self.INTERRUPT_START_TIME = time.time() 
            return

        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            # --- 2.1 检测用户结束讲话（end事件）---
            if event and "end" in event:
                self.SILENCE_COUNTER = 1
                await self.send_control("vad_done", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
                self.INTERRUPT_END_TIME = time.time()
                return

            # --- 静音确认阶段（640 ms 延迟）---
            if self.SILENCE_COUNTER > 0:
                if event and "start" in event: # 640ms 内出现新语音 → 继续接上
                    self.SILENCE_COUNTER = 0
                    return
                else:
                    elapsed_silence = time.time() - self.INTERRUPT_END_TIME   # 用真实时间判断静音
                    if elapsed_silence >= 0.64:  # 超过640ms，认为打断结束
                        seg_audio = np.concatenate(self.interrupt_buf)
                        intent = await self.async_llm(self.INTERRUPT_PROMPT, seg_audio, self.TURN_IDX, add_to_history=False)
                        print(f"出现了打断，打断意图判定: {intent}")

                        if "interrupt" in intent.lower():
                            await self.send_control("shot_interrupt", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
                            self.BUFFER = self.interrupt_buf.copy()
                            self.TURN_IDX += 1
                            user_audio = np.concatenate(self.interrupt_buf)
                            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                            prompt_res_3 = self.build_res_prompt()
                            await self.send_control("prompt_inter", {"p": prompt_res_3,"timestamp": round(time.time() - self.start_wall, 3)})
                            decision = await self.async_llm(prompt_res_3, user_audio, self.TURN_IDX, add_to_history=True)
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))

                            # 清理状态
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

                        else:
                            await self.send_control("no_interrupt", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            # 不切换 state，继续 SPEAK
                            return
            # 2.2 长打断：只有没有出现end的时候，也就是self.SILENCE_COUNTER == 0
            # 然后必须要buffer里面有东西
            #累计达到/超过 1.5s，无需等待 end，直接切 LISTEN
            if (self.interrupt_buf and 
                self.SILENCE_COUNTER == 0 and
                time.time() - self.INTERRUPT_START_TIME >= 1.5):
                self.TURN_IDX += 1
                await self.send_control("long_interrupt", {"turn": self.TURN_IDX,"timestamp": round(time.time() - self.start_wall, 3)})
                self.STATE = "LISTEN"
                self.BUFFER = self.interrupt_buf.copy()  # 交给下一轮继续累计
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                self.SILENCE_COUNTER = 0
                return

        # 3) 未检测到用户发声：SPEAK 持续（离线不模拟播放结束）
        return


    async def run_realtime(self, websocket: WebSocket):
        await websocket.accept()
        print("✅ 前端已连接，进入实时会话循环")
        self.start_wall = time.time() #是否要初始化？是否需要reset？
        # self.MEDIA_TIME = 0.0
        # self.FRAME_IDX = 0

        try:
            while True:
                message = await websocket.receive()
                # message 是 dict，形如：{"type": "websocket.receive", "bytes": b'...', "text": "..."}
                if message["type"] == "websocket.disconnect":
                    break

                if "text" in message:
                    text = message["text"]
                    obj = json.loads(text)
                    if obj.get("event") == "end":
                        print("前端结束")
                        # if self.IN_SPEECH and self.BUFFER:
                        #     await self.process_user_segment(self.BUFFER)
                        self.vad_iterator.reset_states()
                        self.reset()
                        continue
                    # 其他文本消息可选处理

                elif "bytes" in message:
                    raw_bytes = message["bytes"]
                    if len(raw_bytes) == 0:
                        continue
                    frame = np.frombuffer(raw_bytes, dtype=np.float32)
                    if frame.size == 0:
                        continue

                    # self.MEDIA_TIME += self.FRAME_SEC
                    self.FRAME_IDX += 1

                    event = self.detect_vad_frame(frame)
                    if event and ( "start" in event or "end" in event):
                        print(f"[Frame {self.FRAME_IDX}] VAD Event: {event}, STATE: {self.STATE}")
                    if self.STATE == "LISTEN":
                        await self.handle_listen(frame, event)
                    elif self.STATE == "SPEAK":
                        await self.handle_speak(frame, event)

        except WebSocketDisconnect:
            print("⚠️ WebSocket 断开，自动清理状态")
        except Exception as e:
            print(f"❌ 实时运行出错: {e}")
            import traceback
            traceback.print_exc()  # 👈 强烈建议加上这行！
        finally:
            self.vad_iterator.reset_states()
            self.reset()
            print("会话结束，状态已清理")


# ============================================================
# FastAPI 应用
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI()
    @app.websocket("/realtime")
    async def realtime_ws(websocket: WebSocket):
        engine = ConversationEngine(websocket=websocket)
        await engine.run_realtime(websocket)
    return app


def main():
    parser = argparse.ArgumentParser(description="Realtime Voice WS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=18010, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable reload (dev)")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()