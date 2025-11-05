import os, json, asyncio, wave, time, torch, soundfile as sf, numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts

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
        self.MEDIA_TIME = 0.0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.INTERRUPT_COUNT = 0
        self.FILE_NAME = "stream"
        self.END_HOLD_FRAMES = int(0.64 / self.FRAME_SEC)
        self.SILENCE_COUNTER = 0
        self.CONTINUE_INFER_TIMES = []
        self.AFTER_CONTINUE_SILENT_FRAMES = 0
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = max(1, int(round(2.0 / self.FRAME_SEC)))
        self.CONTINUE_ARMED = False
        self.FROM_INTERRUPT = False
        self.history = []

        self.output_dir = Path("realtime_out")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)
        self.websocket = websocket  # ✅ 保存 websocket 引用用于事件推送

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

    async def process_user_segment(self, audio_buf):
        basename = self.output_dir.name
        turn_id = self.TURN_IDX

        
        user_audio = np.concatenate(audio_buf) if isinstance(audio_buf, list) else audio_buf

        # --- 构建上下文 prompt ---
        history_text = ""
        for turn in self.history[-3:]:
            role = "用户" if turn["role"] == "user" else "助手"
            history_text += f"{role}:{turn['content']}\n"
        listen_prompt = f'''
        你是一个自然聊天的语音助手，要像朋友一样回答用户的问题。
        不要反问，也不要解释，不要输出任何格式说明。
        {history_text}
        现在，请继续回应用户的最新语音：
        '''
        api_start = time.perf_counter()
        decision = llm_qwen3o(listen_prompt, user_audio)
        api_time = time.perf_counter() - api_start

        # --- ASR ---
        tmp_path = self.output_dir / f"{self.FILE_NAME}_turn{self.TURN_IDX}_input.wav"
        sf.write(tmp_path, user_audio, self.SAMPLE_RATE)
        user_text = asr(str(tmp_path))
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": decision})
        print(f"决策结果: {decision}")

        
        if ("continue" not in decision.lower()) and ("继续" not in decision.lower()):
            tts_path = self.output_dir / f"{basename}_r{turn_id}.wav"
            tts_start = time.perf_counter()
            tts_file = tts(decision, tts_path)
            tts_time = time.perf_counter() - tts_start

            # ✅ 将 TTS 音频实时发送到前端
            with open(tts_file, "rb") as f:
                await self.websocket.send_bytes(f.read())
            await self.send_control("new_tts", {"turn": turn_id})

            audio_data, sr = sf.read(tts_file)
            tts_dur = len(audio_data) / sr
            sys_start = self.MEDIA_TIME + api_time + tts_time
            self.CURRENT_TURN = {
                "turn": turn_id,
                "user_end": round(self.MEDIA_TIME, 3),
                "api_time": round(api_time, 3),
                "tts_time": round(tts_time, 3),
                "tts_dur": round(tts_dur, 3),
                "sys_start": round(sys_start, 3),
                "tts_file": Path(tts_file).name
            }
            self.TURN_IDX += 1
            self.STATE = "SPEAK"
            self.IN_SPEECH = False
            self.BUFFER.clear()
        else:
            self.STATE = "LISTEN"
            self.IN_SPEECH = True

    async def handle_speak(self, frame, event):
        """在 SPEAK 状态检测打断"""
        if event and "start" in event and not self.IN_SPEECH:
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.INTERRUPT_COUNT = 1
            self.SILENCE_COUNTER = 0
            return

        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            # 长打断：直接通知前端停止播放
            if self.SILENCE_COUNTER == 0 and self.INTERRUPT_COUNT >= self.INTERRUPT_LIMIT:
                print("✅出现长打断，切换到listen继续听")
                await self.send_control("stop_audio", {"reason": "long_interrupt"})
                self.STATE = "LISTEN"
                self.BUFFER = self.interrupt_buf.copy()
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                return

    async def handle_listen(self, frame, event):
        """保留原 LISTEN 逻辑，只在检测到结束后调用异步 process"""
        if event and "start" in event and not self.IN_SPEECH:
            self.IN_SPEECH = True
            self.BUFFER = [frame]
            return
        if not self.IN_SPEECH:
            return
        self.BUFFER.append(frame)
        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            return
        if self.SILENCE_COUNTER > 0:
            self.SILENCE_COUNTER += 1
            if self.SILENCE_COUNTER >= self.END_HOLD_FRAMES:
                self.SILENCE_COUNTER = 0
                await self.process_user_segment(self.BUFFER)
                return

    async def run_realtime(self, websocket: WebSocket):
        """核心实时循环：接收帧并驱动状态机"""
        self.websocket = websocket
        await websocket.accept()
        print("✅ WebSocket 已连接，开始接收音频流")
        try:
            async for message in websocket.iter_bytes():
                frame = np.frombuffer(message, dtype=np.float32)
                event = self.detect_vad_frame(frame)
                self.MEDIA_TIME = self.FRAME_IDX * self.FRAME_SEC
                self.FRAME_IDX += 1

                if self.STATE == "LISTEN":
                    await self.handle_listen(frame, event)
                elif self.STATE == "SPEAK":
                    await self.handle_speak(frame, event)
        except WebSocketDisconnect:
            print("❌ WebSocket 断开")
        finally:
            self.vad_iterator.reset_states()

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI()

@app.websocket("/realtime")
async def websocket_endpoint(websocket: WebSocket):
    engine = ConversationEngine(websocket=websocket)
    await engine.run_realtime(websocket)

@app.get("/")
async def index():
    html = """
    <html><body>
    <h3>WebSocket 语音对话接口已启动</h3>
    </body></html>
    """
    return HTMLResponse(html)
