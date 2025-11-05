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

    def reset(self):
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        self.MEDIA_TIME = 0.0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.BUFFER.clear()
        self.history.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)


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

    # -------------------------------------------------------
    async def handle_listen(self, frame, event):
        """LISTEN 状态：检测用户语音、判断是否说完、决定是否进入 SPEAK"""
        # --- 1. 用户开始说话 ---
        if event and "start" in event and not self.IN_SPEECH:
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
            # self.process_user_segment(self.BUFFER)
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
                    user_audio = np.concatenate(self.BUFFER)
                    judge_prompt = (
                        "你需要从日常对话的角度,而不是语法的角度去判断这句话是否说完了。"
                        "首先，如果你认为用户这句话明显没有说完，请只输出字符串'continue'。"
                        "如果你认为用户已经说完，请只输出字符串'end'。不要输出其他内容。"
                    )

                    start_time = time.time()
                    judge_result = llm_qwen3o(judge_prompt, user_audio).strip().lower()
                    infer_time = time.time() - start_time
                    self.CONTINUE_INFER_TIMES.append(infer_time)

                    print(f"用户语音完整性判定: {judge_result}")
                    if "continue" in judge_result:
                        self.CONTINUE_ARMED = True
                        prefill = int(round(infer_time / self.FRAME_SEC))
                        self.AFTER_CONTINUE_SILENT_FRAMES = prefill
                        print("🔁 用户未说完，继续累积帧")
                        self.IN_SPEECH = True
                        return

                    # === 说完了，进入完整流程 ===
                    await self.process_user_segment(self.BUFFER)
                    return

        # --- 仅在上一次判定为 continue 且已武装时才计数 ---
        if self.CONTINUE_ARMED:
            if not event:
                # 无事件帧：累加空白帧
                self.AFTER_CONTINUE_SILENT_FRAMES += 1

                # 计算触发阈值 = 2s 对应帧数 + 本次推理耗时折算的帧数
                last_infer = self.CONTINUE_INFER_TIMES[-1] if self.CONTINUE_INFER_TIMES else 0.0
                infer_frames = int(round(last_infer / self.FRAME_SEC))
                trigger_frames = self.AFTER_CONTINUE_TIMEOUT_FRAMES + infer_frames

                # 满足条件：空白帧数超过 (2s + 推理耗时)
                if self.AFTER_CONTINUE_SILENT_FRAMES >= trigger_frames:
                    print(
                        f"⚠️ continue 后空白累计 {self.AFTER_CONTINUE_SILENT_FRAMES} 帧 "
                        f"(阈值 {trigger_frames} 帧 ≈ 2s+{last_infer:.3f}s)，强制处理"
                    )
                    await self.process_user_segment(self.BUFFER)
                    # 清理状态
                    self.CONTINUE_INFER_TIMES.clear()
                    self.CONTINUE_ARMED = False
                    self.AFTER_CONTINUE_SILENT_FRAMES = 0
                    self.IN_SPEECH = False
                    self.BUFFER.clear()
                    return
            else:
                # 任意事件（start / end）打断空白 → 解除武装并清零
                self.CONTINUE_ARMED = False
                self.AFTER_CONTINUE_SILENT_FRAMES = 0


    async def handle_speak(self, frame, event):
        """SPEAK 状态：检测短打断或长打断"""
        if event and "start" in event and not self.IN_SPEECH:
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.INTERRUPT_COUNT = 1
            self.SILENCE_COUNTER = 0
            return

        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            # --- 检测到用户结束讲话 ---
            if event and "end" in event:
                self.SILENCE_COUNTER = 1
                return

            # --- 静音确认阶段（640 ms 延迟）---
            if self.SILENCE_COUNTER > 0:
                if event and "start" in event:
                    # 640ms 内出现新语音 → 继续接上
                    self.SILENCE_COUNTER = 0
                    return
                else:
                    self.SILENCE_COUNTER += 1
                    if self.SILENCE_COUNTER >= self.END_HOLD_FRAMES:
                        # ✅ 确认打断结束
                        seg_audio = np.concatenate(self.interrupt_buf)
                        speak_prompt = (
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
                        intent = llm_qwen3o(speak_prompt, seg_audio)
                        print(f"出现了打断，打断意图判定: {intent}")

                        if "interrupt" in intent.lower():
                            # ✅【前端信号】短打断：应立即停止当前 TTS 播放
                            await self.send_control("stop_audio", {"reason": "short_semantic_interrupt"})

                            # 这一句应该传递到前端
                            self.CURRENT_TURN.setdefault("speak", {})["interrupt_time"] = round(self.MEDIA_TIME, 2)
                            # 不需要写json，而是发送到前端
                            # self.STATE = "LISTEN"
                            self.BUFFER = self.interrupt_buf.copy()
                            print("🔁 检测到短打断，启动新一轮 listen→speak")
                            await self.process_user_segment(self.BUFFER)

                            # 清理状态
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

                        else:
                            # ❌ backchannel / 继续说：忽略打断，保持SPEAK
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            # 不切换 state，继续 SPEAK
                            return
            # 2.2 长打断：累计达到/超过 1.5s，无需等待 end，直接切 LISTEN
            if self.SILENCE_COUNTER == 0 and self.INTERRUPT_COUNT >= self.INTERRUPT_LIMIT:
                print("✅出现长打断，切换到listen继续听，正确开启第二轮")
                # ✅【前端信号】长打断：应立即停止当前 TTS 播放
                await self.send_control("stop_audio", {"reason": "long_interrupt"})

                self.CURRENT_TURN.setdefault("speak", {})["interrupt_time"] = round(self.MEDIA_TIME, 2)
                self.STATE = "LISTEN"
                self.BUFFER = self.interrupt_buf.copy()  # 交给下一轮继续累计
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                return

        # 3) 未检测到用户发声：SPEAK 持续（离线不模拟播放结束）
        return


    async def run_realtime(self, websocket: WebSocket):
        await websocket.accept()
        print("✅ 前端已连接，进入实时会话循环")

        self.MEDIA_TIME = 0.0
        self.FRAME_IDX = 0

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
                        if self.IN_SPEECH and self.BUFFER:
                            await self.process_user_segment(self.BUFFER)
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

                    self.MEDIA_TIME += self.FRAME_SEC
                    self.FRAME_IDX += 1

                    event = self.detect_vad_frame(frame)
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
    parser.add_argument("--port", type=int, default=18000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable reload (dev)")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()