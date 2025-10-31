import os, wave, json, time, torch, soundfile as sf, numpy as np
from pathlib import Path
from tqdm import tqdm
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts, get_wav
class ConversationEngine:
    """
    Full-duplex Conversation Engine
    实现 LISTEN → SPEAK 循环的基础框架
    每一轮输出 listen/speak 两个子块
    """

    def __init__(self, sample_rate=16000, window_size=256):
        # ========== 常量 ==========
        self.SAMPLE_RATE = sample_rate #16kHz
        self.WINDOW_SIZE = window_size #256 samples (~16ms)
        self.FRAME_SEC = window_size / sample_rate # 256 / 16000 = 0.016s
        self.INTERRUPT_LIMIT = int(1.5 / self.FRAME_SEC) #1.5 / 0.016 = 94 frames

        # ========== 状态变量 ==========
        self.STATE = "LISTEN"       # 当前状态
        self.IN_SPEECH = False      # 当前是否处于一段语音中
        self.BUFFER = []            # 累积帧缓冲
        self.TURN_IDX = 0           # 全局轮数
        self.MEDIA_TIME = 0.0       # 累计音频时间（秒）
        self.FRAME_IDX = 0          # 帧计数
        self.CURRENT_TURN = None    # 当前轮的 listen + speak 数据
        self.SILENCE_COUNTER = 0    # 静音计数器
        self.INTERRUPT_COUNT = 0    # 打断帧计数

        # ========== 路径与模型接口 ==========
        self.output_dir = None
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)

    # -------------------------------------------------------
    def stream_audio(self, audio_path):
        """逐帧读取音频流（16ms一帧）"""
        with wave.open(str(audio_path), "rb") as wf:
            while True:
                data = wf.readframes(self.WINDOW_SIZE)
                if not data:
                    break
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(chunk) < self.WINDOW_SIZE:
                    break
                yield chunk

    # -------------------------------------------------------
    def detect_vad_frame(self, chunk):
        """VAD 检测函数（返回 start / end / None）"""
        if not hasattr(self, "_vad_buf"):
            self._vad_buf = np.zeros(0, dtype=np.float32)
        self._vad_buf = np.concatenate([self._vad_buf, chunk])
        if len(self._vad_buf) >= 2 * self.WINDOW_SIZE:
            tensor = torch.from_numpy(self._vad_buf[: 2 * self.WINDOW_SIZE])
            event = self.vad_iterator(tensor, return_seconds=True)
            self._vad_buf = np.zeros(0, dtype=np.float32)
            return event
        return None


    def process_user_segment(self, audio_buf):
        """对一段完整的用户语音执行 ASR→LLM→TTS 并写入结果"""
        asr_start = time.perf_counter()
        asr_text = asr(audio_buf)
        asr_time = time.perf_counter() - asr_start

        llm_start = time.perf_counter()
        decision = llm(asr_text)  # {"is_finished": bool, "reply": "..."}
        llm_time = time.perf_counter() - llm_start

        if decision.get("is_finished", True):
            # ---- 正常进入TTS阶段 ----
            tts_start = time.perf_counter()
            tts_file = tts(decision.get("reply", ""))
            tts_time = time.perf_counter() - tts_start

            audio_data, sr = sf.read(tts_file)
            tts_dur = len(audio_data) / sr
            sys_start = self.MEDIA_TIME + asr_time + llm_time + tts_time

            self.CURRENT_TURN = {
                "turn": self.TURN_IDX + 1,
                "user_end": round(self.MEDIA_TIME, 2),
                "asr_time": round(asr_time, 2),
                "llm_time": round(llm_time, 2),
                "tts_time": round(tts_time, 2),
                "tts_dur": round(tts_dur, 2),
                "sys_start": round(sys_start, 2),
                "tts_file": Path(tts_file).name
            }

            self.STATE = "SPEAK"
            self.IN_SPEECH = False
            self.BUFFER.clear()
        else:
            # ---- 未结束，继续监听 ----
            self.STATE = "LISTEN"
            self.IN_SPEECH = True


    # -------------------------------------------------------
    def handle_listen(self, frame, event):
        """LISTEN 状态：检测用户语音、判断是否说完、决定是否进入 SPEAK"""

        # --- 特殊入口：来自短打断 ---
        if self.BUFFER and not self.IN_SPEECH:
            self.process_user_segment(self.BUFFER)
            return

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
            self.process_user_segment(self.BUFFER)
            return


    def handle_speak(self, frame, event):
        """SPEAK：系统说话中。检测用户短/长打断：
        - 短打断：<1.5s 且出现 end → ASR+LLM 判定；interrupt=True 才切 LISTEN
        - 长打断：≥1.5s，无需 end → 直接切 LISTEN
        """

        # 1) 首次检测到用户开口：开始累计打断缓冲
        if event and "start" in event and not self.IN_SPEECH:
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.interrupt_start_time = self.MEDIA_TIME
            self.INTERRUPT_COUNT = 1
            return

        # 2) 正在累计可能的打断片段
        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            # 2.1 短打断：在达到 1.5s 之前出现了 end → 做一次语义判定
            if event and "end" in event and self.INTERRUPT_COUNT < self.INTERRUPT_LIMIT:
                seg_audio = np.concatenate(self.interrupt_buf)
                seg_text  = asr(seg_audio)
                intent    = llm(seg_text)   # 期望 {"interrupt": bool}

                if intent.get("interrupt", False):
                    # —— 真正打断：记录时间，写入本轮，切 LISTEN，并把这段语音交给下一轮
                    self.CURRENT_TURN.setdefault("speak", {})["interrupt_time"] = round(self.MEDIA_TIME, 2)
                    self.write_turn()
                    self.STATE = "LISTEN"
                    self.BUFFER = self.interrupt_buf.copy()  # 种子给下一轮，避免丢帧
                    self.IN_SPEECH = True
                else:
                    # —— 只是backchannel/鼓励继续：忽略，留在 SPEAK
                    self.IN_SPEECH = False

                # 清理本段缓存
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                return

            # 2.2 长打断：累计达到/超过 1.5s，无需等待 end，直接切 LISTEN
            if self.INTERRUPT_COUNT >= self.INTERRUPT_LIMIT:
                self.CURRENT_TURN.setdefault("speak", {})["interrupt_time"] = round(self.MEDIA_TIME, 2)
                self.write_turn()
                self.STATE = "LISTEN"
                self.BUFFER = self.interrupt_buf.copy()  # 交给下一轮继续累计
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                return

        # 3) 未检测到用户发声：SPEAK 持续（离线不模拟播放结束）
        return
    # -------------------------------------------------------
    def run(self, audio_path, output_dir):
        """主循环：逐帧执行"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for frame in self.stream_audio(audio_path):
            event = self.detect_vad_frame(frame)
            self.MEDIA_TIME = self.FRAME_IDX * self.FRAME_SEC
            self.FRAME_IDX += 1

            if self.STATE == "LISTEN":
                self.handle_listen(frame, event)
            elif self.STATE == "SPEAK":
                self.handle_speak(frame, event)

        # 收尾：最后一轮未写入时写入
        if self.CURRENT_TURN is not None:
            self.write_turn()

    # -------------------------------------------------------
    def write_turn(self):
        """将当前 turn 写入 JSONL"""
        jsonl_path = self.output_dir / f"turns.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.CURRENT_TURN, ensure_ascii=False) + "\n")
        self.CURRENT_TURN = None

def main():
    exp_name = "exp1"
    data_lang = "dev_zh"
    out_lang = "medium_zh"
    category_dev = ["Follow-up Questions"]

    data_root = Path("exp") / exp_name / "dev" / data_lang
    output_root = Path("exp") / exp_name / "medium" / out_lang
    for category in category_dev:
        wav_path = data_root / category
        medium_path = output_root / category
        medium_path.mkdir(parents=True, exist_ok=True)
        wav_files = get_wav(wav_path)
        for wav in tqdm(wav_files, desc=category):
            wav_file = wav_path / wav
            output_dir = medium_path / wav_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            stream_vad_asr_pipeline(wav_file, output_dir)
            wav_files_in_dir = list(output_dir.glob("*.wav"))
            json_files_in_dir = list(output_dir.glob("*.jsonl"))
            if not (len(wav_files_in_dir) == 4 and len(json_files_in_dir) == 1):
                raise RuntimeError(f"输出数量错误: {output_dir} 中找到 {len(wav_files_in_dir)} 个wav, {len(json_files_in_dir)} 个json")
            process_folder(output_dir, wav_path)

if __name__ == "__main__":
    main()
