import os, wave, json, time, torch, soundfile as sf, numpy as np
from pathlib import Path
from tqdm import tqdm
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts, get_wav, api_qwen3o
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
        self.FILE_NAME = None
        #end后延续一下，不要立刻判断
        self.END_HOLD_SEC = 0.64
        self.END_HOLD_FRAMES = int(self.END_HOLD_SEC / self.FRAME_SEC) #0.64 / 0.016 = 40
        self.SILENCE_COUNTER = 0
        #打断后处理
        self.FROM_INTERRUPT = False
        #历史上下文
        self.history = []


        # ========== 路径与模型接口 ==========
        self.output_dir = None
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)


    def reset(self):
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        self.MEDIA_TIME = 0.0
        self.FRAME_IDX = 0
        self.CURRENT_TURN = None
        self.BUFFER.clear()
        self.history.clear()

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
        basename = self.output_dir.name
        turn_id = self.TURN_IDX
        #  # ========== ASR ==========
        # asr_start = time.perf_counter()
        # asr_text = asr(audio_buf)
        # asr_time = time.perf_counter() - asr_start
        # # ========== LLM ==========
        # llm_start = time.perf_counter()
        # decision = llm(asr_text)  # {"is_finished": bool, "reply": "..."}
        # llm_time = time.perf_counter() - llm_start

        # ========== api ==========
        api_start = time.perf_counter()
        # 拼接音频帧
        user_audio = np.concatenate(audio_buf) if isinstance(audio_buf, list) else audio_buf
        history_text = ""
        for turn in self.history[-3:]:  # 保留最近5轮，防止prompt太长
            role = "用户" if turn["role"] == "user" else "助手"
            history_text += f"{role}:{turn['content']}\n"
        listen_prompt = f'''
            首先，如果你认为用户这句话明显没有说完，请只输出字符串'continue'。然后给出这段音频的asr转录,如果你认为用户明显说完：
            你是一个自然聊天的语音助手，要像朋友一样回答用户的问题。
            不要反问，也不要解释，不要输出任何格式说明。
            如果用户问到的内容需要你编造，比如不知道答案，也要自然地编造一个合理的回答。
            语气要口语化、简短，像日常聊天。允许出现幻觉。

            以下是一些示例：
            用户：我上次吃的那家火锅店叫什么来着？
            助手：老灶火锅呀。

            以下是部分历史对话：
            {history_text}

            现在，请继续回应用户的最新语音：
        '''
        
        decision = llm_qwen3o(listen_prompt, user_audio)
    

        # ✅ 写入临时 wav 再喂给 asr
        user_audio = np.concatenate(audio_buf) if isinstance(audio_buf, list) else audio_buf
        tmp_path = self.output_dir / f"{self.FILE_NAME}_turn{self.TURN_IDX}_input.wav"
        sf.write(tmp_path, user_audio, self.SAMPLE_RATE)
        user_text = asr(str(tmp_path)) # if "asr" in globals() else "<user audio>"
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": decision})

        print(f"决策结果: {decision}")
        # exit(0)
        api_time = time.perf_counter() - api_start        
        if ("continue" not in decision.lower()) and ("继续" not in decision.lower()):

            #========== TTS ==========
            tts_path = self.output_dir / f"{basename}_r{turn_id}.wav"
            tts_start = time.perf_counter()
            tts_file = tts(decision, tts_path)
            tts_time = time.perf_counter() - tts_start

            audio_data, sr = sf.read(tts_file)
            tts_dur = len(audio_data) / sr
            # sys_start = self.MEDIA_TIME + asr_time + llm_time + tts_time
            sys_start = self.MEDIA_TIME + api_time + tts_time

            self.CURRENT_TURN = {
                "turn": turn_id,
                "user_end": round(self.MEDIA_TIME, 3),
                # "asr_time": round(asr_time, 3),
                # "llm_time": round(llm_time, 3),
                "api_time": round(api_time, 3),
                "tts_time": round(tts_time, 3),
                "tts_dur": round(tts_dur, 3),
                "sys_start": round(sys_start, 3),
                "tts_file": Path(tts_file).name
            }
            # self.write_turn()
            self.TURN_IDX += 1
            # print(self.CURRENT_TURN)
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
                    self.process_user_segment(self.BUFFER)
                    return

    def handle_speak(self, frame, event):
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
                            self.CURRENT_TURN.setdefault("speak", {})["interrupt_time"] = round(self.MEDIA_TIME, 2)
                            self.write_turn()
                            # self.STATE = "LISTEN"
                            self.BUFFER = self.interrupt_buf.copy()
                            print("🔁 检测到短打断，启动新一轮 listen→speak")
                            self.process_user_segment(self.BUFFER)

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
        self.FILE_NAME = audio_path.stem
        print(f"Processing file: {self.FILE_NAME}")
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
        self.vad_iterator.reset_states()

    # -------------------------------------------------------
    def write_turn(self):
        """将当前 turn 写入 JSONL"""
        jsonl_path = self.output_dir / f"{self.FILE_NAME}_r.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.CURRENT_TURN, ensure_ascii=False) + "\n")
        self.CURRENT_TURN = None


import shutil
import json
import numpy as np
import soundfile as sf

def process_folder(folder, save_root):
    """
    处理单个对话文件夹：
    - 如果文件夹以 clean_ 开头：直接把 *_r1.wav 改名为 *_output.wav 复制到 save_root。
    - 否则按 JSONL 拼接多个 TTS wav 生成 output.wav。
    """
    # --- 情况 1：以 clean_ 开头，直接复制 ---
    if folder.name.lower().startswith("clean_"):
        # 找出 _r1.wav 文件
        r1_files = list(folder.glob("*_r0.wav"))
        if not r1_files:
            print(f"未找到 {folder} 下的 *_r0.wav 文件")
            return
        
        src = r1_files[0]
        dst = save_root / f"{folder.name}_output.wav"
        save_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"复制clean音频: {src.name} → {dst.name}")
        return

    # --- 情况 2：普通文件夹，按 JSONL 拼接 ---
    jsonl = folder / f"{folder.name}_r.jsonl"
    if not jsonl.exists():
        print(f"JSONL 不存在: {jsonl}")
        return

    data = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    data.sort(key=lambda x: x["sys_start"])

    segs, cur_len, sr = [], 0, 16000
    for s in data:
        wav, sr = sf.read(folder / s["tts_file"])
        if wav.ndim > 1:
            wav = wav[:, 0]
        pad = int(max(0.0, s["sys_start"] - cur_len) * sr)
        if pad > 0:
            segs.append(np.zeros(pad, dtype=wav.dtype))
            cur_len += pad / sr

        cut = None
        if "interrupt_time" in s and s["sys_start"] + s["tts_dur"] > s["interrupt_time"]:
            cut = int(max(0.0, s["interrupt_time"] - s["sys_start"]) * sr)
        if cut is not None:
            wav = wav[:cut]

        segs.append(wav)
        cur_len += len(wav) / sr

    if segs:
        save_root.mkdir(parents=True, exist_ok=True)
        out = save_root / f"{folder.name}_output.wav"
        sf.write(out, np.concatenate(segs), sr)
        print(f"拼接生成: {out}")


def main():
    exp_name = "exp1"
    data_lang = "dev_zh"
    out_lang = "medium_zh"
    category_dev = ["Pause Handling"]

    data_root = Path("exp") / exp_name / "dev" / data_lang
    output_root = Path("exp") / exp_name / "medium" / out_lang

    if output_root.exists():
        for category in category_dev:
            cat_dir = output_root / category
            if cat_dir.exists():
                shutil.rmtree(cat_dir)  # 直接删除整个类别目录
                print(f"⚠️ 清空输出子目录: {cat_dir}")

    else:
        output_root.mkdir(parents=True, exist_ok=True)
        
    engine = ConversationEngine()

    for category in category_dev:
        wav_path = data_root / category         # 原始数据区
        out_path = output_root / category       # 生成区
        out_path.mkdir(parents=True, exist_ok=True)

        wav_files = get_wav(wav_path, "all")

        for wav in tqdm(wav_files, desc=f"Processing {category}"):
            wav_file = wav_path / wav
            # if wav_file.stem != "0005_0019_add":
            #     continue
            output_dir = out_path / wav_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            engine.reset()      
            engine.run(wav_file, output_dir)

            jsonl_path = output_dir / f"{wav_file.stem}_r.jsonl"
            if not jsonl_path.exists():
                raise RuntimeError(f"未生成结果文件: {jsonl_path}")

            lines = [l.strip() for l in open(jsonl_path, "r", encoding="utf-8") if l.strip()]
            print(f"{wav_file.name} 完成, 共 {len(lines)} 轮对话")

            # 输出结果写回原始数据目录（你的评测区）
            process_folder(output_dir, wav_path)

if __name__ == "__main__":
    main()
