import os
import gc
import re
import configparser
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from faster_whisper import WhisperModel, BatchedInferencePipeline
import opencc
import sys

converter = opencc.OpenCC("t2s")
if getattr(sys, "frozen", False):
    # 打包后：exe 所在目录（models 和 exe 同级）
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 开发时：main.py 所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

MODEL_CHOICES = {
    "small（快，准确率一般）": "faster-whisper-small",
    "medium（较慢，中文更准）": "faster-whisper-medium",
}

SIMPLE_PROMPT = "以下是普通话的句子，请使用简体中文和正确的标点符号。"


class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Faster-Whisper CPU 转录工具")
        self.root.geometry("640x600")
        self.root.resizable(False, False)

        self.model = None
        self._model_lock = threading.Lock()
        self.audio_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.mode = tk.StringVar(value="均衡")
        self.with_timestamp = tk.BooleanVar(value=False)  # 默认不输出时间戳
        self.status = tk.StringVar(value="就绪")
        self.model_path = tk.StringVar()

        self._build_ui()

        # 主窗口置底，弹窗置顶选择
        self.root.lower()
        chosen = self._ask_model()
        if not chosen:
            self.root.destroy()
            return

        if not self._apply_model_choice(chosen):
            self.root.destroy()
            return

        # 选完模型后恢复主窗口
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    # ---------- 配置读写 ----------
    def _load_model_choice(self):
        if not os.path.isfile(CONFIG_PATH):
            return None
        cfg = configparser.ConfigParser()
        try:
            cfg.read(CONFIG_PATH, encoding="utf-8")
            choice = cfg.get("model", "last", fallback=None)
            if choice in MODEL_CHOICES.values():
                return choice
        except Exception:
            pass
        return None

    def _save_model_choice(self, folder_name):
        cfg = configparser.ConfigParser()
        cfg["model"] = {"last": folder_name}
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                cfg.write(f)
        except Exception:
            pass

    # ---------- 应用模型选择 ----------
    def _apply_model_choice(self, folder_name):
        """切换模型：检查目录、更新路径、写配置、重置指示条、后台重新加载"""
        full = os.path.join(BASE_DIR, "models", folder_name)
        if not os.path.isdir(full):
            messagebox.showerror(
                "模型不存在",
                f"未找到模型目录：\n{full}\n\n请先放入模型文件。",
            )
            return False

        self.model_path.set(full)
        self._save_model_choice(folder_name)
        self._set_indicator(False)

        # 释放旧模型
        with self._model_lock:
            if self.model is not None:
                del self.model
                self.model = None
        gc.collect()

        threading.Thread(target=self._preload_model, daemon=True).start()
        return True

    # ---------- 启动选择弹窗 ----------
    def _ask_model(self):
        last = self._load_model_choice()

        dlg = tk.Toplevel(self.root)
        dlg.title("选择模型")
        w, h = 360, 220
        dlg.resizable(False, False)
        dlg.geometry(f"{w}x{h}")
        dlg.update_idletasks()
        x = (dlg.winfo_screenwidth() - w) // 2
        y = (dlg.winfo_screenheight() - h) // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        ttk.Label(dlg, text="请选择要加载的模型：", font=("", 10, "bold")).pack(
            anchor="w", padx=16, pady=(16, 8)
        )

        default_label = list(MODEL_CHOICES.keys())[0]
        if last:
            for label, folder in MODEL_CHOICES.items():
                if folder == last:
                    default_label = label
                    break
        choice = tk.StringVar(value=default_label)

        for label in MODEL_CHOICES:
            ttk.Radiobutton(dlg, text=label, value=label, variable=choice).pack(
                anchor="w", padx=24, pady=2
            )

        result = {"value": None}

        def on_ok():
            result["value"] = MODEL_CHOICES[choice.get()]
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        frm_btn = ttk.Frame(dlg)
        frm_btn.pack(fill="x", padx=16, pady=16)
        ttk.Button(frm_btn, text="取消", command=on_cancel).pack(side="right")
        ttk.Button(frm_btn, text="确定", command=on_ok).pack(side="right", padx=(8, 0))
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        dlg.update()
        dlg.deiconify()
        dlg.lift()
        dlg.focus_force()
        dlg.grab_set()

        self.root.wait_window(dlg)
        return result["value"]

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm_indicator = ttk.Frame(self.root)
        frm_indicator.pack(fill="x", padx=10, pady=(10, 0))

        self.indicator_canvas = tk.Canvas(
            frm_indicator, width=16, height=16, highlightthickness=0
        )
        self.indicator_canvas.pack(side="left", padx=(0, 6))
        self._dot = self.indicator_canvas.create_oval(
            3, 3, 13, 13, fill="#f0c000", outline=""
        )

        self.indicator_label = tk.Label(
            frm_indicator, text="加载模型", anchor="w", fg="#333", cursor="hand2"
        )
        self.indicator_label.pack(side="left", fill="x", expand=True)
        self.indicator_label.bind("<Button-1>", lambda e: self._on_indicator_click())
        self.indicator_canvas.bind("<Button-1>", lambda e: self._on_indicator_click())

        frm_audio = ttk.LabelFrame(self.root, text="音频文件")
        frm_audio.pack(fill="x", **pad)
        ttk.Entry(frm_audio, textvariable=self.audio_path).pack(
            side="left", fill="x", expand=True, padx=(10, 5), pady=8
        )
        ttk.Button(frm_audio, text="选择...", command=self.choose_audio).pack(
            side="right", padx=(0, 10), pady=8
        )

        frm_output = ttk.LabelFrame(self.root, text="输出 TXT 位置")
        frm_output.pack(fill="x", **pad)
        ttk.Entry(frm_output, textvariable=self.output_path).pack(
            side="left", fill="x", expand=True, padx=(10, 5), pady=8
        )
        ttk.Button(frm_output, text="选择...", command=self.choose_output).pack(
            side="right", padx=(0, 10), pady=8
        )

        frm_mode = ttk.LabelFrame(self.root, text="推理模式")
        frm_mode.pack(fill="x", **pad)
        for text, value in [
            ("均衡", "均衡"),
            ("高性能（内存 ≥16GB）", "高速"),
            ("节能（省内存，效果差）", "低速"),
        ]:
            ttk.Radiobutton(frm_mode, text=text, value=value, variable=self.mode).pack(
                anchor="w", padx=10, pady=2
            )

        frm_ts = ttk.LabelFrame(self.root, text="输出选项")
        frm_ts.pack(fill="x", **pad)
        ttk.Checkbutton(frm_ts, text="输出带时间戳", variable=self.with_timestamp).pack(
            anchor="w", padx=10, pady=8
        )

        frm_log = ttk.LabelFrame(self.root, text="转录文本（实时）")
        frm_log.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(frm_log, height=10, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(8, 8))

        self.btn_start = ttk.Button(
            self.root, text="开始转录", command=self.start_transcribe
        )
        self.btn_start.pack(fill="x", padx=10, pady=8)

        ttk.Label(self.root, textvariable=self.status, foreground="gray").pack(
            anchor="w", padx=10, pady=(0, 8)
        )

    def _on_indicator_click(self):
        """点击指示条重新选模型"""
        chosen = self._ask_model()
        if chosen:
            self._apply_model_choice(chosen)

    def _set_indicator(self, loaded, model_name=None):
        """loaded=True 绿色+已加载（可带模型名），False 黄色+加载模型"""
        def _do():
            if loaded:
                self.indicator_canvas.itemconfigure(self._dot, fill="#2ea44f")
                text = f"模型已加载：{model_name}" if model_name else "模型已加载"
                self.indicator_label.configure(text=text, fg="#2ea44f", cursor="hand2")
            else:
                self.indicator_canvas.itemconfigure(self._dot, fill="#f0c000")
                self.indicator_label.configure(text="加载模型", fg="#333", cursor="hand2")
        self.root.after(0, _do)

    def choose_audio(self):
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.audio_path.set(path)
            if not self.output_path.get():
                base = os.path.splitext(os.path.basename(path))[0]
                self.output_path.set(os.path.join(os.path.dirname(path), base + ".txt"))

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="保存 TXT 到",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt")],
        )
        if path:
            self.output_path.set(path)

    # ---------- 模型加载 ----------
    def load_model(self):
        with self._model_lock:
            if self.model is None:
                if not self.model_path.get():
                    raise RuntimeError("未选择模型目录")
                self.status.set("正在加载模型...")
                self.model = WhisperModel(
                    self.model_path.get(), device="cpu", compute_type="int8"
                )

    def _preload_model(self):
        try:
            self.load_model()
            folder = os.path.basename(self.model_path.get())
            self._set_indicator(True, folder)
            self.root.after(0, lambda: self.status.set("模型已就绪"))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: self.status.set("模型加载失败：" + err_msg))

    def _append_log(self, text):
        def _do():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(0, _do)

    # ---------- 拆句 ----------
    @staticmethod
    def _split_sentences(text):
        """按中文句末标点拆句，保留标点，返回非空句子列表"""
        parts = re.split(r"(?<=[。！？；])", text)
        return [p.strip() for p in parts if p.strip()]

    def start_transcribe(self):
        audio = self.audio_path.get().strip()
        output = self.output_path.get().strip()

        if not audio or not os.path.isfile(audio):
            messagebox.showwarning("提示", "请选择有效的音频文件。")
            return
        if not output:
            messagebox.showwarning("提示", "请选择输出 TXT 的位置。")
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self.btn_start.configure(state="disabled", text="文本转录中...")
        self.status.set("正在处理，请稍候...")
        threading.Thread(
            target=self._transcribe_worker, args=(audio, output), daemon=True
        ).start()

    def _transcribe_worker(self, audio, output):
        try:
            self.load_model()
            mode = self.mode.get()

            if mode == "均衡":
                segments, info = self.model.transcribe(
                    audio, beam_size=5, language="zh",
                    vad_filter=False, initial_prompt=SIMPLE_PROMPT,
                )
            elif mode == "高速":
                batched = BatchedInferencePipeline(model=self.model)
                segments, info = batched.transcribe(
                    audio, batch_size=8, beam_size=5, language="zh",
                    vad_filter=False, initial_prompt=SIMPLE_PROMPT,
                )
            else:
                segments, info = self.model.transcribe(
                    audio, beam_size=1, language="zh",
                    vad_filter=False, initial_prompt=SIMPLE_PROMPT,
                )

            lines = []
            for seg in segments:
                text = converter.convert(seg.text.strip())
                sentences = self._split_sentences(text)

                if self.with_timestamp.get():
                    # 同一 segment 内多句共用该段起止时间
                    for sent in sentences:
                        line = "[%.2fs -> %.2fs] %s" % (seg.start, seg.end, sent)
                        lines.append(line)
                        self._append_log(line)
                else:
                    # 不开时间戳：每句一行
                    for sent in sentences:
                        lines.append(sent)
                        self._append_log(sent)

            with open(output, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self.status.set("完成：%s" % output)
            self.root.after(
                0, lambda: messagebox.showinfo("完成", "转录完成！\n" + output)
            )
        except Exception as e:
            err_msg = str(e)
            self.status.set("出错")
            self.root.after(0, lambda: messagebox.showerror("出错", err_msg))
        finally:
            self.root.after(
                0, lambda: self.btn_start.configure(state="normal", text="开始转录")
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()