"""TinyReadAloud - Select text, press Ctrl+Alt+R, hear it read aloud."""

import asyncio
import ctypes
import ctypes.wintypes
import json
import os
import queue
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import urllib.request

from version import __version__

# When frozen as a windowless app (console=False), stdout/stderr are None.
# Redirect to a log file so print() calls don't crash.
if getattr(sys, 'frozen', False) and sys.stdout is None:
    _data_base = os.environ.get("LOCALAPPDATA",
                                os.path.expanduser("~\\AppData\\Local"))
    _log_dir = os.path.join(_data_base, "TinyReadAloud")
    os.makedirs(_log_dir, exist_ok=True)
    sys.stdout = open(os.path.join(_log_dir, "tinyreadaloud.log"), "a", encoding="utf-8")
    sys.stderr = sys.stdout

# Add pip-installed NVIDIA CUDA DLLs to PATH so onnxruntime can find them
try:
    import nvidia
    _nv_root = os.path.dirname(nvidia.__path__[0] if hasattr(nvidia.__path__, '__iter__') else nvidia.__path__)
    for _subpkg in ("cublas", "cuda_runtime", "cudnn", "cufft", "nvjitlink"):
        _bin = os.path.join(_nv_root, "nvidia", _subpkg, "bin")
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)
            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

import keyboard
import numpy as np
import onnxruntime as ort
import pystray
import sounddevice as sd
from kokoro_onnx import Kokoro
from langdetect import detect as langdetect_detect
from PIL import Image, ImageDraw

# ── Constants ────────────────────────────────────────────────────────────────

HOTKEY = "ctrl+alt+r"
DEFAULT_VOICE_EN = "af_heart"
DEFAULT_VOICE_ES = "ef_dora"
DEFAULT_SPEED = 1.0
COPY_WAIT_INTERVAL = 0.02
COPY_WAIT_TIMEOUT = 0.5
AUDIO_CHUNK_SECS = 0.05  # 50ms playback granularity for stop responsiveness


def _get_app_dir():
    """Return the application directory (where the exe/script lives)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _get_data_dir():
    """Return %LOCALAPPDATA%/TinyReadAloud, creating it if needed."""
    base = os.environ.get("LOCALAPPDATA",
                          os.path.expanduser("~\\AppData\\Local"))
    d = os.path.join(base, "TinyReadAloud")
    os.makedirs(d, exist_ok=True)
    return d


APP_DIR = _get_app_dir()
DATA_DIR = _get_data_dir()
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
MODEL_PATH_FP16 = os.path.join(DATA_DIR, "kokoro-v1.0.fp16.onnx")
MODEL_PATH_INT8 = os.path.join(DATA_DIR, "kokoro-v1.0.int8.onnx")
VOICES_PATH = os.path.join(DATA_DIR, "voices-v1.0.bin")
MODEL_URL_FP16 = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx"
MODEL_URL_INT8 = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
GMEM_ZEROINIT = 0x0040

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.OpenClipboard.argtypes = [ctypes.wintypes.HWND]
user32.OpenClipboard.restype = ctypes.wintypes.BOOL
user32.CloseClipboard.argtypes = []
user32.CloseClipboard.restype = ctypes.wintypes.BOOL
user32.EmptyClipboard.argtypes = []
user32.EmptyClipboard.restype = ctypes.wintypes.BOOL
user32.GetClipboardData.argtypes = [ctypes.wintypes.UINT]
user32.GetClipboardData.restype = ctypes.c_void_p
user32.SetClipboardData.argtypes = [ctypes.wintypes.UINT, ctypes.c_void_p]
user32.SetClipboardData.restype = ctypes.c_void_p
user32.IsClipboardFormatAvailable.argtypes = [ctypes.wintypes.UINT]
user32.IsClipboardFormatAvailable.restype = ctypes.wintypes.BOOL
kernel32.GlobalAlloc.argtypes = [ctypes.wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
kernel32.GlobalUnlock.restype = ctypes.wintypes.BOOL


# ── Model Download ───────────────────────────────────────────────────────────

def _has_cuda():
    """Check if CUDA execution provider is available."""
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


USE_GPU = _has_cuda()


def ensure_models():
    """Download model files if they don't exist. Returns True if ready."""
    model_path = MODEL_PATH_FP16 if USE_GPU else MODEL_PATH_INT8
    model_url = MODEL_URL_FP16 if USE_GPU else MODEL_URL_INT8
    # Remove the other model variant to save disk space
    other = MODEL_PATH_INT8 if USE_GPU else MODEL_PATH_FP16
    if os.path.exists(other):
        os.remove(other)
    for path, url in [(model_path, model_url), (VOICES_PATH, VOICES_URL)]:
        if os.path.exists(path):
            continue
        name = os.path.basename(path)
        print(f"Downloading {name} (first run only)...")

        def reporthook(block, block_size, total):
            done = block * block_size
            pct = min(100, done * 100 // max(total, 1))
            mb_done = done / 1048576
            mb_total = total / 1048576
            print(f"\r  {pct}%  ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, path, reporthook=reporthook)
            print()
        except Exception as e:
            print(f"\nDownload failed: {e}", file=sys.stderr)
            if os.path.exists(path):
                os.remove(path)
            return False
    return True


SPEED_OPTIONS = [("Slow", 0.8), ("Normal", 1.0), ("Fast", 1.2), ("Very Fast", 1.5)]
SPEED_BY_LABEL = {label: spd for label, spd in SPEED_OPTIONS}
SPEED_BY_VALUE = {spd: label for label, spd in SPEED_OPTIONS}


# ── Config ──────────────────────────────────────────────────────────────────

def load_config():
    """Load settings from config.json, returning defaults for missing keys."""
    defaults = {"hotkey": HOTKEY, "voice_en": DEFAULT_VOICE_EN,
                "voice_es": DEFAULT_VOICE_ES, "speed": DEFAULT_SPEED}
    if not os.path.exists(CONFIG_PATH):
        return defaults
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate old single-voice config
        if "voice" in data and "voice_en" not in data:
            data["voice_en"] = data.pop("voice")
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return defaults


def save_config(cfg):
    """Save settings dict to config.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── Clipboard ────────────────────────────────────────────────────────────────

def _open_clipboard(retries=2):
    for i in range(retries):
        if user32.OpenClipboard(0):
            return True
        time.sleep(0.05)
    return False


def clipboard_get_text():
    if not _open_clipboard():
        return ""
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return ""
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = ctypes.c_wchar_p(handle)
        return ptr.value or ""
    finally:
        user32.CloseClipboard()


def clipboard_set_text(text):
    if not _open_clipboard():
        return
    try:
        user32.EmptyClipboard()
        if not text:
            return
        buf = (text + "\0").encode("utf-16-le")
        hmem = kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, len(buf))
        if not hmem:
            return
        ptr = kernel32.GlobalLock(hmem)
        if not ptr:
            return
        ctypes.memmove(ptr, buf, len(buf))
        kernel32.GlobalUnlock(hmem)
        user32.SetClipboardData(CF_UNICODETEXT, hmem)
    finally:
        user32.CloseClipboard()


def clipboard_clear():
    if not _open_clipboard():
        return
    try:
        user32.EmptyClipboard()
    finally:
        user32.CloseClipboard()


def capture_selected_text():
    old = clipboard_get_text()
    clipboard_clear()
    keyboard.send("ctrl+c")

    elapsed = 0.0
    text = ""
    while elapsed < COPY_WAIT_TIMEOUT:
        time.sleep(COPY_WAIT_INTERVAL)
        elapsed += COPY_WAIT_INTERVAL
        text = clipboard_get_text()
        if text:
            break

    if old:
        clipboard_set_text(old)
    else:
        clipboard_clear()

    return text.strip()


# ── Icon ─────────────────────────────────────────────────────────────────────

def create_tray_icon(size=64, speaking=False):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg = "#2ECC71" if speaking else "#E74C3C"
    pad = 2
    draw.ellipse([pad, pad, size - pad, size - pad], fill=bg)

    cx, cy = size // 2, size // 2

    # Speaker body
    draw.rectangle([cx - 12, cy - 6, cx - 4, cy + 6], fill="white")
    # Speaker cone
    draw.polygon(
        [(cx - 4, cy - 6), (cx + 6, cy - 14), (cx + 6, cy + 14), (cx - 4, cy + 6)],
        fill="white",
    )

    if speaking:
        draw.rectangle([cx + 12, cy - 7, cx + 17, cy + 7], fill="white")
        draw.rectangle([cx + 20, cy - 7, cx + 25, cy + 7], fill="white")
    else:
        for r in [14, 21]:
            bbox = [cx + 6 - r, cy - r, cx + 6 + r, cy + r]
            draw.arc(bbox, start=-35, end=35, fill="white", width=2)

    return img


# ── Voice Helpers ────────────────────────────────────────────────────────────

_LANG_NAMES = {"a": "American", "b": "British", "e": "Spanish", "f": "French",
               "h": "Hindi", "i": "Italian", "j": "Japanese", "p": "Portuguese",
               "z": "Chinese"}
_GENDER_NAMES = {"f": "Female", "m": "Male"}


def voice_display_name(code):
    """Convert 'af_heart' to 'Heart (American, Female)'."""
    parts = code.split("_", 1)
    if len(parts) != 2 or len(parts[0]) < 2:
        return code
    prefix, name = parts
    lang = _LANG_NAMES.get(prefix[0], prefix[0].upper())
    gender = _GENDER_NAMES.get(prefix[1], prefix[1].upper())
    return f"{name.title()} ({lang}, {gender})"


def is_english_voice(code):
    """Voice codes starting with 'a' (American) or 'b' (British) are English."""
    return len(code) >= 2 and code[0] in ("a", "b")


def is_spanish_voice(code):
    """Voice codes starting with 'e' are Spanish."""
    return len(code) >= 2 and code[0] == "e"


# Kokoro lang parameter mapping
_KOKORO_LANG = {"en": "en-us", "es": "es"}


def detect_language(text):
    """Detect whether text is English or Spanish. Returns 'en' or 'es'."""
    try:
        lang = langdetect_detect(text)
        if lang.startswith("es"):
            return "es"
    except Exception:
        pass
    return "en"


# ── Settings Window ─────────────────────────────────────────────────────────

class SettingsWindow:
    """Tkinter settings dialog. Runs its own mainloop in a separate thread."""

    _instance_lock = threading.Lock()
    _instance = None

    @classmethod
    def open(cls, app):
        """Open the settings window, or focus it if already open."""
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance._root.after(0, cls._instance._root.lift)
                    return
                except Exception:
                    cls._instance = None
            win = cls(app)
            cls._instance = win
        threading.Thread(target=win._run, daemon=True).start()

    def __init__(self, app):
        self._app = app
        self._root = None
        self._hotkey_var = None
        self._voice_en_var = None
        self._voice_es_var = None
        self._speed_var = None
        self._recording = False
        self._en_codes = []
        self._en_labels = []
        self._es_codes = []
        self._es_labels = []

    def _run(self):
        root = tk.Tk()
        self._root = root
        root.title("TinyReadAloud Settings")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Try to set icon to match tray
        try:
            icon_img = create_tray_icon(size=32, speaking=False)
            self._tk_icon = tk.PhotoImage(data=icon_img.tobytes())
        except Exception:
            pass

        pad = {"padx": 10, "pady": 5}
        row = 0

        # ── Hotkey ──
        ttk.Label(root, text="Read-aloud shortcut:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        hotkey_frame = ttk.Frame(root)
        hotkey_frame.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)

        self._hotkey_var = tk.StringVar(value=self._app._hotkey)
        hotkey_entry = ttk.Entry(hotkey_frame, textvariable=self._hotkey_var, state="readonly", width=25)
        hotkey_entry.pack(side="left", padx=(0, 5))

        self._record_btn = ttk.Button(hotkey_frame, text="Record", command=self._record_hotkey)
        self._record_btn.pack(side="left")
        row += 1

        # ── English Voice ──
        ttk.Label(root, text="English Voice:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        all_voices = self._app.tts.voices
        self._en_codes = [v for v in all_voices if is_english_voice(v)]
        self._en_labels = [voice_display_name(v) for v in self._en_codes]
        current_en_label = voice_display_name(self._app.tts.voice_en)

        self._voice_en_var = tk.StringVar(value=current_en_label)
        en_combo = ttk.Combobox(root, textvariable=self._voice_en_var, values=self._en_labels,
                                state="readonly", width=30)
        en_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        en_combo.bind("<<ComboboxSelected>>", self._on_en_voice_changed)
        row += 1

        # ── Spanish Voice ──
        ttk.Label(root, text="Spanish Voice:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        self._es_codes = [v for v in all_voices if is_spanish_voice(v)]
        self._es_labels = [voice_display_name(v) for v in self._es_codes]
        current_es_label = voice_display_name(self._app.tts.voice_es)

        self._voice_es_var = tk.StringVar(value=current_es_label)
        es_combo = ttk.Combobox(root, textvariable=self._voice_es_var, values=self._es_labels,
                                state="readonly", width=30)
        es_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        es_combo.bind("<<ComboboxSelected>>", self._on_es_voice_changed)
        row += 1

        # ── Speed ──
        ttk.Label(root, text="Speed:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        speed_labels = [label for label, _ in SPEED_OPTIONS]
        current_speed_label = SPEED_BY_VALUE.get(self._app.tts.current_speed, "Normal")
        self._speed_var = tk.StringVar(value=current_speed_label)
        speed_combo = ttk.Combobox(root, textvariable=self._speed_var, values=speed_labels,
                                   state="readonly", width=30)
        speed_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_changed)
        row += 1

        # ── Separator ──
        ttk.Separator(root, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        row += 1

        # ── Save / Cancel ──
        btn_frame = ttk.Frame(root)
        btn_frame.grid(row=row, column=0, columnspan=2, **pad)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_close).pack(side="left", padx=5)

        # Center the window on screen
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"+{x}+{y}")

        root.mainloop()

    def _record_hotkey(self):
        if self._recording:
            return
        self._recording = True
        self._hotkey_var.set("Press a key combo...")
        self._record_btn.config(state="disabled")

        def capture():
            try:
                combo = keyboard.read_hotkey(suppress=False)
                self._root.after(0, self._finish_record, combo)
            except Exception:
                self._root.after(0, self._finish_record, None)

        threading.Thread(target=capture, daemon=True).start()

    def _finish_record(self, combo):
        self._recording = False
        self._record_btn.config(state="normal")
        if combo:
            self._hotkey_var.set(combo)

    def _on_en_voice_changed(self, event):
        """Auto-preview when English voice dropdown changes."""
        label = self._voice_en_var.get()
        if label in self._en_labels:
            idx = self._en_labels.index(label)
            code = self._en_codes[idx]
            speed = SPEED_BY_LABEL.get(self._speed_var.get(), DEFAULT_SPEED)
            self._app.tts.stop()
            self._app.tts.set_speed(speed)
            self._app.tts.speak_preview("Hello! This is a preview.", code, "en-us")

    def _on_es_voice_changed(self, event):
        """Auto-preview when Spanish voice dropdown changes."""
        label = self._voice_es_var.get()
        if label in self._es_labels:
            idx = self._es_labels.index(label)
            code = self._es_codes[idx]
            speed = SPEED_BY_LABEL.get(self._speed_var.get(), DEFAULT_SPEED)
            self._app.tts.stop()
            self._app.tts.set_speed(speed)
            self._app.tts.speak_preview("Hola, esta es una vista previa.", code, "es")

    def _on_speed_changed(self, event):
        """Auto-preview when speed dropdown changes."""
        # Preview with the current English voice at new speed
        en_label = self._voice_en_var.get()
        if en_label in self._en_labels:
            idx = self._en_labels.index(en_label)
            code = self._en_codes[idx]
        else:
            code = self._app.tts.voice_en
        speed = SPEED_BY_LABEL.get(self._speed_var.get(), DEFAULT_SPEED)
        self._app.tts.stop()
        self._app.tts.set_speed(speed)
        self._app.tts.speak_preview("Hello! This is a preview.", code, "en-us")

    def _save(self):
        # Resolve selected values
        en_label = self._voice_en_var.get()
        idx_en = self._en_labels.index(en_label) if en_label in self._en_labels else 0
        voice_en = self._en_codes[idx_en]

        es_label = self._voice_es_var.get()
        idx_es = self._es_labels.index(es_label) if es_label in self._es_labels else 0
        voice_es = self._es_codes[idx_es]

        speed = SPEED_BY_LABEL.get(self._speed_var.get(), DEFAULT_SPEED)
        hotkey = self._hotkey_var.get()

        # Apply to TTS worker
        self._app.tts.set_voice_en(voice_en)
        self._app.tts.set_voice_es(voice_es)
        self._app.tts.set_speed(speed)

        # Re-register hotkey if changed
        if hotkey != self._app._hotkey:
            self._app._update_hotkey(hotkey)

        # Persist
        save_config({"hotkey": hotkey, "voice_en": voice_en, "voice_es": voice_es, "speed": speed})

        self._on_close()

    def _on_close(self):
        with SettingsWindow._instance_lock:
            SettingsWindow._instance = None
        try:
            self._root.destroy()
        except Exception:
            pass


# ── TTS Worker ───────────────────────────────────────────────────────────────

class TTSWorker:
    def __init__(self):
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._speaking = False
        self._voices = []
        self._voice_en = DEFAULT_VOICE_EN
        self._voice_es = DEFAULT_VOICE_ES
        self._current_speed = DEFAULT_SPEED
        self.on_state_change = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    @property
    def is_speaking(self):
        return self._speaking

    @property
    def voices(self):
        return list(self._voices)

    @property
    def voice_en(self):
        return self._voice_en

    @property
    def voice_es(self):
        return self._voice_es

    @property
    def current_speed(self):
        return self._current_speed

    def speak(self, text):
        if not self._speaking:
            lang = detect_language(text)
            voice = self._voice_es if lang == "es" else self._voice_en
            kokoro_lang = _KOKORO_LANG.get(lang, "en-us")
            self._queue.put(("speak", (text, voice, kokoro_lang)))

    def speak_preview(self, text, voice, kokoro_lang):
        """Speak a preview with an explicit voice and lang (skips detection).
        Always queues even if currently speaking (caller should call stop() first).
        """
        self._queue.put(("speak", (text, voice, kokoro_lang)))

    def stop(self):
        self._stop_event.set()

    def set_voice_en(self, voice):
        self._queue.put(("set_voice_en", voice))

    def set_voice_es(self, voice):
        self._queue.put(("set_voice_es", voice))

    def set_speed(self, speed):
        self._queue.put(("set_speed", speed))

    def quit(self):
        self._queue.put(("quit", None))

    def _set_speaking(self, value):
        self._speaking = value
        if self.on_state_change:
            try:
                self.on_state_change(value)
            except Exception:
                pass

    def _run(self):
        # Use CUDA with fp16 model if available, otherwise CPU with int8
        if USE_GPU:
            use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            model_path = MODEL_PATH_FP16
            print("Using GPU (CUDA) with fp16 model for TTS inference.")
        else:
            use_providers = ["CPUExecutionProvider"]
            model_path = MODEL_PATH_INT8
            print("Using CPU with int8 model for TTS inference.")
        session = ort.InferenceSession(model_path, providers=use_providers)
        kokoro = Kokoro.from_session(session, VOICES_PATH)

        # Read available voice names from the NPZ file
        data = np.load(VOICES_PATH)
        self._voices = sorted(data.files)
        data.close()

        # Warm up ONNX runtime so first real request is fast
        asyncio.run(self._warmup(kokoro))

        while True:
            cmd, arg = self._queue.get()  # block until command arrives

            try:
                if cmd == "speak":
                    text, voice, kokoro_lang = arg
                    self._stop_event.clear()
                    self._set_speaking(True)
                    asyncio.run(self._speak_stream(kokoro, text, voice, kokoro_lang))
                    self._set_speaking(False)

                elif cmd == "set_voice_en":
                    self._voice_en = arg

                elif cmd == "set_voice_es":
                    self._voice_es = arg

                elif cmd == "set_speed":
                    self._current_speed = arg

                elif cmd == "quit":
                    break

            except Exception as e:
                print(f"TTS error: {e}", file=sys.stderr)
                self._set_speaking(False)

    async def _warmup(self, kokoro):
        """Run a tiny inference to pre-warm ONNX runtime."""
        async for _, _ in kokoro.create_stream(".", voice=self._voice_en, speed=1.0, lang="en-us"):
            break  # only need the first chunk

    async def _speak_stream(self, kokoro, text, voice, kokoro_lang):
        stream = kokoro.create_stream(
            text, voice=voice, speed=self._current_speed, lang=kokoro_lang
        )
        async for samples, sample_rate in stream:
            if self._stop_event.is_set():
                break
            if not self._play_audio(samples, sample_rate):
                break

    def _play_audio(self, samples, sample_rate):
        """Play audio with 50ms stop-check granularity."""
        samples = np.asarray(samples, dtype=np.float32)
        chunk_size = int(sample_rate * AUDIO_CHUNK_SECS)
        try:
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out:
                i = 0
                while i < len(samples):
                    if self._stop_event.is_set():
                        return False
                    end = min(i + chunk_size, len(samples))
                    out.write(samples[i:end].reshape(-1, 1))
                    i = end
        except Exception as e:
            print(f"Audio playback error: {e}", file=sys.stderr)
            return False
        return True


# ── Main App ─────────────────────────────────────────────────────────────────

class TinyReadAloud:
    def __init__(self):
        cfg = load_config()
        self._hotkey = cfg["hotkey"]
        self.tts = TTSWorker()
        self.tts._voice_en = cfg["voice_en"]
        self.tts._voice_es = cfg["voice_es"]
        self.tts._current_speed = cfg["speed"]
        self.tts.on_state_change = self._on_speaking_changed
        self.icon = None
        self._hotkey_handle = None
        self._update_info = None
        self._icon_idle = create_tray_icon(speaking=False)
        self._icon_speaking = create_tray_icon(speaking=True)

    def run(self):
        self.icon = pystray.Icon(
            name="TinyReadAloud",
            icon=self._icon_idle,
            title=f"TinyReadAloud v{__version__}  [{self._hotkey}]",
            menu=self._build_menu(),
        )
        self.icon.run(setup=self._on_ready)

    def _on_ready(self, icon):
        icon.visible = True
        self.tts.start()
        time.sleep(1.0)  # give Kokoro time to load
        icon.menu = self._build_menu()
        icon.update_menu()
        self._hotkey_handle = keyboard.add_hotkey(self._hotkey, self._on_hotkey, suppress=True)
        print(f"TinyReadAloud v{__version__} ready. Press {self._hotkey} to read selected text.")
        # Check for updates in background after 5 seconds
        threading.Timer(5.0, self._check_for_updates_background).start()

    def _build_menu(self):
        items = [
            pystray.MenuItem(
                "Stop Reading",
                self._cmd_stop,
                enabled=lambda item: self.tts.is_speaking,
            ),
        ]

        # English voice submenu
        if self.tts.voices:
            en_items = []
            for v in self.tts.voices:
                if is_english_voice(v):
                    en_items.append(
                        pystray.MenuItem(
                            voice_display_name(v),
                            self._make_voice_en_setter(v),
                            checked=lambda item, vc=v: self.tts.voice_en == vc,
                            radio=True,
                        )
                    )
            if en_items:
                items.append(pystray.MenuItem("English Voice", pystray.Menu(*en_items)))

            # Spanish voice submenu
            es_items = []
            for v in self.tts.voices:
                if is_spanish_voice(v):
                    es_items.append(
                        pystray.MenuItem(
                            voice_display_name(v),
                            self._make_voice_es_setter(v),
                            checked=lambda item, vc=v: self.tts.voice_es == vc,
                            radio=True,
                        )
                    )
            if es_items:
                items.append(pystray.MenuItem("Spanish Voice", pystray.Menu(*es_items)))

        # Speed submenu
        speed_items = [
            pystray.MenuItem(
                label,
                self._make_speed_setter(spd),
                checked=lambda item, s=spd: self.tts.current_speed == s,
                radio=True,
            )
            for label, spd in SPEED_OPTIONS
        ]
        items.append(pystray.MenuItem("Speed", pystray.Menu(*speed_items)))

        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Settings", self._cmd_settings))
        items.append(pystray.MenuItem("Check for Updates", self._cmd_check_updates))
        if self._update_info and self._update_info.available:
            items.append(pystray.MenuItem(
                f"Download v{self._update_info.tag}",
                self._cmd_download_update))
        items.append(pystray.MenuItem("Exit", self._cmd_exit))

        return pystray.Menu(*items)

    def _make_voice_en_setter(self, voice):
        def setter(icon, item):
            self.tts.set_voice_en(voice)
        return setter

    def _make_voice_es_setter(self, voice):
        def setter(icon, item):
            self.tts.set_voice_es(voice)
        return setter

    def _make_speed_setter(self, speed):
        def setter(icon, item):
            self.tts.set_speed(speed)
        return setter

    def _on_hotkey(self):
        if self.tts.is_speaking:
            self.tts.stop()
        else:
            threading.Thread(target=self._capture_and_speak, daemon=True).start()

    def _capture_and_speak(self):
        text = capture_selected_text()
        if text:
            self.tts.speak(text)

    def _on_speaking_changed(self, is_speaking):
        if self.icon:
            self.icon.icon = (
                self._icon_speaking if is_speaking else self._icon_idle
            )

    def _cmd_stop(self, icon, item):
        self.tts.stop()

    def _cmd_settings(self, icon, item):
        SettingsWindow.open(self)

    def _check_for_updates_background(self):
        """Run update check in background, notify if update found."""
        from updater import check_for_update
        info = check_for_update(__version__)
        if info and info.available:
            self._update_info = info
            if self.icon:
                self.icon.menu = self._build_menu()
                self.icon.update_menu()
                self.icon.notify(
                    f"TinyReadAloud v{info.tag} is available!",
                    "Update Available")

    def _cmd_check_updates(self, icon, item):
        """Manual update check from tray menu."""
        threading.Thread(target=self._manual_update_check, daemon=True).start()

    def _manual_update_check(self):
        from updater import check_for_update
        info = check_for_update(__version__)
        if info and info.available:
            self._update_info = info
            if self.icon:
                self.icon.menu = self._build_menu()
                self.icon.update_menu()
                self.icon.notify(
                    f"TinyReadAloud v{info.tag} is available!",
                    "Update Available")
        else:
            if self.icon:
                self.icon.notify("You are running the latest version.",
                                 "No Update Available")

    def _cmd_download_update(self, icon, item):
        """Download and run the new installer."""
        if not self._update_info or not self._update_info.download_url:
            return
        threading.Thread(target=self._do_download_update, daemon=True).start()

    def _do_download_update(self):
        from updater import download_and_run_installer
        if self.icon:
            self.icon.notify("Downloading update...", "TinyReadAloud")
        success = download_and_run_installer(self._update_info.download_url)
        if success:
            self._cmd_exit(self.icon, None)
        else:
            if self.icon:
                self.icon.notify("Download failed. Try again later.",
                                 "Update Error")

    def _update_hotkey(self, new_hotkey):
        """Re-register the global hotkey and update tooltip."""
        if self._hotkey_handle is not None:
            keyboard.remove_hotkey(self._hotkey_handle)
        self._hotkey = new_hotkey
        self._hotkey_handle = keyboard.add_hotkey(self._hotkey, self._on_hotkey, suppress=True)
        if self.icon:
            self.icon.title = f"TinyReadAloud v{__version__}  [{self._hotkey}]"

    def _cmd_exit(self, icon, item):
        if self._hotkey_handle is not None:
            keyboard.remove_hotkey(self._hotkey_handle)
        self.tts.quit()
        icon.stop()


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    if not ensure_models():
        print("Cannot start without model files.", file=sys.stderr)
        sys.exit(1)

    print("Loading Kokoro TTS model...")
    app = TinyReadAloud()

    def _sigint_handler(sig, frame):
        if app.icon:
            app.icon.stop()

    signal.signal(signal.SIGINT, _sigint_handler)
    app.run()


if __name__ == "__main__":
    main()
