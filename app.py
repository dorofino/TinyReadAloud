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
from tkinter import messagebox, ttk
import urllib.error
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
DICTATION_HOTKEY = "ctrl+alt+d"
GRAMMAR_HOTKEY = "ctrl+alt+g"
REPHRASE_HOTKEY = "ctrl+alt+p"
DEFAULT_VOICE_EN = "af_heart"
DEFAULT_VOICE_ES = "ef_dora"
DEFAULT_SPEED = 1.0
DEFAULT_GRAMMAR_MODE = "manual"
GRAMMAR_MODES = ["off", "manual", "after_dictation"]
DEFAULT_DICTATION_PROVIDER = "windows"
DEFAULT_GRAMMAR_PROVIDER = "anthropic"
DICTATION_PROVIDERS = ["windows"]
GRAMMAR_PROVIDERS = ["anthropic"]
REPHRASE_STYLES = ["Natural", "Formal", "Casual", "Concise", "Expanded", "Professional"]
DEFAULT_REPHRASE_STYLE = "Natural"
COPY_WAIT_INTERVAL = 0.02
COPY_WAIT_TIMEOUT = 0.5
AUDIO_CHUNK_SECS = 0.05  # 50ms playback granularity for stop responsiveness
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-6"
ANTHROPIC_VERSION = "2023-06-01"


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
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = ctypes.wintypes.HWND
user32.SetForegroundWindow.argtypes = [ctypes.wintypes.HWND]
user32.SetForegroundWindow.restype = ctypes.wintypes.BOOL
kernel32.GlobalAlloc.argtypes = [ctypes.wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
kernel32.GlobalUnlock.restype = ctypes.wintypes.BOOL
kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
kernel32.GlobalFree.restype = ctypes.c_void_p


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
                "voice_es": DEFAULT_VOICE_ES, "speed": DEFAULT_SPEED,
                "dictation_hotkey": DICTATION_HOTKEY,
                "grammar_hotkey": GRAMMAR_HOTKEY,
                "rephrase_hotkey": REPHRASE_HOTKEY,
                "grammar_mode": DEFAULT_GRAMMAR_MODE,
                "dictation_provider": DEFAULT_DICTATION_PROVIDER,
                "grammar_provider": DEFAULT_GRAMMAR_PROVIDER,
                "rephrase_style": DEFAULT_REPHRASE_STYLE,
                "anthropic_api_key": "",
                "anthropic_model": ANTHROPIC_MODEL_DEFAULT}
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


def grammar_check_text_anthropic(text, api_key="", model=""):
    """Use Anthropic to correct grammar and return corrected plain text."""
    api_key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    model = (model or "").strip() or ANTHROPIC_MODEL_DEFAULT

    prompt = (
        "Correct grammar and punctuation for the text below. "
        "Do not add commentary. Return only corrected text in the same language.\n\n"
        f"{text}"
    )
    payload = {
        "model": model,
        "max_tokens": max(256, min(2048, len(text) * 3)),
        "messages": [{"role": "user", "content": prompt}],
    }

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {body or e.reason}") from e

    parts = []
    for item in data.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
    corrected = "".join(parts).strip()
    if not corrected:
        return text
    return corrected


def grammar_check_text(text, provider, api_key="", model=""):
    """Dispatch grammar check to the selected provider."""
    if provider == "anthropic":
        return grammar_check_text_anthropic(text, api_key=api_key, model=model)
    raise RuntimeError(f"Unsupported grammar provider: {provider}")


_REPHRASE_STYLE_PROMPTS = {
    "Natural":      (
        "Rewrite the text below using different words and sentence structure. "
        "Do NOT just fix grammar or punctuation — actively rephrase with new wording so the output "
        "sounds noticeably different from the input while keeping the same meaning and language."
    ),
    "Formal":       (
        "Rewrite the text below in a formal, professional tone using different vocabulary and sentence structure. "
        "Do NOT just fix grammar — actively rephrase with elevated, polished wording. Keep the same meaning and language."
    ),
    "Casual":       (
        "Rewrite the text below in a relaxed, conversational tone using different words and phrasing. "
        "Do NOT just fix grammar — actively rephrase to sound friendly and informal. Keep the same meaning and language."
    ),
    "Concise":      (
        "Rewrite the text below to be significantly shorter and more direct. "
        "Cut unnecessary words, merge sentences where possible, and use tighter phrasing. "
        "Keep the core meaning and language but make it noticeably more compact."
    ),
    "Expanded":     (
        "Rewrite the text below to be longer and more detailed. "
        "Add descriptive language, elaborate on ideas, and use fuller sentences. "
        "Keep the same meaning and language but make the output noticeably richer and more developed."
    ),
    "Professional": (
        "Rewrite the text below in polished business language suitable for a workplace email or report. "
        "Use different words and sentence structure — do NOT just fix grammar. "
        "Keep the same meaning and language."
    ),
}


def rephrase_text_anthropic(text, api_key="", model="", style=""):
    """Use Anthropic to rephrase/reword text and return the result."""
    api_key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    model = (model or "").strip() or ANTHROPIC_MODEL_DEFAULT
    style = (style or DEFAULT_REPHRASE_STYLE).strip()
    style_instr = _REPHRASE_STYLE_PROMPTS.get(style, _REPHRASE_STYLE_PROMPTS[DEFAULT_REPHRASE_STYLE])

    system_prompt = (
        "You are a professional writing assistant. "
        "Your sole task is to rewrite the text the user gives you according to the style instruction. "
        "Output ONLY the rewritten text — no quotes, no labels, no commentary, no explanation."
    )
    user_message = f"{style_instr}\n\nText to rewrite:\n{text}"

    payload = {
        "model": model,
        "max_tokens": max(256, min(2048, len(text) * 4)),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {body or e.reason}") from e

    print(f"[Rephrase] Raw API response type={data.get('type')} stop_reason={data.get('stop_reason')}", flush=True)

    # Anthropic sometimes returns a 200 with {"type": "error", ...}
    if data.get("type") == "error":
        err = data.get("error", {})
        raise RuntimeError(f"{err.get('type', 'api_error')}: {err.get('message', str(data))}")

    parts = []
    for item in data.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
    result = "".join(parts).strip()
    print(f"[Rephrase] Extracted result ({len(result)} chars): {result[:120]!r}", flush=True)
    if not result:
        raise RuntimeError("API returned empty content — check model name and API key tier.")
    return result


def rephrase_text(text, provider, api_key="", model="", style=""):
    """Dispatch rephrase to the selected provider."""
    if provider == "anthropic":
        return rephrase_text_anthropic(text, api_key=api_key, model=model, style=style)
    raise RuntimeError(f"Unsupported rephrase provider: {provider}")


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
    """Write text to clipboard, retrying up to 10 times. Returns True on success."""
    for attempt in range(10):
        if _open_clipboard():
            try:
                user32.EmptyClipboard()
                if not text:
                    return True
                buf = (text + "\0").encode("utf-16-le")
                hmem = kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, len(buf))
                if not hmem:
                    continue
                ptr = kernel32.GlobalLock(hmem)
                if not ptr:
                    kernel32.GlobalFree(hmem)
                    continue
                ctypes.memmove(ptr, buf, len(buf))
                kernel32.GlobalUnlock(hmem)
                user32.SetClipboardData(CF_UNICODETEXT, hmem)
                return True
            finally:
                user32.CloseClipboard()
        time.sleep(0.05)
    return False


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


# ── Floating Status Bar ──────────────────────────────────────────────────────

_STATUSBAR_TRANSPARENT = "#010203"


class FloatingStatusBar:
    """Tiny draggable always-on-top bar: style switcher + live status + settings gear."""

    BG     = "#1a1a2e"
    FG     = "#e2e8f0"
    DIM    = "#8892a0"
    ACCENT = "#a78bfa"
    SEP    = "#2d3748"
    W, H   = 300, 36
    RADIUS = 10
    _FONT  = ("Segoe UI", 9)
    _FONTB = ("Segoe UI", 9, "bold")

    _instance_lock = threading.Lock()
    _instance = None

    # ── Singleton ────────────────────────────────────────────────────────────

    @classmethod
    def open(cls, app):
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance._root.lift()
                    return
                except Exception:
                    pass
            inst = cls(app)
            cls._instance = inst
            threading.Thread(target=inst._run, daemon=True).start()

    @classmethod
    def get(cls):
        with cls._instance_lock:
            return cls._instance

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, app):
        self._app = app
        self._root = None
        self._status_var = None
        self._style_var = None
        self._drag_x = self._drag_y = 0
        self._own_hwnd = 0          # set once the Tk window is created
        self._last_target_hwnd = 0  # last foreground HWND that isn't the status bar
        try:
            self._style_idx = REPHRASE_STYLES.index(app._rephrase_style)
        except (ValueError, AttributeError):
            self._style_idx = 0

    # ── Thread-safe updates ───────────────────────────────────────────────────

    def set_status(self, text):
        if self._root and self._status_var:
            try:
                self._root.after(0, lambda t=text: self._status_var.set(t))
            except Exception:
                pass

    def sync_style(self):
        """Called after Settings saves a new rephrase style."""
        style = getattr(self._app, "_rephrase_style", REPHRASE_STYLES[0])
        try:
            self._style_idx = REPHRASE_STYLES.index(style)
        except ValueError:
            pass
        if self._root and self._style_var:
            try:
                self._root.after(0, lambda s=style: self._style_var.set(s))
            except Exception:
                pass

    # ── Canvas helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
        pts = [
            x1 + r, y1,   x2 - r, y1,
            x2,     y1,   x2,     y1 + r,
            x2,     y2 - r, x2,   y2,
            x2 - r, y2,   x1 + r, y2,
            x1,     y2,   x1,     y2 - r,
            x1,     y1 + r, x1,   y1,
        ]
        canvas.create_polygon(pts, smooth=True, **kw)

    # ── Drag ─────────────────────────────────────────────────────────────────

    def _drag_start(self, event):
        self._drag_x = event.x_root - self._root.winfo_x()
        self._drag_y = event.y_root - self._root.winfo_y()

    def _drag_move(self, event):
        self._root.geometry(
            f"+{event.x_root - self._drag_x}+{event.y_root - self._drag_y}"
        )

    # ── Style cycling ─────────────────────────────────────────────────────────

    def _prev_style(self, _=None):
        self._style_idx = (self._style_idx - 1) % len(REPHRASE_STYLES)
        self._commit_style()

    def _next_style(self, _=None):
        self._style_idx = (self._style_idx + 1) % len(REPHRASE_STYLES)
        self._commit_style()

    def _commit_style(self):
        style = REPHRASE_STYLES[self._style_idx]
        if self._style_var:
            self._style_var.set(style)
        self._app._rephrase_style = style
        try:
            cfg = load_config()
            cfg["rephrase_style"] = style
            save_config(cfg)
        except Exception:
            pass

    # ── Open settings / trigger rephrase ─────────────────────────────────────

    def _open_settings(self, _=None):
        SettingsWindow.open(self._app)

    def _trigger_rephrase(self, _=None):
        """Triggered when the user clicks the style name — rephrases with current style."""
        # Use the last tracked editor window, NOT GetForegroundWindow() which at
        # click time returns the status bar's own HWND.
        target_hwnd = self._last_target_hwnd or None
        style = self._app._rephrase_style
        target_hex = f"{target_hwnd:#010x}" if target_hwnd else "0"
        print(f"[Rephrase] Status bar click. last_target_hwnd={target_hex}  style={style!r}", flush=True)
        threading.Thread(
            target=self._app._run_rephrase_select_all,
            args=(target_hwnd,),
            daemon=True,
        ).start()

    # ── Tk main loop ──────────────────────────────────────────────────────────

    def _run(self):
        T = _STATUSBAR_TRANSPARENT
        W, H, R = self.W, self.H, self.RADIUS
        ymid = H // 2

        root = tk.Tk()
        self._root = root
        root.overrideredirect(True)
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-alpha", 0.95)
        root.wm_attributes("-transparentcolor", T)
        root.configure(bg=T)
        root.resizable(False, False)

        sw = root.winfo_screenwidth()
        root.geometry(f"{W}x{H}+{(sw - W) // 2}+48")

        # Win11 rounded corners via DWM (safe no-op on Win10)
        try:
            root.update_idletasks()
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                root.winfo_id(), 33,
                ctypes.byref(ctypes.c_int(2)), ctypes.sizeof(ctypes.c_int),
            )
        except Exception:
            pass

        # Prevent this window from ever stealing keyboard focus
        GWL_EXSTYLE      = -20
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        SWP_NOMOVE       = 0x0002
        SWP_NOSIZE       = 0x0001
        SWP_NOZORDER     = 0x0004
        SWP_FRAMECHANGED = 0x0020
        hwnd = root.winfo_id()
        self._own_hwnd = hwnd
        try:
            cur = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                cur | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            )
            # Force the extended-style change to take effect immediately
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
            )
        except Exception:
            pass

        # Background thread: track the last non-statusbar foreground window
        def _fg_tracker():
            while self._root is not None:
                try:
                    fg = user32.GetForegroundWindow()
                    if fg and fg != self._own_hwnd:
                        self._last_target_hwnd = fg
                except Exception:
                    pass
                time.sleep(0.15)
        threading.Thread(target=_fg_tracker, daemon=True).start()

        # ── Canvas ───────────────────────────────────────────────────────────
        canvas = tk.Canvas(root, width=W, height=H, bg=T, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        self._rounded_rect(canvas, 0, 0, W, H, R, fill=self.BG, outline=self.BG)

        def _lbl(text="", textvariable=None, font=None, fg=None,
                 width=None, anchor="center", cursor="arrow"):
            kw = dict(bg=self.BG, fg=fg or self.FG,
                      font=font or self._FONT, cursor=cursor)
            if textvariable is not None:
                kw["textvariable"] = textvariable
            else:
                kw["text"] = text
            if width:
                kw["width"] = width
            if anchor:
                kw["anchor"] = anchor
            return tk.Label(root, **kw)

        # ── ◀ left arrow ─────────────────────────────────────────────────────
        btn_l = _lbl("◀", fg=self.ACCENT, cursor="hand2")
        btn_l.bind("<Button-1>", self._prev_style)
        btn_l.bind("<Enter>", lambda e: btn_l.config(fg="white"))
        btn_l.bind("<Leave>", lambda e: btn_l.config(fg=self.ACCENT))
        canvas.create_window(10, ymid, window=btn_l, anchor="w")

        # ── style name (clickable — triggers rephrase) ───────────────────────────
        self._style_var = tk.StringVar(value=REPHRASE_STYLES[self._style_idx])
        lbl_style = _lbl(textvariable=self._style_var,
                         font=self._FONTB, width=13, anchor="center", cursor="hand2")
        lbl_style.bind("<Button-1>", self._trigger_rephrase)
        lbl_style.bind("<Enter>", lambda e: lbl_style.config(fg=self.ACCENT))
        lbl_style.bind("<Leave>", lambda e: lbl_style.config(fg=self.FG))
        canvas.create_window(26, ymid, window=lbl_style, anchor="w")

        # ── ▶ right arrow ────────────────────────────────────────────────────
        btn_r = _lbl("▶", fg=self.ACCENT, cursor="hand2")
        btn_r.bind("<Button-1>", self._next_style)
        btn_r.bind("<Enter>", lambda e: btn_r.config(fg="white"))
        btn_r.bind("<Leave>", lambda e: btn_r.config(fg=self.ACCENT))
        canvas.create_window(118, ymid, window=btn_r, anchor="w")

        # ── separator ────────────────────────────────────────────────────────
        canvas.create_line(134, 8, 134, H - 8, fill=self.SEP, width=1)

        # ── status label ─────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready")
        lbl_status = _lbl(textvariable=self._status_var,
                          fg=self.DIM, width=14, anchor="w")
        canvas.create_window(140, ymid, window=lbl_status, anchor="w")

        # ── gear ⚙ ───────────────────────────────────────────────────────────
        btn_gear = _lbl("⚙", font=("Segoe UI", 11), fg=self.DIM, cursor="hand2")
        btn_gear.bind("<Button-1>", self._open_settings)
        btn_gear.bind("<Enter>", lambda e: btn_gear.config(fg=self.ACCENT))
        btn_gear.bind("<Leave>", lambda e: btn_gear.config(fg=self.DIM))
        canvas.create_window(W - 10, ymid, window=btn_gear, anchor="e")

        # Prevent all widgets from grabbing keyboard focus when clicked
        for w in (canvas, btn_l, btn_r, btn_gear, lbl_style, lbl_status):
            try:
                w.configure(takefocus=0)
            except Exception:
                pass
        root.attributes("-topmost", True)

        # drag on background + status label (NOT on style label — it triggers rephrase)
        for w in (canvas, lbl_status):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

        root.protocol("WM_DELETE_WINDOW", lambda: None)
        root.mainloop()


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
        self._dictation_hotkey_var = None
        self._grammar_mode_var = None
        self._dictation_provider_var = None
        self._grammar_provider_var = None
        self._anthropic_api_key_var = None
        self._anthropic_model_var = None
        self._rephrase_hotkey_var = None
        self._rephrase_style_var = None
        self._voice_en_var = None
        self._voice_es_var = None
        self._speed_var = None
        self._recording = False
        self._record_target = None
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

        # ── Dictation Hotkey ──
        ttk.Label(root, text="Dictation shortcut:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        dict_frame = ttk.Frame(root)
        dict_frame.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)

        self._dictation_hotkey_var = tk.StringVar(value=self._app._dictation_hotkey)
        dict_entry = ttk.Entry(dict_frame, textvariable=self._dictation_hotkey_var, state="readonly", width=25)
        dict_entry.pack(side="left", padx=(0, 5))

        self._record_dict_btn = ttk.Button(dict_frame, text="Record", command=self._record_dictation_hotkey)
        self._record_dict_btn.pack(side="left")
        row += 1

        # ── Grammar Hotkey ──
        ttk.Label(root, text="Grammar shortcut:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        grammar_hk_frame = ttk.Frame(root)
        grammar_hk_frame.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)

        self._grammar_hotkey_var = tk.StringVar(value=self._app._grammar_hotkey)
        grammar_hk_entry = ttk.Entry(grammar_hk_frame, textvariable=self._grammar_hotkey_var, state="readonly", width=25)
        grammar_hk_entry.pack(side="left", padx=(0, 5))

        self._record_grammar_btn = ttk.Button(grammar_hk_frame, text="Record", command=self._record_grammar_hotkey)
        self._record_grammar_btn.pack(side="left")
        row += 1

        # ── Rephrase Hotkey ──
        ttk.Label(root, text="Rephrase shortcut:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        rephrase_hk_frame = ttk.Frame(root)
        rephrase_hk_frame.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)

        self._rephrase_hotkey_var = tk.StringVar(value=self._app._rephrase_hotkey)
        rephrase_hk_entry = ttk.Entry(rephrase_hk_frame, textvariable=self._rephrase_hotkey_var, state="readonly", width=25)
        rephrase_hk_entry.pack(side="left", padx=(0, 5))

        self._record_rephrase_btn = ttk.Button(rephrase_hk_frame, text="Record", command=self._record_rephrase_hotkey)
        self._record_rephrase_btn.pack(side="left")
        row += 1

        # ── Rephrase Style ──
        ttk.Label(root, text="Rephrase style:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        self._rephrase_style_var = tk.StringVar(value=self._app._rephrase_style)
        rephrase_style_combo = ttk.Combobox(
            root,
            textvariable=self._rephrase_style_var,
            values=REPHRASE_STYLES,
            state="readonly",
            width=30,
        )
        rephrase_style_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1
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

        # ── Grammar Mode ──
        ttk.Label(root, text="Grammar mode:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        grammar_values = ["off", "manual", "after_dictation"]
        self._grammar_mode_var = tk.StringVar(value=self._app._grammar_mode)
        grammar_combo = ttk.Combobox(root, textvariable=self._grammar_mode_var, values=grammar_values,
                         state="readonly", width=30)
        grammar_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1

        # ── Dictation Provider ──
        ttk.Label(root, text="Dictation provider:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        self._dictation_provider_var = tk.StringVar(value=self._app._dictation_provider)
        dict_provider_combo = ttk.Combobox(
            root,
            textvariable=self._dictation_provider_var,
            values=DICTATION_PROVIDERS,
            state="readonly",
            width=30,
        )
        dict_provider_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1

        # ── Grammar Provider ──
        ttk.Label(root, text="Grammar provider:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        self._grammar_provider_var = tk.StringVar(value=self._app._grammar_provider)
        grammar_provider_combo = ttk.Combobox(
            root,
            textvariable=self._grammar_provider_var,
            values=GRAMMAR_PROVIDERS,
            state="readonly",
            width=30,
        )
        grammar_provider_combo.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1

        # ── Anthropic API Key ──
        ttk.Label(root, text="Anthropic API key:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        self._anthropic_api_key_var = tk.StringVar(value=self._app._anthropic_api_key)
        anthropic_entry = ttk.Entry(root, textvariable=self._anthropic_api_key_var, show="*", width=40)
        anthropic_entry.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1

        # ── Anthropic Model ──
        ttk.Label(root, text="Anthropic model:").grid(row=row, column=0, sticky="w", **pad)
        row += 1

        self._anthropic_model_var = tk.StringVar(value=self._app._anthropic_model)
        anthropic_model_entry = ttk.Entry(root, textvariable=self._anthropic_model_var, width=40)
        anthropic_model_entry.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1
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
        self._start_hotkey_record("read")

    def _record_dictation_hotkey(self):
        self._start_hotkey_record("dictation")

    def _record_grammar_hotkey(self):
        self._start_hotkey_record("grammar")

    def _record_rephrase_hotkey(self):
        self._start_hotkey_record("rephrase")

    def _start_hotkey_record(self, target):
        if self._recording:
            return
        self._recording = True
        self._record_target = target
        if target == "read":
            self._hotkey_var.set("Press a key combo...")
        elif target == "dictation":
            self._dictation_hotkey_var.set("Press a key combo...")
        elif target == "grammar":
            self._grammar_hotkey_var.set("Press a key combo...")
        else:
            self._rephrase_hotkey_var.set("Press a key combo...")
        self._record_btn.config(state="disabled")
        self._record_dict_btn.config(state="disabled")
        self._record_grammar_btn.config(state="disabled")
        self._record_rephrase_btn.config(state="disabled")

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
        self._record_dict_btn.config(state="normal")
        self._record_grammar_btn.config(state="normal")
        self._record_rephrase_btn.config(state="normal")
        if combo:
            if self._record_target == "dictation":
                self._dictation_hotkey_var.set(combo)
            elif self._record_target == "grammar":
                self._grammar_hotkey_var.set(combo)
            elif self._record_target == "rephrase":
                self._rephrase_hotkey_var.set(combo)
            else:
                self._hotkey_var.set(combo)
        self._record_target = None

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
        dictation_hotkey = self._dictation_hotkey_var.get()
        grammar_hotkey = self._grammar_hotkey_var.get()
        rephrase_hotkey = self._rephrase_hotkey_var.get()
        rephrase_style = self._rephrase_style_var.get() or DEFAULT_REPHRASE_STYLE
        grammar_mode = self._grammar_mode_var.get()
        dictation_provider = self._dictation_provider_var.get()
        grammar_provider = self._grammar_provider_var.get()
        anthropic_api_key = self._anthropic_api_key_var.get().strip()
        anthropic_model = self._anthropic_model_var.get().strip() or ANTHROPIC_MODEL_DEFAULT

        shortcuts = {"Read-aloud": hotkey, "Dictation": dictation_hotkey, "Grammar": grammar_hotkey, "Rephrase": rephrase_hotkey}
        seen = {}
        for name, key in shortcuts.items():
            if key in seen:
                messagebox.showerror("Shortcut conflict",
                    f"{name} and {seen[key]} shortcuts must be different.")
                return
            seen[key] = name
        if grammar_mode not in GRAMMAR_MODES:
            messagebox.showerror("Invalid setting", "Please select a valid grammar mode.")
            return
        if dictation_provider not in DICTATION_PROVIDERS:
            messagebox.showerror("Invalid setting", "Please select a valid dictation provider.")
            return
        if grammar_provider not in GRAMMAR_PROVIDERS:
            messagebox.showerror("Invalid setting", "Please select a valid grammar provider.")
            return

        # Apply to TTS worker
        self._app.tts.set_voice_en(voice_en)
        self._app.tts.set_voice_es(voice_es)
        self._app.tts.set_speed(speed)

        # Re-register hotkey if changed
        if hotkey != self._app._hotkey:
            self._app._update_hotkey(hotkey)
        if dictation_hotkey != self._app._dictation_hotkey:
            self._app._update_dictation_hotkey(dictation_hotkey)
        if grammar_hotkey != self._app._grammar_hotkey:
            self._app._update_grammar_hotkey(grammar_hotkey)
        if rephrase_hotkey != self._app._rephrase_hotkey:
            self._app._update_rephrase_hotkey(rephrase_hotkey)
        self._app._grammar_mode = grammar_mode
        self._app._dictation_provider = dictation_provider
        self._app._grammar_provider = grammar_provider
        self._app._rephrase_style = rephrase_style
        _sb = FloatingStatusBar.get()
        if _sb:
            _sb.sync_style()
        self._app._anthropic_api_key = anthropic_api_key
        self._app._anthropic_model = anthropic_model
        self._app._refresh_menu()

        # Persist
        save_config({
            "hotkey": hotkey,
            "dictation_hotkey": dictation_hotkey,
            "grammar_hotkey": grammar_hotkey,
            "rephrase_hotkey": rephrase_hotkey,
            "rephrase_style": rephrase_style,
            "voice_en": voice_en,
            "voice_es": voice_es,
            "speed": speed,
            "grammar_mode": grammar_mode,
            "dictation_provider": dictation_provider,
            "grammar_provider": grammar_provider,
            "anthropic_api_key": anthropic_api_key,
            "anthropic_model": anthropic_model,
        })

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
        self._dictation_hotkey = cfg["dictation_hotkey"]
        self._grammar_hotkey = cfg["grammar_hotkey"]
        self._rephrase_hotkey = cfg["rephrase_hotkey"]
        self._grammar_mode = cfg["grammar_mode"]
        self._dictation_provider = cfg["dictation_provider"]
        self._grammar_provider = cfg["grammar_provider"]
        self._rephrase_style = cfg["rephrase_style"]
        self._anthropic_api_key = cfg["anthropic_api_key"]
        self._anthropic_model = cfg["anthropic_model"]
        self.tts = TTSWorker()
        self.tts._voice_en = cfg["voice_en"]
        self.tts._voice_es = cfg["voice_es"]
        self.tts._current_speed = cfg["speed"]
        self.tts.on_state_change = self._on_speaking_changed
        self.icon = None
        self._hotkey_handle = None
        self._dictation_hotkey_handle = None
        self._grammar_hotkey_handle = None
        self._rephrase_hotkey_handle = None
        self._status_bar = None
        self._dictation_listening = False
        self._update_info = None
        self._icon_idle = create_tray_icon(speaking=False)
        self._icon_speaking = create_tray_icon(speaking=True)

    def run(self):
        self.icon = pystray.Icon(
            name="TinyReadAloud",
            icon=self._icon_idle,
            title=f"TinyReadAloud v{__version__}  [Read: {self._hotkey} | Dictation: {self._dictation_hotkey} | Grammar: {self._grammar_hotkey} | Rephrase: {self._rephrase_hotkey}]",
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
        self._dictation_hotkey_handle = keyboard.add_hotkey(
            self._dictation_hotkey, self._on_dictation_hotkey, suppress=True
        )
        self._grammar_hotkey_handle = keyboard.add_hotkey(
            self._grammar_hotkey, self._on_grammar_hotkey, suppress=True
        )
        self._rephrase_hotkey_handle = keyboard.add_hotkey(
            self._rephrase_hotkey, self._on_rephrase_hotkey, suppress=True
        )
        FloatingStatusBar.open(self)
        self._status_bar = FloatingStatusBar.get()
        print(
            f"TinyReadAloud v{__version__} ready. "
            f"Read: {self._hotkey} | Dictation: {self._dictation_hotkey} | Grammar: {self._grammar_hotkey} | Rephrase: {self._rephrase_hotkey}"
        )
        # Check for updates in background after 5 seconds
        threading.Timer(5.0, self._check_for_updates_background).start()

    def _build_menu(self):
        items = [
            pystray.MenuItem(
                "Stop Reading",
                self._cmd_stop,
                enabled=lambda item: self.tts.is_speaking,
            ),
            pystray.MenuItem(
                "Stop Dictation" if self._dictation_listening else "Start Dictation",
                self._cmd_toggle_dictation,
            ),
            pystray.MenuItem(
                "Check Grammar",
                self._cmd_check_grammar,
                enabled=lambda item: self._grammar_mode != "off",
            ),
            pystray.MenuItem(
                "Rephrase",
                self._cmd_rephrase,
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

    def _on_dictation_hotkey(self):
        self._toggle_dictation()

    def _on_grammar_hotkey(self):
        if self._grammar_mode == "off":
            return
        target_hwnd = (
            self._status_bar._last_target_hwnd
            if self._status_bar is not None
            else user32.GetForegroundWindow()
        ) or user32.GetForegroundWindow()
        print(f"[Grammar] Hotkey fired. target_hwnd={target_hwnd:#010x}", flush=True)
        threading.Thread(target=self._run_grammar_select_all, args=(target_hwnd,), daemon=True).start()

    def _on_rephrase_hotkey(self):
        target_hwnd = (
            self._status_bar._last_target_hwnd
            if self._status_bar is not None
            else user32.GetForegroundWindow()
        ) or user32.GetForegroundWindow()
        style = self._rephrase_style
        print(f"[Rephrase] Hotkey fired. target_hwnd={target_hwnd:#010x}  style={style!r}", flush=True)
        if self.icon:
            self.icon.notify(f"Rephrase hotkey fired ({style})", "Rephrase")
        threading.Thread(target=self._run_rephrase_select_all, args=(target_hwnd,), daemon=True).start()

    def _capture_and_speak(self):
        text = capture_selected_text()
        if text:
            self.tts.speak(text)

    def _on_speaking_changed(self, is_speaking):
        if self.icon:
            self.icon.icon = (
                self._icon_speaking if is_speaking else self._icon_idle
            )
        self._set_status("Speaking…" if is_speaking else "Ready")

    def _set_status(self, text):
        """Update the floating status bar (no-op if not open)."""
        if self._status_bar is not None:
            self._status_bar.set_status(text)

    def _cmd_stop(self, icon, item):
        self.tts.stop()

    def _cmd_toggle_dictation(self, icon, item):
        self._toggle_dictation()

    def _toggle_dictation(self):
        try:
            if self._dictation_provider == "windows":
                keyboard.send("windows+h")
            else:
                raise RuntimeError(f"Unsupported dictation provider: {self._dictation_provider}")
            self._dictation_listening = not self._dictation_listening
            self._refresh_menu()
            self._set_status("Listening…" if self._dictation_listening else "Ready")
            if self.icon:
                state = "Dictation listening started" if self._dictation_listening else "Dictation listening stopped"
                self.icon.notify(state, "TinyReadAloud")
            if (not self._dictation_listening) and self._grammar_mode == "after_dictation":
                threading.Thread(
                    target=self._run_grammar_select_all,
                    daemon=True,
                ).start()
        except Exception as e:
            if self.icon:
                self.icon.notify(f"Dictation toggle failed: {e}", "Dictation Error")

    def _cmd_check_grammar(self, icon, item):
        threading.Thread(
            target=self._run_grammar_select_all,
            daemon=True,
        ).start()

    def _cmd_rephrase(self, icon, item):
        threading.Thread(
            target=self._run_rephrase_select_all,
            daemon=True,
        ).start()

    def _reset_hotkeys(self):
        """Re-register all hotkeys to reset the keyboard library suppress state.
        Called after any keyboard.send() sequence so the next hotkey press is
        recognised correctly."""
        for attr, hotkey, cb in (
            ("_hotkey_handle",           self._hotkey,           self._on_hotkey),
            ("_dictation_hotkey_handle", self._dictation_hotkey, self._on_dictation_hotkey),
            ("_grammar_hotkey_handle",   self._grammar_hotkey,   self._on_grammar_hotkey),
            ("_rephrase_hotkey_handle",  self._rephrase_hotkey,  self._on_rephrase_hotkey),
        ):
            try:
                keyboard.remove_hotkey(getattr(self, attr))
            except Exception:
                pass
            try:
                setattr(self, attr, keyboard.add_hotkey(hotkey, cb, suppress=True))
            except Exception:
                pass

    def _run_grammar_select_all(self, target_hwnd=None):
        """Ctrl+A → Ctrl+C → grammar check → Ctrl+A → Ctrl+V corrected text → Ctrl+End."""
        try:
            self._run_grammar_inner(target_hwnd)
        except Exception:
            import traceback
            print(f"[Grammar] UNHANDLED EXCEPTION:\n{traceback.format_exc()}", flush=True)
            self._set_status("Grammar error.")
        finally:
            # Re-register hotkeys so suppress=True state is fresh for the next press
            self._reset_hotkeys()

    def _run_grammar_inner(self, target_hwnd=None):

        # Restore keyboard focus to the target window (prevents FloatingStatusBar from
        # intercepting our ctrl+a / ctrl+c / ctrl+v sends)
        if target_hwnd:
            try:
                user32.SetForegroundWindow(target_hwnd)
                time.sleep(0.1)
            except Exception:
                pass

        # Clear clipboard so we can detect when Ctrl+C actually updates it
        clipboard_clear()
        time.sleep(0.1)

        keyboard.send("ctrl+a")
        time.sleep(0.2)
        keyboard.send("ctrl+c")

        # Poll until clipboard has content (up to 1.5s)
        text = ""
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            time.sleep(0.05)
            text = clipboard_get_text().strip()
            if text:
                break

        print(f"[Grammar] Captured {len(text)} chars from clipboard.", flush=True)

        if not text:
            self._set_status("Nothing selected.")
            if self.icon:
                self.icon.notify("Nothing to grammar-check — clipboard was empty.", "Grammar Check")
            return

        self._set_status("Checking grammar…")
        if self.icon:
            self.icon.notify(f"Checking grammar ({len(text)} chars)…", "Grammar Check")

        try:
            corrected = grammar_check_text(
                text,
                self._grammar_provider,
                api_key=self._anthropic_api_key,
                model=self._anthropic_model,
            )
        except Exception as e:
            print(f"[Grammar] API error: {e}", flush=True)
            err_str = str(e)
            if "not_found_error" in err_str or "404" in err_str:
                msg = "Model not found. Open Settings → Anthropic model and enter a valid model for your account."
            elif "authentication" in err_str.lower() or "401" in err_str:
                msg = "Invalid API key. Open Settings → Anthropic API key."
            else:
                msg = f"Grammar check failed: {e}"
            self._set_status("Grammar error.")
            if self.icon:
                self.icon.notify(msg, "Grammar Check")
            return

        print(f"[Grammar] Corrected {len(corrected)} chars.", flush=True)

        if corrected.strip() == text.strip():
            keyboard.send("ctrl+end")
            self._set_status("No changes.")
            if self.icon:
                self.icon.notify("No grammar changes suggested.", "Grammar Check")
            return

        # ctrl+a to re-select all, then overwrite clipboard and paste.
        if target_hwnd:
            try:
                user32.SetForegroundWindow(target_hwnd)
                time.sleep(0.1)
            except Exception:
                pass
        keyboard.send("ctrl+a")
        time.sleep(0.3)
        ok = clipboard_set_text(corrected)
        if not ok:
            print("[Grammar] clipboard_set_text failed — aborting paste.", flush=True)
            self._set_status("Clipboard error.")
            if self.icon:
                self.icon.notify("Could not write to clipboard. Try again.", "Grammar Check")
            return
        # Verify the write took before issuing ctrl+v
        verify = clipboard_get_text()
        print(f"[Grammar] Clipboard verify: {len(verify)} chars", flush=True)
        time.sleep(0.05)
        keyboard.send("ctrl+v")
        time.sleep(0.15)
        keyboard.send("ctrl+end")
        self._set_status("Grammar applied.")
        if self.icon:
            self.icon.notify("Grammar applied. Ready to continue.", "Grammar Check")

    def _run_rephrase_select_all(self, target_hwnd=None):
        """Ctrl+A → Ctrl+C → rephrase via AI → Ctrl+A → Ctrl+V rephrased text → Ctrl+End."""
        try:
            self._run_rephrase_inner(target_hwnd)
        except Exception:
            import traceback
            print(f"[Rephrase] UNHANDLED EXCEPTION:\n{traceback.format_exc()}", flush=True)
            self._set_status("Rephrase error.")
        finally:
            # Re-register hotkeys so suppress=True state is fresh for the next press
            self._reset_hotkeys()

    def _run_rephrase_inner(self, target_hwnd=None):
        time.sleep(0.3)

        # --- Phase 1: Focus capture window ---
        fg_before = user32.GetForegroundWindow()
        target_hex = f"{target_hwnd:#010x}" if target_hwnd else "0"
        print(f"[Rephrase] Phase1 focus: current_fg={fg_before:#010x}  target={target_hex}", flush=True)
        if target_hwnd:
            ok_focus = user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.15)
            fg_after = user32.GetForegroundWindow()
            print(f"[Rephrase] SetForegroundWindow({target_hex}) -> ok={ok_focus}  now_fg={fg_after:#010x}", flush=True)

        clipboard_clear()
        time.sleep(0.1)

        fg_pre_copy = user32.GetForegroundWindow()
        print(f"[Rephrase] Before ctrl+a/ctrl+c: fg={fg_pre_copy:#010x}", flush=True)
        keyboard.send("ctrl+a")
        time.sleep(0.2)
        keyboard.send("ctrl+c")

        # Poll until clipboard has content (up to 1.5s)
        text = ""
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            time.sleep(0.05)
            text = clipboard_get_text().strip()
            if text:
                break

        print(f"[Rephrase] Captured {len(text)} chars. Text[:80]={text[:80]!r}", flush=True)

        if not text:
            self._set_status("Nothing selected.")
            if self.icon:
                self.icon.notify("Nothing to rephrase — clipboard was empty.", "Rephrase")
            return

        print(f"[Rephrase] Calling API with style={self._rephrase_style!r}  model={self._anthropic_model!r}", flush=True)
        self._set_status("Rephrasing…")
        if self.icon:
            self.icon.notify(f"Rephrasing ({len(text)} chars)…", "Rephrase")

        try:
            rephrased = rephrase_text(
                text,
                self._grammar_provider,
                api_key=self._anthropic_api_key,
                model=self._anthropic_model,
                style=self._rephrase_style,
            )
        except Exception as e:
            print(f"[Rephrase] API error: {e}", flush=True)
            err_str = str(e)
            if "not_found_error" in err_str or "404" in err_str:
                msg = "Model not found. Open Settings → Anthropic model and enter a valid model."
            elif "authentication" in err_str.lower() or "401" in err_str:
                msg = "Invalid API key. Open Settings → Anthropic API key."
            else:
                msg = f"Rephrase failed: {e}"
            self._set_status("Rephrase error.")
            if self.icon:
                self.icon.notify(msg, "Rephrase")
            return

        print(f"[Rephrase] API done. Input[:80]={text[:80]!r}", flush=True)
        print(f"[Rephrase] API done. Output[:200]={rephrased[:200]!r}", flush=True)
        same = rephrased.strip() == text.strip()
        print(f"[Rephrase] Same as input? {same}", flush=True)

        # --- Phase 2: Focus capture window again, paste result ---
        fg_pre_paste = user32.GetForegroundWindow()
        print(f"[Rephrase] Phase2 focus before paste: fg={fg_pre_paste:#010x}  target={target_hex}", flush=True)
        if target_hwnd:
            ok_focus2 = user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.15)
            fg_after2 = user32.GetForegroundWindow()
            print(f"[Rephrase] SetForegroundWindow({target_hex}) -> ok={ok_focus2}  now_fg={fg_after2:#010x}", flush=True)

        keyboard.send("ctrl+a")
        time.sleep(0.3)
        ok = clipboard_set_text(rephrased)
        print(f"[Rephrase] clipboard_set_text returned {ok}", flush=True)
        if not ok:
            self._set_status("Clipboard error.")
            if self.icon:
                self.icon.notify("Could not write to clipboard. Try again.", "Rephrase")
            return
        # Verify the write took before issuing ctrl+v
        verify = clipboard_get_text()
        print(f"[Rephrase] Clipboard verify: {len(verify)} chars: {verify[:80]!r}", flush=True)
        fg_pre_v = user32.GetForegroundWindow()
        print(f"[Rephrase] fg before ctrl+v: {fg_pre_v:#010x}", flush=True)
        time.sleep(0.05)
        keyboard.send("ctrl+v")
        time.sleep(0.15)
        keyboard.send("ctrl+end")
        self._set_status("Rephrased.")
        if self.icon:
            self.icon.notify("Rephrased. Ready to continue.", "Rephrase")

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
            self.icon.title = f"TinyReadAloud v{__version__}  [Read: {self._hotkey} | Dictation: {self._dictation_hotkey} | Grammar: {self._grammar_hotkey} | Rephrase: {self._rephrase_hotkey}]"

    def _update_dictation_hotkey(self, new_hotkey):
        if self._dictation_hotkey_handle is not None:
            keyboard.remove_hotkey(self._dictation_hotkey_handle)
        self._dictation_hotkey = new_hotkey
        self._dictation_hotkey_handle = keyboard.add_hotkey(
            self._dictation_hotkey, self._on_dictation_hotkey, suppress=True
        )
        if self.icon:
            self.icon.title = f"TinyReadAloud v{__version__}  [Read: {self._hotkey} | Dictation: {self._dictation_hotkey} | Grammar: {self._grammar_hotkey} | Rephrase: {self._rephrase_hotkey}]"

    def _update_grammar_hotkey(self, new_hotkey):
        if self._grammar_hotkey_handle is not None:
            keyboard.remove_hotkey(self._grammar_hotkey_handle)
        self._grammar_hotkey = new_hotkey
        self._grammar_hotkey_handle = keyboard.add_hotkey(
            self._grammar_hotkey, self._on_grammar_hotkey, suppress=True
        )
        if self.icon:
            self.icon.title = f"TinyReadAloud v{__version__}  [Read: {self._hotkey} | Dictation: {self._dictation_hotkey} | Grammar: {self._grammar_hotkey} | Rephrase: {self._rephrase_hotkey}]"

    def _update_rephrase_hotkey(self, new_hotkey):
        if self._rephrase_hotkey_handle is not None:
            keyboard.remove_hotkey(self._rephrase_hotkey_handle)
        self._rephrase_hotkey = new_hotkey
        self._rephrase_hotkey_handle = keyboard.add_hotkey(
            self._rephrase_hotkey, self._on_rephrase_hotkey, suppress=True
        )
        if self.icon:
            self.icon.title = f"TinyReadAloud v{__version__}  [Read: {self._hotkey} | Dictation: {self._dictation_hotkey} | Grammar: {self._grammar_hotkey} | Rephrase: {self._rephrase_hotkey}]"

    def _refresh_menu(self):
        if self.icon:
            self.icon.menu = self._build_menu()
            self.icon.update_menu()

    def _cmd_exit(self, icon, item):
        if self._hotkey_handle is not None:
            keyboard.remove_hotkey(self._hotkey_handle)
        if self._dictation_hotkey_handle is not None:
            keyboard.remove_hotkey(self._dictation_hotkey_handle)
        if self._grammar_hotkey_handle is not None:
            keyboard.remove_hotkey(self._grammar_hotkey_handle)
        if self._rephrase_hotkey_handle is not None:
            keyboard.remove_hotkey(self._rephrase_hotkey_handle)
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
