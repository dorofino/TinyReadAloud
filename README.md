# TinyReadAloud

Select text, press **Ctrl+Alt+R**, hear it read aloud. A lightweight Windows system tray app powered by [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) text-to-speech.

## Features

- **Global hotkey** — configurable shortcut (default `Ctrl+Alt+R`) to read any selected text aloud; press again to stop
- **Automatic language detection** — detects English or Spanish text and switches to the appropriate voice automatically
- **Multi-language voices** — supports American English, British English, Spanish, French, Hindi, Italian, Japanese, Portuguese, and Chinese voice codes from Kokoro
- **GPU acceleration** — uses CUDA with fp16 model on NVIDIA GPUs; falls back to CPU with int8 model automatically
- **Streaming TTS** — audio is streamed in chunks for fast first-word latency and responsive stop (50ms granularity)
- **Settings window** — Tkinter dialog to configure hotkey, English/Spanish voices, and speed with live audio preview on every change
- **System tray** — runs in the background with a dynamic icon (red = idle, green = speaking) and right-click menu for quick access to voices, speed, settings, and updates
- **Auto-update** — checks GitHub Releases 5 seconds after launch; shows a tray notification and "Download" menu item when a new version is found
- **Silent update** — downloads the installer to temp and runs it with `/SILENT`; the running app is killed automatically before upgrade
- **Optional startup with Windows** — installer checkbox to add a registry Run entry

## Install (Windows Setup)

1. Download `TinyReadAloud-X.Y.Z-Setup.exe` (or `setup.exe`) from [Releases](https://github.com/dorofino/TinyReadAloud/releases)
2. Run the installer and keep the default path in `Program Files`
3. Re-run any newer installer to upgrade in place; existing install is overwritten
4. Model files (~200MB) download automatically on first launch

### From source

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Windows 10/11

### Setup

```bash
git clone https://github.com/dorofino/TinyReadAloud.git
cd TinyReadAloud
pip install -r requirements.txt
```

For **GPU acceleration** (NVIDIA GPUs with CUDA support):

```bash
pip install onnxruntime-gpu
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12
```

For **CPU only**:

```bash
pip install onnxruntime
```

### Run

```bash
python app.py
```

On first run, the app downloads the TTS model and voice files (~200MB) to `%LOCALAPPDATA%\TinyReadAloud\`.

## Usage

1. Select any text in any application
2. Press **Ctrl+Alt+R** (or your configured hotkey) to hear it read aloud
3. Press the hotkey again to stop playback immediately
4. Right-click the tray icon for quick access to:
   - **Stop Reading** — stop current playback
   - **English Voice** — submenu to pick from all available English voices
   - **Spanish Voice** — submenu to pick from all available Spanish voices
   - **Speed** — Slow (0.8×), Normal (1.0×), Fast (1.2×), Very Fast (1.5×)
   - **Settings** — open the settings window
   - **Check for Updates** — manually check for a new version
   - **Download vX.Y.Z** — appears when an update is available

## Settings Window

Open from the tray menu → **Settings**. All changes take effect after clicking **Save**.

| Setting | Control | Description |
|---------|---------|-------------|
| **Read-aloud shortcut** | Entry + **Record** button | Click Record, then press any key combo (e.g. `F3`, `ctrl+shift+s`). The new hotkey is shown in the entry field. |
| **English Voice** | Dropdown | Lists all Kokoro English voices (American & British). Selecting a voice plays a live preview: *"Hello! This is a preview."* |
| **Spanish Voice** | Dropdown | Lists all Kokoro Spanish voices. Selecting a voice plays a live preview: *"Hola, esta es una vista previa."* |
| **Speed** | Dropdown | Slow (0.8×) · Normal (1.0×) · Fast (1.2×) · Very Fast (1.5×). Changing speed plays a live preview with the current English voice. |

Voice display format: `Name (Language, Gender)` — e.g. *Heart (American, Female)*.

## Build installer

### Prerequisites

- [Inno Setup 6](https://jrsoftware.org/issetup.exe)
- PyInstaller (`pip install pyinstaller`)

### Build

```bash
build.bat
```

This produces `installer_output\TinyReadAloud-x.x.x-Setup.exe`.

Build also creates stable aliases:

- `installer_output\TinyReadAloud-Setup.exe`
- `installer_output\setup.exe`

For GPU variant (includes NVIDIA CUDA DLLs, ~800MB+ installer):

```bash
build.bat --gpu
```

## Release workflow

1. Bump version in `version.py`
2. Run `build.bat`
3. Create a GitHub Release tagged `vX.Y.Z`
4. Attach `installer_output\TinyReadAloud-X.Y.Z-Setup.exe` as a release asset

Running instances will detect the new version and offer to download/install it.

## Configuration

Settings are stored in `%LOCALAPPDATA%\TinyReadAloud\config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey` | `ctrl+alt+r` | Global read-aloud shortcut |
| `voice_en` | `af_heart` | English voice code (e.g. `af_heart`, `bm_george`) |
| `voice_es` | `ef_dora` | Spanish voice code |
| `speed` | `1.0` | Playback speed — `0.8`, `1.0`, `1.2`, or `1.5` |

Model and voice data files are also stored in `%LOCALAPPDATA%\TinyReadAloud\`:

| File | Size | Description |
|------|------|-------------|
| `kokoro-v1.0.fp16.onnx` | ~160 MB | FP16 model (used with GPU/CUDA) |
| `kokoro-v1.0.int8.onnx` | ~80 MB | INT8 model (used on CPU) |
| `voices-v1.0.bin` | ~15 MB | Voice embeddings (NPZ) |

Only the model matching your hardware is kept; the other variant is deleted automatically.

## Dependencies

| Package | Purpose |
|---------|---------|
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Kokoro TTS inference via ONNX Runtime |
| onnxruntime / onnxruntime-gpu | ONNX model execution (CPU or CUDA) |
| sounddevice | Audio playback via PortAudio |
| numpy | Audio sample manipulation |
| pystray | System tray icon and menu |
| Pillow | Tray icon image generation |
| keyboard | Global hotkey registration and capture |
| langdetect | Automatic English/Spanish language detection |

## Project structure

```
TinyReadAloud/
├── app.py              Main application (TTS worker, settings UI, tray, hotkey)
├── version.py          Version string (__version__)
├── updater.py          GitHub Releases update checker & silent installer download
├── config.json         Default config (dev only; runtime config in %LOCALAPPDATA%)
├── requirements.txt    Python dependencies
├── generate_icon.py    Creates assets/app.ico from the tray icon renderer
├── tinyreadaloud.spec  PyInstaller build spec (CPU + GPU variants)
├── installer.iss       Inno Setup installer script (startup option, silent upgrade)
├── build.bat           Build automation (PyInstaller → Inno Setup)
└── assets/
    └── app.ico         Application icon (multi-size ICO)
```

## License

This project is for personal use.
