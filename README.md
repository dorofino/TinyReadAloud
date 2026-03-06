# TinyReadAloud

Select text, press **Ctrl+Alt+R**, hear it read aloud. A lightweight Windows system tray app powered by [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) text-to-speech.

## Features

- **Global hotkey** (Ctrl+Alt+R) to read any selected text aloud
- **Automatic language detection** — switches between English and Spanish voices
- **GPU acceleration** — uses CUDA with fp16 model on NVIDIA GPUs, falls back to CPU with int8
- **Settings window** — configure hotkey, English/Spanish voices, speed, with live preview
- **Auto-update** — checks GitHub Releases for new versions on startup
- **System tray** — runs quietly in the background with right-click menu

## Install (from release)

1. Download the latest `TinyReadAloud-x.x.x-Setup.exe` from [Releases](https://github.com/dorof/TinyReadAloud/releases)
2. Run the installer
3. Model files (~200MB) download automatically on first launch

## Install (from source)

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Windows 10/11

### Setup

```bash
git clone https://github.com/dorof/TinyReadAloud.git
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
2. Press **Ctrl+Alt+R** to hear it read aloud
3. Press **Ctrl+Alt+R** again to stop
4. Right-click the tray icon for settings, voice selection, and speed control

## Build installer

### Prerequisites

- [Inno Setup 6](https://jrsoftware.org/issetup.exe)
- PyInstaller (`pip install pyinstaller`)

### Build

```bash
build.bat
```

This produces `installer_output\TinyReadAloud-x.x.x-Setup.exe`.

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
| `voice_en` | `af_heart` | English voice code |
| `voice_es` | `ef_dora` | Spanish voice code |
| `speed` | `1.0` | Playback speed (0.8 - 1.5) |

## Project structure

```
TinyReadAloud/
├── app.py              Main application
├── version.py          Version string
├── updater.py          GitHub Releases update checker
├── requirements.txt    Python dependencies
├── generate_icon.py    Creates assets/app.ico from tray icon
├── tinyreadaloud.spec  PyInstaller build spec
├── installer.iss       Inno Setup installer script
└── build.bat           Build automation
```
