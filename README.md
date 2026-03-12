# TinyReadAloud

Select text, press a hotkey, hear it read aloud. A lightweight Windows system tray app powered by [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) text-to-speech — with built-in dictation, AI grammar checking, and rephrase styles.

## Features

### Text-to-Speech
- **Global hotkey** — configurable shortcut (default `Ctrl+Alt+R`) to read any selected text aloud; press again to stop
- **Automatic language detection** — detects English or Spanish text and switches to the appropriate voice automatically
- **Multi-language voices** — American English, British English, and Spanish voices from Kokoro
- **GPU acceleration** — uses CUDA with fp16 model on NVIDIA GPUs; falls back to CPU with int8 model automatically
- **Streaming TTS** — audio is streamed in chunks for fast first-word latency and responsive stop (50ms granularity)
- **Any-key stops playback** — press any key during speech to stop instantly (0.5s debounce to avoid hotkey self-cancel)
- **Speed control** — Slow (0.8×), Normal (1.0×), Fast (1.2×), Very Fast (1.5×)

### Dictation
- **Windows voice dictation** — toggle dictation via hotkey (default `Ctrl+Alt+D`); uses the built-in Windows `Win+H` speech recognition
- **Any-key stops dictation** — press any key during listening to stop (0.8s debounce)
- **Text field focus check** — dictation only starts if a text input field is focused; shows a notification otherwise
- **Microphone selector** — choose a specific input device or use the system default

### Grammar Checking
- **AI-powered grammar correction** — uses the Anthropic Claude API to fix grammar and punctuation
- **Three modes:** Off, Manual (hotkey/menu only), or After Dictation (auto-runs when dictation stops)
- **Hotkey trigger** — default `Ctrl+Alt+G` to check grammar on the current document (Select All → Copy → Correct → Paste)
- **Auto-trigger after playback** — grammar check runs automatically when any-key stops playback (if grammar mode is enabled)

### Rephrase Styles
- **6 built-in styles:** Natural, Formal, Casual, Concise, Expanded, Professional
- **Per-style hotkeys** — assign a unique hotkey to each style for one-press rephrase
- **Recall last style hotkey** — switch back to the previously used rephrase style
- **Hotkey trigger** — default `Ctrl+Alt+P` to rephrase with the active style

### Floating Status Bar
- **Always-on-top draggable bar** — shows current status (Ready, Speaking…, Listening…, Checking grammar…, Rephrasing…)
- **Style switcher** — ◀/▶ arrow buttons to cycle through rephrase styles; click the style name to rephrase
- **Settings gear ⚙** — quick access to the Settings window
- **Non-focus-stealing** — uses Win32 `WS_EX_NOACTIVATE` so it never takes keyboard focus from your editor

### System Tray
- **Dynamic icon** — red when idle, green when speaking
- **Right-click menu** — Stop Reading, Start/Stop Dictation, Check Grammar, Rephrase, voice submenus, speed, Settings, update options, Exit

### Updates
- **Auto-update check** — checks GitHub Releases 5 seconds after launch; shows a tray notification when a new version is found
- **Silent update** — downloads the installer to temp and runs it with `/SILENT`; the running app exits automatically before upgrade
- **Optional startup with Windows** — installer checkbox to add a registry Run entry

### Dark Theme UI
- **Modern dark settings window** — two-column layout with Catppuccin-inspired colors and purple accent
- **Rounded corners** — uses DWM attributes for native Win11 rounded corners

## Install (Windows Setup)

1. Download `TinyReadAloud-X.Y.Z-Setup.exe` (or `setup.exe`) from [Releases](https://github.com/dorofino/TinyReadAloud/releases)
2. Run the installer and keep the default path in `Program Files`
3. Re-run any newer installer to upgrade in place; existing install is overwritten
4. Model files (~200 MB) download automatically on first launch

### From source

#### Prerequisites

- Python 3.10+ (tested on 3.13)
- Windows 10/11

#### Setup

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

#### Run

```bash
python app.py
```

On first run, the app downloads the TTS model and voice files (~200 MB) to `%LOCALAPPDATA%\TinyReadAloud\`.

## Usage

1. Select any text in any application
2. Press **Ctrl+Alt+R** (or your configured hotkey) to hear it read aloud
3. Press the hotkey again — or any other key — to stop playback
4. Press **Ctrl+Alt+D** to start/stop dictation (a text field must be focused)
5. Press **Ctrl+Alt+G** to run grammar check on the current document
6. Press **Ctrl+Alt+P** to rephrase with the active style
7. Right-click the tray icon for quick access to all features

### Tray Menu

| Item | Description |
|------|-------------|
| **Stop Reading** | Stop current playback (visible only when speaking) |
| **Start / Stop Dictation** | Toggle Windows voice dictation |
| **Check Grammar** | Run grammar check on the active document |
| **Rephrase** | Rephrase with the current style |
| **English Voice** | Submenu — pick from all American & British English voices |
| **Spanish Voice** | Submenu — pick from all Spanish voices |
| **Speed** | Submenu — Slow / Normal / Fast / Very Fast |
| **Settings** | Open the Settings window |
| **Check for Updates** | Manually check for a new version |
| **Download vX.Y.Z** | Appears when an update is available |
| **Exit** | Quit the app |

## Settings Window

Open from the tray menu → **Settings** (or click ⚙ on the floating status bar). All changes take effect after clicking **Save**.

### Left Column

| Section | Setting | Control | Description |
|---------|---------|---------|-------------|
| **Voice & Playback** | English voice | Dropdown | All Kokoro English voices with live preview |
| | Spanish voice | Dropdown | All Kokoro Spanish voices with live preview |
| | Speed | Dropdown | Slow · Normal · Fast · Very Fast (with live preview) |
| **Shortcuts** | Read aloud | Entry + Record ⏺ + Clear ✕ | Press Record, then type any key combo |
| | Dictation | Entry + Record + Clear | Hotkey to toggle dictation |
| | Grammar | Entry + Record + Clear | Hotkey to run grammar check |
| **Grammar & Dictation** | Grammar mode | Dropdown | Off · Manual · After Dictation |
| | Dictation provider | Dropdown | Windows (uses Win+H) |
| | Microphone | Dropdown | System default or a specific input device |
| | Grammar provider | Dropdown | Anthropic |
| **Anthropic API** | API key | Password entry | Your Anthropic API key (masked) |
| | Model | Text entry | e.g. `claude-sonnet-4-6` |

### Right Column

| Section | Setting | Control | Description |
|---------|---------|---------|-------------|
| **Rephrase Styles** | Active style | Dropdown | Natural · Formal · Casual · Concise · Expanded · Professional |
| | Per-style hotkeys | 6 × Entry + Record + Clear | One hotkey per style for instant rephrase |
| | Recall last style | Entry + Record + Clear | Hotkey to revert to the previous style |

Voice display format: `Name (Language, Gender)` — e.g. *Heart (American, Female)*.

Hotkey conflict detection prevents assigning the same shortcut to two actions.

## Configuration

Settings are stored in `%LOCALAPPDATA%\TinyReadAloud\config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey` | `ctrl+alt+r` | Global read-aloud shortcut |
| `dictation_hotkey` | `ctrl+alt+d` | Start/stop dictation |
| `grammar_hotkey` | `ctrl+alt+g` | Run grammar check |
| `rephrase_hotkey` | `ctrl+alt+p` | Rephrase with current style |
| `rephrase_style` | `Natural` | Active rephrase style |
| `voice_en` | `af_jessica` | English voice code |
| `voice_es` | `ef_dora` | Spanish voice code |
| `speed` | `1.0` | Playback speed (`0.8`, `1.0`, `1.2`, `1.5`) |
| `grammar_mode` | `manual` | `off`, `manual`, or `after_dictation` |
| `grammar_provider` | `anthropic` | Grammar/rephrase API provider |
| `anthropic_api_key` | *(empty)* | Anthropic API key |
| `anthropic_model` | `claude-sonnet-4-6` | Anthropic model name |
| `dictation_provider` | `windows` | Dictation backend |
| `style_hotkeys` | `{}` | Per-style hotkey mappings |
| `recall_style_hotkey` | *(empty)* | Hotkey to recall last style |

### Model & data files

Stored in `%LOCALAPPDATA%\TinyReadAloud\`:

| File | Size | Description |
|------|------|-------------|
| `kokoro-v1.0.fp16.onnx` | ~160 MB | FP16 model (GPU/CUDA) |
| `kokoro-v1.0.int8.onnx` | ~80 MB | INT8 model (CPU) |
| `voices-v1.0.bin` | ~15 MB | Voice embeddings |

Only the model matching your hardware is kept; the other variant is deleted automatically.

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

For GPU variant (includes NVIDIA CUDA DLLs, ~1.7 GB installer):

```bash
build.bat --gpu
```

## Release workflow

1. Bump version in `version.py`
2. Run `build.bat` (or `build.bat --gpu`)
3. Create a GitHub Release tagged `vX.Y.Z`
4. Attach `installer_output\TinyReadAloud-X.Y.Z-Setup.exe` as a release asset

Running instances will detect the new version and offer to download/install it.

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

The Anthropic Python SDK is **not** required — API calls are made directly via `urllib`.

## Project structure

```
TinyReadAloud/
├── app.py              Main application (TTS, dictation, grammar, rephrase, settings UI, tray, hotkeys)
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
