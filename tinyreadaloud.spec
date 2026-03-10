# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TinyReadAloud."""

import importlib
import os

block_cipher = None


def get_pkg_dir(pkg_name):
    mod = importlib.import_module(pkg_name)
    mod_file = getattr(mod, "__file__", None)
    if mod_file:
        return os.path.dirname(mod_file)

    # Namespace packages (e.g. nvidia) may not define __file__.
    mod_path = getattr(mod, "__path__", None)
    if mod_path:
        for p in mod_path:
            if p:
                return p

    raise ImportError(f"Could not resolve package directory for {pkg_name}")


# Locate package data directories
espeakng_dir = get_pkg_dir("espeakng_loader")
langdetect_dir = get_pkg_dir("langdetect")
kokoro_dir = get_pkg_dir("kokoro_onnx")
sounddevice_data_dir = get_pkg_dir("_sounddevice_data")
phonemizer_dir = get_pkg_dir("phonemizer")
language_tags_dir = get_pkg_dir("language_tags")

# GPU build toggle (set TINYREADALOUD_GPU=1 for GPU variant)
gpu_build = os.environ.get("TINYREADALOUD_GPU", "0") == "1"

# Data files that must be bundled
datas = [
    # espeak-ng runtime (DLL + data)
    (os.path.join(espeakng_dir, "espeak-ng.dll"), "espeakng_loader"),
    (os.path.join(espeakng_dir, "espeak-ng-data"), "espeakng_loader/espeak-ng-data"),
    # langdetect language profiles
    (os.path.join(langdetect_dir, "profiles"), "langdetect/profiles"),
    # kokoro_onnx tokenizer config
    (os.path.join(kokoro_dir, "config.json"), "kokoro_onnx"),
    # phonemizer shared data
    (os.path.join(phonemizer_dir, "share"), "phonemizer/share"),
    # portaudio DLL for sounddevice
    (os.path.join(sounddevice_data_dir, "portaudio-binaries"), "_sounddevice_data/portaudio-binaries"),
    # language_tags JSON data (required by csvw -> segments -> phonemizer)
    (os.path.join(language_tags_dir, "data"), "language_tags/data"),
]

# For GPU builds, include NVIDIA CUDA DLLs from pip packages
binaries = []
if gpu_build:
    try:
        nvidia_base = os.path.dirname(get_pkg_dir("nvidia"))
        for subpkg in ("cublas", "cuda_runtime", "cudnn", "cufft", "nvjitlink"):
            bin_dir = os.path.join(nvidia_base, "nvidia", subpkg, "bin")
            if os.path.isdir(bin_dir):
                for dll in os.listdir(bin_dir):
                    if dll.lower().endswith(".dll"):
                        binaries.append((os.path.join(bin_dir, dll), "."))
    except ImportError:
        pass

hiddenimports = [
    "pystray._win32",
    "keyboard._winkeyboard",
    "PIL._tkinter_finder",
    "phonemizer.backend.espeak",
    "phonemizer.backend.espeak.espeak",
    "phonemizer.backend.espeak.wrapper",
    "phonemizer.backend.espeak.api",
    "phonemizer.backend.espeak.mbrola",
    "phonemizer.backend.espeak.voice",
    "espeakng_loader",
    "kokoro_onnx.config",
    "kokoro_onnx.tokenizer",
    "kokoro_onnx.trim",
    "kokoro_onnx.log",
    "version",
    "updater",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "pandas", "pytest", "IPython",
        "notebook", "sphinx", "docutils",
        # Prevent PyInstaller from traversing unrelated heavyweight ML stacks
        # that may exist in the local Python environment.
        "torch", "torchvision", "torchaudio",
        "tensorflow", "keras", "transformers",
        "sklearn", "cv2", "numba", "librosa", "soundfile",
        "xgboost", "lightgbm", "catboost", "faiss", "openvino",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TinyReadAloud",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="assets/app.ico",
    uac_admin=False,
)

# For CPU builds, exclude the huge CUDA/TensorRT provider DLLs
# that the onnxruntime hook pulls in automatically (~712MB)
_excluded_binaries = set()
if not gpu_build:
    _excluded_binaries = {
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_tensorrt.dll",
    }

# Filter out excluded binaries
_filtered_binaries = [(name, path, typecode) for name, path, typecode in a.binaries
                      if os.path.basename(path) not in _excluded_binaries]

coll = COLLECT(
    exe,
    _filtered_binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TinyReadAloud",
)
