# -*- mode: python ; coding: utf-8 -*-
"""Podvia Windows 可重复打包规格（PyInstaller，阶段 7）。

构建：powershell packaging/build.ps1
产物：dist/Podvia/（onedir，双击 Podvia.exe 启动 GUI；命令行见 --help）

安全要求（build.ps1 会再次检查）：
- 绝不把 .env / API Key 打入包内；datas 只放 config.yaml 模板。
- tools/bin 下存在 ffmpeg.exe/ffprobe.exe 时才随包分发，否则要求用户单独安装
  （包启动后 config.yaml 的合并会给出明确下载指引）。
"""
from pathlib import Path

ROOT = Path(SPECPATH).parent

datas = [(str(ROOT / "config.yaml"), ".")]
binaries = []

tool_dir = ROOT / "tools" / "bin"
for tool_name in ("ffmpeg.exe", "ffprobe.exe"):
    tool_path = tool_dir / tool_name
    if tool_path.is_file():
        binaries.append((str(tool_path), "tools/bin"))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "yaml",
        "dotenv",
        "podcast_tts",
        "podcast_tts.gui",
        "podcast_tts.gui.app",
        "podcast_tts.gui.controller",
        "podcast_tts.gui.settings",
        "podcast_tts.gui.theme",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["Podvia-Android", "pytest", "pip", "setuptools"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Podvia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # CLI 与 GUI 共用一个 exe，保留控制台以便查看日志
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Podvia",
)
