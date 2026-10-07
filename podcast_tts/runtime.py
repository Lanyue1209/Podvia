"""运行环境路径处理（阶段 7：PyInstaller 冻结后也能在"无开发环境"目录运行）。

目录策略：
- 开发态：仓库根（config.yaml / scripts/ / projects/ 都在仓库内）。
- 冻结态（PyInstaller）：程序目录只读且可能位于 Program Files / 中文或带空格路径，
  所有可写内容（外部配置、脚本、缓存 projects、日志、.env）一律放
  %APPDATA%\\Podvia（用户目录），首次启动从包内模板复制 config.yaml。
- 打包产物永远不包含 .env / API Key；首启创建的 .env 只有空键，由用户填写
  或在 GUI 中输入（GUI 的 Key 存 %APPDATA%/Podvia/gui_settings.json）。
- FFmpeg：随包分发时位于包内 tools/bin；否则回退用户目录 tools/bin，再回退 PATH。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "Podvia"
_USER_SUBDIRS = ("scripts", "projects", "logs", "tools", "tools/bin")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """PyInstaller 解包目录（onefile 时是临时 _MEIPASS；onedir 时是 exe 目录）。"""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else Path(sys.executable).resolve().parent


def user_data_dir() -> Path:
    base = os.getenv("APPDATA") or str(Path.home())
    return Path(base) / APP_DIR_NAME


def ensure_user_root(root: Path, template_config: Path) -> Path:
    """首次启动初始化用户数据目录。返回 root。

    - config.yaml 缺失时从包内模板复制；已存在则绝不覆盖（保留用户配置）。
    - 创建空 .env（无密钥）；绝不复制开发机的任何 .env。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in _USER_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    target_cfg = root / "config.yaml"
    if not target_cfg.exists():
        if not Path(template_config).is_file():
            raise FileNotFoundError(
                f"包内缺少配置模板 {template_config}，请重新获取安装包。"
            )
        shutil.copyfile(template_config, target_cfg)

    env_file = root / ".env"
    if not env_file.exists():
        env_file.write_text("# 在此粘贴百炼 API Key（或使用 GUI 配置）\n"
                           "DASHSCOPE_API_KEY=\n", encoding="utf-8")
    return root


def bundled_ffmpeg_dir() -> Path | None:
    """随包分发的 FFmpeg 目录（含 ffmpeg.exe 时才返回）。"""
    candidate = bundle_dir() / "tools" / "bin"
    if (candidate / "ffmpeg.exe").is_file():
        return candidate
    return None
