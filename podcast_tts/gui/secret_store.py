"""API Key 本机安全存储（阶段"通用语音合成工具"）。

取舍说明：
- Windows 优先使用 DPAPI（CryptProtectData，当前用户作用域）：密文只能由同一
  Windows 用户在同一台机器上解开，不引入任何第三方依赖，PyInstaller 打包零成本。
  这比 Windows Credential Manager（需要额外封装 CredWrite API/依赖 keyring 库）
  更简单可靠；密文以 base64 存放在 %APPDATA%/Podvia/gui_settings.json。
- 非 Windows（开发/测试环境）没有 DPAPI：降级为仅 base64 混淆并显式打标
  "plain:"，绝不能误认为加密。生产只发布 Windows 包。
- API Key 永远不写入仓库 .env / config.yaml / manifest / 日志 / 打包目录。
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wt
import getpass
import os
import subprocess
from pathlib import Path

_ENTROPY = b"Podvia-tts-key-v1"
_PREFIX = "dpapi:"
_PLAIN_PREFIX = "plain:"


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wt.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_available() -> bool:
    return os.name == "nt" and hasattr(ctypes, "windll")


def _blob_from_bytes(data: bytes) -> _DATA_BLOB:
    buf = ctypes.create_string_buffer(data, len(data))
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def protect(plaintext: str) -> str:
    """加密；返回可直接写入 JSON 的字符串令牌。"""
    if plaintext is None:
        plaintext = ""
    raw = plaintext.encode("utf-8")
    if not _dpapi_available():
        # 非 Windows：明确标记的降级存储（仅开发环境使用）
        return _PLAIN_PREFIX + base64.b64encode(raw).decode("ascii")

    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    in_blob = _blob_from_bytes(raw)
    ent_blob = _blob_from_bytes(_ENTROPY)
    out_blob = _DATA_BLOB()
    # CRYPTPROTECT_UI_FORBIDDEN=0x1（不弹任何 UI）；当前用户作用域
    ok = crypt32.CryptProtectData(
        ctypes.byref(in_blob), None, ctypes.byref(ent_blob),
        None, None, 0x1, ctypes.byref(out_blob),
    )
    if not ok:
        raise ctypes.WinError()  # type: ignore[attr-defined]
    try:
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)
    return _PREFIX + base64.b64encode(encrypted).decode("ascii")


def unprotect(token: str) -> str:
    """解密 protect() 的产物；无法识别时抛 ValueError。"""
    if not token:
        return ""
    if token.startswith(_PLAIN_PREFIX):
        return base64.b64decode(token[len(_PLAIN_PREFIX):]).decode("utf-8")
    if not token.startswith(_PREFIX):
        # 历史明文：直接返回（迁移在 settings 层完成）
        return token
    if not _dpapi_available():
        raise ValueError("该 API Key 令牌由 Windows DPAPI 加密，当前系统无法解密。")

    raw = base64.b64decode(token[len(_PREFIX):])
    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    in_blob = _blob_from_bytes(raw)
    ent_blob = _blob_from_bytes(_ENTROPY)
    out_blob = _DATA_BLOB()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob), None, ctypes.byref(ent_blob),
        None, None, 0x1, ctypes.byref(out_blob),
    )
    if not ok:
        raise ctypes.WinError()  # type: ignore[attr-defined]
    try:
        data = ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)
    return data.decode("utf-8")


def mask_key(key: str) -> str:
    """日志/界面展示用脱敏：sk-abc...xyz，永不输出完整 Key。"""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}…{key[-3:]}"


def restrict_file_permissions(path: Path) -> bool:
    """尽量把设置文件限制为当前 Windows 用户可访问；失败不阻断保存。

    PyInstaller/普通用户环境下 icacls 始终可用；权限收紧失败时只返回 False，
    由调用方记录（DPAPI 密文本身已是用户作用域，这里是纵深防御）。
    """
    path = Path(path)
    if not _dpapi_available():
        return False
    user = os.getenv("USERNAME") or getpass.getuser()
    try:
        # 取消继承、移除既有授权、只给当前用户完全控制
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
            capture_output=True, check=False,
        )
    except OSError:
        return False
    return True
