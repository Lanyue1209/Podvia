"""日志模块。

控制台：INFO 级，[INFO]/[OK]/[RETRY]/[WARNING]/[ERROR] 前缀，无时间戳（人读体验）。
文件：仅当指定 episode 时，写 logs/<episode>.log，DEBUG 级，含时间戳（排查用）。
"""
from __future__ import annotations

import logging
from pathlib import Path

# 自定义日志级别：RETRY、OK 位于 INFO(20) 与 WARNING(30) 之间
RETRY = 23
OK = 25
logging.addLevelName(RETRY, "RETRY")
logging.addLevelName(OK, "OK")

_NAMESPACE = "podcast_tts"
_configured = False


def get_logger() -> logging.Logger:
    return logging.getLogger(_NAMESPACE)


def setup_logging() -> None:
    """初始化控制台日志（幂等，重复调用无副作用）。"""
    global _configured
    if _configured:
        return
    root = logging.getLogger(_NAMESPACE)
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)

    _configured = True


def enable_file_logging(logs_dir: Path, episode: str) -> None:
    """为 episode 启用文件日志 logs/<episode>.log（DEBUG 级，含时间戳）。幂等。"""
    root = logging.getLogger(_NAMESPACE)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "_episode", None) == episode:
            return
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / f"{episode}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    file_handler._episode = episode
    root.addHandler(file_handler)


# 模块级便捷函数：业务代码用 logs.info(...) / logs.ok(...) 等
def debug(msg: str) -> None:
    get_logger().debug(msg)


def info(msg: str) -> None:
    get_logger().info(msg)


def ok(msg: str) -> None:
    get_logger().log(OK, msg)


def retry(msg: str) -> None:
    get_logger().log(RETRY, msg)


def warning(msg: str) -> None:
    get_logger().warning(msg)


def error(msg: str) -> None:
    get_logger().error(msg)
