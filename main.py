"""Podvia 通用语音合成工具 —— CLI 入口。

命令总览：
  parse     解析脚本，打印 segments 摘要
  generate  解析→生成→自动合并（全流程）
  retry     重新生成 failed/pending 片段，成功后自动合并
  merge     仅重新合并
  srt       按最终拼接时间轴导出句级 SRT 字幕
  status    查看 manifest 摘要
  book      导入章节文件（可选输入方式），分段→生成→自动合并
  regen     局部重生成指定编号片段
  gui       启动图形界面（不带参数时默认启动 GUI）

退出码：0 成功 | 1 一般错误 | 2 配置错误 | 3 脚本解析失败 | 4 存在失败片段 | 5 FFmpeg 缺失/合并失败
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

# Windows 控制台强制 UTF-8，避免中文输出乱码（GBK 控制台）
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from podcast_tts import log as logs
from podcast_tts.audio import MergeError, run_merge, segments_signature
from podcast_tts.config import AppConfig, ConfigError, load_config
from podcast_tts.manifest import load as load_manifest
from podcast_tts.manifest import manifest_path
from podcast_tts.models import Segment
from podcast_tts.book import BookError, parse_book
from podcast_tts.parser import ParseError, ScriptParser
from podcast_tts.pipeline import run_book, run_generate, run_regenerate, run_retry
from podcast_tts.subtitles import SrtError, export_srt, export_subtitle_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podvia",
        description="Podvia 通用语音合成工具：输入文本或章节文件，按角色与音色参数生成完整音频。",
    )
    sub = parser.add_subparsers(dest="command")

    p_parse = sub.add_parser("parse", help="解析脚本，打印 segments 摘要（调试用）")
    p_parse.add_argument("script", help="脚本 Markdown 路径，如 scripts/episode001.md")

    p_gen = sub.add_parser("generate", help="解析→生成→自动合并（全流程）")
    p_gen.add_argument("script", help="脚本路径或 episode id，如 scripts/episode001.md 或 episode001")
    p_gen.add_argument("--force", action="store_true", help="忽略缓存，全部重新生成")
    p_gen.add_argument("--no-merge", action="store_true", help="生成后不自动合并")
    p_gen.add_argument("--no-resume", action="store_true", help="忽略已有成功记录，重新生成")

    p_retry = sub.add_parser("retry", help="重新生成 failed/pending 的片段，成功后自动合并")
    p_retry.add_argument("episode", help="episode id，如 episode001")

    p_merge = sub.add_parser("merge", help="仅重新合并")
    p_merge.add_argument("episode", help="episode id，如 episode001")
    p_merge.add_argument("--force", action="store_true", help="跳过签名检查，强制重新合并")

    p_srt = sub.add_parser(
        "srt", help="按最终拼接时间轴导出句级 SRT 字幕（与成品音频同源）")
    p_srt.add_argument("episode", help="episode id，如 episode001")
    p_srt.add_argument("--output", default=None,
                       help="输出 .srt 路径（默认 projects/<episode>/output/<episode>.srt）")

    p_sjson = sub.add_parser(
        "subtitle-json",
        help="导出 subtitle.json 结构化字幕数据（与 SRT 同一时间轴，供外部渲染系统使用）")
    p_sjson.add_argument("episode", help="episode id，如 episode001")
    p_sjson.add_argument("--output", default=None,
                         help="输出路径（默认 projects/<episode>/output/<episode>.subtitle.json）")

    p_status = sub.add_parser("status", help="查看 manifest 摘要：总数/成功/失败/最近错误")
    p_status.add_argument("episode", help="episode id，如 episode001")

    p_book = sub.add_parser("book", help="导入章节文件（可选输入方式），分段→生成→自动合并")
    p_book.add_argument("book", help="章节/文本文件路径（.md/.txt，UTF-8）")
    p_book.add_argument("--speaker", default=None, help="全文默认朗读者（配置中的角色名）")
    p_book.add_argument("--force", action="store_true", help="忽略缓存，全部重新生成")
    p_book.add_argument("--no-merge", action="store_true", help="生成后不自动合并")
    p_book.add_argument("--no-resume", action="store_true", help="忽略已有成功记录，重新生成")

    p_gui = sub.add_parser("gui", help="启动图形界面（不带任何命令行参数时默认启动 GUI）")

    p_regen = sub.add_parser("regen", help="局部重生成指定编号的台词片段（忽略缓存）")
    p_regen.add_argument("episode", help="episode id，如 mybook")
    p_regen.add_argument("indices", nargs="+", type=int, help="片段编号（一个或多个）")

    return parser


def _load_config_or_exit(key_required: bool = True) -> AppConfig | None:
    """加载配置；失败打印 [ERROR] 并返回 None（调用方返回退出码 2）。"""
    try:
        return load_config(key_required=key_required)
    except ConfigError as exc:
        logs.error(str(exc))
        return None


def _resolve_script_path(cfg: AppConfig, ref: str) -> Path:
    """把命令行引用解析为脚本路径：优先按给定路径，其次 scripts_dir/<ref>.md。"""
    candidate = Path(ref)
    if candidate.exists():
        return candidate
    # 引用已带 .md 后缀时不再重复拼接（避免 scripts/x.md.md 误导排查）
    by_id = cfg.scripts_dir / (ref if ref.lower().endswith(".md") else f"{ref}.md")
    if by_id.exists():
        return by_id
    raise FileNotFoundError(f"找不到脚本文件：'{ref}'（尝试过 '{candidate}' 和 '{by_id}'）。")


def _display_width(text: str) -> int:
    """按终端显示宽度计算（中日韩全角字符计 2）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def _fit(text: str, width: int) -> str:
    """按显示宽度截断（超出加省略号）并补空格对齐。仅影响展示，不改数据。"""
    if _display_width(text) > width:
        out = ""
        for ch in text:
            if _display_width(out) + _display_width(ch) > width - 1:
                break
            out += ch
        text = out + "…"
    return text + " " * (width - _display_width(text))


def _print_segment_table(segments: list[Segment]) -> None:
    speech_count = sum(1 for seg in segments if seg.kind == "speech")
    logs.info(f"共 {len(segments)} 段：{speech_count} 台词 / {len(segments) - speech_count} 停顿")
    for seg in segments:
        if seg.kind == "pause":
            print(f"  {seg.index:>4}  PAUSE   {seg.pause_seconds:.1f}s")
            continue
        direction = f"({seg.direction})" if seg.direction else ""
        print(
            f"  {seg.index:>4}  {seg.speaker:<6} {_fit(direction, 28)}  {_fit(seg.text or '', 40)}"
        )


def _cross_check_voice_config(cfg: AppConfig, parser: ScriptParser) -> None:
    """脚本 [全局配置] 与 config.yaml 交叉校验：不一致时警告，以 config.yaml 为准。"""
    for name, setting in cfg.voices.items():
        script_voice = parser.voice_map.get(name)
        if script_voice and script_voice != setting.voice:
            logs.warning(
                f"脚本 [全局配置] VOICE：{name}={script_voice} 与 config.yaml"
                f"（{setting.voice}）不一致，以 config.yaml 为准。"
            )
        script_language = parser.language_map.get(name)
        if script_language and setting.language_type and script_language != setting.language_type:
            logs.warning(
                f"脚本 [全局配置] LANGUAGE：{name}={script_language} 与 config.yaml"
                f"（{setting.language_type}）不一致，以 config.yaml 为准。"
            )


def cmd_parse(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(key_required=False)
    except ConfigError as exc:
        logs.error(str(exc))
        return 2

    try:
        script_path = _resolve_script_path(cfg, args.script)
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1

    logs.info(f"解析脚本：{script_path}")
    try:
        source = script_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        logs.error(f"读取脚本失败：{exc}")
        return 1

    # 与生成链路一致：脚本说话人按"显示名 → 稳定ID"解析（CLI 配置显示名通常即键）
    parser = ScriptParser(
        {(s.display_name or sid): sid for sid, s in cfg.voices.items()}
    )
    try:
        segments = parser.parse(source)
    except ParseError as exc:
        logs.error(f"脚本解析失败：{exc}")
        return 3

    _cross_check_voice_config(cfg, parser)
    _print_segment_table(segments)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit()
    if cfg is None:
        return 2
    try:
        script_path = _resolve_script_path(cfg, args.script)
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1

    logs.enable_file_logging(cfg.logs_dir, script_path.stem)
    try:
        outcome = run_generate(
            script_path, cfg, force=args.force, no_resume=args.no_resume
        )
    except ParseError as exc:
        logs.error(f"脚本解析失败：{exc}")
        return 3
    except ConfigError as exc:
        logs.error(str(exc))
        return 2

    if outcome.failed == 0 and not args.no_merge:
        return _auto_merge(cfg, script_path.stem)
    return outcome.exit_code


def _auto_merge(cfg: AppConfig, episode: str) -> int:
    """generate/retry 成功后的自动合并（文档 17）。合并失败 → 退出码 5。"""
    try:
        run_merge(episode, cfg)
    except MergeError as exc:
        logs.error(str(exc))
        return 5
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit()
    if cfg is None:
        return 2

    logs.enable_file_logging(cfg.logs_dir, args.episode)
    try:
        outcome = run_retry(args.episode, cfg)
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1
    except ConfigError as exc:
        logs.error(str(exc))
        return 2

    if outcome.failed == 0:
        return _auto_merge(cfg, args.episode)
    return outcome.exit_code


def cmd_merge(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit(key_required=False)
    if cfg is None:
        return 2

    logs.enable_file_logging(cfg.logs_dir, args.episode)
    try:
        run_merge(args.episode, cfg, force=args.force)
    except MergeError as exc:
        logs.error(str(exc))
        return 5
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1
    return 0


def cmd_srt(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit(key_required=False)
    if cfg is None:
        return 2

    logs.enable_file_logging(cfg.logs_dir, args.episode)
    out = Path(args.output) if args.output else None
    try:
        path, count = export_srt(args.episode, cfg, out)
    except SrtError as exc:
        logs.error(str(exc))
        return 1
    except MergeError as exc:
        logs.error(str(exc))
        return 5
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1
    if count == 0:
        logs.info(f"已写出 {path}，但项目没有台词片段（纯停顿），字幕为 0 条。")
    else:
        logs.ok(f"已导出 {count} 条字幕：{path}")
    return 0


def cmd_subtitle_json(args: argparse.Namespace) -> int:
    """导出 subtitle.json（结构化字幕数据，供 Remotion 等外部渲染系统使用）。

    与 srt 子命令同源：同一份整数毫秒时间轴，绝不解析 SRT。
    """
    cfg = _load_config_or_exit(key_required=False)
    if cfg is None:
        return 2

    logs.enable_file_logging(cfg.logs_dir, args.episode)
    out = Path(args.output) if args.output else None
    try:
        path, count = export_subtitle_json(args.episode, cfg, out)
    except SrtError as exc:
        logs.error(str(exc))
        return 1
    except MergeError as exc:
        logs.error(str(exc))
        return 5
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1
    if count == 0:
        logs.info(f"已写出 {path}，但项目没有台词片段（纯停顿），字幕数据为 0 条。")
    else:
        logs.ok(f"已导出 {count} 条字幕数据：{path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit(key_required=False)
    if cfg is None:
        return 2
    episode_dir = cfg.projects_dir / args.episode
    manifest = load_manifest(episode_dir)
    if manifest is None:
        logs.error(
            f"未找到 {manifest_path(episode_dir)}（episode 可能尚未生成，"
            f"先运行 generate 命令）。"
        )
        return 1

    speech = [r for r in manifest.segments if r.kind == "speech"]
    pauses = [r for r in manifest.segments if r.kind == "pause"]
    succeeded = sum(1 for r in speech if r.status == "success")
    failed = sum(1 for r in speech if r.status == "failed")
    pending = len(speech) - succeeded - failed
    logs.info(f"episode={manifest.episode}  模型={manifest.provider}/{manifest.model}")
    logs.info(
        f"共 {len(manifest.segments)} 段（{len(speech)} 台词 / {len(pauses)} 停顿）："
        f"台词成功 {succeeded} / 失败 {failed} / 待处理 {pending}"
    )
    if manifest.merge.get("status"):
        stale = ""
        output = episode_dir / str(manifest.merge.get("output") or "")
        if manifest.merge.get("segments_signature") != segments_signature(episode_dir, manifest):
            stale = "（片段已变化，建议重新 merge）"
        elif not output.is_file():
            stale = "（输出文件丢失，建议重新 merge）"
        logs.info(
            f"合并：{manifest.merge.get('output')}（{manifest.merge.get('merged_at')}）{stale}"
        )

    errors = [
        (r.index, r.speaker or "", r.error or "")
        for r in speech
        if r.status == "failed" and r.error
    ]
    if errors:
        logs.warning("失败片段（最多显示 10 条）：")
        for index, speaker, error in errors[:10]:
            print(f"  {index:04d} {speaker}: {error}")
    return 0


def cmd_book(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit()
    if cfg is None:
        return 2
    book_path = Path(args.book)
    if not book_path.exists():
        logs.error(f"找不到章节文件：{book_path}")
        return 1

    logs.enable_file_logging(cfg.logs_dir, book_path.stem)
    try:
        outcome = run_book(
            book_path, cfg, force=args.force, no_resume=args.no_resume,
            default_speaker=args.speaker,
        )
    except BookError as exc:
        logs.error(f"章节文件解析失败：{exc}")
        return 3
    except ConfigError as exc:
        logs.error(str(exc))
        return 2

    if outcome.failed == 0 and not args.no_merge:
        return _auto_merge(cfg, book_path.stem)
    return outcome.exit_code


def cmd_regen(args: argparse.Namespace) -> int:
    cfg = _load_config_or_exit()
    if cfg is None:
        return 2

    logs.enable_file_logging(cfg.logs_dir, args.episode)
    try:
        outcome = run_regenerate(args.episode, args.indices, cfg)
    except FileNotFoundError as exc:
        logs.error(str(exc))
        return 1
    except ConfigError as exc:
        logs.error(str(exc))
        return 2

    if outcome.failed == 0:
        return _auto_merge(cfg, args.episode)
    return outcome.exit_code


def _launch_gui() -> int:
    from podcast_tts.gui.app import main as gui_main
    gui_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # 无参数（双击 exe）默认启动 GUI
    if not argv:
        logs.setup_logging()
        return _launch_gui()

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 初始化控制台日志；文件日志在 Phase 5 拿到 episode 后再启用
    logs.setup_logging()

    handlers = {
        "parse": cmd_parse,
        "generate": cmd_generate,
        "retry": cmd_retry,
        "merge": cmd_merge,
        "srt": cmd_srt,
        "subtitle-json": cmd_subtitle_json,
        "status": cmd_status,
        "book": cmd_book,
        "regen": cmd_regen,
        "gui": lambda _args: _launch_gui(),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
