"""SRT 句级字幕测试：时间轴必须与最终音频拼接同源。

覆盖：
- 时间格式化（0 / 毫秒 3 位 / 超 1 小时）、标签清理（官方情绪标签/SSML/普通方括号）
- 整数毫秒时间线：开头/中间/连续/结尾停顿、空正文跳过但时间不错位
- build_timeline 严格错误（缺音频、非 success、时长不可读、序列数量不一致）
- 端到端（FakeProvider 封网 + mock ffmpeg）：
  纯英文/中英交替/各类停顿/无停顿、导演指令与标签不入字幕、
  片段音频时长变化、改正文重生成、删除片段、调整顺序、改停顿时长（0 TTS）、
  独立导出（未合并时停顿用精确时长）、纯停顿项目、manifest 字幕快照
- GUI 控制器守卫：有 dirty 片段时拒绝导出
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_tts import audio as audio_mod
from podcast_tts.audio import AudioFormat, run_merge
from podcast_tts.config import ConfigError
from podcast_tts.gui.controller import GuiController
from podcast_tts.gui.settings import GuiSettings, RoleVoice
from podcast_tts.manifest import (
    Manifest,
    SegmentRecord,
    load as load_manifest,
    save as save_manifest,
)
from podcast_tts.models import TTSResult
from podcast_tts.pipeline import run_project
from podcast_tts.project import WorkingScript
from podcast_tts.providers.base import TTSProvider
from podcast_tts.subtitles import (
    SrtError,
    SubtitleCue,
    TimelineEntry,
    build_timeline,
    clean_spoken_text,
    cues_from_timeline,
    export_srt,
    format_srt_time,
    render_for_episode,
    render_srt,
)

from tests.test_pipeline import FakeProvider, make_cfg

NOW = "2026-10-06T00:00:00Z"


# —— 1. 纯函数：时间格式 ——————————————————————————————————————————————

def test_format_srt_time_boundaries():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(1) == "00:00:00,001"
    assert format_srt_time(999) == "00:00:00,999"
    assert format_srt_time(1000) == "00:00:01,000"
    assert format_srt_time(59_999) == "00:00:59,999"
    assert format_srt_time(60_000) == "00:01:00,000"
    assert format_srt_time(3_599_999) == "00:59:59,999"
    assert format_srt_time(3_600_000) == "01:00:00,000"
    # 超过 1 小时（长音频）
    assert format_srt_time(3_723_456) == "01:02:03,456"
    assert format_srt_time(37_234_567) == "10:20:34,567"
    # 负值防御（不应出现）
    assert format_srt_time(-5) == "00:00:00,000"


# —— 2. 纯函数：字幕文本清洗 ————————————————————————————————————————————

def test_clean_spoken_text_remends_control_and_ssml_only():
    # 官方情绪/拟声标签删除，正文与标点保留
    assert clean_spoken_text("[laughing] 你终于来了！") == "你终于来了！"
    assert clean_spoken_text("[excited] Really? [gasp] No way.") == "Really? No way."
    # SSML 标签删除
    assert clean_spoken_text("<speak>Dorothy <break time=\"300ms\"/>lived.</speak>") \
        == "Dorothy lived."
    # 普通方括号正文（非官方标签）原样保留
    assert clean_spoken_text("见[注]一。") == "见[注]一。"
    # 换行/制表/连续空白折成单空格
    assert clean_spoken_text("Hello\n  world\t!") == "Hello world !"
    # 导演指令/角色名不会混进来（清洗对象只有正文；这里验证不会凭空添加内容）
    assert clean_spoken_text("Speak slowly. Dorothy lived in Kansas.") == \
        "Speak slowly. Dorothy lived in Kansas."
    assert clean_spoken_text("") == ""
    assert clean_spoken_text(None) == ""
    assert clean_spoken_text("[laughing][sighing]") == ""


# —— 3. 纯函数：整数毫秒时间线 ——————————————————————————————————————————

def test_cues_follow_integer_ms_timeline_with_all_pause_positions():
    # 需求示例：2.0 开头停顿 + A 3.2 + 0.7 + B 4.1 + 0.5 + C 2.8
    entries = [
        TimelineEntry(1, "s1", "pause", 2000),
        TimelineEntry(2, "s2", "speech", 3200, "英文片段 A"),
        TimelineEntry(3, "s3", "pause", 700),
        TimelineEntry(4, "s4", "speech", 4100, "中文片段 B"),
        TimelineEntry(5, "s5", "pause", 500),
        TimelineEntry(6, "s6", "speech", 2800, "英文片段 C"),
        TimelineEntry(7, "s7", "pause", 600),   # 结尾停顿：无字幕
    ]
    cues = cues_from_timeline(entries)
    assert [(c.start_ms, c.end_ms, c.text) for c in cues] == [
        (2000, 5200, "英文片段 A"),
        (5900, 10000, "中文片段 B"),
        (10500, 13300, "英文片段 C"),
    ]
    assert [c.number for c in cues] == [1, 2, 3]
    # 总时长含结尾停顿（13.9s），但最后字幕结束于 13.3s —— 结尾停顿不出字幕
    assert cues[-1].end_ms == 13300


def test_cues_consecutive_pauses_accumulate():
    entries = [
        TimelineEntry(1, "p1", "pause", 500),
        TimelineEntry(2, "p2", "pause", 500),
        TimelineEntry(3, "s1", "speech", 1000, "你好"),
    ]
    cues = cues_from_timeline(entries)
    assert cues == [SubtitleCue(1, 1000, 2000, "你好", seg_id="s1", index=3)]


def test_cues_empty_speech_text_skips_cue_but_keeps_time():
    entries = [
        TimelineEntry(1, "s0", "speech", 1500, ""),   # 异常空正文：占位但无字幕
        TimelineEntry(2, "s1", "speech", 2000, "下一句"),
    ]
    cues = cues_from_timeline(entries)
    assert len(cues) == 1
    assert cues[0].number == 1                # 编号连续
    assert (cues[0].start_ms, cues[0].end_ms) == (1500, 3500)  # 时间轴不缩


def test_render_srt_standard_format():
    content = render_srt([
        SubtitleCue(1, 0, 3240, "Dorothy lived in Kansas."),
        SubtitleCue(2, 3940, 7520, "多萝西住在堪萨斯。"),
    ])
    assert content == (
        "1\n"
        "00:00:00,000 --> 00:00:03,240\n"
        "Dorothy lived in Kansas.\n\n"
        "2\n"
        "00:00:03,940 --> 00:00:07,520\n"
        "多萝西住在堪萨斯。\n"
    )
    # 中文 UTF-8 往返不乱码
    assert "多萝西住在堪萨斯。" in content.encode("utf-8").decode("utf-8")


def test_render_empty_srt():
    assert render_srt([]) == ""


# —— 4. build_timeline：严格错误与来源 ———————————————————————————————————

def _record(index, kind, text=None, status="success", pause=None,
            file=None):
    if kind == "pause":
        return SegmentRecord(index=index, kind="pause",
                             pause_seconds=pause, status="success",
                             file=file or f"segments/{index:04d}_PAUSE_{pause:g}s.wav")
    return SegmentRecord(
        index=index, kind="speech", speaker="Lily", speaker_name="Lily",
        text=text, status=status,
        file=file or f"segments/{index:04d}_Lily.wav",
        content_hash="h")


def _manifest(records, episode="epx"):
    return Manifest(
        episode=episode, created_at=NOW, updated_at=NOW,
        script_file="scripts/x.md", script_hash="dead",
        provider="qwen", model="qwen3-tts-instruct-flash",
        voices={"Lily": "Cherry"}, segments=records)


def test_build_timeline_speech_missing_or_failed_raises(tmp_path):
    ep = tmp_path / "projects" / "epx"
    man = _manifest([
        _record(1, "speech", "A"),
        _record(2, "speech", "B", status="failed", file="segments/0002_Leo.wav"),
    ])
    prober = lambda ff, f: 3.0  # noqa: E731
    with pytest.raises(SrtError, match="0002"):
        build_timeline(ep, man, "ffprobe", prober)


def test_build_timeline_unreadable_duration_raises(tmp_path):
    ep = tmp_path / "projects" / "epx"
    f = ep / "segments" / "0001_Lily.wav"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    man = _manifest([_record(1, "speech", "A")])
    with pytest.raises(SrtError, match="无法读取"):
        build_timeline(ep, man, "ffprobe", lambda ff, p: None)


def test_build_timeline_pause_without_file_uses_exact_seconds(tmp_path):
    ep = tmp_path / "projects" / "epx"
    man = _manifest([
        _record(1, "pause", pause=1.5),
        _record(2, "speech", "A"),
    ])
    f = ep / "segments" / "0002_Lily.wav"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    entries = build_timeline(ep, man, "ffprobe",
                             lambda ff, p: 2.0 if p == f else None)
    assert entries[0].kind == "pause" and entries[0].duration_ms == 1500
    assert entries[1].duration_ms == 2000


def test_build_timeline_sources_must_match_records(tmp_path):
    man = _manifest([_record(1, "pause", pause=1.0)])
    with pytest.raises(SrtError, match="数量不一致"):
        build_timeline(tmp_path, man, "ffprobe", lambda ff, f: 1.0,
                       sources=[Path("a.wav"), Path("b.wav")])


def test_build_timeline_from_explicit_sources(tmp_path):
    # merge 路径：显式 sources 与记录一一对应（可指向规范化临时文件）
    ep = tmp_path / "epx"
    ep.mkdir(parents=True)
    a, p = ep / "tmp_0001.wav", ep / "tmp_0002.wav"
    a.write_bytes(b"x")
    p.write_bytes(b"x")
    man = _manifest([_record(1, "speech", "A"), _record(2, "pause", pause=9.0)])
    entries = build_timeline(
        ep, man, "ffprobe", lambda ff, f: 3.2 if f == a else 0.7,
        sources=[a, p])
    assert entries[0].duration_ms == 3200
    assert entries[1].duration_ms == 700   # 以实际拼接文件探测为准，忽略 9.0


# —— 5. mock ffmpeg 端到端 ——————————————————————————————————————————————

class _Recorder:
    """模拟 ffmpeg：anullsrc/normalize/concat 均落盘假文件。"""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> None:
        self.commands.append(list(cmd))
        joined = " ".join(cmd)
        if "anullsrc" in joined or ("-f" in cmd and "concat" in joined) \
                or "-ar" in cmd:
            Path(cmd[-1]).write_bytes(b"RIFFfake")


@pytest.fixture
def ffmpeg_env(monkeypatch):
    rec = _Recorder()
    # speech_durations：默认按片段编号映射；需要消歧（如重排后同编号不同角色）
    # 时可用完整文件名 stem 覆盖
    env = SimpleNamespace(recorder=rec, speech_durations={})
    monkeypatch.setattr(audio_mod, "_run_command", rec)
    monkeypatch.setattr(
        audio_mod, "probe",
        lambda ff, file: AudioFormat("pcm_s16le", 24000, 1))

    def _dur(ff, file):
        stem = Path(file).stem
        if len(stem) < 4 or not stem[:4].isdigit():
            return 3.0                    # 最终成品等非片段文件
        if stem in env.speech_durations:
            return env.speech_durations[stem]
        idx = int(stem[:4])
        if "_PAUSE_" in stem:
            return float(stem.split("_PAUSE_")[1].rstrip("s"))
        return env.speech_durations.get(idx, 3.0)

    monkeypatch.setattr(audio_mod, "probe_duration", _dur)
    return env


def _generate(cfg, text, episode="ep1", provider=None):
    ws = WorkingScript.from_paste(episode, text)
    fake = provider or FakeProvider()
    run_project(episode, cfg, ws, default_speaker_id="Lily",
                provider=fake, persist=True)
    return ws, fake


def _srt_text(cfg, episode="ep1") -> str:
    path = cfg.projects_dir / episode / "output" / f"{episode}.srt"
    assert path.is_file(), "合并应同步写出 srt"
    return path.read_text(encoding="utf-8")


def _cues(content):
    out = []
    for block in content.strip().split("\n\n"):
        lines = block.splitlines()
        num = int(lines[0])
        start, end = lines[1].split(" --> ")
        out.append((num, start, end, "\n".join(lines[2:])))
    return out


def test_e2e_alternating_en_zh_with_pauses(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 3.2, 3: 4.1, 5: 2.8}
    _generate(cfg,
              "Lily: Segment A.\n[PAUSE:0.7]\nLeo: 中文片段 B。\n"
              "[PAUSE:0.5]\nLily: Segment C.\n")
    outcome = run_merge("ep1", cfg)
    assert outcome.srt == "output/ep1.srt"
    cues = _cues(_srt_text(cfg))
    assert cues == [
        (1, "00:00:00,000", "00:00:03,200", "Segment A."),
        (2, "00:00:03,900", "00:00:08,000", "中文片段 B。"),
        (3, "00:00:08,500", "00:00:11,300", "Segment C."),
    ]


def test_e2e_leading_pause_shifts_first_cue(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {2: 3.0}
    _generate(cfg, "[PAUSE:2.0]\nLily: 你好。")
    run_merge("ep1", cfg)
    cues = _cues(_srt_text(cfg))
    assert cues == [(1, "00:00:02,000", "00:00:05,000", "你好。")]


def test_e2e_consecutive_and_trailing_pauses(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {3: 1.0, 5: 2.0}
    _generate(cfg,
              "[PAUSE:0.5]\n[PAUSE:0.5]\nLily: 甲\n"
              "[PAUSE:1.0]\nLeo: 乙\n[PAUSE:1.0]")
    run_merge("ep1", cfg)
    cues = _cues(_srt_text(cfg))
    assert cues == [
        (1, "00:00:01,000", "00:00:02,000", "甲"),
        (2, "00:00:03,000", "00:00:05,000", "乙"),  # 结尾 1s 停顿不产生字幕
    ]


def test_e2e_no_pause_starts_at_zero(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 2.5}
    _generate(cfg, "Lily: 直接开始。")
    run_merge("ep1", cfg)
    assert _cues(_srt_text(cfg)) == [
        (1, "00:00:00,000", "00:00:02,500", "直接开始。")]


def test_e2e_direction_and_emotags_excluded_from_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path, provider="qwen_audio",
                   model="qwen-audio-3.0-tts-plus")
    ffmpeg_env.speech_durations = {1: 1.0}
    _generate(cfg, "Lily(轻快地): [laughing] 你终于来了！")
    run_merge("ep1", cfg)
    text = _srt_text(cfg)
    assert "你终于来了！" in text
    assert "laughing" not in text
    assert "轻快" not in text          # 导演指令不进字幕
    assert "Lily" not in _cues(text)[0][3]  # 角色名不进字幕


def test_e2e_regenerated_longer_audio_shifts_later_cues(tmp_path, ffmpeg_env):
    """片段重新生成后实际音频 3.2→3.8s：重新导出必须自动使用新时长。"""
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 3.2, 3: 4.1}
    _generate(cfg, "Lily: A。\n[PAUSE:0.7]\nLeo: B。")
    run_merge("ep1", cfg)
    assert _cues(_srt_text(cfg))[1] == (
        2, "00:00:03,900", "00:00:08,000", "B。")

    # 模拟"重新生成后第一段变长"：替换文件内容（size 变化驱动重合并）+ 新探测时长
    ep = cfg.projects_dir / "ep1"
    first = ep / "segments" / "0001_Lily.wav"
    first.write_bytes(b"X" * 3200)
    ffmpeg_env.speech_durations[1] = 3.8
    run_merge("ep1", cfg, force=True)
    cues = _cues(_srt_text(cfg))
    assert cues[0] == (1, "00:00:00,000", "00:00:03,800", "A。")
    assert cues[1] == (2, "00:00:04,500", "00:00:08,600", "B。")
    # 合并产物总时长也随之增长（manifest 记录的 cues 同步刷新）
    man = load_manifest(ep)
    assert man.merge["subtitles"]["cues"][-1]["end_ms"] == 8600


def test_e2e_edit_text_then_regenerate_updates_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 1.0, 3: 1.0}
    ws, fake = _generate(cfg, "Lily: 旧文本。\n[PAUSE:0.5]\nLeo: 保持。")
    run_merge("ep1", cfg)
    assert "旧文本。" in _srt_text(cfg)

    # 主文本框改正文 → 重新生成该段（调 1 次 TTS）→ 重合并 → 新字幕文本
    ws.update_text("Lily: 新文本！\n[PAUSE:0.5]\nLeo: 保持。\n")
    ws.save(cfg.projects_dir)
    fake2 = FakeProvider()
    run_project("ep1", cfg, ws, default_speaker_id="Lily",
                provider=fake2, persist=True)
    assert len(fake2.calls) == 1
    run_merge("ep1", cfg)
    text = _srt_text(cfg)
    assert "新文本！" in text and "旧文本。" not in text
    assert "保持。" in text


def test_e2e_deleted_segment_disappears_from_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 1.0, 3: 1.0}
    _generate(cfg, "Lily: 甲。\n[PAUSE:0.5]\nLeo: 乙。")
    run_merge("ep1", cfg)
    assert len(_cues(_srt_text(cfg))) == 2

    ws2 = WorkingScript.from_paste("ep1", "Lily: 甲。\n")
    fake = FakeProvider()
    run_project("ep1", cfg, ws2, default_speaker_id="Lily",
                provider=fake, persist=True)
    assert fake.calls == []                    # 幸存者沿用缓存
    run_merge("ep1", cfg)
    cues = _cues(_srt_text(cfg))
    assert [c[3] for c in cues] == ["甲。"]
    # 被删片段的旧音频已不在 segments 目录
    assert not list((cfg.projects_dir / "ep1" / "segments").glob("*Leo*"))


def test_e2e_reorder_segments_reorders_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 1.0, 2: 2.0}
    _generate(cfg, "Lily: 第一句。\nLeo: 第二句。")
    run_merge("ep1", cfg)
    assert [c[3] for c in _cues(_srt_text(cfg))] == ["第一句。", "第二句。"]

    ws2 = WorkingScript.from_paste("ep1", "Leo: 第二句。\nLily: 第一句。")
    run_project("ep1", cfg, ws2, default_speaker_id="Lily",
                provider=FakeProvider(), persist=True)
    # 重排后编号重排（0001_Leo / 0001_Lily），按实际文件给出时长
    ffmpeg_env.speech_durations = {"0001_Leo": 2.0, "0001_Lily": 1.0}
    run_merge("ep1", cfg)
    cues = _cues(_srt_text(cfg))
    assert [c[3] for c in cues] == ["第二句。", "第一句。"]
    assert cues[0] == (1, "00:00:00,000", "00:00:02,000", "第二句。")
    assert cues[1] == (2, "00:00:02,000", "00:00:03,000", "第一句。")


def test_e2e_pause_duration_edit_remerges_with_zero_tts(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 1.0, 3: 1.0}
    ws, fake = _generate(cfg, "Lily: 甲。\n[PAUSE:0.7]\nLeo: 乙。")
    run_merge("ep1", cfg)

    ws.update_text("Lily: 甲。\n[PAUSE:1.2]\nLeo: 乙。\n")
    ws.save(cfg.projects_dir)
    fake2 = FakeProvider()
    run_project("ep1", cfg, ws, default_speaker_id="Lily",
                provider=fake2, persist=True)
    assert fake2.calls == []                   # 改停顿不调 TTS
    run_merge("ep1", cfg)
    cues = _cues(_srt_text(cfg))
    assert cues[1][1] == "00:00:02,200"        # 1.0 + 1.2
    assert cues[1][2] == "00:00:03,200"


def test_standalone_export_after_merge_equals_merged_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {2: 1.5}
    _generate(cfg, "[PAUSE:0.5]\nLily: 独立导出。")
    run_merge("ep1", cfg)
    content, count = render_for_episode("ep1", cfg)
    assert count == 1
    assert content == _srt_text(cfg)
    # export_srt 写到指定路径
    out = tmp_path / "sub" / "book.srt"
    path, n = export_srt("ep1", cfg, out)
    assert path == out and n == 1
    assert path.read_text(encoding="utf-8") == content


def test_standalone_export_before_merge_uses_pause_seconds(tmp_path, ffmpeg_env):
    """未合并就导出：停顿静音文件尚不存在，用 pause_seconds 精确值。"""
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {2: 2.0}
    _generate(cfg, "[PAUSE:1.5]\nLily: 先导出。")
    # 不运行 run_merge → segments 下没有 PAUSE 静音文件
    assert not list(
        (cfg.projects_dir / "ep1" / "segments").glob("*PAUSE*"))
    content, count = render_for_episode("ep1", cfg)
    assert count == 1
    assert _cues(content) == [
        (1, "00:00:01,500", "00:00:03,500", "先导出。")]


def test_pure_pause_project_has_no_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    _generate(cfg, "[PAUSE:1.5]")
    outcome = run_merge("ep1", cfg)
    assert outcome.srt is None
    assert not (cfg.projects_dir / "ep1" / "output" / "ep1.srt").exists()
    # 独立导出：0 条字幕、空文件
    path, count = export_srt("ep1", cfg)
    assert count == 0
    assert path.read_text(encoding="utf-8") == ""


def test_export_without_manifest_raises(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    with pytest.raises(SrtError, match="manifest"):
        render_for_episode("ghost", cfg)


# —— 6. GUI 控制器守卫 —————————————————————————————————————————————————

def _gui_settings(tmp_path):
    return GuiSettings(
        api_key="sk-test", provider="qwen",
        model="qwen3-tts-instruct-flash",
        roles=[
            RoleVoice(role="Lily", voice="Cherry", language_type="English"),
            RoleVoice(role="Leo", voice="Ethan", language_type="Chinese"),
        ],
        default_role="Lily", episode_name="epgui",
        projects_dir=str(tmp_path / "projects"),
    )


def test_controller_build_srt_blocks_dirty_and_allows_clean(tmp_path,
                                                             ffmpeg_env):
    s = _gui_settings(tmp_path)
    cfg = s.to_app_config()
    ffmpeg_env.speech_durations = {2: 1.0}

    c = GuiController(provider_factory=lambda c: FakeProvider(),
                      merge_func=lambda *a, **k: None)
    c.set_settings(s)
    c.ws = WorkingScript.from_paste("epgui", "[PAUSE:0.5]\nLily: 你好。")
    c.episode = "epgui"
    # 未生成 → 守卫拒绝
    with pytest.raises(ConfigError, match="不是最新"):
        c.build_srt()

    # 生成落盘（不经过 worker，直接让片段全部 clean）
    run_project("epgui", cfg, c.ws, default_speaker_id="Lily",
                provider=FakeProvider(), persist=True)
    content, count = c.build_srt()
    assert count == 1
    assert "你好。" in content

    # 文本改动未重新生成 → 再次拒绝（防止字幕与听到的成品不一致）
    c.ws.update_text("[PAUSE:0.5]\nLily: 改了但没生成。\n")
    with pytest.raises(ConfigError, match="不是最新"):
        c.build_srt()
