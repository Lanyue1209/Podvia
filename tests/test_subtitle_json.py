"""subtitle.json 结构化字幕数据导出测试。

验证（对应需求测试要求）：
1. subtitle.json 随合并正常生成 / 独立导出可用；
2. JSON 条目数 == SRT 字幕条数；
3. start_time/end_time 与 SRT 时间戳完全一致（同一批 cues，毫秒/1000）；
4. segment_id 使用稳定 seg_id，重解析/编辑后不变；
5. 不改动 SRT 既有行为（同一文件内断言 SRT 内容原样）。

全程 FakeProvider + mock ffmpeg，封网运行。
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_tts import audio as audio_mod
from podcast_tts.audio import AudioFormat, run_merge
from podcast_tts.manifest import load as load_manifest
from podcast_tts.pipeline import run_project
from podcast_tts.project import WorkingScript
from podcast_tts.subtitles import (
    cues_from_timeline,
    default_subtitle_json_path,
    export_subtitle_json,
    TimelineEntry,
    SubtitleCue,
    build_subtitle_json_data,
    render_subtitle_json,
)

from tests.test_pipeline import FakeProvider, make_cfg


# —— 测试辅助（与 test_subtitles.py 同款的 mock ffmpeg 环境） ———————————————————

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
    env = SimpleNamespace(recorder=rec, speech_durations={})
    monkeypatch.setattr(audio_mod, "_run_command", rec)
    monkeypatch.setattr(
        audio_mod, "probe",
        lambda ff, file: AudioFormat("pcm_s16le", 24000, 1))

    def _dur(ff, file):
        stem = Path(file).stem
        if len(stem) < 4 or not stem[:4].isdigit():
            return 3.0
        if stem in env.speech_durations:
            return env.speech_durations[stem]
        idx = int(stem[:4])
        if "_PAUSE_" in stem:
            return float(stem.split("_PAUSE_")[1].rstrip("s"))
        return env.speech_durations.get(idx, 3.0)

    monkeypatch.setattr(audio_mod, "probe_duration", _dur)
    return env


def _generate(cfg, text, episode="ep1"):
    ws = WorkingScript.from_paste(episode, text)
    fake = FakeProvider()
    run_project(episode, cfg, ws, default_speaker_id="Lily",
                provider=fake, persist=True)
    return ws, fake


def _json_path(cfg, episode="ep1") -> Path:
    return cfg.projects_dir / episode / "output" / f"{episode}.subtitle.json"


def _srt_ms(t: str) -> int:
    """"HH:MM:SS,mmm" → 毫秒。"""
    hms, ms = t.split(",")
    h, m, s = hms.split(":")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def _srt_time_pairs(cfg, episode="ep1") -> list[tuple[int, int]]:
    srt = (cfg.projects_dir / episode / "output" / f"{episode}.srt") \
        .read_text(encoding="utf-8")
    pairs = []
    for block in srt.strip().split("\n\n"):
        lines = block.splitlines()
        start, end = lines[1].split(" --> ")
        pairs.append((_srt_ms(start), _srt_ms(end)))
    return pairs


# —— 纯函数：字段与时间 ————————————————————————————————————————————————————

def test_json_data_fields_and_time_conversion(tmp_path):
    cfg = make_cfg(tmp_path)
    # 构造 manifest 记录（不依赖真实生成）
    from podcast_tts.manifest import Manifest, SegmentRecord
    manifest = Manifest(
        episode="epx", created_at="t", updated_at="t",
        script_file="x", script_hash="h", provider="qwen", model="m",
        voices={"Lily": "Cherry", "Leo": "Ethan"},
        segments=[
            SegmentRecord(index=1, kind="speech", seg_id="seg-aaa",
                          speaker="Lily", speaker_name="Lily", text="你好。"),
            SegmentRecord(index=2, kind="pause", seg_id="seg-bbb",
                          pause_seconds=0.5),
            SegmentRecord(index=3, kind="speech", seg_id="seg-ccc",
                          speaker="Leo", speaker_name="Leo", text="Hello."),
        ],
    )
    cues = [
        SubtitleCue(1, 0, 1594, "你好。", seg_id="seg-aaa", index=1),
        SubtitleCue(2, 2094, 3919, "Hello.", seg_id="seg-ccc", index=3),
    ]
    data = build_subtitle_json_data(cues, manifest, cfg)
    assert data == [
        {
            "segment_id": "seg-aaa",
            "speaker": "Lily",
            "language": "English",
            "start_time": 0.0,
            "end_time": 1.594,
            "text": "你好。",
            "translation": "",
            "voice": "Cherry",
        },
        {
            "segment_id": "seg-ccc",
            "speaker": "Leo",
            "language": "Chinese",
            "start_time": 2.094,
            "end_time": 3.919,
            "text": "Hello.",
            "translation": "",
            "voice": "Ethan",
        },
    ]


def test_render_subtitle_json_is_valid_utf8_json(tmp_path):
    cfg = make_cfg(tmp_path)
    from podcast_tts.manifest import Manifest, SegmentRecord
    manifest = Manifest(
        episode="epx", created_at="t", updated_at="t",
        script_file="x", script_hash="h", provider="qwen", model="m",
        voices={},
        segments=[SegmentRecord(index=1, kind="speech", seg_id="s1",
                                speaker=None, text="无角色")],
    )
    cues = [SubtitleCue(1, 0, 1000, "无角色", seg_id="s1", index=1)]
    text = render_subtitle_json(cues, manifest, cfg)
    parsed = json.loads(text)
    assert parsed[0]["speaker"] == "" and parsed[0]["language"] == "" \
        and parsed[0]["voice"] == ""
    assert parsed[0]["segment_id"] == "s1"


# —— 端到端：随合并生成 + 与 SRT 完全一致 ————————————————————————————————————

def test_e2e_merge_writes_subtitle_json_matching_srt(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 1.594, 3: 2.325}
    _generate(cfg,
              "Lily: 你终于来了！\n[PAUSE:1.0]\nLeo: 抱歉，我来晚了。\n")
    outcome = run_merge("ep1", cfg)

    # 1. 文件生成 + MergeOutcome 记录
    assert outcome.srt == "output/ep1.srt"
    assert outcome.subtitle_json == "output/ep1.subtitle.json"
    jpath = _json_path(cfg)
    assert jpath.is_file()

    data = json.loads(jpath.read_text(encoding="utf-8"))

    # 2. 条目数 == SRT 条数（停顿不出字幕）
    srt_pairs = _srt_time_pairs(cfg)
    assert len(data) == len(srt_pairs) == 2

    # 3. 时间与 SRT 完全一致（SRT ms / 1000 == JSON 秒）
    for item, (sms, ems) in zip(data, srt_pairs):
        assert item["start_time"] == round(sms / 1000, 3)
        assert item["end_time"] == round(ems / 1000, 3)
    assert (data[0]["start_time"], data[0]["end_time"]) == (0.0, 1.594)
    assert (data[1]["start_time"], data[1]["end_time"]) == (2.594, 4.919)

    # 5. SRT 行为不变：文件内容仍是标准 SRT
    srt_text = (cfg.projects_dir / "ep1" / "output" / "ep1.srt") \
        .read_text(encoding="utf-8")
    assert srt_text.startswith("1\n00:00:00,000 --> 00:00:01,594\n你终于来了！")

    # manifest 快照记录 json_file
    man = load_manifest(cfg.projects_dir / "ep1")
    assert man.merge["subtitles"]["json_file"] == "output/ep1.subtitle.json"


def test_e2e_json_uses_stable_seg_id_and_voice_language(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ws, _fake = _generate(cfg, "Lily: 你好。\nLeo: 我在。\n")
    run_merge("ep1", cfg)

    man = load_manifest(cfg.projects_dir / "ep1")
    ids_before = [r.seg_id for r in man.segments]
    data = json.loads(_json_path(cfg).read_text(encoding="utf-8"))
    assert [d["segment_id"] for d in data] == ids_before
    assert all(d["segment_id"] for d in data)
    assert data[0]["speaker"] == "Lily" and data[0]["voice"] == "Cherry" \
        and data[0]["language"] == "English"
    assert data[1]["speaker"] == "Leo" and data[1]["voice"] == "Ethan" \
        and data[1]["language"] == "Chinese"

    # 编辑正文后重解析：seg_id 不变，JSON 仍用同一 ID
    ws.text = ws.text.replace("你好。", "你好呀。")
    run_project("ep1", cfg, ws, default_speaker_id="Lily",
                provider=FakeProvider(), persist=True)
    man2 = load_manifest(cfg.projects_dir / "ep1")
    assert [r.seg_id for r in man2.segments] == ids_before
    run_merge("ep1", cfg, force=True)
    data2 = json.loads(_json_path(cfg).read_text(encoding="utf-8"))
    assert [d["segment_id"] for d in data2] == ids_before
    assert data2[0]["text"] == "你好呀。"


def test_e2e_merge_skipped_reuses_existing_json(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    _generate(cfg, "Lily: 你好。\n")
    first = run_merge("ep1", cfg)
    assert first.subtitle_json == "output/ep1.subtitle.json"
    second = run_merge("ep1", cfg)   # 签名一致 → 跳过
    assert second.skipped is True
    assert second.subtitle_json == "output/ep1.subtitle.json"


# —— 独立导出 ————————————————————————————————————————————————————————————

def test_export_subtitle_json_default_and_custom_path(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    ffmpeg_env.speech_durations = {1: 1.594}
    _generate(cfg, "Lily: 你终于来了！\n")

    # 未合并也可独立导出（停顿用精确时长，台词读片段文件）
    path, count = export_subtitle_json("ep1", cfg)
    assert path == default_subtitle_json_path("ep1", cfg)
    assert count == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["start_time"] == 0.0 and data[0]["end_time"] == 1.594

    out = tmp_path / "custom.json"
    path2, _ = export_subtitle_json("ep1", cfg, out)
    assert path2 == out and out.is_file()


def test_export_subtitle_json_pure_pause_project_is_empty(tmp_path, ffmpeg_env):
    cfg = make_cfg(tmp_path)
    _generate(cfg, "[PAUSE:1.0]\n")
    path, count = export_subtitle_json("ep1", cfg)
    assert count == 0
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_export_subtitle_json_no_manifest_raises(tmp_path):
    from podcast_tts.subtitles import SrtError
    cfg = make_cfg(tmp_path)
    with pytest.raises(SrtError, match="manifest"):
        export_subtitle_json("nope", cfg)


def test_pause_only_pushes_timeline_not_json_entry(tmp_path):
    # 纯函数级：PAUSE 不产生 JSON 条目，只推进时间轴（与 SRT 同规则）
    entries = [
        TimelineEntry(1, "p1", "pause", 1500),
        TimelineEntry(2, "s2", "speech", 2000, "你好。"),
    ]
    cues = cues_from_timeline(entries)
    assert len(cues) == 1
    assert cues[0].start_ms == 1500 and cues[0].seg_id == "s2"
