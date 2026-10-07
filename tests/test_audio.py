"""Phase 7 单元测试：FFmpeg 合并（文档第 16 节）。

全程 mock 外部命令与 ffprobe（不依赖真实 FFmpeg）。覆盖：
- 合并前置校验：manifest 缺失、存在未完成片段（提示先 retry）、片段文件缺失
- 正常合并：静音参数取自探测结果、concat 顺序与绝对路径、MP3 编码参数、
  manifest.merge 字段（status/output/merged_at/segments_signature）
- 签名跳过：重跑零命令；--force 强制重并；台词文件变化 → 签名变化 → 重新合并
- 格式不一致：先原地标准化（-ar 24000 -ac 1）再合并，manifest.merge.normalized=True
- merge.format=wav：流复制（-c:a copy）
- FFmpeg 不可用：MergeError 含安装指引
"""
from __future__ import annotations

from pathlib import Path

import pytest

from podcast_tts.audio import (
    AudioFormat,
    MergeError,
    run_merge,
    segments_signature,
    verify_tools,
)
from podcast_tts.manifest import (
    Manifest,
    SegmentRecord,
    content_hash,
    load as load_manifest,
    save,
)
from podcast_tts.models import Segment as _Segment
from podcast_tts.pipeline import plan_segment
from test_pipeline import make_cfg

NOW = "2026-08-30T12:00:00+08:00"


def planned_hash(cfg, speaker: str, text: str = "x") -> str:
    """按当前 cfg 规划出片段的请求指纹（夹具里的"已生成"记录必须与之一致）。"""
    seg = _Segment(index=1, kind="speech", speaker=speaker,
                   speaker_name=speaker, text=text)
    setting, instruction, extras = plan_segment(cfg, seg)
    return content_hash(
        cfg, text, setting.voice, setting.language_type, instruction, extras)


def make_episode(cfg, spec: list[tuple[str, object]], episode: str = "episode001"):
    """按 spec 构建全 success 的 manifest 与假片段文件。

    spec 元素：("speech", "Lily") 或 ("pause", 0.8)。
    返回 episode_dir。
    """
    episode_dir = cfg.projects_dir / episode
    records: list[SegmentRecord] = []
    for i, (kind, payload) in enumerate(spec, start=1):
        if kind == "pause":
            records.append(SegmentRecord(
                index=i, kind="pause", pause_seconds=float(payload),
                status="success", file=f"segments/{i:04d}_PAUSE.wav",
            ))
        else:
            speaker = str(payload)
            rel = f"segments/{i:04d}_{speaker}.wav"
            f = episode_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"RIFFfake")
            records.append(SegmentRecord(
                index=i, kind="speech", speaker=speaker, speaker_name=speaker,
                text="x", status="success", file=rel,
                content_hash=planned_hash(cfg, speaker),
            ))
    manifest = Manifest(
        episode=episode, created_at=NOW, updated_at=NOW,
        script_file="scripts/episode001.md", script_hash="deadbeef",
        provider="qwen", model="qwen3-tts-instruct-flash",
        voices={"Lily": "Cherry", "Leo": "Ethan"}, segments=records,
    )
    save(episode_dir, manifest)
    return episode_dir


class CommandRecorder:
    """替代 audio._run_command：记录命令并模拟产物落盘。"""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> None:
        self.commands.append(list(cmd))
        joined = " ".join(cmd)
        if "anullsrc" in joined:          # 静音生成：末位为输出路径
            Path(cmd[-1]).write_bytes(b"SILENCE")
        elif "-f" in cmd and "concat" in cmd:   # 拼接编码：末位为输出路径
            Path(cmd[-1]).write_bytes(b"MP3DATA")
        elif "-ar" in cmd:                # 标准化：末位为临时文件
            Path(cmd[-1]).write_bytes(b"RIFFfake")

    @property
    def concat_cmds(self) -> list[list[str]]:
        return [c for c in self.commands if "-f" in c and "concat" in c]

    @property
    def silence_cmds(self) -> list[list[str]]:
        return [c for c in self.commands if "anullsrc" in " ".join(c)]

    @property
    def normalize_cmds(self) -> list[list[str]]:
        return [c for c in self.commands if "-ar" in c]


@pytest.fixture
def recorder(monkeypatch):
    rec = CommandRecorder()
    monkeypatch.setattr("podcast_tts.audio._run_command", rec)
    monkeypatch.setattr(
        "podcast_tts.audio.probe",
        lambda ffprobe, file: AudioFormat("pcm_s16le", 24000, 1),
    )
    monkeypatch.setattr("podcast_tts.audio.probe_duration", lambda ffprobe, file: 3.0)
    return rec


# —— 前置校验 ————————————————————————————————————————————————————————


def test_merge_missing_manifest_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_merge("episode001", cfg)


def test_merge_refuses_incomplete(tmp_path):
    cfg = make_cfg(tmp_path)
    episode_dir = make_episode(cfg, [("speech", "Lily")])
    manifest = load_manifest(episode_dir)
    manifest.segments[0].status = "failed"
    save(episode_dir, manifest)

    with pytest.raises(MergeError, match="retry"):
        run_merge("episode001", cfg)


def test_merge_missing_segment_file(tmp_path, recorder):
    cfg = make_cfg(tmp_path)
    make_episode(cfg, [("speech", "Lily"), ("pause", 0.5), ("speech", "Leo")])
    (cfg.projects_dir / "episode001" / "segments" / "0001_Lily.wav").unlink()

    with pytest.raises(MergeError, match="缺失"):
        run_merge("episode001", cfg)


# —— 正常合并 ————————————————————————————————————————————————————————


def test_merge_happy_path(tmp_path, recorder):
    cfg = make_cfg(tmp_path)
    make_episode(cfg, [("speech", "Lily"), ("pause", 0.8), ("speech", "Leo")])

    outcome = run_merge("episode001", cfg)

    assert outcome.skipped is False
    assert outcome.output == "output/episode001.mp3"
    assert outcome.duration == 3.0

    # 静音：参数取自探测结果（24000/mono），时长 0.8
    assert len(recorder.silence_cmds) == 1
    assert "anullsrc=r=24000:cl=mono" in " ".join(recorder.silence_cmds[0])
    assert recorder.silence_cmds[0][recorder.silence_cmds[0].index("-t") + 1] == "0.800"

    # 拼接：-safe 0 + libmp3lame 192k；清单为 3 个绝对路径、按脚本顺序
    assert len(recorder.concat_cmds) == 1
    concat_cmd = recorder.concat_cmds[0]
    assert "-safe" in concat_cmd and "0" in concat_cmd
    assert "libmp3lame" in concat_cmd and "192k" in concat_cmd
    list_file = Path(concat_cmd[concat_cmd.index("-i") + 1])
    lines = list_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # 行格式：file '<绝对路径>'（"file '" 共 6 字符）
    assert all(line.startswith("file '") and Path(line[6:-1]).is_absolute() for line in lines)
    assert [Path(line[6:-1]).name for line in lines] == [
        "0001_Lily.wav", "0002_PAUSE.wav", "0003_Leo.wav",
    ]

    # manifest.merge 落盘
    manifest = load_manifest(cfg.projects_dir / "episode001")
    assert manifest.merge["status"] == "success"
    assert manifest.merge["output"] == "output/episode001.mp3"
    assert manifest.merge["merged_at"]
    assert manifest.merge["segments_signature"] == segments_signature(
        cfg.projects_dir / "episode001", manifest
    )
    assert "normalized" not in manifest.merge
    # 静音文件已落盘（模拟）
    assert (cfg.projects_dir / "episode001" / "segments" / "0002_PAUSE.wav").read_bytes() == b"SILENCE"


def test_merge_signature_skips_rerun(tmp_path, recorder):
    cfg = make_cfg(tmp_path)
    make_episode(cfg, [("speech", "Lily"), ("pause", 0.5)])
    run_merge("episode001", cfg)
    calls_after_first = len(recorder.commands)

    outcome = run_merge("episode001", cfg)
    assert outcome.skipped is True
    assert len(recorder.commands) == calls_after_first  # 零新命令


def test_merge_force_remerges(tmp_path, recorder):
    cfg = make_cfg(tmp_path)
    make_episode(cfg, [("speech", "Lily"), ("pause", 0.5)])
    run_merge("episode001", cfg)

    outcome = run_merge("episode001", cfg, force=True)
    assert outcome.skipped is False
    assert len(recorder.concat_cmds) == 2


def test_merge_remerges_when_segment_changed(tmp_path, recorder):
    cfg = make_cfg(tmp_path)
    episode_dir = make_episode(cfg, [("speech", "Lily"), ("pause", 0.5)])
    run_merge("episode001", cfg)

    # 片段重新生成（内容变化 → 大小变化 → 签名变化）
    (episode_dir / "segments" / "0001_Lily.wav").write_bytes(b"RIFFnew-size")
    outcome = run_merge("episode001", cfg)
    assert outcome.skipped is False
    assert len(recorder.concat_cmds) == 2


def test_merge_inconsistent_formats_normalizes_in_temp_dir(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    episode_dir = make_episode(
        cfg, [("speech", "Lily"), ("pause", 0.5), ("speech", "Leo")]
    )
    # 记录原始片段哈希（规范化前后必须不变）
    orig_hashes = {
        name: (episode_dir / "segments" / name).read_bytes()
        for name in ("0001_Lily.wav", "0003_Leo.wav")
    }

    rec = CommandRecorder()
    monkeypatch.setattr("podcast_tts.audio._run_command", rec)
    formats = {
        "0001_Lily.wav": AudioFormat("pcm_s16le", 24000, 1),
        "0003_Leo.wav": AudioFormat("mp3", 44100, 2),
    }

    def fake_probe(ffprobe, file):
        return formats.get(Path(file).name, AudioFormat("pcm_s16le", 24000, 1))

    monkeypatch.setattr("podcast_tts.audio.probe", fake_probe)
    monkeypatch.setattr("podcast_tts.audio.probe_duration", lambda ffprobe, file: 3.0)

    outcome = run_merge("episode001", cfg)

    assert outcome.skipped is False
    assert len(rec.normalize_cmds) == 2  # 两个台词片段都转码
    for cmd in rec.normalize_cmds:
        assert cmd[cmd.index("-ar") + 1] == "24000"
        assert cmd[cmd.index("-ac") + 1] == "1"
        inp = Path(cmd[cmd.index("-i") + 1])
        outp = Path(cmd[-1])
        # 输入仍是 segments 下的原文件；输出必须在独立临时目录
        assert inp.parent.name == "segments"
        assert outp.parent.name == "merge_tmp"
        assert outp.suffix == ".wav"
    assert "anullsrc=r=24000:cl=mono" in " ".join(rec.silence_cmds[0])
    # PAUSE 静音恒为 WAV 并缓存在 segments 目录（可跨次复用）
    silence_out = Path(rec.silence_cmds[0][-1])
    assert silence_out.parent.name == "segments"
    assert silence_out.name == "0002_PAUSE.wav"

    # 拼接清单：台词来自临时规范化目录，PAUSE 来自 segments，顺序严格按 manifest
    concat_cmd = rec.concat_cmds[0]
    list_file = Path(concat_cmd[concat_cmd.index("-i") + 1])
    lines = list_file.read_text(encoding="utf-8").strip().splitlines()
    names = [Path(line[6:-1]).name for line in lines]
    assert names == ["0001.wav", "0002_PAUSE.wav", "0003.wav"]
    assert "merge_tmp" in lines[0] and "segments" in lines[1] and "merge_tmp" in lines[2]

    manifest = load_manifest(episode_dir)
    assert manifest.merge["normalized"] is True

    # 原片段内容不变；临时目录已清理；segments 下没有留下规范化残文件
    for name, raw in orig_hashes.items():
        assert (episode_dir / "segments" / name).read_bytes() == raw
    assert not (episode_dir / "metadata" / "merge_tmp").exists()
    assert not list((episode_dir / "segments").glob("*.norm.tmp.wav"))


def test_merge_mp3_only_normalizes_without_touching_originals(tmp_path, monkeypatch):
    """全部是 MP3 片段：转码进临时目录，原 .mp3 文件一个字节都不改。"""
    cfg = make_cfg(tmp_path)
    episode_dir = cfg.projects_dir / "episode001"
    (episode_dir / "segments").mkdir(parents=True)
    mp3_files = {}
    records = []
    for i, speaker in enumerate(("Lily", "Leo"), start=1):
        rel = f"segments/{i:04d}_{speaker}.mp3"
        p = episode_dir / rel
        p.write_bytes(b"ID3fake-mp3-content-%d" % i)
        mp3_files[rel] = p.read_bytes()
        records.append(SegmentRecord(
            index=i, kind="speech", speaker=speaker, speaker_name=speaker,
            text="x",
            status="success", file=rel, content_hash=planned_hash(cfg, speaker)))
    man = Manifest(
        episode="episode001", created_at=NOW, updated_at=NOW,
        script_file="x.md", script_hash="deadbeef", provider="qwen_audio",
        model="qwen-audio-3.0-tts-flash", voices={}, segments=records)
    save(episode_dir, man)

    rec = CommandRecorder()
    monkeypatch.setattr("podcast_tts.audio._run_command", rec)
    monkeypatch.setattr(
        "podcast_tts.audio.probe",
        lambda ffprobe, file: AudioFormat("mp3", 44100, 2)
        if Path(file).suffix == ".mp3" and "merge_tmp" not in str(file)
        else AudioFormat("pcm_s16le", 24000, 1))
    monkeypatch.setattr("podcast_tts.audio.probe_duration", lambda ffprobe, file: 2.0)

    run_merge("episode001", cfg)

    # 两个原 mp3 均为转码输入，没有任何命令的输出落在 segments 目录
    for cmd in rec.normalize_cmds:
        assert Path(cmd[-1]).parent.name == "merge_tmp"
        assert Path(cmd[cmd.index("-i") + 1]).suffix == ".mp3"
    for rel, raw in mp3_files.items():
        assert (episode_dir / rel).read_bytes() == raw
    assert not (episode_dir / "metadata" / "merge_tmp").exists()


def test_merge_cleans_temp_dir_on_failure(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    make_episode(cfg, [("speech", "Lily"), ("speech", "Leo")])

    def fake_run(cmd):
        joined = " ".join(cmd)
        if "anullsrc" in joined:
            Path(cmd[-1]).write_bytes(b"SILENCE")
        elif "-ar" in cmd:
            Path(cmd[-1]).write_bytes(b"RIFFfake")
        elif "concat" in joined:
            raise MergeError("模拟 concat 失败")

    monkeypatch.setattr("podcast_tts.audio._run_command", fake_run)
    monkeypatch.setattr(
        "podcast_tts.audio.probe",
        lambda ffprobe, file: AudioFormat("mp3", 44100, 2)
        if Path(file).suffix == ".mp3" else AudioFormat("pcm_s16le", 24000, 1))

    with pytest.raises(MergeError, match="concat"):
        run_merge("episode001", cfg)
    episode_dir = cfg.projects_dir / "episode001"
    # 失败也要清理临时目录，原片段保留
    assert not (episode_dir / "metadata" / "merge_tmp").exists()
    assert (episode_dir / "segments" / "0001_Lily.wav").is_file()


def test_merge_wav_uses_stream_copy(tmp_path, recorder):
    cfg = make_cfg(tmp_path, merge_format="wav")
    make_episode(cfg, [("speech", "Lily")])

    outcome = run_merge("episode001", cfg)
    assert outcome.output == "output/episode001.wav"
    assert "copy" in recorder.concat_cmds[0]
    assert "libmp3lame" not in recorder.concat_cmds[0]


# —— FFmpeg 不可用 ——————————————————————————————————————————————————


def test_tools_missing_gives_guidance(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    make_episode(cfg, [("speech", "Lily")])

    def broken_run(cmd):
        raise MergeError("命令执行失败：not found")

    monkeypatch.setattr("podcast_tts.audio._run_command", broken_run)
    with pytest.raises(MergeError, match="安装指引"):
        run_merge("episode001", cfg)


def test_verify_tools_ok(monkeypatch):
    calls = []
    monkeypatch.setattr("podcast_tts.audio._run_command", calls.append)
    verify_tools("ffmpeg", "ffprobe")
    assert [c[0] for c in calls] == ["ffprobe", "ffmpeg"]
