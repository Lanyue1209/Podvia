"""真实 FFmpeg 合并验证（不触网；仅在本机存在 ffmpeg/ffprobe 时运行）。

覆盖需求：
- MP3-only / WAV-only / MP3+WAV 混合 / PAUSE WAV
- 中文路径、空格路径
- 规范化前后原片段哈希不变（.mp3 不会被写成 WAV 内容）
- 最终产物可被 ffprobe 真实识别
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from podcast_tts import audio
from podcast_tts.config import load_config
from podcast_tts.manifest import Manifest, SegmentRecord, save

REPO = Path(__file__).resolve().parent.parent
FFMPEG = REPO / "tools" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "tools" / "bin" / "ffprobe.exe"

pytestmark = pytest.mark.skipif(
    not (FFMPEG.is_file() and FFPROBE.is_file()),
    reason="本机 tools/bin 无 ffmpeg/ffprobe，真实合并验证跳过",
)


class _Tools:
    ffmpeg = str(FFMPEG)
    ffprobe = str(FFPROBE)


def _make_cfg(tmp_path: Path, merge_format: str):
    cfg = load_config(key_required=False)
    cfg.projects_dir = tmp_path / "项目 输出"  # 中文 + 空格同时覆盖
    cfg.merge_format = merge_format
    cfg.bitrate = "128k"
    cfg.ffmpeg_dir = REPO / "tools" / "bin"
    return cfg


def _silence_wav(cfg, path: Path, seconds: float, sr: int, ch: int) -> None:
    audio.make_silence(
        _Tools.ffmpeg, seconds, audio.AudioFormat("pcm_s16le", sr, ch), path
    )


def _silence_mp3(cfg, path: Path, seconds: float, sr: int = 44100, ch: int = 2) -> None:
    tmp = path.with_suffix(".src.wav")
    audio.make_silence(_Tools.ffmpeg, seconds, audio.AudioFormat("pcm_s16le", sr, ch), tmp)
    audio._run_command(
        [_Tools.ffmpeg, "-y", "-i", str(tmp), "-c:a", "libmp3lame", "-b:a", "128k", str(path)]
    )
    tmp.unlink()


def _build_episode(cfg, episode: str, kinds: list[str], merge_format: str):
    """kinds: ["wav", "pause", "mp3", ...] 按顺序构造片段与 manifest。"""
    ep_dir = cfg.projects_dir / episode
    (ep_dir / "segments").mkdir(parents=True, exist_ok=True)
    records = []
    n_speech = 0
    for i, kind in enumerate(kinds, start=1):
        if kind == "pause":
            records.append(SegmentRecord(
                index=i, kind="pause", pause_seconds=0.6, status="success",
                file=f"segments/{i:04d}_PAUSE.wav"))
            continue
        n_speech += 1
        speaker = f"角色{n_speech}"
        ext = kind
        rel = f"segments/{i:04d}_{speaker}.{ext}"
        p = ep_dir / rel
        if ext == "wav":
            _silence_wav(cfg, p, 1.0, 24000, 1)
        else:
            _silence_mp3(cfg, p, 1.0)
        records.append(SegmentRecord(
            index=i, kind="speech", speaker=speaker, text=f"第{n_speech}句",
            status="success", file=rel, content_hash=hashlib.sha1(rel.encode()).hexdigest()))
    man = Manifest(
        episode=episode, created_at="2026-10-02T00:00:00+08:00",
        updated_at="2026-10-02T00:00:00+08:00", script_file=f"{episode}.md",
        script_hash="0" * 64, provider="qwen_audio",
        model="qwen-audio-3.0-tts-flash",
        voices={r.speaker: "longanfengyue" for r in records if r.kind == "speech"},
        segments=records)
    save(ep_dir, man)
    return ep_dir, records


def _file_hashes(ep_dir: Path, records) -> dict[str, bytes]:
    return {
        r.file: (ep_dir / r.file).read_bytes()
        for r in records if r.kind == "speech"
    }


@pytest.mark.parametrize("merge_format", ["mp3", "wav"])
@pytest.mark.parametrize("kinds", [
    ["wav", "wav"],                       # WAV-only
    ["mp3", "mp3"],                       # MP3-only
    ["wav", "pause", "mp3", "wav"],      # 混合 + PAUSE，顺序交错
    ["mp3", "pause", "wav", "pause", "mp3"],
])
def test_real_merge_preserves_originals(tmp_path, merge_format, kinds):
    cfg = _make_cfg(tmp_path, merge_format)
    ep_name = "合 集"  # episode 名也带空格
    ep_dir, records = _build_episode(cfg, ep_name, kinds, merge_format)
    before = _file_hashes(ep_dir, records)

    outcome = audio.run_merge(ep_name, cfg)

    out = ep_dir / outcome.output
    assert out.is_file() and out.stat().st_size > 0
    # 最终文件必须被真实 ffprobe 识别
    fmt = audio.probe(_Tools.ffprobe, out)
    assert fmt.codec  # 能解析出编码器即通过
    dur = audio.probe_duration(_Tools.ffprobe, out)
    expected = sum(1.0 for k in kinds if k != "pause") + \
        sum(0.6 for k in kinds if k == "pause")
    assert dur is not None and dur >= expected - 0.5

    # 原片段哈希不变（关键：.mp3 文件内容仍是 MP3，未被 WAV 覆盖）
    after = _file_hashes(ep_dir, records)
    assert before == after
    for r in records:
        if r.kind == "speech" and r.file.endswith(".mp3"):
            probed = audio.probe(_Tools.ffprobe, ep_dir / r.file)
            assert probed.codec == "mp3"

    # 临时目录已清理；PAUSE 仍是 WAV
    assert not (ep_dir / "metadata" / "merge_tmp").exists()
    pause_files = list((ep_dir / "segments").glob("*_PAUSE.wav"))
    if any(k == "pause" for k in kinds):
        assert pause_files and audio.probe(
            _Tools.ffprobe, pause_files[0]).codec.startswith("pcm_")


def test_real_merge_rerun_skips_and_force_remerges(tmp_path):
    cfg = _make_cfg(tmp_path, "mp3")
    ep_dir, _ = _build_episode(cfg, "再合并", ["wav", "pause", "mp3"], "mp3")
    first = audio.run_merge("再合并", cfg)
    assert first.skipped is False
    second = audio.run_merge("再合并", cfg)
    assert second.skipped is True
    forced = audio.run_merge("再合并", cfg, force=True)
    assert forced.skipped is False
    assert (ep_dir / forced.output).is_file()
