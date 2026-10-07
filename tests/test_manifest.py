"""阶段 2 单元测试：content_hash v2 与旧 manifest 兼容。

不触网、不需要 FFmpeg。
"""
from __future__ import annotations

import hashlib
import json
import logging

from podcast_tts.capabilities import HASH_VERSION
from podcast_tts.manifest import (
    content_hash,
    load as load_manifest,
    load_or_init,
    save,
)
from podcast_tts.models import Segment, SegmentRecord
from podcast_tts.parser import ScriptParser
from test_providers import make_cfg


def _hash_manual(cfg, params: dict) -> str:
    params_key = json.dumps(
        params, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    payload = "|".join([
        HASH_VERSION, cfg.provider, cfg.model, "Cherry", "English",
        "Hello", "", params_key,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_hash_v2_matches_manual_formula():
    cfg = make_cfg()
    params = {"format": "wav", "sample_rate": 24000, "language_hints": ["en"]}
    assert content_hash(cfg, "Hello", "Cherry", "English", None, params) == _hash_manual(cfg, params)


def test_hash_changes_when_params_change():
    cfg = make_cfg()
    base = {"format": "wav", "sample_rate": 24000}
    cases = [
        {"format": "mp3", "sample_rate": 24000},
        {"format": "wav", "sample_rate": 48000},
        {"format": "wav", "sample_rate": 24000, "rate": 1.5},
        {"format": "wav", "sample_rate": 24000, "volume": 80},
        {"format": "wav", "sample_rate": 24000, "language_hints": ["fr"]},
    ]
    h0 = content_hash(cfg, "Hello", "Cherry", "English", None, base)
    for params in cases:
        assert content_hash(cfg, "Hello", "Cherry", "English", None, params) != h0


def test_hash_key_order_independent():
    cfg = make_cfg()
    a = {"rate": 1.0, "format": "wav", "sample_rate": 24000}
    b = {"sample_rate": 24000, "format": "wav", "rate": 1.0}
    assert content_hash(cfg, "Hello", "Cherry", "English", None, a) == \
        content_hash(cfg, "Hello", "Cherry", "English", None, b)


def test_hash_none_params_equals_empty():
    cfg = make_cfg()
    assert content_hash(cfg, "Hello", "Cherry", "English", None, None) == \
        content_hash(cfg, "Hello", "Cherry", "English", None, {})


def test_old_manifest_loads_as_v1_with_empty_snapshot(tmp_path):
    """旧文件无 hash_version/audio_params：加载视为 v1，记录快照为空 dict。"""
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    old = {
        "episode": "ep1", "created_at": "t", "updated_at": "t",
        "script_file": "scripts/ep1.md", "script_hash": "abc",
        "provider": "qwen_audio", "model": "qwen-audio-3.0-tts-plus",
        "voices": {"Lily": "longanlingxin"},
        "segments": [
            {"index": 1, "kind": "speech", "speaker": "Lily",
             "text": "Hi", "status": "success", "file": "segments/0001_Lily.wav",
             "content_hash": "oldhash"},
        ],
        "merge": {},
    }
    (meta / "manifest.json").write_text(json.dumps(old), encoding="utf-8")

    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest.hash_version == "v1"
    assert manifest.record(1).audio_params == {}


def test_v1_manifest_upgrade_warns_and_marks_v2(tmp_path, caplog):
    """旧版 manifest 经 load_or_init：WARNING 一次并升级 hash_version=v2。"""
    cfg = make_cfg()
    cfg = type(cfg)(**{**cfg.__dict__, "projects_dir": tmp_path / "projects"})
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "ep1.md"
    script.write_text("Lily:\nHi\n", encoding="utf-8")
    episode_dir = cfg.projects_dir / "ep1"

    # 先放一个 v1 manifest
    old_manifest_path = episode_dir / "metadata" / "manifest.json"
    old_manifest_path.parent.mkdir(parents=True)
    old = {
        "episode": "ep1", "created_at": "t", "updated_at": "t",
        "script_file": "scripts/ep1.md",
        "script_hash": "deadbeef",
        "provider": cfg.provider, "model": cfg.model,
        "voices": {"Lily": "Cherry", "Leo": "Ethan"},
        "segments": [], "merge": {},
    }
    old_manifest_path.write_text(json.dumps(old), encoding="utf-8")

    segments: list[Segment] = []
    with caplog.at_level(logging.WARNING, logger="podcast_tts"):
        manifest = load_or_init(episode_dir, script, cfg, segments)
    assert manifest.hash_version == HASH_VERSION
    assert any("旧版缓存公式" in r.message for r in caplog.records)
    save(episode_dir, manifest)
    assert load_manifest(episode_dir).hash_version == HASH_VERSION


def test_new_manifest_records_snapshot_roundtrip(tmp_path):
    """新 manifest 落盘后 audio_params 快照可完整回读。"""
    cfg = make_cfg()
    cfg = type(cfg)(**{**cfg.__dict__, "projects_dir": tmp_path / "projects"})
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "ep2.md"
    script.write_text("Lily:\nHi\n", encoding="utf-8")
    episode_dir = cfg.projects_dir / "ep2"
    segments = ScriptParser(set(cfg.voices)).parse(script.read_text(encoding="utf-8"))
    manifest = load_or_init(episode_dir, script, cfg, segments)
    rec = SegmentRecord(
        index=1, kind="speech", speaker="Lily", text="Hi", status="success",
        file="segments/0001_Lily.mp3",
        audio_params={"format": "mp3", "sample_rate": 48000, "rate": 1.2},
    )
    manifest.segments = [rec]
    save(episode_dir, manifest)

    reloaded = load_manifest(episode_dir)
    assert reloaded.record(1).audio_params["format"] == "mp3"
    assert reloaded.record(1).audio_params["rate"] == 1.2
    assert reloaded.hash_version == HASH_VERSION
