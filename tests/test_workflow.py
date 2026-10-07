"""工作流收敛回归测试（需求三十八 ~ 四十二）。

纯本地 + FakeProvider：不触网、不调用真实 TTS、不写真实 Key。
覆盖：
1. 统一片段状态机（missing/pause/clean/dirty/failed + 配置/音色变化）；
2. 整章成品 freshness（未生成/最新/待更新，含 PAUSE 时长变化）；
3. 两个编辑入口 latest-edit-wins（字段级 base 快照冲突消解）；
4. merge 严格化（dirty/missing/failed 不得合并为最新；force 仅高级旁路）；
5. 入口数据源唯一（预览/管理器/规划同一结果）；
6. 文件清理（orphan 音频、output 保留、试听只留最新）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from podcast_tts import audio as audio_mod
from podcast_tts.audio import (
    AudioFormat,
    MergeError,
    run_merge,
    segments_signature,
)
from podcast_tts.config import VoiceSetting
from podcast_tts.gui.controller import (
    GuiController,
    preview_working,
)
from podcast_tts.gui.settings import GuiSettings, RoleVoice
from podcast_tts.manifest import (
    Manifest,
    load as load_manifest,
    save as save_manifest,
)
from podcast_tts.project import WorkingScript
from podcast_tts.pipeline import (
    SEG_STATE_CLEAN,
    SEG_STATE_DIRTY,
    SEG_STATE_FAILED,
    SEG_STATE_MISSING,
    SEG_STATE_PAUSE,
    chapter_freshness,
    guarded_run_merge,
    inspect_segments,
    plan_and_persist,
    run_project,
    source_snapshot_for_edit,
)

from tests.test_pipeline import FakeProvider, make_cfg

# 3 台词 + 1 停顿（Lily / PAUSE / Leo / Lily）
SCRIPT = """Lily:
第一句，你好。

[PAUSE:0.5]

Leo:
第二句，我是 Leo。

Lily:
第三句，再见。
"""


def _ws(episode: str = "ep1", text: str = SCRIPT) -> WorkingScript:
    return WorkingScript.from_paste(episode, text)


def _generate(cfg, ws, *, provider=None):
    return run_project(
        ws.episode, cfg, ws, default_speaker_id="Lily",
        provider=provider or FakeProvider(), persist=True)


def _rows(cfg, ws, *, persist=False):
    rows, _ = inspect_segments(
        ws.episode, cfg, ws, default_speaker_id="Lily", persist=persist)
    return rows


def _state_by_index(cfg, ws):
    return {r["index"]: r["state"] for r in _rows(cfg, ws)}


def _row_by_text(cfg, ws, text_part):
    return next(r for r in _rows(cfg, ws) if text_part in (r["text"] or ""))


# —— 1. 统一片段状态机 ————————————————————————————————————————————————

def test_states_initial_are_missing_and_pause(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    states = _state_by_index(cfg, ws)
    assert states == {1: SEG_STATE_MISSING, 2: SEG_STATE_PAUSE,
                      3: SEG_STATE_MISSING, 4: SEG_STATE_MISSING}


def test_states_all_clean_after_generation(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    states = _state_by_index(cfg, ws)
    assert states == {1: SEG_STATE_CLEAN, 2: SEG_STATE_PAUSE,
                      3: SEG_STATE_CLEAN, 4: SEG_STATE_CLEAN}


def test_state_dirty_after_segment_text_override(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    rows = _rows(cfg, ws)
    sid = rows[2]["seg_id"]  # 第二段台词（Leo）
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"text"})
    ws.set_override(sid, {"text": "片段管理改后的新台词。"}, base=base)
    ws.save(cfg.projects_dir)
    new_state = {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}[sid]
    assert new_state == SEG_STATE_DIRTY


def test_state_dirty_when_audio_file_missing(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    rows = _rows(cfg, ws)
    speech = [r for r in rows if r["kind"] == "speech"]
    (cfg.projects_dir / "ep1" / speech[0]["file"]).unlink()
    assert _state_by_index(cfg, ws)[speech[0]["index"]] == SEG_STATE_DIRTY


def test_state_failed_record(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    # 直接把一段成功记录置为 failed（模拟上一轮失败）
    ep = cfg.projects_dir / "ep1"
    manifest = load_manifest(ep)
    manifest.segments[0].status = "failed"
    save_manifest(ep, manifest)
    assert _state_by_index(cfg, ws)[1] == SEG_STATE_FAILED


def test_state_dirty_after_project_instruction_change(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    cfg2 = make_cfg(tmp_path, default_instruction="全局都要温柔地说")
    assert all(r["state"] == SEG_STATE_DIRTY
               for r in _rows(cfg2, ws) if r["kind"] == "speech")


def test_state_dirty_after_voice_change(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    cfg2 = make_cfg(tmp_path, voices={
        "Lily": VoiceSetting("Cherry", "English"),
        "Leo": VoiceSetting("Ethan", "Chinese"),
    })
    # 把 Leo 换成另一音色 → 仅 Leo 片段 dirty，Lily 仍 clean
    cfg3 = make_cfg(tmp_path, voices={
        "Lily": VoiceSetting("Cherry", "English"),
        "Leo": VoiceSetting("longanlufeng", "Chinese"),
    })
    states = _state_by_index(cfg3, ws)
    assert states[3] == SEG_STATE_DIRTY
    assert states[1] == SEG_STATE_CLEAN and states[4] == SEG_STATE_CLEAN


def test_dirty_back_to_clean_after_regenerate(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    rows = _rows(cfg, ws)
    sid = rows[2]["seg_id"]
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"text"})
    ws.set_override(sid, {"text": "重做后又干净了。"}, base=base)
    ws.save(cfg.projects_dir)
    assert {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}[sid] == "dirty"
    # 再跑一轮：dirty 被拾起重做，完成后恢复 clean
    _generate(cfg, ws, provider=FakeProvider())
    assert {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}[sid] == "clean"


def test_state_dirty_after_segment_direction_override(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    sid = _rows(cfg, ws)[0]["seg_id"]
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"direction"})
    ws.set_override(sid, {"direction": "贴着耳朵悄悄说"}, base=base)
    ws.save(cfg.projects_dir)
    assert {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}[sid] == "dirty"


def test_state_dirty_after_rate_override(tmp_path):
    # rate/pitch 音频参数仅在 qwen-audio plus 发送，hash 也随之变化
    cfg = make_cfg(tmp_path, provider="qwen_audio",
                   model="qwen-audio-3.0-tts-plus")
    ws = _ws()
    _generate(cfg, ws)
    sid = _rows(cfg, ws)[0]["seg_id"]
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"rate"})
    ws.set_override(sid, {"rate": 0.8}, base=base)
    ws.save(cfg.projects_dir)
    assert {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}[sid] == "dirty"


def test_failed_generation_keeps_old_audio_and_marks_failed(tmp_path):
    from podcast_tts.providers.base import FatalProviderError

    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    sid = _rows(cfg, ws)[0]["seg_id"]
    old_file = cfg.projects_dir / "ep1" / _rows(cfg, ws)[0]["file"]
    assert old_file.is_file()
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"text"})
    ws.set_override(sid, {"text": "改完就遇到生成失败。"}, base=base)
    ws.save(cfg.projects_dir)

    failing = FakeProvider(script=[FatalProviderError("模拟云端 400")])
    _generate(cfg, ws, provider=failing)
    states = {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}
    assert states[sid] == SEG_STATE_FAILED
    # 旧音频保留作回退/试听，但状态绝不是 clean
    assert old_file.is_file()

    # 再次补齐（普通"生成 / 更新音频"语义）必须发送当前新内容
    ok = FakeProvider()
    _generate(cfg, ws, provider=ok)
    assert len(ok.calls) == 1
    assert ok.calls[0].text == "改完就遇到生成失败。"
    states = {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}
    assert states[sid] == SEG_STATE_CLEAN


def test_force_regen_clean_segment_calls_tts_once(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    sid = _rows(cfg, ws)[2]["seg_id"]
    fake = FakeProvider()
    run_project("ep1", cfg, ws, only_seg_ids=[sid], force=True,
                default_speaker_id="Lily", provider=fake, persist=True)
    # 显式重做 clean 段：仍强制调用且只调用这一段
    assert len(fake.calls) == 1
    assert {r["seg_id"]: r["state"] for r in _rows(cfg, ws)}[sid] == "clean"


# —— 2. 整章成品 freshness ————————————————————————————————————————————

def _mark_chapter_merged(cfg, episode: str) -> None:
    """模拟一次成功合并：写成品文件 + 存一致签名。"""
    ep = cfg.projects_dir / episode
    manifest = load_manifest(ep)
    out = ep / "output" / f"{episode}.{cfg.merge_format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"CHAPTER-AUDIO")
    manifest.merge = {
        "status": "success",
        "segments_signature": segments_signature(ep, manifest),
        "output": f"output/{episode}.{cfg.merge_format}",
        "merged_at": "2026-09-01T00:00:00+08:00",
    }
    save_manifest(ep, manifest)


def _freshness(cfg, ws):
    return chapter_freshness(ws.episode, cfg, ws, default_speaker_id="Lily")


def test_chapter_missing_before_and_fresh_after_merge(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    assert _freshness(cfg, ws)["state"] == "missing"
    _generate(cfg, ws)
    # 片段全部 clean 但还没合并成品 → 仍"未生成"
    assert _freshness(cfg, ws)["state"] == "missing"
    _mark_chapter_merged(cfg, "ep1")
    assert _freshness(cfg, ws)["state"] == "fresh"


def test_chapter_stale_when_segment_dirty(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    _mark_chapter_merged(cfg, "ep1")
    rows = _rows(cfg, ws)
    sid = rows[0]["seg_id"]
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"text"})
    ws.set_override(sid, {"text": "改了一个字。"}, base=base)
    ws.save(cfg.projects_dir)
    info = _freshness(cfg, ws)
    assert info["state"] == "stale"
    assert info["counts"]["dirty"] == 1
    # 旧成品文件不会被删除（保留可回退），只是状态不再"最新"
    assert (cfg.projects_dir / "ep1" / "output" / "ep1.mp3").is_file()


def test_chapter_stale_when_only_pause_duration_changes(tmp_path):
    """PAUSE 时长变化不 dirty 任何台词段，但整章签名变化 → 待更新（只需重合并）。"""
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    _mark_chapter_merged(cfg, "ep1")
    pause_row = next(r for r in _rows(cfg, ws) if r["kind"] == "pause")
    base = source_snapshot_for_edit("ep1", cfg, ws, pause_row["seg_id"],
                                    {"pause_seconds"})
    ws.set_override(pause_row["seg_id"], {"pause_seconds": 2.0}, base=base)
    ws.save(cfg.projects_dir)
    info = _freshness(cfg, ws)
    assert info["state"] == "stale"
    assert info["counts"]["clean"] == 3 and info["counts"]["dirty"] == 0


# —— 3. 两个编辑入口 latest-edit-wins ——————————————————————————————————

def _override_segment(cfg, ws, index, patch):
    rows = _rows(cfg, ws)
    sid = rows[index - 1]["seg_id"]
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, set(patch))
    ws.set_override(sid, patch, base=base)
    ws.save(cfg.projects_dir)
    return sid


def test_main_textbox_edit_alone_marks_segment_dirty(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    ws.update_text(SCRIPT.replace("第一句，你好。", "直接在主文本框改的句子。"))
    row = _row_by_text(cfg, ws, "直接在主文本框改的句子")
    assert row["state"] == SEG_STATE_DIRTY


def test_conflict_main_textbox_wins_over_segment_text_override(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    plan_and_persist("ep1", cfg, ws, default_speaker_id="Lily")
    sid = _override_segment(cfg, ws, 1, {"text": "片段管理改的 B。"})
    # 保存后规划生效：B
    assert _row_by_text(cfg, ws, "片段管理改的 B")["seg_id"] == sid
    # 用户随后在主文本框把同一句改成 C
    ws.update_text(SCRIPT.replace("第一句，你好。", "主文本框后改的 C。"))
    row = next(r for r in _rows(cfg, ws) if r["seg_id"] == sid)
    assert row["text"] == "主文本框后改的 C。"
    # 失效覆盖已被清除，不再残留
    assert sid not in ws.overrides


def test_conflict_rate_override_kept_when_only_main_text_changes(tmp_path):
    # rate/pitch 参数测试按约定使用 qwen_audio plus 模型
    cfg = make_cfg(tmp_path, provider="qwen_audio",
                   model="qwen-audio-3.0-tts-plus")
    ws = _ws()
    plan_and_persist("ep1", cfg, ws, default_speaker_id="Lily")
    sid = _override_segment(cfg, ws, 1, {"rate": 1.5})
    # 主文本框只改正文，不动 rate → rate 覆盖保留
    ws.update_text(SCRIPT.replace("第一句，你好。", "只改了正文内容。"))
    row = next(r for r in _rows(cfg, ws) if r["seg_id"] == sid)
    assert row["text"] == "只改了正文内容。"
    assert ws.overrides[sid].get("rate") == 1.5
    assert float(row["rate"]) == 1.5


def test_conflict_direction_override_kept_when_only_main_text_changes(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    plan_and_persist("ep1", cfg, ws, default_speaker_id="Lily")
    sid = _override_segment(cfg, ws, 1, {"direction": "悄悄地说"})
    ws.update_text(SCRIPT.replace("第一句，你好。", "正文变了指令不变。"))
    row = next(r for r in _rows(cfg, ws) if r["seg_id"] == sid)
    assert row["direction"] == "悄悄地说"
    assert ws.overrides[sid].get("direction") == "悄悄地说"


def test_conflict_re_edit_refreshes_base_chain(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    plan_and_persist("ep1", cfg, ws, default_speaker_id="Lily")
    sid = _override_segment(cfg, ws, 1, {"text": "第一次片段编辑 B。"})
    # 主文本框改 C（下次规划时 B 覆盖失效）
    ws.update_text(SCRIPT.replace("第一句，你好。", "主文本改 C。"))
    _ = _rows(cfg, ws)
    assert sid not in ws.overrides
    # 片段管理再次基于"当前源 C"编辑为 D
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"text"})
    ws.set_override(sid, {"text": "第二次片段编辑 D。"}, base=base)
    ws.save(cfg.projects_dir)
    row = next(r for r in _rows(cfg, ws) if r["seg_id"] == sid)
    assert row["text"] == "第二次片段编辑 D。"
    # 主文本框再改 E → D 同样失效，永远是最后一次编辑生效
    ws.update_text(ws.text.replace("主文本改 C。", "主文本又改 E。"))
    row = next(r for r in _rows(cfg, ws) if r["seg_id"] == sid)
    assert row["text"] == "主文本又改 E。"
    assert sid not in ws.overrides


def test_pause_override_yields_to_later_main_text_duration(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    plan_and_persist("ep1", cfg, ws, default_speaker_id="Lily")
    pause_sid = _override_segment(cfg, ws, 2, {"pause_seconds": 3.0})
    assert next(r for r in _rows(cfg, ws)
                if r["kind"] == "pause")["pause_seconds"] == 3.0
    # 主文本框把停顿改成 1.2 秒 → 片段管理的时长覆盖失效
    ws.update_text(SCRIPT.replace("[PAUSE:0.5]", "[PAUSE:1.2]"))
    row = next(r for r in _rows(cfg, ws) if r["seg_id"] == pause_sid)
    assert row["pause_seconds"] == 1.2
    assert pause_sid not in ws.overrides


# —— 4. merge 严格化 ————————————————————————————————————————————————

class _CmdRecorder:
    """test_audio.CommandRecorder 的本地副本（模拟 ffmpeg/ffprobe）。"""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> None:
        self.commands.append(list(cmd))
        joined = " ".join(cmd)
        if "anullsrc" in joined or ("-f" in cmd and "concat" in joined) or "-ar" in cmd:
            Path(cmd[-1]).write_bytes(b"RIFFfake")

    @property
    def concat_count(self) -> int:
        return sum(1 for c in self.commands if "-f" in c and "concat" in " ".join(c))


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    rec = _CmdRecorder()
    monkeypatch.setattr(audio_mod, "_run_command", rec)
    monkeypatch.setattr(
        audio_mod, "probe",
        lambda ffprobe, file: AudioFormat("pcm_s16le", 24000, 1))
    monkeypatch.setattr(audio_mod, "probe_duration",
                        lambda ffprobe, file: 3.0)
    return rec


def _clean_episode(cfg, episode="ep1", spec=None):
    """用真实规划指纹构造全 success 的 episode（夹具音频与 clean 状态一致）。"""
    from tests.test_audio import make_episode
    spec = spec or [("speech", "Lily"), ("pause", 0.5), ("speech", "Leo")]
    return make_episode(cfg, spec, episode=episode)


def test_merge_clean_episode_ok(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    _clean_episode(cfg)
    outcome = run_merge("ep1", cfg)
    assert outcome.skipped is False
    assert (cfg.projects_dir / "ep1" / "output" / "ep1.mp3").is_file()


def test_merge_blocks_dirty_via_self_recheck(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    ep = _clean_episode(cfg)
    # 模拟"规划已变但未重新生成"：记录文本（请求组成部分）变化
    manifest = load_manifest(ep)
    manifest.segments[0].text = "改了台词但音频还是旧的"
    save_manifest(ep, manifest)
    with pytest.raises(MergeError, match="待更新"):
        run_merge("ep1", cfg)
    # 被拒后不得产出/覆盖成品
    assert not (ep / "output" / "ep1.mp3").exists()


def test_merge_blocks_failed_segments(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    ep = _clean_episode(cfg)
    manifest = load_manifest(ep)
    manifest.segments[0].status = "failed"
    save_manifest(ep, manifest)
    with pytest.raises(MergeError, match="未完成"):
        run_merge("ep1", cfg)


def test_merge_blocks_pending_segments(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    ep = _clean_episode(cfg)
    manifest = load_manifest(ep)
    # 新片段已入清单但从未生成（文件即便残留也不允许合并为"最新"）
    manifest.segments[0].status = "pending"
    save_manifest(ep, manifest)
    with pytest.raises(MergeError, match="未完成"):
        run_merge("ep1", cfg)
    assert not (ep / "output" / "ep1.mp3").exists()


def test_merge_blocks_missing_segment_file(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    _clean_episode(cfg)
    (cfg.projects_dir / "ep1" / "segments" / "0001_Lily.wav").unlink()
    with pytest.raises(MergeError, match="缺失"):
        run_merge("ep1", cfg)


def test_merge_blocks_when_expected_hashes_mismatch(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    ep = _clean_episode(cfg)
    manifest = load_manifest(ep)
    for i, r in enumerate(manifest.segments):
        if r.kind == "speech":
            r.seg_id = f"sid-{i}"
    save_manifest(ep, manifest)
    with pytest.raises(MergeError, match="待更新"):
        run_merge("ep1", cfg, expected_hashes={"sid-0": "stale-hash",
                                               "sid-2": "stale-hash"})


def test_merge_force_bypasses_strict_check_advanced(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    ep = _clean_episode(cfg)
    manifest = load_manifest(ep)
    manifest.segments[0].text = "规划变了但高级用户明确要强制重拼"
    save_manifest(ep, manifest)
    # force=True 是高级旁路：用现有片段文件强制合并（不静默、调用方已警告用户）
    outcome = run_merge("ep1", cfg, force=True)
    assert outcome.skipped is False
    assert (ep / "output" / "ep1.mp3").is_file()


def test_merge_remerges_when_pause_duration_changes(tmp_path, fake_ffmpeg):
    cfg = make_cfg(tmp_path)
    ep = _clean_episode(cfg)
    run_merge("ep1", cfg)
    assert fake_ffmpeg.concat_count == 1
    # 只改 PAUSE 时长（台词音频不动）→ 签名变化 → 重新合并且严格检查放行
    manifest = load_manifest(ep)
    manifest.segments[1].pause_seconds = 2.0
    save_manifest(ep, manifest)
    outcome = run_merge("ep1", cfg)
    assert outcome.skipped is False
    assert fake_ffmpeg.concat_count == 2


def test_guarded_merge_blocks_when_segments_pending(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    # 制造一个 dirty 片段但不重新生成
    sid = _rows(cfg, ws)[0]["seg_id"]
    base = source_snapshot_for_edit("ep1", cfg, ws, sid, {"text"})
    ws.set_override(sid, {"text": "还没重做。"}, base=base)
    ws.save(cfg.projects_dir)

    captured = {}

    def _fake_run_merge(episode, cfg, *, force=False, expected_hashes=None):
        captured.update(force=force, expected_hashes=expected_hashes)
        raise AssertionError("有未更新片段时不应调用 run_merge")

    monkeypatch.setattr("podcast_tts.pipeline.run_merge", _fake_run_merge)
    with pytest.raises(MergeError, match="暂不能生成最新整章"):
        guarded_run_merge("ep1", cfg, ws, default_speaker_id="Lily")
    assert captured == {}


def test_guarded_merge_passes_expected_hashes_when_all_clean(tmp_path,
                                                             monkeypatch):
    cfg = make_cfg(tmp_path)
    ws = _ws()
    _generate(cfg, ws)
    captured = {}

    def _fake_run_merge(episode, cfg, *, force=False, expected_hashes=None):
        captured.update(force=force, expected_hashes=expected_hashes)
        return object()

    monkeypatch.setattr("podcast_tts.pipeline.run_merge", _fake_run_merge)
    guarded_run_merge("ep1", cfg, ws, default_speaker_id="Lily")
    assert captured["force"] is False
    speech_ids = {r["seg_id"] for r in _rows(cfg, ws) if r["kind"] == "speech"}
    assert set(captured["expected_hashes"]) == speech_ids
    assert all(captured["expected_hashes"].values())


# —— 5. 入口数据源唯一 ——————————————————————————————————————————————

def _gui_settings(tmp_path, episode: str = "epgui"):
    return GuiSettings(
        api_key="sk-test", provider="qwen", model="qwen3-tts-instruct-flash",
        roles=[
            RoleVoice(role="Lily", voice="Cherry", language_type="English"),
            RoleVoice(role="Leo", voice="Ethan", language_type="Chinese"),
        ],
        default_role="Lily", episode_name=episode,
        projects_dir=str(tmp_path / "projects"),
    )


def test_preview_and_manager_share_same_planned_result(tmp_path):
    s = _gui_settings(tmp_path)
    c = GuiController()
    ws = WorkingScript.from_paste("epgui", SCRIPT)
    # 先完整生成一轮（之后单句编辑才会表现为 dirty 而非 missing）
    run_project("epgui", s.to_app_config(), ws,
                default_speaker_id="Lily", provider=FakeProvider(), persist=True)
    manager_rows = c.segment_rows(s, ws)
    sid = next(r["seg_id"] for r in manager_rows if r["speaker_name"] == "Leo")
    c.edit_segment(s, sid, {"text": "两个入口看到同一份文本。"}, ws=ws)
    manager_rows = c.segment_rows(s, ws)
    preview_rows, errors = preview_working(s, ws)
    assert errors == []
    m = next(r for r in manager_rows if r["seg_id"] == sid)
    p = next(r for r in preview_rows if r["index"] == m["index"])
    assert p["text"] == m["text"] == "两个入口看到同一份文本。"
    assert p["state"] == m["state"] == "dirty"
    assert p["action_zh"] and p["state_zh"]


# —— 6. 文件清理 ————————————————————————————————————————————————————

def test_plan_persist_removes_orphan_audio_but_keeps_output(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws("epclean")
    _generate(cfg, ws)
    ep = cfg.projects_dir / "epclean"
    seg_dir = ep / "segments"
    stray_audio = seg_dir / "9999_OLD.wav"
    stray_audio.write_bytes(b"OLD")
    note = seg_dir / "readme.txt"          # 非白名单扩展名：绝不动
    note.write_text("keep", encoding="utf-8")
    out_dir = ep / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    chapter = out_dir / "epclean.mp3"
    chapter.write_bytes(b"CHAPTER")

    referenced = {r.file for r in load_manifest(ep).segments if r.file}
    from podcast_tts.manifest import cleanup_orphan_audio
    removed = cleanup_orphan_audio(ep, referenced)
    assert "segments/9999_OLD.wav" in removed
    assert not stray_audio.exists()
    assert note.exists() and chapter.exists()
    # 受引用的台词片段一个不少（PAUSE 静音文件在首次合并时才生成）
    for rel in referenced:
        if "_PAUSE" not in Path(rel).stem:
            assert (ep / rel).is_file()


def test_removing_segment_deletes_its_audio_via_plan_persist(tmp_path):
    """需求四十二：删除 Segment → manifest 不再引用 → orphan 音频安全删除；
    正式 output 不受影响（端到端走 plan_and_persist，而非直接调清理函数）。"""
    cfg = make_cfg(tmp_path)
    ws = _ws("epcut")
    _generate(cfg, ws)
    ep = cfg.projects_dir / "epcut"
    out_dir = ep / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    chapter = out_dir / "epcut.mp3"
    chapter.write_bytes(b"CHAPTER")
    rows = _rows(cfg, ws)
    removed_audio = ep / rows[-1]["file"]
    assert removed_audio.is_file()

    # 用户在主文本框删掉最后一段并重新规划
    ws.update_text(SCRIPT.rsplit("Lily:\n第三句，再见。", 1)[0].strip())
    ws.save(cfg.projects_dir)
    _segments, manifest = plan_and_persist(
        "epcut", cfg, ws, default_speaker_id="Lily")

    assert not removed_audio.exists()                 # 孤儿音频已删
    assert chapter.is_file()                          # 成品不动
    kept = [r for r in manifest.segments if r.kind == "speech"]
    assert len(kept) == 2
    for rec in kept:
        assert (ep / rec.file).is_file()
    # 非白名单文件绝不动
    note = ep / "segments" / "notes.txt"
    note.parent.mkdir(exist_ok=True)
    note.write_text("keep", encoding="utf-8")
    plan_and_persist("epcut", cfg, ws, default_speaker_id="Lily")
    assert note.exists()


def test_audition_dir_keeps_only_latest_for_role(tmp_path):
    s = _gui_settings(tmp_path)
    c = GuiController(provider_factory=lambda cfg: FakeProvider())
    aud_dir = Path(s.projects_dir) / "_audition"
    aud_dir.mkdir(parents=True, exist_ok=True)
    sid = s.default_role_id()  # 试听文件按角色稳定 ID 命名
    # 同角色历史试听残留（不同文件名/内容）；其他角色文件不碰
    old = aud_dir / f"audition_{sid}.old.wav"
    old.write_bytes(b"OLD-TRIAL")
    other = aud_dir / "audition_other-role.wav"
    other.write_bytes(b"OTHER")
    c.audition(s, "你好，这是音色试听。")
    c.join(10)
    lily_files = list(aud_dir.glob(f"audition_{sid}.*"))
    assert len(lily_files) == 1            # 同角色只留本次一份
    assert not old.exists()
    assert lily_files[0].read_bytes() == b"FAKEWAVDATA"  # 留的是最新内容
    assert other.exists()


# —— 7. 单段重生成：其他段 dirty 时绝不自动拼接旧音频（需求四十）————————

def _drain_events(q, timeout: float = 15.0):
    import queue as _queue
    import time as _time
    events = []
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        try:
            events.append(q.get(timeout=0.1))
        except _queue.Empty:
            if any(e[0] == "finished" for e in events):
                break
    return events


def test_single_regen_skips_merge_when_another_segment_dirty(tmp_path):
    import queue
    from tests.test_gui import FakeMerge

    merge = FakeMerge()
    q: queue.Queue = queue.Queue()
    c = GuiController(q, provider_factory=lambda cfg: FakeProvider(),
                      merge_func=merge)
    s = _gui_settings(tmp_path, episode="epguard")
    c.start_text(s, SCRIPT)
    c.join(20)
    _drain_events(q)
    assert len(merge.calls) == 1  # 首轮全 clean：自动合并一次

    ws = c.ws
    rows = c.segment_rows(s, ws)
    leo = next(r for r in rows if r["speaker_name"] == "Leo")
    lily = next(r for r in rows if r["speaker_name"] == "Lily")
    # 只改 Leo（dirty），随后仅强制重生成另一段 Lily → Leo 仍旧
    c.edit_segment(s, leo["seg_id"], {"text": "Leo 改了但这轮先不重做。"})
    c.regenerate_segments(s, [lily["seg_id"]], ws)
    c.join(20)
    events = _drain_events(q)

    # 守卫拦截：FakeMerge 没有第二次调用，旧成品不会被"伪最新"覆盖
    assert len(merge.calls) == 1
    finished = next(e for e in events if e[0] == "finished")
    assert finished[1]["merged"] is False
    assert finished[1]["merge_skipped"] is True
    skipped = [e for e in events if e[0] == "merge_skipped"]
    assert skipped
    # 待更新列表带片段编号与中文状态，供主界面提示
    assert any("待更新" in p for p in skipped[0][1]["pending"])


# —— 8. PAUSE 可在任意合法位置（需求：开头/中间/结尾停顿）————————————————

from podcast_tts.project import parse_working  # noqa: E402


def _parse_kinds(cfg, text, *, episode="epp"):
    ws = WorkingScript.from_paste(episode, text)
    segs = parse_working(ws, cfg, default_speaker_id="Lily")
    return segs


def test_pause_at_script_start_then_speech(tmp_path):
    """测试 1：[PAUSE:1.5] + Lily → 2 段（pause 1.5, speech）。"""
    cfg = make_cfg(tmp_path)
    segs = _parse_kinds(cfg, "[PAUSE:1.5]\nLily: 你好。")
    assert [(s.kind, s.pause_seconds, s.speaker_name, s.text) for s in segs] == [
        ("pause", 1.5, None, None),
        ("speech", None, "Lily", "你好。"),
    ]
    assert [s.index for s in segs] == [1, 2]


def test_pause_at_start_and_end(tmp_path):
    """测试 2：开头 + 结尾 PAUSE → pause / speech / pause 共 3 段。"""
    cfg = make_cfg(tmp_path)
    segs = _parse_kinds(cfg, "[PAUSE:1.5]\nLily: 你好。\n[PAUSE:2.0]")
    assert [s.kind for s in segs] == ["pause", "speech", "pause"]
    assert segs[0].pause_seconds == 1.5 and segs[2].pause_seconds == 2.0
    assert segs[1].text == "你好。"


def test_pause_between_and_after_multiple_speakers(tmp_path):
    """开头/中间/结尾 PAUSE 与多角色混排，序号不错乱。"""
    cfg = make_cfg(tmp_path)
    segs = _parse_kinds(
        cfg, "[PAUSE:1.5]\nLily: 你终于来了。\n[PAUSE:2.0]\nLeo: 抱歉。\n[PAUSE:1.0]")
    assert [s.kind for s in segs] == ["pause", "speech", "pause", "speech", "pause"]
    assert [s.index for s in segs] == [1, 2, 3, 4, 5]
    assert segs[1].speaker_name == "Lily" and segs[3].speaker_name == "Leo"


def test_book_chapter_leading_pause(tmp_path):
    """测试 3：章节模式章首元数据后直接 [PAUSE:2.0] → pause + 默认角色 speech。"""
    cfg = make_cfg(tmp_path)
    book = ("# 第一章\n\n- 角色: Lily\n- 导演: 悬疑、缓慢\n\n"
            "[PAUSE:2.0]\n\n夜色渐渐笼罩城市。")
    ws = WorkingScript(episode="epb", mode="book", text=book)
    segs = parse_working(ws, cfg, default_speaker_id="Lily")
    assert [s.kind for s in segs] == ["pause", "speech"]
    assert segs[0].pause_seconds == 2.0
    assert segs[1].speaker_name == "Lily"
    assert segs[1].text == "夜色渐渐笼罩城市。"
    assert segs[1].chapter_direction == "悬疑、缓慢"


def test_pure_pause_project_silence_only(tmp_path, fake_ffmpeg):
    """测试 4/5：纯 [PAUSE:1.5] 项目——不调 TTS，合成 1.5s 静音成品。"""
    cfg = make_cfg(tmp_path)
    ws = WorkingScript.from_paste("eppure", "[PAUSE:1.5]")
    fake = FakeProvider()
    run_project("eppure", cfg, ws, default_speaker_id="Lily",
                provider=fake, persist=True)
    assert fake.calls == []                      # 绝不调用 TTS
    outcome = run_merge("eppure", cfg)           # 不因"第一段是 PAUSE"报错
    assert outcome.skipped is False
    assert (cfg.projects_dir / "eppure" / outcome.output).is_file()
    # 只有一条静音生成命令，时长 1.500
    silences = [c for c in fake_ffmpeg.commands if "anullsrc" in " ".join(c)]
    assert len(silences) == 1
    assert "1.500" in silences[0]


def test_merge_keeps_leading_and_trailing_silence(tmp_path, fake_ffmpeg):
    """测试 6：merge 后拼接顺序 = 头静音 + 台词 + 尾静音（第一条输入是静音）。"""
    cfg = make_cfg(tmp_path)
    ws = WorkingScript.from_paste("eplead", "[PAUSE:1.5]\nLily: 你好。\n[PAUSE:1.0]")
    fake = FakeProvider()
    run_project("eplead", cfg, ws, default_speaker_id="Lily",
                provider=fake, persist=True)
    assert len(fake.calls) == 1                  # 只有 Lily 一句调 TTS
    run_merge("eplead", cfg)

    ep = cfg.projects_dir / "eplead"
    concat = (ep / "metadata" / "concat_list.txt").read_text(encoding="utf-8")
    ordered = [ln.split("'")[1].replace("\\", "/")
               for ln in concat.splitlines() if ln.startswith("file ")]
    manifest = load_manifest(ep)
    by_index = {r.index: r for r in manifest.segments}
    # 第一条输入就是 PAUSE 静音，最后一条是结尾 PAUSE 静音
    assert ordered[0].endswith(by_index[1].file)
    assert by_index[1].kind == "pause" and by_index[1].pause_seconds == 1.5
    assert ordered[-1].endswith(by_index[3].file)
    assert by_index[3].kind == "pause" and by_index[3].pause_seconds == 1.0
    assert len(ordered) == 3
    # 两条静音生成命令，时长与顺序对应
    silences = [c for c in fake_ffmpeg.commands if "anullsrc" in " ".join(c)]
    durations = [c[c.index("-t") + 1] for c in silences]
    assert durations == ["1.500", "1.000"]


def test_edit_leading_pause_only_remerges_no_tts(tmp_path, fake_ffmpeg):
    """测试 7：开头 PAUSE 1.5→2.0 → 0 次 TTS，整章待更新，重合并生效。"""
    cfg = make_cfg(tmp_path)
    ws = WorkingScript.from_paste("epedit", "[PAUSE:1.5]\nLily: 你好。")
    fake = FakeProvider()
    run_project("epedit", cfg, ws, default_speaker_id="Lily",
                provider=fake, persist=True)
    run_merge("epedit", cfg)
    _mark_chapter_merged(cfg, "epedit")
    assert len(fake.calls) == 1

    # 主文本框把开头 PAUSE 改成 2.0
    ws.update_text("[PAUSE:2.0]\nLily: 你好。")
    ws.save(cfg.projects_dir)
    info = _freshness(cfg, ws)
    assert info["state"] == "stale"             # 只需要重新合并
    assert info["counts"]["dirty"] == 0         # 台词不 dirty

    fake2 = FakeProvider()
    run_project("epedit", cfg, ws, default_speaker_id="Lily",
                provider=fake2, persist=True)
    assert fake2.calls == []                    # 不调用 TTS
    before = fake_ffmpeg.concat_count
    outcome = run_merge("epedit", cfg)
    assert outcome.skipped is False
    assert fake_ffmpeg.concat_count == before + 1
    silences = [c for c in fake_ffmpeg.commands if "anullsrc" in " ".join(c)]
    assert silences[-1][silences[-1].index("-t") + 1] == "2.000"


def test_preview_shows_leading_pause_row(tmp_path):
    """需求六：生成预览第一条正常显示 PAUSE（类型/时长），序号与后续 speech 不丢。"""
    s = _gui_settings(tmp_path)
    ws = WorkingScript.from_paste("epgui", "[PAUSE:1.5]\nLily: 你好。")
    rows, errors = preview_working(s, ws)
    assert errors == []
    assert rows[0]["kind"] == "pause" and rows[0]["index"] == 1
    assert rows[0]["pause_seconds"] == 1.5
    assert rows[1]["kind"] == "speech" and rows[1]["index"] == 2
    assert rows[1]["speaker_name"] == "Lily"
    # 状态/动作列也不因首段 PAUSE 错乱
    assert rows[0]["state"] == SEG_STATE_PAUSE
    assert rows[1]["state"] == SEG_STATE_MISSING
