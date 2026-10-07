"""Working Script 机制端到端测试（用户验收 12 场景）。

纯本地 + FakeProvider：不触网、不调用真实 TTS、不写真实 Key。
覆盖：导入只读、粘贴汇合、单段编辑 dirty、seg_id 稳定、单段重生成、
失败隔离、PAUSE 不调 TTS、合并顺序、旧项目兼容、单/多角色解析。
"""
from __future__ import annotations

import queue
import time
from pathlib import Path

import pytest

from podcast_tts.audio import segments_signature
from podcast_tts.config import ConfigError, VoiceSetting
from podcast_tts.gui.controller import GuiController
from podcast_tts.gui.settings import GuiSettings, RoleVoice
from podcast_tts.manifest import load as load_manifest
from podcast_tts.project import (
    WorkingScript,
    apply_overrides,
    parse_working,
    working_path,
)
from podcast_tts.pipeline import run_project, run_regenerate, segment_states
from podcast_tts.providers.base import FatalProviderError

from tests.test_gui import FakeMerge
from tests.test_pipeline import FakeProvider, make_cfg

# 3 台词 + 1 停顿（顺序：Lily / PAUSE / Leo / Lily）
SCRIPT = """Lily:
第一句，你好。

[PAUSE:0.5]

Leo:
第二句，我是 Leo。

Lily:
第三句，再见。
"""

BOOK_MULTI = """# 第一章 出发

- 角色: Lily

清晨，Lily 出门了。

# 第二章 到达

- 角色: Leo

终于到了！
"""


def _ws_paste(episode: str = "ep1", text: str = SCRIPT) -> WorkingScript:
    return WorkingScript.from_paste(episode, text)


def _generate(cfg, ws, *, provider=None, default_speaker_id="Lily"):
    return run_project(
        ws.episode, cfg, ws,
        default_speaker_id=default_speaker_id,
        provider=provider or FakeProvider(), persist=True)


def _states(cfg, ws, default_speaker_id="Lily") -> dict[str, dict]:
    return {r["seg_id"]: r for r in
            segment_states(ws.episode, cfg, ws,
                           default_speaker_id=default_speaker_id)}


def _speech_ids(cfg, ws, default_speaker_id="Lily") -> list[str]:
    rows = segment_states(ws.episode, cfg, ws,
                          default_speaker_id=default_speaker_id)
    return [r["seg_id"] for r in rows if r["kind"] == "speech"]


def _pause_id(cfg, ws, default_speaker_id="Lily") -> str:
    rows = segment_states(ws.episode, cfg, ws,
                          default_speaker_id=default_speaker_id)
    return next(r["seg_id"] for r in rows if r["kind"] == "pause")


# —— 1. 文件导入 → Working Script，原始文件不被修改 ————————————————————————

def test_file_import_creates_working_script_source_readonly(tmp_path):
    src = tmp_path / "chapter01.md"
    src.write_text("# 第一章\n\n- 角色: Lily\n\n你好，世界。\n", encoding="utf-8")
    before = src.read_bytes()
    stat_before = (src.stat().st_mtime_ns, src.stat().st_size)

    cfg = make_cfg(tmp_path)
    ws = WorkingScript.import_file("chapter01", src, mode="book")
    assert ws.source_type == "file" and ws.source_name == "chapter01.md"
    assert "你好，世界" in ws.text

    _generate(cfg, ws)  # 完整跑一遍：解析/对齐/生成/落工作脚本
    ws.save(cfg.projects_dir)

    assert working_path(cfg.projects_dir, "chapter01").is_file()
    assert src.read_bytes() == before
    assert (src.stat().st_mtime_ns, src.stat().st_size) == stat_before


# —— 2. 文本粘贴 → 直接成为 Working Script ——————————————————————————————————

def test_pasted_text_becomes_working_script(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = WorkingScript.from_paste("ep1", "Lily:\n你终于来了。\n")
    segs = parse_working(ws, cfg)
    assert len(segs) == 1 and segs[0].speaker == "Lily"
    assert segs[0].text == "你终于来了。"
    assert ws.source_type == "paste" and ws.source_name is None


def test_plain_paste_uses_default_role_and_never_silent(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = WorkingScript.from_paste("ep1", "没有角色头的普通文本。")
    # 有默认角色：统一补角色头
    segs = parse_working(ws, cfg, default_speaker_id="Lily")
    assert len(segs) == 1 and segs[0].speaker == "Lily"
    assert segs[0].text == "没有角色头的普通文本。"
    # 原文不被改写
    assert ws.text == "没有角色头的普通文本。"
    # 无默认角色：报错，绝不静默使用任何角色
    with pytest.raises(ConfigError, match="默认角色"):
        parse_working(ws, cfg)


# —— 3. 修改一个 Segment → 只有该 Segment dirty ————————————————————————————

def test_edit_one_segment_only_that_one_is_dirty(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)

    states = _states(cfg, ws)
    assert {r["state"] for r in states.values()} >= {"clean", "pause"}
    target = _speech_ids(cfg, ws)[1]  # 第 2 句（Leo）

    ws.set_override(target, {"text": "说真的……我现在有点害怕。"})
    states = _states(cfg, ws)
    assert states[target]["state"] == "dirty"
    dirty = [sid for sid, r in states.items() if r["state"] == "dirty"]
    assert dirty == [target]


# —— 4. 修改一个 Segment → 其他 Segment ID 保持稳定 ————————————————————————

def test_edit_keeps_other_segment_ids_stable(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)

    ids_before = _speech_ids(cfg, ws)
    pause_before = _pause_id(cfg, ws)
    ws.set_override(ids_before[1], {"text": "只改这一句。"})

    ids_after = _speech_ids(cfg, ws)
    assert ids_after == ids_before          # 顺序与身份都不变
    assert _pause_id(cfg, ws) == pause_before
    assert _states(cfg, ws)[ids_before[1]]["text"] == "只改这一句。"


def test_insert_line_keeps_existing_segment_ids(tmp_path):
    """在中间插入一句：旧片段 seg_id 沿用，新句才是新身份。"""
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)
    ids_before = _speech_ids(cfg, ws)

    ws.update_text(SCRIPT.replace(
        "Leo:\n第二句，我是 Leo。\n",
        "Leo:\n第二句，我是 Leo。\n\nLily:\n插入的新句。\n"))
    # 落盘规划后，新插入片段也获得持久化 seg_id（与管理器打开时同一路径）
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    ids_after = _speech_ids(cfg, ws)
    states = _states(cfg, ws)
    # 旧 3 句全部沿用（LCS 对齐，插入不重新编号身份）
    for sid in ids_before:
        assert sid in ids_after
    assert len(ids_after) == 4
    new_ids = [s for s in ids_after if s not in ids_before]
    assert len(new_ids) == 1
    assert states[new_ids[0]]["state"] == "missing"


# —— 5. seg_id 单段重生成 → 只调用一次 TTS —————————————————————————————————

def test_regenerate_one_segment_calls_tts_once(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws, provider=FakeProvider())

    target = _speech_ids(cfg, ws)[1]
    ws.set_override(target, {"text": "只重做这一句。"})
    fake2 = FakeProvider()
    outcome = run_regenerate(
        "ep1", [], cfg, seg_ids=[target], ws=ws, provider=fake2)
    assert len(fake2.calls) == 1
    assert fake2.calls[0].text == "只重做这一句。"
    assert outcome.succeeded == 1 and outcome.failed == 0


# —— 6. 单段生成成功 → dirty 变 clean ———————————————————————————————————————

def test_successful_regen_marks_segment_clean(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)
    target = _speech_ids(cfg, ws)[2]
    ws.set_override(target, {"text": "重做成功后应恢复 clean。"})
    assert _states(cfg, ws)[target]["state"] == "dirty"

    run_regenerate("ep1", [], cfg, seg_ids=[target], ws=ws,
                   provider=FakeProvider())
    assert _states(cfg, ws)[target]["state"] == "clean"


# —— 7. 单段生成失败 → 其他 Segment 不受影响 ————————————————————————————————

def test_failed_segment_does_not_affect_others(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)
    ids = _speech_ids(cfg, ws)
    target = ids[0]
    ws.set_override(target, {"text": "这句一定会失败。"})

    failing = FakeProvider([FatalProviderError("模拟 TTS 失败")])
    outcome = run_regenerate("ep1", [], cfg, seg_ids=[target], ws=ws,
                             provider=failing)
    assert outcome.failed == 1 and outcome.succeeded == 0
    states = _states(cfg, ws)
    assert states[target]["state"] == "failed"
    for sid in ids[1:]:
        assert states[sid]["state"] == "clean"


# —— 8. voice / rate / instruction 变化 → 当前片段 dirty ————————————————————

def test_rate_and_instruction_override_dirties_segment(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio",
                   model="qwen-audio-3.0-tts-plus")
    ws = _ws_paste()
    _generate(cfg, ws)
    target = _speech_ids(cfg, ws)[1]

    ws.set_override(target, {"rate": 1.1})
    assert _states(cfg, ws)[target]["state"] == "dirty"
    # 重新生成恢复 clean 后，再改导演指令仍应 dirty
    run_regenerate("ep1", [], cfg, seg_ids=[target], ws=ws,
                   provider=FakeProvider())
    assert _states(cfg, ws)[target]["state"] == "clean"
    ws.set_override(target, {"direction": "压低声音"})
    assert _states(cfg, ws)[target]["state"] == "dirty"


def test_voice_change_dirties_only_matching_role(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)
    ids = _speech_ids(cfg, ws)

    # 改配置：Leo 换音色（不碰脚本文本）
    cfg2 = make_cfg(tmp_path, voices={
        "Lily": VoiceSetting("Cherry", "English"),
        "Leo": VoiceSetting("Cherry", "Chinese"),
    })
    states = _states(cfg2, ws)
    speakers = {sid: states[sid]["speaker"] for sid in ids}
    leo = [sid for sid in ids if speakers[sid] == "Leo"]
    lily = [sid for sid in ids if speakers[sid] == "Lily"]
    assert all(states[sid]["state"] == "dirty" for sid in leo)
    assert all(states[sid]["state"] == "clean" for sid in lily)


# —— 9. PAUSE 时长修改 → 不调 TTS，记录时长更新并需要重新合并 ——————————————

def test_pause_duration_change_does_not_call_tts(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)
    ep_dir = cfg.projects_dir / "ep1"
    sig_before = segments_signature(ep_dir, load_manifest(ep_dir))
    pause = _pause_id(cfg, ws)

    ws.set_override(pause, {"pause_seconds": 2.0})
    fake = FakeProvider()
    outcome = run_project("ep1", cfg, ws, provider=fake, persist=True)
    assert outcome.succeeded == 0 and len(fake.calls) == 0

    manifest = load_manifest(ep_dir)
    rec = next(r for r in manifest.segments if r.kind == "pause")
    assert rec.pause_seconds == 2.0
    assert segments_signature(ep_dir, manifest) != sig_before


# —— 10. 重新导出章节 → 使用最新所有 Segment 音频（按清单顺序） —————————————

def make_gui_settings(tmp_path):
    return GuiSettings(
        api_key="sk-test", provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        roles=[
            RoleVoice(role="Lily", voice="longanlingxin",
                      language_type="English"),
            RoleVoice(role="Leo", voice="longanlufeng",
                      language_type="Chinese"),
        ],
        default_role="Lily", episode_name="epgui",
        projects_dir=str(tmp_path / "projects"),
    )


def _drain(q, timeout=10.0):
    events = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            events.append(q.get(timeout=0.1))
        except queue.Empty:
            if any(e[0] == "finished" for e in events):
                break
    return events


def test_merge_after_single_regen_uses_latest_segments(tmp_path):
    merge = FakeMerge()
    q: queue.Queue = queue.Queue()
    c = GuiController(q, provider_factory=lambda cfg: FakeProvider(),
                      merge_func=merge)
    s = make_gui_settings(tmp_path)
    c.start_text(s, SCRIPT)
    c.join(15)
    _drain(q)
    assert merge.speech_order[-1] == ["Lily", "Leo", "Lily"]

    # 单句编辑 + 单段重生成 → 自动再次合并
    ws = c.ws
    rows = c.segment_rows(s, ws)
    leo = next(r for r in rows if r["speaker_name"] == "Leo")
    c.edit_segment(s, leo["seg_id"], {"text": "Leo 的新台词。"})
    c.regenerate_segments(s, [leo["seg_id"]], ws)
    c.join(15)
    _drain(q)

    assert len(merge.calls) == 2
    # 合并清单仍按脚本顺序，且该段音频文件已被新内容覆盖
    assert merge.speech_order[-1] == ["Lily", "Leo", "Lily"]
    manifest = load_manifest(Path(s.projects_dir) / "epgui")
    rec = next(r for r in manifest.segments if r.seg_id == leo["seg_id"])
    assert rec.text == "Leo 的新台词。" and rec.status == "success"
    assert (Path(s.projects_dir) / "epgui" / rec.file).is_file()


# —— 11. 旧项目兼容：无 working_script.json、旧 manifest 无 seg_id ———————————

def test_legacy_project_opens_without_regenerating(tmp_path):
    cfg = make_cfg(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    script_path = scripts / "epold.md"
    script_path.write_text(SCRIPT, encoding="utf-8")

    # CLI 旧路径：只生成 manifest（不落 working_script.json）
    from podcast_tts.pipeline import run_generate
    run_generate(script_path, cfg, provider=FakeProvider())
    ep_dir = cfg.projects_dir / "epold"
    assert not working_path(cfg.projects_dir, "epold").exists()

    # 模拟更旧的 manifest：JSON 中没有 seg_id 字段
    mpath = ep_dir / "metadata" / "manifest.json"
    import json
    data = json.loads(mpath.read_text(encoding="utf-8"))
    for rec in data["segments"]:
        rec.pop("seg_id", None)
    mpath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    assert WorkingScript.load(cfg.projects_dir, "epold") is None
    manifest = load_manifest(ep_dir)
    assert all(r.seg_id is None for r in manifest.segments)

    # 打开旧项目：从 manifest 重建工作脚本，重跑不应重新生成任何片段
    ws = WorkingScript.from_manifest("epold", manifest)
    assert ws.source_type == "legacy"
    fake = FakeProvider()
    outcome = run_project("epold", cfg, ws, provider=fake, persist=True)
    assert outcome.succeeded == 0 and len(fake.calls) == 0
    # 旧记录已补上稳定 seg_id，工作脚本也已持久化
    manifest2 = load_manifest(ep_dir)
    assert all(r.seg_id for r in manifest2.segments)
    assert working_path(cfg.projects_dir, "epold").is_file()


# —— 12. 单角色 / 多角色章节解析不被破坏 ———————————————————————————————————

def test_book_multirole_and_single_role_parse(tmp_path):
    cfg = make_cfg(tmp_path)

    ws_multi = WorkingScript(
        episode="book1", mode="book", text=BOOK_MULTI,
        source_type="file", source_name="novel.md")
    segs = parse_working(ws_multi, cfg, default_speaker_id="Lily")
    speakers = {s.speaker for s in segs if s.kind == "speech"}
    assert speakers == {"Lily", "Leo"}
    assert any(s.kind == "pause" for s in segs)

    single = "# 第一章\n\n没有角色头的正文，全部归默认角色。\n"
    ws_one = WorkingScript(episode="book2", mode="book", text=single,
                           source_type="paste")
    segs2 = parse_working(ws_one, cfg, default_speaker_id="Lily")
    assert segs2 and all(s.speaker == "Lily" for s in segs2
                        if s.kind == "speech")

    # 脚本模式显式多角色头同样正常
    ws_script = _ws_paste("book3")
    segs3 = parse_working(ws_script, cfg)
    assert {s.speaker for s in segs3 if s.kind == "speech"} == {"Lily", "Leo"}


# —— 附：覆盖层随 working_script.json 持久化 ————————————————————————————————

def test_overrides_persist_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    ws = _ws_paste()
    _generate(cfg, ws)
    target = _speech_ids(cfg, ws)[0]
    ws.set_override(target, {"text": "持久化的修改。", "rate": 0.9})
    ws.save(cfg.projects_dir)

    loaded = WorkingScript.load(cfg.projects_dir, "ep1")
    assert loaded is not None
    assert loaded.overrides[target]["text"] == "持久化的修改。"
    assert loaded.modified is True


# =====================================================================
# 重新导入 = 全新工作状态（本次修复重点）
# =====================================================================

def _reimport_file(cfg, episode: str, path: Path,
                   default_speaker_id: str | None = "Lily") -> WorkingScript:
    """模拟 GUI 重新导入：import_file（fresh=True）→ plan_and_persist。"""
    from podcast_tts.pipeline import plan_and_persist
    ws = WorkingScript.import_file(episode, path)
    plan_and_persist(episode, cfg, ws, default_speaker_id=default_speaker_id)
    return ws


# —— 1. 导入脚本 A → 生成 → 全部已生成 ——————————————————————————————————

def test_reimport_1_script_a_all_generated(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    ws = _ws_paste(text="Lily:\nA 第一句。\n\nLeo:\nA 第二句。\n")
    _generate(cfg, ws)
    states = _states(cfg, ws)
    speech = [r for r in states.values() if r["kind"] == "speech"]
    assert all(r["state"] == "clean" for r in speech)


# —— 2. 重新导入完全不同的脚本 B → 全部 未生成（不能 待更新） ———————————————

def test_reimport_2_script_b_all_missing_not_dirty(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    wsA = _ws_paste(text="Lily:\nA 第一句。\n\nLeo:\nA 第二句。\n")
    _generate(cfg, wsA)

    fb = tmp_path / "scriptB.md"
    fb.write_text("Lily:\nB 完全不同的一。\n\nLeo:\nB 完全不同的二。\n\nLily:\nB 第三句。\n",
                  encoding="utf-8")
    wsB = _reimport_file(cfg, "ep1", fb)

    states = _states(cfg, wsB)
    speech = [r for r in states.values() if r["kind"] == "speech"]
    # 重新导入的新片段只能是 missing（未生成），绝不能是 dirty（待更新）
    assert len(speech) == 3
    assert all(r["state"] == "missing" for r in speech)
    assert all(r["state"] != "dirty" for r in speech)
    assert wsB.fresh is False  # 首次规划后 fresh 已复位


# —— 3. 脚本 B 新导入后所有 Segment 都可以立即编辑 ——————————————————————————

def test_reimport_3_new_segments_editable_immediately(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    fb = tmp_path / "scriptB.md"
    fb.write_text("Lily:\nB1。\n\nLeo:\nB2。\n", encoding="utf-8")
    wsB = _reimport_file(cfg, "ep1", fb)
    target = _speech_ids(cfg, wsB)[0]

    # 未生成状态下编辑并保存（走控制器同一路径：set_override + plan_and_persist）
    wsB.set_override(target, {"text": "编辑后的 B1。"})
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, wsB)

    row = _states(cfg, wsB)[target]
    assert row["text"] == "编辑后的 B1。"
    assert row["state"] in ("missing", "dirty")  # 未生成过，仍属未生成/待更新范畴


# —— 4. 未生成 Segment 修改后仍可再次编辑，状态合理 ————————————————————————

def test_reimport_4_missing_segment_repeated_edit(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    fb = tmp_path / "scriptB.md"
    fb.write_text("Lily:\nB1。\n", encoding="utf-8")
    wsB = _reimport_file(cfg, "ep1", fb)
    target = _speech_ids(cfg, wsB)[0]
    from podcast_tts.pipeline import plan_and_persist

    wsB.set_override(target, {"text": "第一次修改。"})
    plan_and_persist("ep1", cfg, wsB)
    assert _states(cfg, wsB)[target]["text"] == "第一次修改。"

    wsB.set_override(target, {"text": "第二次修改。"})
    plan_and_persist("ep1", cfg, wsB)
    row = _states(cfg, wsB)[target]
    assert row["text"] == "第二次修改。"
    # 从未生成过 → 仍是 missing（不会变成 dirty，因为没有旧音频可过期）
    assert row["state"] == "missing"


# —— 5. 已生成 Segment：编辑 → 待更新 —————————————————————————————————————

def test_reimport_5_generated_segment_edit_becomes_dirty(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    ws = _ws_paste(text="Lily:\n原句。\n")
    _generate(cfg, ws)
    target = _speech_ids(cfg, ws)[0]
    assert _states(cfg, ws)[target]["state"] == "clean"

    ws.set_override(target, {"text": "改了。"})
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    assert _states(cfg, ws)[target]["state"] == "dirty"


# —— 6. 待更新 Segment：再次编辑仍可保存，仍待更新 ——————————————————————————

def test_reimport_6_dirty_segment_repeated_edit(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    ws = _ws_paste(text="Lily:\n原句。\n")
    _generate(cfg, ws)
    target = _speech_ids(cfg, ws)[0]
    from podcast_tts.pipeline import plan_and_persist

    ws.set_override(target, {"text": "第一次改。"})
    plan_and_persist("ep1", cfg, ws)
    assert _states(cfg, ws)[target]["state"] == "dirty"

    ws.set_override(target, {"text": "第二次改。"})
    plan_and_persist("ep1", cfg, ws)
    row = _states(cfg, ws)[target]
    assert row["state"] == "dirty"
    assert row["text"] == "第二次改。"


# —— 7. 待更新 Segment 单段重生成：只调一次 TTS，成功后变已生成 ————————————

def test_reimport_7_dirty_segment_regen_only_one_tts(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    ws = _ws_paste(text="Lily:\n第一句。\n\nLeo:\n第二句。\n\nLily:\n第三句。\n")
    _generate(cfg, ws)
    ids = _speech_ids(cfg, ws)
    target = ids[1]  # 中间一句

    ws.set_override(target, {"text": "改了第二句。"})
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    assert _states(cfg, ws)[target]["state"] == "dirty"

    fake = FakeProvider()
    run_regenerate("ep1", [], cfg, seg_ids=[target], ws=ws, provider=fake)
    assert len(fake.calls) == 1
    assert fake.calls[0].text == "改了第二句。"
    assert _states(cfg, ws)[target]["state"] == "clean"
    # 其他段未被触动
    for sid in ids:
        if sid != target:
            assert _states(cfg, ws)[sid]["state"] == "clean"


# —— 8. 重新导入后旧 Segment 临时音频正确清理 ——————————————————————————————

def test_reimport_8_old_segment_audio_cleaned(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    wsA = _ws_paste(text="Lily:\nA1。\n\nLeo:\nA2。\n")
    _generate(cfg, wsA)
    ep_dir = cfg.projects_dir / "ep1"
    seg_dir = ep_dir / "segments"
    old_files = list(seg_dir.glob("*")) if seg_dir.exists() else []
    assert len(old_files) > 0

    fb = tmp_path / "scriptB.md"
    fb.write_text("Lily:\nB1。\n", encoding="utf-8")
    _reimport_file(cfg, "ep1", fb)

    # 旧片段音频文件已清理
    if seg_dir.exists():
        remaining = list(seg_dir.glob("*"))
        assert remaining == []
    # 新 manifest 存在
    assert (ep_dir / "metadata" / "manifest.json").is_file()


# —— 9. 用户正式导出的最终音频不会被误删 ————————————————————————————————————

def test_reimport_9_final_output_preserved(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    wsA = _ws_paste(text="Lily:\nA1。\n")
    _generate(cfg, wsA)
    ep_dir = cfg.projects_dir / "ep1"

    # 模拟用户导出的最终成品
    out_dir = ep_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "chapter_final.mp3"
    final.write_bytes(b"USER-FINAL-EXPORT")

    fb = tmp_path / "scriptB.md"
    fb.write_text("Lily:\nB1。\n", encoding="utf-8")
    _reimport_file(cfg, "ep1", fb)

    assert final.is_file()
    assert final.read_bytes() == b"USER-FINAL-EXPORT"


# —— 10. 单角色/多角色/PAUSE/导演指令解析不受影响 ————————————————————————————

def test_reimport_10_parsing_behaviors_unchanged(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")

    # 多角色 + PAUSE + 行内导演指令
    text = "Lily(温柔地):\n你好。\n\n[PAUSE:1.2]\n\nLeo(兴奋地):\n嗨！\n"
    fb = tmp_path / "multi.md"
    fb.write_text(text, encoding="utf-8")
    ws = _reimport_file(cfg, "ep1", fb)
    rows = segment_states("ep1", cfg, ws)
    kinds = [(r["kind"], r["speaker_name"]) for r in rows]
    assert kinds[0] == ("speech", "Lily")
    assert kinds[1][0] == "pause"  # 停顿段 speaker_name 为 None，GUI 显示"停顿"
    assert kinds[2] == ("speech", "Leo")
    # 未导入过 → 全部未生成（台词 missing，pause 状态为 pause）
    speech = [r for r in rows if r["kind"] == "speech"]
    assert all(r["state"] == "missing" for r in speech)

    # 单角色默认角色
    single = tmp_path / "single.md"
    single.write_text("# 单章\n\n没有角色头的正文。\n", encoding="utf-8")
    ws2 = _reimport_file(cfg, "ep2", single, default_speaker_id="Lily")
    rows2 = segment_states("ep2", cfg, ws2, default_speaker_id="Lily")
    speech2 = [r for r in rows2 if r["kind"] == "speech"]
    assert speech2 and all(r["speaker"] == "Lily" for r in speech2)


# =====================================================================
# 片段编辑持久化：局部 direction/text/role/rate 修改必须真正成为生成依据
# =====================================================================

def _ws_with_direction(tmp_path, direction: str = "轻快地") -> tuple:
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    ws = WorkingScript(
        episode="ep1", mode="script", source_type="paste",
        text=f"Lily({direction}): [excited] 你好。\n",
    )
    return cfg, ws


# —— 1/2. 清空 direction 后，预览与 TTS request 都不再出现旧值 ————————————

def test_segment_clear_direction_not_restored(tmp_path):
    cfg, ws = _ws_with_direction(tmp_path)
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    rows = segment_states("ep1", cfg, ws)
    assert rows[0]["direction"] == "轻快地"
    sid = rows[0]["seg_id"]

    # 模拟片段管理：清空 direction
    ws.set_override(sid, {"direction": "", "text": "你好。"})
    ws.save(cfg.projects_dir)
    plan_and_persist("ep1", cfg, ws)

    rows2 = segment_states("ep1", cfg, ws)
    assert rows2[0]["direction"] is None  # 局部指令已清除

    # 真正 TTS request 不含旧 direction
    fake = FakeProvider()
    run_project("ep1", cfg, ws, provider=fake)
    assert fake.calls[0].direction is None
    assert "轻快地" not in (fake.calls[0].direction or "")


# —— 3. 修改 direction 为新值后，预览/TTS 使用新值 ——————————————————————————

def test_segment_change_direction_uses_new_value(tmp_path):
    cfg, ws = _ws_with_direction(tmp_path)
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    sid = segment_states("ep1", cfg, ws)[0]["seg_id"]

    ws.set_override(sid, {"direction": "压低声音，略带犹豫", "text": "你好。"})
    ws.save(cfg.projects_dir)
    plan_and_persist("ep1", cfg, ws)

    fake = FakeProvider()
    run_project("ep1", cfg, ws, provider=fake)
    assert fake.calls[0].direction == "压低声音，略带犹豫"
    assert "轻快地" not in (fake.calls[0].direction or "")


# —— 4. 编辑后刷新片段列表，修改仍然存在 ————————————————————————————————————

def test_segment_edit_persists_after_reload(tmp_path):
    cfg, ws = _ws_with_direction(tmp_path)
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    sid = segment_states("ep1", cfg, ws)[0]["seg_id"]

    ws.set_override(sid, {"direction": "新指令", "text": "改过的正文"})
    ws.save(cfg.projects_dir)

    # 模拟关闭窗口后重新打开：重新 load ws + plan
    ws2 = WorkingScript.load(cfg.projects_dir, "ep1")
    assert ws2 is not None
    plan_and_persist("ep1", cfg, ws2)
    rows = segment_states("ep1", cfg, ws2)
    by_id = {r["seg_id"]: r for r in rows}
    assert by_id[sid]["direction"] == "新指令"
    assert by_id[sid]["text"] == "改过的正文"


# —— 5. 关闭再重新打开片段管理，修改持久存在 —————————————————————————————————

def test_segment_edit_survives_window_reopen(tmp_path):
    cfg, ws = _ws_with_direction(tmp_path)
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    sid = segment_states("ep1", cfg, ws)[0]["seg_id"]

    ws.set_override(sid, {"direction": "", "rate": 0.8})
    ws.save(cfg.projects_dir)

    # 重新从磁盘加载 working_script.json
    ws_reloaded = WorkingScript.load(cfg.projects_dir, "ep1")
    assert ws_reloaded is not None
    assert ws_reloaded.overrides[sid]["direction"] == ""
    assert ws_reloaded.overrides[sid]["rate"] == 0.8


# —— 6. 修改局部 direction 不影响项目默认/角色默认指令 ——————————————————————

def test_segment_edit_direction_keeps_defaults(tmp_path):
    cfg = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    # 项目级默认指令
    cfg.default_instruction = "项目全局指令"
    ws = WorkingScript(
        episode="ep1", mode="script", source_type="paste",
        text="Lily(轻快地): 你好。\n",
    )
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    sid = segment_states("ep1", cfg, ws)[0]["seg_id"]

    # 只清除局部 direction，项目默认应保留
    ws.set_override(sid, {"direction": ""})
    ws.save(cfg.projects_dir)
    plan_and_persist("ep1", cfg, ws)

    fake = FakeProvider()
    run_project("ep1", cfg, ws, provider=fake)
    # 最终 direction 是合并后的；局部被清除但项目默认仍在
    assert "轻快地" not in (fake.calls[0].direction or "")
    assert "项目全局指令" in (fake.calls[0].direction or "")


# —— 7. text/role/rate 等字段同样不会因重新 parse 恢复 ——————————————————————

def test_segment_edit_text_role_rate_not_restored(tmp_path):
    cfg, ws = _ws_with_direction(tmp_path)
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    sid = segment_states("ep1", cfg, ws)[0]["seg_id"]

    # 同时改 text / role / rate
    ws.set_override(sid, {
        "text": "完全不同的句子",
        "speaker": "Leo",
        "rate": 1.3,
    })
    ws.save(cfg.projects_dir)
    plan_and_persist("ep1", cfg, ws)

    fake = FakeProvider()
    run_project("ep1", cfg, ws, provider=fake)
    assert fake.calls[0].text == "完全不同的句子"
    assert fake.calls[0].voice == cfg.voices["Leo"].voice
    assert fake.calls[0].audio_params.get("rate") == 1.3
    # direction 未被修改 → 保持脚本原值（不会因改其他字段而丢失）
    assert fake.calls[0].direction == "轻快地"


# —— 8. 清空 rate 覆盖后回退默认，不保留旧值 ————————————————————————————————

def test_segment_clear_rate_falls_back_to_default(tmp_path):
    cfg, ws = _ws_with_direction(tmp_path)
    from podcast_tts.pipeline import plan_and_persist
    plan_and_persist("ep1", cfg, ws)
    sid = segment_states("ep1", cfg, ws)[0]["seg_id"]

    # 先设 rate=0.9
    ws.set_override(sid, {"rate": 0.9})
    ws.save(cfg.projects_dir)
    plan_and_persist("ep1", cfg, ws)
    fake1 = FakeProvider()
    run_project("ep1", cfg, ws, provider=fake1)
    assert fake1.calls[0].audio_params.get("rate") == 0.9

    # 再清除 rate（None）
    ws.set_override(sid, {"rate": None})
    ws.save(cfg.projects_dir)
    plan_and_persist("ep1", cfg, ws)
    fake2 = FakeProvider()
    run_project("ep1", cfg, ws, provider=fake2)
    # 回退到项目默认 rate（make_cfg 默认 None → provider 默认 1.0）
    assert fake2.calls[0].audio_params.get("rate") is None or \
        fake2.calls[0].audio_params.get("rate") == 1.0
