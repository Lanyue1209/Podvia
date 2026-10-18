"""Segment 级 SSML（文本模式）测试：每个片段独立决定 plain / ssml。

全程 FakeProvider（不触网）。对应需求（需求十八）：
 1. 普通台词 → plain；          2. 单行 <speak> → ssml；
 3. 跨行 SSML 块只解析为一个 Segment；4. plain/ssml/plain 三段互不影响；
 5. SSML 格式错误明确报错、绝不静默按 plain 发送；
 6. <speak rate> 冲突时去掉该片段 HTTP rate（仅该片段）；
 7. break-only SSML 保留 HTTP rate； 8. plain→ssml content_hash 变化 → 重新生成；
 9. SSML 内 500ms→900ms 仅该片段 dirty；10. 局部重生成只调用该片段 TTS；
11. PAUSE+plain+SSML+plain+PAUSE 全链路（生成+合并顺序正确）；
12. 旧项目 manifest 无 text_mode 字段仍可正常加载、缓存命中。
另加两个强制模式测试：覆盖层 text_mode 强制 plain / 强制 ssml。
"""
from __future__ import annotations

import json
from pathlib import Path

from podcast_tts.audio import AudioFormat, run_merge
from podcast_tts.config import VoiceSetting
from podcast_tts.manifest import load as load_manifest, manifest_path
from podcast_tts.parser import ScriptParser
from podcast_tts.pipeline import (
    inspect_segments,
    plan_segment,
    run_generate,
    run_regenerate,
)
from podcast_tts.project import WorkingScript
from podcast_tts.ssml import resolve_text_mode, validate_ssml

from test_pipeline import FakeProvider, make_cfg, write_script


def ssml_cfg(tmp_path: Path, **voice_over):
    """qwen_audio + 已核实支持 SSML 的模型；Lily/Leo 默认中文。"""
    defaults = {"Lily": VoiceSetting("Cherry", "Chinese"),
                "Leo": VoiceSetting("Ethan", "Chinese")}
    defaults.update(voice_over)
    return make_cfg(
        tmp_path,
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        voices=defaults,
    )


# —— 测试 1：普通台词 → plain ————————————————————————————————————————————


def test_plain_text_is_plain(tmp_path):
    cfg = ssml_cfg(tmp_path)
    segs = ScriptParser(set(cfg.voices)).parse("Lily: 你好。\n")
    assert len(segs) == 1
    assert resolve_text_mode(segs[0].text, segs[0].text_mode) == "plain"
    _setting, _inst, extras = plan_segment(cfg, segs[0])
    assert "enable_ssml" not in extras


# —— 测试 2：单行 <speak> → ssml ———————————————————————————————————————————


def test_single_line_speak_is_ssml(tmp_path):
    cfg = ssml_cfg(tmp_path)
    segs = ScriptParser(set(cfg.voices)).parse(
        "Leo: <speak>你好。</speak>\n")
    assert len(segs) == 1
    assert resolve_text_mode(segs[0].text, segs[0].text_mode) == "ssml"
    _setting, _inst, extras = plan_segment(cfg, segs[0])
    assert extras.get("enable_ssml") is True


# —— 测试 3：跨行 SSML 块只解析为一个 Segment ———————————————————————————————


def test_multiline_ssml_block_is_single_segment(tmp_path):
    cfg = ssml_cfg(tmp_path)
    script = (
        "Leo:\n"
        "<speak>\n"
        "你好。\n"
        '<break time="500ms"/>\n'
        "我来了。\n"
        "</speak>\n"
        "\n"
        "Lily: 收到。\n"
    )
    segs = ScriptParser(set(cfg.voices)).parse(script)
    assert len(segs) == 2  # Leo 的 SSML 整块 1 段 + Lily 1 段
    leo = segs[0]
    assert leo.speaker == "Leo" and leo.kind == "speech"
    # 标签原文保留：开/闭标签与 break 都在最终发送文本里
    assert leo.text.count("<speak") == 1 and "</speak>" in leo.text
    assert '<break time="500ms"/>' in leo.text
    assert "我来了" in leo.text
    assert resolve_text_mode(leo.text, leo.text_mode) == "ssml"
    assert validate_ssml(leo.text) is not None  # 合并后仍是合法 SSML 文档
    _setting, _inst, extras = plan_segment(cfg, leo)
    assert extras.get("enable_ssml") is True


# —— 测试 4：plain / ssml / plain 三段互不影响 ———————————————————————————————


def test_plain_ssml_plain_independent(tmp_path):
    cfg = ssml_cfg(tmp_path)
    script = "Lily: 你好。\n\nLeo: <speak>我在。</speak>\n\nLily: 再见。\n"
    segs = ScriptParser(set(cfg.voices)).parse(script)
    assert len(segs) == 3
    flags = [("enable_ssml" in plan_segment(cfg, s)[2]) for s in segs]
    assert flags == [False, True, False]


# —— 测试 5：SSML 格式错误明确报错，绝不静默按 plain 发送 ————————————————————


def test_malformed_ssml_fails_loudly(tmp_path):
    cfg = ssml_cfg(tmp_path)
    provider = FakeProvider()
    script = write_script(tmp_path, "Leo:\n<speak>\n你好\n")
    outcome = run_generate(script, cfg, provider=provider)

    assert provider.calls == []          # 没有任何请求发出（不是按 plain 发送）
    assert outcome.failed == 1
    rec = outcome.manifest.record(1)
    assert rec.status == "failed"
    assert "SSML 格式错误" in (rec.error or "")

    # 预览同样提前提示（不触网）
    ws = WorkingScript.from_paste("ep1", "Leo:\n<speak>\n你好\n")
    rows, _manifest = inspect_segments("ep1", cfg, ws)
    assert rows[0]["ssml_error"]
    assert "SSML 格式错误" in rows[0]["error"]
    assert rows[0]["mode_zh"] == "SSML"


# —— 测试 6：<speak rate> 冲突时去掉该片段的 HTTP rate（仅该片段）————————————


def test_speak_rate_attribute_removes_http_rate_for_that_segment(tmp_path):
    cfg = ssml_cfg(
        tmp_path,
        Lily=VoiceSetting("Cherry", "Chinese", rate=1.1),
        Leo=VoiceSetting("Ethan", "Chinese", rate=0.9),
    )
    script = 'Lily: 你好。\n\nLeo: <speak rate="0.8">我在。</speak>\n'
    provider = FakeProvider()
    outcome = run_generate(write_script(tmp_path, script), cfg, provider=provider)
    assert outcome.succeeded == 2
    lily, leo = provider.calls
    assert lily.audio_params.get("rate") == 1.1
    assert "enable_ssml" not in lily.audio_params
    assert leo.audio_params.get("enable_ssml") is True
    assert "rate" not in leo.audio_params  # SSML 已控制 rate → 不重复发送


# —— 测试 7：break-only SSML 保留 HTTP rate —————————————————————————————————


def test_break_only_ssml_keeps_http_rate(tmp_path):
    cfg = ssml_cfg(tmp_path, Leo=VoiceSetting("Ethan", "Chinese", rate=0.9))
    script = 'Leo: <speak>嗯。<break time="500ms"/>好。</speak>\n'
    provider = FakeProvider()
    outcome = run_generate(write_script(tmp_path, script), cfg, provider=provider)
    assert outcome.succeeded == 1
    req = provider.calls[0]
    assert req.audio_params.get("enable_ssml") is True
    assert req.audio_params.get("rate") == 0.9  # 无 rate 属性 → 参数保留


# —— 测试 8：plain → ssml，content_hash 改变 → 重新生成（待更新）——————————————


def test_plain_to_ssml_changes_hash_and_regenerates(tmp_path):
    cfg = ssml_cfg(tmp_path)
    script = write_script(tmp_path, "Leo:\n我来晚了。\n")
    first = run_generate(script, cfg, provider=FakeProvider())
    assert first.succeeded == 1
    old_hash = first.manifest.record(1).content_hash

    script.write_text("Leo:\n<speak>我来晚了。</speak>\n", encoding="utf-8")
    provider2 = FakeProvider()
    second = run_generate(script, cfg, provider=provider2)

    assert len(provider2.calls) == 1     # 朗读文字相同但模式变了 → 必须重新生成
    assert provider2.calls[0].audio_params.get("enable_ssml") is True
    rec = second.manifest.record(1)
    assert rec.status == "success" and rec.content_hash != old_hash


# —— 测试 9：SSML 内 500ms → 900ms，仅该片段 dirty ——————————————————————————


def test_break_change_only_dirties_that_segment(tmp_path):
    cfg = ssml_cfg(tmp_path)
    script = write_script(
        tmp_path,
        'Lily: 你好。\n\nLeo: <speak>嗯。<break time="500ms"/>好。</speak>'
        "\n\nLily: 再见。\n",
    )
    first = run_generate(script, cfg, provider=FakeProvider())
    assert first.succeeded == 3

    script.write_text(
        'Lily: 你好。\n\nLeo: <speak>嗯。<break time="900ms"/>好。</speak>'
        "\n\nLily: 再见。\n",
        encoding="utf-8",
    )
    provider2 = FakeProvider()
    second = run_generate(script, cfg, provider=provider2)

    assert len(provider2.calls) == 1
    assert "900ms" in provider2.calls[0].text
    assert second.succeeded == 1 and second.skipped == 2


# —— 测试 10：局部重生成 SSML 片段，只调用该片段 TTS ————————————————————————


def test_regenerate_single_ssml_segment(tmp_path):
    cfg = ssml_cfg(tmp_path)
    script = (
        'Lily: 你好。\n\nLeo: <speak>嗯。<break time="500ms"/>好。</speak>'
        "\n\nLily: 再见。\n"
    )
    ws = WorkingScript.from_paste("ep1", script)
    first = run_regenerate("ep1", [], cfg, ws=ws, provider=FakeProvider())
    assert first.succeeded == 3
    sid = first.manifest.record(2).seg_id

    ws.text = ws.text.replace("500ms", "900ms")
    provider2 = FakeProvider()
    second = run_regenerate("ep1", [], cfg, seg_ids={sid}, ws=ws, provider=provider2)

    assert len(provider2.calls) == 1                      # 只调用该片段
    assert "900ms" in provider2.calls[0].text
    assert provider2.calls[0].audio_params.get("enable_ssml") is True
    assert second.manifest.record(2).status == "success"
    assert second.manifest.record(1).status == "success"  # 其他片段不受影响


# —— 测试 11：PAUSE + plain + SSML + plain + PAUSE 全链路 ———————————————————


CHAIN_SCRIPT = """[PAUSE:1.0]

Narrator: 夜色渐渐笼罩城市。

Lily(紧张): 谁在那里？

Leo(低沉):
<speak>
别害怕。
<break time="700ms"/>
是我。
</speak>

Lily(松了一口气): 原来是你。

[PAUSE:2.0]
"""


def test_full_chain_pause_plain_ssml_plain_pause(tmp_path, monkeypatch):
    cfg = ssml_cfg(tmp_path, Narrator=VoiceSetting("Cherry", "Chinese"))
    provider = FakeProvider()
    outcome = run_generate(write_script(tmp_path, CHAIN_SCRIPT), cfg,
                           provider=provider)

    # 结构：001 PAUSE / 002 Narrator / 003 Lily / 004 Leo(ssml) / 005 Lily / 006 PAUSE
    recs = outcome.manifest.segments
    assert [r.kind for r in recs] == [
        "pause", "speech", "speech", "speech", "speech", "pause"]
    assert [r.index for r in recs] == [1, 2, 3, 4, 5, 6]

    # 只发起 4 个 TTS 请求；enable_ssml 仅出现在 Leo（第 4 段）上
    assert len(provider.calls) == 4
    flags = [bool(r.audio_params.get("enable_ssml")) for r in provider.calls]
    assert flags == [False, False, True, False]
    assert "别害怕" in provider.calls[2].text and "是我" in provider.calls[2].text

    # 合并：静音 2 处（1s / 2s），拼接清单严格按 manifest 顺序
    class _Rec:
        def __init__(self):
            self.commands = []

        def __call__(self, cmd):
            self.commands.append(list(cmd))
            joined = " ".join(cmd)
            if "anullsrc" in joined:
                Path(cmd[-1]).write_bytes(b"SILENCE")
            elif "-f" in cmd and "concat" in cmd:
                Path(cmd[-1]).write_bytes(b"MP3DATA")

    rec = _Rec()
    monkeypatch.setattr("podcast_tts.audio._run_command", rec)
    monkeypatch.setattr(
        "podcast_tts.audio.probe",
        lambda ffprobe, file: AudioFormat("pcm_s16le", 24000, 1))
    monkeypatch.setattr(
        "podcast_tts.audio.probe_duration", lambda ffprobe, file: 3.0)

    mout = run_merge("episode001", cfg)
    assert mout.skipped is False and mout.output == "output/episode001.mp3"
    silences = [c for c in rec.commands if "anullsrc" in " ".join(c)]
    assert [c[c.index("-t") + 1] for c in silences] == ["1.000", "2.000"]
    concats = [c for c in rec.commands if "-f" in c and "concat" in c]
    assert len(concats) == 1
    concat_cmd = concats[0]
    list_file = Path(concat_cmd[concat_cmd.index("-i") + 1])
    lines = list_file.read_text(encoding="utf-8").strip().splitlines()
    names = [Path(line[6:-1]).name for line in lines]
    assert names == [
        "0001_PAUSE_1s.wav", "0002_Narrator.wav", "0003_Lily.wav",
        "0004_Leo.wav", "0005_Lily.wav", "0006_PAUSE_2s.wav",
    ]


# —— 测试 12：旧项目 manifest 无 text_mode 字段仍可正常加载 —————————————————


def test_legacy_manifest_without_text_mode_loads(tmp_path):
    cfg = ssml_cfg(tmp_path)
    script = write_script(tmp_path, "Lily:\n你好。\n")
    first = run_generate(script, cfg, provider=FakeProvider())
    assert first.succeeded == 1

    # 模拟旧版 manifest：删除 text_mode 字段后重新加载
    mp = manifest_path(cfg.projects_dir / "episode001")
    raw = json.loads(mp.read_text(encoding="utf-8"))
    for seg in raw["segments"]:
        seg.pop("text_mode", None)
    mp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    m = load_manifest(cfg.projects_dir / "episode001")
    assert m is not None
    assert all(r.text_mode is None for r in m.segments)

    # 旧记录继续有效：同一脚本缓存命中，零 API 调用
    provider2 = FakeProvider()
    second = run_generate(script, cfg, provider=provider2)
    assert provider2.calls == [] and second.skipped == 1


# —— 附加：覆盖层强制 text_mode ————————————————————————————————————————


def test_force_plain_via_override_sends_speak_doc_as_plain(tmp_path):
    cfg = ssml_cfg(tmp_path)
    ws = WorkingScript.from_paste("ep1", "Leo: <speak>我在。</speak>\n")
    _segs, manifest = _persist(cfg, ws)
    ws.set_override(manifest.record(1).seg_id, {"text_mode": "plain"})

    provider = FakeProvider()
    outcome = run_regenerate("ep1", [], cfg, ws=ws, provider=provider)
    assert outcome.succeeded == 1
    req = provider.calls[0]
    assert req.text == "<speak>我在。</speak>"      # 原文按普通文本发送
    assert "enable_ssml" not in req.audio_params    # 不做 SSML 校验/包装


def test_force_ssml_on_plain_text_fails_that_segment_only(tmp_path):
    cfg = ssml_cfg(tmp_path)
    ws = WorkingScript.from_paste("ep1", "Lily: 你好。\n\nLeo: 我在。\n")
    _segs, manifest = _persist(cfg, ws)
    ws.set_override(manifest.record(2).seg_id, {"text_mode": "ssml"})

    provider = FakeProvider()
    outcome = run_regenerate("ep1", [], cfg, ws=ws, provider=provider)
    assert len(provider.calls) == 1          # 只有 Lily 正常发出
    assert provider.calls[0].text == "你好。"
    rec2 = outcome.manifest.record(2)
    assert rec2.status == "failed" and "SSML 格式错误" in (rec2.error or "")
    assert outcome.manifest.record(1).status == "success"


def _persist(cfg, ws):
    from podcast_tts.pipeline import plan_and_persist
    return plan_and_persist("ep1", cfg, ws)
