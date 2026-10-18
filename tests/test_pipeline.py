"""Phase 5/6 单元测试：批量生成 Pipeline 与 Manifest（文档第 5.3 / 11 / 14 / 15 / 17 节）。

全程使用 FakeProvider（不真实调用网络）。覆盖：
- manifest 初始化（PAUSE 直接 success）与原子写 roundtrip
- 全量生成成功：manifest 更新、文件落盘、attempts/request_id/content_hash
- 断点续生成：二次运行跳过缓存；文本/voice 变化或文件被删 → 重新生成
- 失败片段记录（Fatal 不重试）→ 继续其他片段 → exit_code 4
- Retryable 重试后成功（attempts 累计）；force 全量重生成；脚本变更 WARNING
- run_retry（Phase 6）：只拾取 failed/pending、成功片段零调用、manifest 缺失报错、
  角色缺配置报错、再次失败记录 error 并 exit_code 4
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from podcast_tts.config import AppConfig, ConfigError, QwenAudioParams, VoiceSetting
from podcast_tts.manifest import (
    Manifest,
    content_hash,
    load as load_manifest,
    load_or_init,
    manifest_path,
    save,
)
from podcast_tts.models import Segment, TTSResult
from podcast_tts.parser import ScriptParser
from podcast_tts.pipeline import run_generate, run_retry
from podcast_tts.providers.base import FatalProviderError, RetryableError, TTSProvider

SCRIPT = """Lily(明亮地):
Hello there!

[PAUSE:0.5]

Leo:
嗨，我是 Leo。
"""

SINGLE_SPEECH = "Lily:\nHello there!\n"


def make_cfg(tmp_path: Path, **overrides) -> AppConfig:
    defaults = dict(
        provider="qwen",
        model="qwen3-tts-instruct-flash",
        voices={
            "Lily": VoiceSetting("Cherry", "English"),
            "Leo": VoiceSetting("Ethan", "Chinese"),
        },
        max_concurrency=2,
        request_timeout=5,
        max_retries=3,
        retry_base_delay=0.01,
        optimize_instructions=True,
        qwen_base_url="https://fake.example/api",
        qwen_audio_base_url="https://fake.example/audio-tts",
        audio_params=QwenAudioParams(),
        merge_format="mp3",
        bitrate="192k",
        scripts_dir=tmp_path / "scripts",
        projects_dir=tmp_path / "projects",
        logs_dir=tmp_path / "logs",
        ffmpeg_dir=tmp_path / "tools",
        api_key="sk-test",
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


def write_script(tmp_path: Path, text: str = SCRIPT) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / "episode001.md"
    path.write_text(text, encoding="utf-8")
    return path


class FakeProvider(TTSProvider):
    """按脚本表返回结果或异常；成功时模拟落盘。"""

    name = "fake"

    def __init__(self, script=None):
        self._script = list(script or [])
        self.calls: list = []

    def generate(self, request):
        self.calls.append(request)
        if self._script:
            action = self._script.pop(0)
            if isinstance(action, Exception):
                raise action
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKEWAVDATA")
        return TTSResult(
            output_path=request.output_path,
            request_id=f"req-{len(self.calls):03d}",
            provider_data={"usage": {"characters": len(request.text)}},
        )


# —— Manifest 基础 ———————————————————————————————————————————————————


def test_manifest_init_pause_success_speech_pending(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    segments = ScriptParser(set(cfg.voices)).parse(script.read_text(encoding="utf-8"))
    episode_dir = cfg.projects_dir / "episode001"

    manifest = load_or_init(episode_dir, script, cfg, segments)
    save(episode_dir, manifest)

    assert [r.status for r in manifest.segments] == ["pending", "success", "pending"]
    assert manifest.segments[1].kind == "pause"
    assert manifest.segments[1].pause_seconds == 0.5
    assert manifest.segments[1].file == "segments/0002_PAUSE_0.5s.wav"
    assert manifest.voices == {"Lily": "Cherry", "Leo": "Ethan"}
    # script_file 记录相对项目根的路径；测试临时目录在项目外，只校验文件名部分
    assert Path(manifest.script_file).name == "episode001.md"

    # roundtrip：重新加载内容一致，无 .tmp 残留
    reloaded = load_manifest(episode_dir)
    assert reloaded is not None
    assert [r.index for r in reloaded.segments] == [1, 2, 3]
    assert reloaded.script_hash == manifest.script_hash
    assert not list(episode_dir.glob("*.tmp"))
    raw = json.loads(manifest_path(episode_dir).read_text(encoding="utf-8"))
    assert raw["episode"] == "episode001"


def test_manifest_load_missing_returns_none(tmp_path):
    assert load_manifest(tmp_path / "nope") is None


def test_manifest_script_change_warning(tmp_path, caplog):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    segments = ScriptParser(set(cfg.voices)).parse(script.read_text(encoding="utf-8"))
    episode_dir = cfg.projects_dir / "episode001"
    save(episode_dir, load_or_init(episode_dir, script, cfg, segments))

    first = load_manifest(episode_dir)
    old_ids = {r.seg_id for r in first.segments}

    script.write_text(SCRIPT + "\nLily:\n新加一句。\n", encoding="utf-8")
    new_segments = ScriptParser(set(cfg.voices)).parse(script.read_text(encoding="utf-8"))
    with caplog.at_level(logging.WARNING, logger="podcast_tts"):
        manifest = load_or_init(episode_dir, script, cfg, new_segments)
    # 新机制：脚本变化时按片段内容 LCS 对齐，只提示"工作脚本已变化"
    assert any("工作脚本与上次生成时相比已变化" in r.message for r in caplog.records)
    # 插入新句后：旧 3 段 seg_id 全部沿用，新增 1 段（pending）
    assert len(manifest.segments) == 4
    assert all(r.seg_id in old_ids for r in manifest.segments[:3])
    new_ones = [r for r in manifest.segments if r.seg_id not in old_ids]
    assert len(new_ones) == 1 and new_ones[0].status == "pending" \
        and new_ones[0].text == "新加一句。"


# —— Pipeline：全量成功 / 断点续生成 —————————————————————————————————————


def test_full_success_flow(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    provider = FakeProvider()

    outcome = run_generate(script, cfg, provider=provider)

    assert (outcome.succeeded, outcome.failed, outcome.skipped) == (2, 0, 0)
    assert outcome.exit_code == 0

    episode_dir = cfg.projects_dir / "episode001"
    rec1 = outcome.manifest.record(1)
    rec3 = outcome.manifest.record(3)
    assert rec1.status == "success" and rec1.speaker == "Lily"
    assert rec1.file == "segments/0001_Lily.wav"
    # 并发完成顺序不定，request_id 只能断言形式（req-001/002 分配与时序相关）
    assert rec1.request_id is not None and rec1.request_id.startswith("req-")
    assert rec1.attempts == 1
    assert rec1.error is None
    assert (episode_dir / rec1.file).read_bytes() == b"FAKEWAVDATA"

    # content_hash 与公式一致（v2 含 audio_params 快照）
    setting = cfg.voices["Lily"]
    assert rec1.content_hash == content_hash(
        cfg, rec1.text, setting.voice, setting.language_type, rec1.direction,
        rec1.audio_params,
    )
    assert rec1.audio_params == {}  # provider=qwen 不携带 qwen_audio 参数
    assert rec3.speaker == "Leo" and rec3.status == "success"


def test_resume_skips_cached(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)

    first = run_generate(script, cfg, provider=FakeProvider())
    assert first.succeeded == 2

    provider2 = FakeProvider()
    second = run_generate(script, cfg, provider=provider2)
    assert second.succeeded == 0 and second.skipped == 2
    assert provider2.calls == []  # 全部缓存命中，零 API 调用


def test_text_change_regenerates(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    run_generate(script, cfg, provider=FakeProvider())

    changed = SCRIPT.replace("Hello there!", "Hello again!")
    script.write_text(changed, encoding="utf-8")
    provider2 = FakeProvider()
    outcome = run_generate(script, cfg, provider=provider2)

    assert len(provider2.calls) == 1
    assert provider2.calls[0].text == "Hello again!"
    assert outcome.skipped == 1  # Leo 未变，仍命中缓存
    assert outcome.manifest.record(1).text == "Hello again!"


def test_voice_change_regenerates(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    run_generate(script, cfg, provider=FakeProvider())

    cfg.voices["Lily"] = VoiceSetting("Serena", "English")  # 换音色
    provider2 = FakeProvider()
    outcome = run_generate(script, cfg, provider=provider2)

    assert len(provider2.calls) == 1
    assert provider2.calls[0].voice == "Serena"
    assert outcome.skipped == 1


def test_deleted_file_regenerates(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    run_generate(script, cfg, provider=FakeProvider())

    (cfg.projects_dir / "episode001" / "segments" / "0001_Lily.wav").unlink()
    provider2 = FakeProvider()
    outcome = run_generate(script, cfg, provider=provider2)

    assert len(provider2.calls) == 1  # 文件丢失 → 重新生成
    assert outcome.skipped == 1


def test_force_regenerates_all(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    run_generate(script, cfg, provider=FakeProvider())

    provider2 = FakeProvider()
    outcome = run_generate(script, cfg, provider=provider2, force=True)
    assert len(provider2.calls) == 2
    assert outcome.skipped == 0


# —— Pipeline：失败与重试 ———————————————————————————————————————————


def test_fatal_failure_recorded_and_others_continue(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    # 第 1 次调用（Lily）Fatal；Leo 正常
    provider = FakeProvider(script=[FatalProviderError("bad key")])

    outcome = run_generate(script, cfg, provider=provider)

    assert (outcome.succeeded, outcome.failed) == (1, 1)
    assert outcome.exit_code == 4
    rec1 = outcome.manifest.record(1)
    assert rec1.status == "failed"
    assert "bad key" in rec1.error
    assert rec1.attempts == 1  # Fatal 不重试
    assert rec1.content_hash is None
    # 失败片段没有落盘文件
    assert not (cfg.projects_dir / "episode001" / rec1.file).exists()
    assert outcome.manifest.record(3).status == "success"


def test_retryable_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # 测试中不真等
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path, SINGLE_SPEECH)  # 单片段，避免并发干扰重试脚本
    provider = FakeProvider(
        script=[RetryableError("http 429"), RetryableError("http 429")]
    )

    outcome = run_generate(script, cfg, provider=provider)

    assert outcome.failed == 0
    rec1 = outcome.manifest.record(1)
    assert rec1.status == "success"
    assert rec1.attempts == 3  # 首次 + 2 次重试


def test_retryable_exhausted_marks_failed(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = make_cfg(tmp_path, max_retries=2)
    script = write_script(tmp_path, SINGLE_SPEECH)
    provider = FakeProvider(
        script=[RetryableError("down"), RetryableError("down"), RetryableError("down")]
    )

    outcome = run_generate(script, cfg, provider=provider)

    rec1 = outcome.manifest.record(1)
    assert rec1.status == "failed"
    assert rec1.attempts == 3  # 首次 + 2 次重试
    assert "down" in rec1.error


def test_overlong_text_is_safely_split_before_api(tmp_path, caplog):
    """700 字无停点文本：生成前安全切分（480 硬切 + 220），不等 API 报 600 错误。"""
    cfg = make_cfg(tmp_path)
    long_line = "x" * 700
    script = write_script(tmp_path, f"Lily:\n{long_line}\n")
    provider = FakeProvider()

    with caplog.at_level(logging.WARNING, logger="podcast_tts"):
        run_generate(script, cfg, provider=provider)

    # 切成 2 个请求，每段都低于 600 上限（安全线约 480），内容不丢
    assert len(provider.calls) == 2
    lengths = sorted(len(req.text) for req in provider.calls)
    assert all(n <= 600 for n in lengths)
    assert sum(lengths) == 700
    # 无句读停点的硬切要如实 WARNING；不应再出现"超过百炼上限"的运行期警告
    assert any("硬切" in r.message for r in caplog.records)
    assert not any("超过百炼实测上限" in r.message for r in caplog.records)
    assert outcome_all_success(cfg, "episode001")


def outcome_all_success(cfg: AppConfig, episode: str) -> bool:
    manifest = load_manifest(cfg.projects_dir / episode)
    assert manifest is not None
    assert all(r.status == "success" for r in manifest.segments)
    return True


def test_concurrency_limited(tmp_path):
    """max_concurrency 生效：并发内同时处理的请求数不超过上限。"""
    import threading

    cfg = make_cfg(tmp_path, max_concurrency=1)
    script = write_script(tmp_path)

    class ConcurrencyProbe(FakeProvider):
        def __init__(self):
            super().__init__()
            self.current = 0
            self.max_seen = 0
            self._lock = threading.Lock()

        def generate(self, request):
            with self._lock:
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)
            try:
                return super().generate(request)
            finally:
                with self._lock:
                    self.current -= 1

    probe = ConcurrencyProbe()
    run_generate(script, cfg, provider=probe)
    assert probe.max_seen == 1


def test_manifest_updated_per_segment(tmp_path):
    """每个片段完成后 manifest 落盘一次（中途可读取到已完成片段）。"""
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    episode_dir = cfg.projects_dir / "episode001"

    class SlowThenFast(FakeProvider):
        def generate(self, request):
            result = super().generate(request)
            # 第一个请求完成后即可从磁盘读到含 success 的 manifest
            snapshot = load_manifest(episode_dir)
            assert snapshot is not None
            done = [r for r in snapshot.segments if r.status == "success" and r.kind == "speech"]
            assert len(done) >= 1
            return result

    run_generate(script, cfg, provider=SlowThenFast())


def test_manifest_syncs_provider_on_config_change(tmp_path):
    """provider/model 变化时 manifest 头部同步为当前配置（status 显示正确）。"""
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    run_generate(script, cfg, provider=FakeProvider())

    cfg2 = make_cfg(tmp_path, provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    episode_dir = cfg.projects_dir / "episode001"
    segments = ScriptParser(set(cfg2.voices)).parse(script.read_text(encoding="utf-8"))
    manifest = load_or_init(episode_dir, script, cfg2, segments)

    assert manifest.provider == "qwen_audio"
    assert manifest.model == "qwen-audio-3.0-tts-plus"
    # 落盘验证
    save(episode_dir, manifest)
    assert load_manifest(episode_dir).provider == "qwen_audio"


# —— run_retry（Phase 6：文档 17 retry 命令）———————————————————————————


def make_manifest_with_states(
    tmp_path: Path, cfg: AppConfig, states: dict[int, str]
) -> Path:
    """生成脚本 → 初始化 manifest → 按索引人为置状态并落盘（模拟失败/中断现场）。"""
    script = write_script(tmp_path)
    segments = ScriptParser(set(cfg.voices)).parse(script.read_text(encoding="utf-8"))
    episode_dir = cfg.projects_dir / "episode001"
    manifest = load_or_init(episode_dir, script, cfg, segments)
    for rec in manifest.segments:
        if rec.index in states:
            rec.status = states[rec.index]
            if rec.status == "failed":
                rec.error = "simulated failure"
    save(episode_dir, manifest)
    return script


def test_retry_missing_manifest_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_retry("episode001", cfg)


def test_retry_picks_failed_and_pending_only(tmp_path):
    cfg = make_cfg(tmp_path)
    # 状态：1=failed、2=PAUSE success、3=pending；同时删掉 1 的残留文件
    script = make_manifest_with_states(tmp_path, cfg, {1: "failed", 3: "pending"})
    stale = cfg.projects_dir / "episode001" / "segments" / "0001_Lily.wav"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"STALE")

    provider = FakeProvider()
    outcome = run_retry("episode001", cfg, provider=provider)

    assert (outcome.succeeded, outcome.failed, outcome.skipped) == (2, 0, 0)
    assert outcome.exit_code == 0
    # 只调用了 failed + pending 两段，文本来自 manifest（而非重新解析脚本）
    assert sorted(req.text for req in provider.calls) == ["Hello there!", "嗨，我是 Leo。"]
    assert all(req.voice == "Cherry" for req in provider.calls if req.text == "Hello there!")

    episode_dir = cfg.projects_dir / "episode001"
    final = load_manifest(episode_dir)
    assert all(r.status == "success" for r in final.segments)
    assert (episode_dir / "segments" / "0001_Lily.wav").read_bytes() == b"FAKEWAVDATA"
    # PAUSE 记录未被触碰
    assert final.record(2).file == "segments/0002_PAUSE_0.5s.wav"
    assert final.record(2).attempts == 0


def test_retry_nothing_to_do_zero_calls(tmp_path):
    cfg = make_cfg(tmp_path)
    script = write_script(tmp_path)
    run_generate(script, cfg, provider=FakeProvider())  # 全部成功

    provider2 = FakeProvider()
    outcome = run_retry("episode001", cfg, provider=provider2)

    assert (outcome.succeeded, outcome.failed) == (0, 0)
    assert outcome.exit_code == 0
    assert provider2.calls == []  # 无需重试，零 API 调用


def test_retry_failure_again_marks_failed(tmp_path):
    cfg = make_cfg(tmp_path, max_retries=1)
    # 3 置为 success，保证本测试只重试片段 1（speech 初始态是 pending，会被 retry 拾取）
    make_manifest_with_states(tmp_path, cfg, {1: "failed", 3: "success"})

    provider = FakeProvider(script=[FatalProviderError("still boom")])
    outcome = run_retry("episode001", cfg, provider=provider)

    assert (outcome.succeeded, outcome.failed) == (0, 1)
    assert outcome.exit_code == 4
    rec1 = outcome.manifest.record(1)
    assert rec1.status == "failed"
    assert "still boom" in rec1.error


def test_retry_speaker_missing_in_config(tmp_path):
    cfg = make_cfg(tmp_path)
    make_manifest_with_states(tmp_path, cfg, {1: "failed"})

    cfg.voices.pop("Lily")
    with pytest.raises(ConfigError):
        run_retry("episode001", cfg)


# —— 阶段 2：音频参数 / hash v2 / 格式路径 ————————————————————————————————


def _audio_cfg(tmp_path: Path, **params) -> AppConfig:
    return make_cfg(
        tmp_path,
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        audio_params=QwenAudioParams(**params),
    )


def test_audio_params_flow_request_manifest_and_hash(tmp_path):
    cfg = _audio_cfg(tmp_path, rate=1.5, volume=80)
    script = write_script(tmp_path)
    provider = FakeProvider()

    outcome = run_generate(script, cfg, provider=provider)
    assert outcome.succeeded == 2

    lily_req = next(r for r in provider.calls if r.voice == "Cherry")
    assert lily_req.audio_params["rate"] == 1.5
    assert lily_req.audio_params["volume"] == 80
    assert lily_req.audio_params["format"] == "wav"
    assert lily_req.audio_params["language_hints"] == ["en"]  # 旧 language_type 映射
    rec1 = outcome.manifest.record(1)
    assert rec1.audio_params == lily_req.audio_params  # manifest 快照与请求一致

    # 参数变化 → 全部重新生成；再改回缺省 → 再次全量（v2 hash 含参数）
    cfg_default = _audio_cfg(tmp_path)
    provider2 = FakeProvider()
    run_generate(script, cfg_default, provider=provider2)
    assert len(provider2.calls) == 2
    provider3 = FakeProvider()
    run_generate(script, cfg_default, provider=provider3)
    assert provider3.calls == []  # 参数稳定后缓存命中


def test_mp3_segment_extensions_and_pause_stays_wav(tmp_path):
    cfg = _audio_cfg(tmp_path, format="mp3", sample_rate=48000)
    script = write_script(tmp_path)

    outcome = run_generate(script, cfg, provider=FakeProvider())
    rec1, rec2, rec3 = outcome.manifest.segments
    assert rec1.file == "segments/0001_Lily.mp3"
    assert rec3.file == "segments/0003_Leo.mp3"
    assert rec2.file == "segments/0002_PAUSE_0.5s.wav"  # PAUSE 恒为 WAV，文件名含时长
    assert (cfg.projects_dir / "episode001" / rec1.file).exists()
    assert rec1.audio_params["format"] == "mp3"

    # 二次运行按 mp3 路径命中缓存（不会去找旧 wav）
    provider2 = FakeProvider()
    run_generate(script, cfg, provider=provider2)
    assert provider2.calls == []


def test_format_change_uses_new_path_and_keeps_old_file(tmp_path):
    """wav→mp3：不把 mp3 写进 .wav 文件名；旧 wav 残留保留不删除。"""
    script = write_script(tmp_path, SINGLE_SPEECH)
    run_generate(script, _audio_cfg(tmp_path), provider=FakeProvider())
    episode_dir = tmp_path / "projects" / "episode001"
    old_wav = episode_dir / "segments" / "0001_Lily.wav"
    assert old_wav.exists()

    provider2 = FakeProvider()
    outcome = run_generate(script, _audio_cfg(tmp_path, format="mp3"), provider=provider2)
    assert len(provider2.calls) == 1
    assert provider2.calls[0].output_path.endswith("0001_Lily.mp3")
    assert outcome.manifest.record(1).file == "segments/0001_Lily.mp3"
    assert old_wav.exists()  # 旧残文件不自动删除


def test_retry_uses_manifest_audio_params_snapshot(tmp_path):
    """首次以 mp3+rate 失败；配置改回 wav 后 retry 仍必须用失败时的快照参数/路径。"""
    script = write_script(tmp_path, SINGLE_SPEECH)
    cfg_fail = _audio_cfg(tmp_path, format="mp3", sample_rate=48000, rate=1.3)
    provider_fail = FakeProvider(script=[FatalProviderError("boom once")])
    outcome = run_generate(script, cfg_fail, provider=provider_fail)
    assert outcome.manifest.record(1).status == "failed"
    failed_snapshot = outcome.manifest.record(1).audio_params
    assert failed_snapshot["format"] == "mp3" and failed_snapshot["rate"] == 1.3

    # 用户随后把配置改回缺省 wav
    cfg_now = _audio_cfg(tmp_path)
    provider_retry = FakeProvider()
    result = run_retry("episode001", cfg_now, provider=provider_retry)
    assert result.succeeded == 1
    req = provider_retry.calls[0]
    assert req.output_path.endswith("0001_Lily.mp3")   # 沿用失败记录路径
    assert req.audio_params["format"] == "mp3"          # 快照参数，不被新配置覆盖
    assert req.audio_params["rate"] == 1.3


# —— 阶段 3：SSML ————————————————————————————————————————————————————————


def _ssml_script(line: str) -> str:
    return f"Lily:\n{line}\n"


def test_ssml_enabled_sends_flag_and_keeps_text(tmp_path):
    cfg = _audio_cfg(tmp_path, enable_ssml=True)
    script = write_script(tmp_path, _ssml_script("<speak>你好<break time=\"500ms\"/>世界</speak>"))
    provider = FakeProvider()

    outcome = run_generate(script, cfg, provider=provider)
    assert outcome.succeeded == 1 and outcome.failed == 0
    req = provider.calls[0]
    assert req.audio_params["enable_ssml"] is True
    assert req.text == "<speak>你好<break time=\"500ms\"/>世界</speak>"  # 原文透传
    assert outcome.manifest.record(1).audio_params["enable_ssml"] is True


def test_ssml_invalid_fails_without_api_call(tmp_path):
    cfg = _audio_cfg(tmp_path, enable_ssml=True)
    script = write_script(tmp_path, _ssml_script("<speak><prosody rate=\"slow\">x</prosody></speak>"))
    provider = FakeProvider()

    outcome = run_generate(script, cfg, provider=provider)
    assert outcome.failed == 1 and outcome.succeeded == 0
    assert provider.calls == []  # 校验失败不调用 API
    assert "prosody" in outcome.manifest.record(1).error


def test_ssml_speak_attr_overrides_http_param(tmp_path):
    # 全局配置了 rate=1.5；SSML 内 speak rate=0.8 → HTTP rate 必须被移除
    cfg = _audio_cfg(tmp_path, enable_ssml=True, rate=1.5, volume=80)
    script = write_script(tmp_path, _ssml_script('<speak rate="0.8">文本</speak>'))
    provider = FakeProvider()

    run_generate(script, cfg, provider=provider)
    extras = provider.calls[0].audio_params
    assert "rate" not in extras          # 被 speak 属性覆盖，不重复发送
    assert extras.get("volume") == 80    # 未冲突的参数保留
    assert extras["enable_ssml"] is True


def test_ssml_disabled_speak_doc_auto_validated(tmp_path):
    # 片段级文本模式（auto）：整段正文是 <speak> 文档 → 自动按 SSML 校验发送；
    # 格式错误（阿里云不支持 prosody）→ 明确报错，绝不静默按普通文本发送
    cfg = _audio_cfg(tmp_path)
    script = write_script(tmp_path, _ssml_script("<speak><prosody>原样透传</prosody></speak>"))
    provider = FakeProvider()

    outcome = run_generate(script, cfg, provider=provider)
    assert outcome.failed == 1
    assert provider.calls == []
    assert "SSML" in outcome.manifest.record(1).error


def test_auto_plain_text_with_angle_brackets_stays_plain(tmp_path):
    # auto 只认整段 <speak>…</speak> 文档；偶尔出现 <xxx> 不盲目判成 SSML
    cfg = _audio_cfg(tmp_path)
    script = write_script(tmp_path, _ssml_script("温度 5<10 且 a<b 正常。"))
    provider = FakeProvider()

    outcome = run_generate(script, cfg, provider=provider)
    assert outcome.succeeded == 1
    assert "enable_ssml" not in provider.calls[0].audio_params
    assert provider.calls[0].text == "温度 5<10 且 a<b 正常。"


def test_ssml_voice_level_override(tmp_path):
    cfg = _audio_cfg(tmp_path, enable_ssml=True)
    cfg.voices["Lily"] = VoiceSetting("Cherry", "English", ssml=False)
    script = write_script(tmp_path, _ssml_script("<speak>文本</speak>"))
    provider = FakeProvider()

    run_generate(script, cfg, provider=provider)
    assert "enable_ssml" not in provider.calls[0].audio_params
