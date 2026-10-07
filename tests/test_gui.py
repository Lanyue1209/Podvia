"""通用语音合成工具 —— GUI 层单元测试（全程 mock，不触网、不使用真实 Key）。

覆盖：
- 通用角色管理：稳定 ID、增删改名、独立音色名额、模型切换失效音色不静默替换
- API Key 持久化：DPAPI 密文落盘、旧明文/读书字段迁移、日志脱敏
- 控制器：文本/章节/失败重试/局部重生成后自动拼接；拼接失败保留片段并可仅重试拼接
- Tk 界面冒烟：动态角色控件、自动保存/重启恢复（无桌面会话时自动 skip）
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from podcast_tts import config as config_mod
from podcast_tts.audio import MergeError, MergeOutcome
from podcast_tts.config import ConfigError
from podcast_tts.gui.controller import GuiController
from podcast_tts.gui.settings import GuiSettings, RoleVoice
from podcast_tts.models import TTSResult
from podcast_tts.providers.base import TTSProvider

CONFIG_YAML = """
provider: qwen_audio
model: qwen-audio-3.0-tts-plus
voices:
  Lily: { voice: "longanlingxin", language_type: "English" }
  Leo:  { voice: "longanlufeng", language_type: "Chinese" }
generate:
  max_concurrency: 1
"""

THREE_SPEECH = "Lily:\n一。\n\nLeo:\n二。\n\nLily:\n三。\n"
SECRET_KEY = "sk-secret-abcdef-1234567890"


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "ROOT_DIR", tmp_path)
    (tmp_path / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    return tmp_path


def make_settings(**kw) -> GuiSettings:
    s = GuiSettings(
        api_key="sk-test",
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        roles=[
            RoleVoice(role="Lily", voice="longanlingxin", language_type="English"),
            RoleVoice(role="Leo", voice="longanlufeng", language_type="Chinese"),
        ],
        default_role="Lily",
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class FakeProvider(TTSProvider):
    name = "fake"

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            self.calls.append(request)
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE")
        return TTSResult(output_path=request.output_path, request_id="r1")


class TestEmotagPreview:
    """情绪/拟声标签在 GUI 预览与试听路径上的行为（封网，不调真实 API）。"""

    def test_preview_passes_with_tags_on_supported_model(self):
        from podcast_tts.gui.controller import preview_text
        s = make_settings()
        rows, errors = preview_text(s, "Lily: [excited] 太好了！[laughing] 走吧！")
        assert errors == []
        assert rows and rows[0]["text"].startswith("[excited]")

    def test_preview_blocks_unknown_tag(self):
        from podcast_tts.gui.controller import preview_text
        s = make_settings()
        rows, errors = preview_text(s, "Lily: [happy] 你好")
        assert rows == []
        assert errors and "不是官方情绪/拟声标签" in errors[0]

    def test_preview_blocks_tags_on_unsupported_model(self):
        from podcast_tts.gui.controller import preview_text
        s = make_settings(
            provider="qwen", model="qwen3-tts-flash",
            roles=[RoleVoice(role="Lily", voice="Cherry", language_type="English"),
                   RoleVoice(role="Leo", voice="Ethan", language_type="Chinese")],
        )
        rows, errors = preview_text(s, "Lily: [excited] 你好")
        assert rows == []
        assert errors and "当前接口" in errors[0]

    def test_standalone_tag_without_speaker_errors(self):
        from podcast_tts.gui.controller import preview_text
        s = make_settings(default_role="")
        rows, errors = preview_text(s, "[laughing]")
        assert rows == []
        assert errors and "角色" in errors[0]

    def test_preview_marks_injected_prefix(self):
        from podcast_tts.gui.controller import preview_text
        s = make_settings()
        body = "今天发生了一件特别有意思的事情，" * 40  # 触发自动分段
        rows, errors = preview_text(s, f"Lily: [excited] {body}。")
        assert errors == []
        assert len(rows) >= 2
        assert rows[0]["injected"] is None
        assert all(r["injected"] == "[excited] " for r in rows[1:])


class FailProvider(TTSProvider):
    """所有片段立即失败（普通 RuntimeError 不触发退避等待）。"""

    name = "fail"

    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        raise RuntimeError("模拟 API 失败")


class FakeMerge:
    """run_merge 替身：记录调用与 manifest（脚本）顺序，按需产出成品文件。"""

    def __init__(self, *, fail: bool = False, lazy: bool = False):
        self.calls: list[tuple[str, bool]] = []
        self.speech_order: list[list[str]] = []
        self.fail = fail
        self.lazy = lazy  # True=声称成功但不落盘（控制器必须判为拼接失败）

    def __call__(self, episode, cfg, *, force=False, expected_hashes=None):
        from podcast_tts.manifest import load as load_manifest

        self.calls.append((episode, force))
        self.expected_hashes = expected_hashes
        manifest = load_manifest(cfg.projects_dir / episode)
        self.speech_order.append(
            [r.speaker_name or r.speaker
             for r in manifest.segments if r.kind == "speech"]
        )
        if self.fail:
            raise MergeError("模拟拼接失败：ffmpeg 不可用，请安装 FFmpeg 后重试。")
        out_rel = f"output/{episode}.{cfg.merge_format}"
        if not self.lazy:
            out = cfg.projects_dir / episode / out_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"FAKE-OUTPUT")
        return MergeOutcome(episode=episode, output=out_rel, duration=1.5,
                            skipped=False)


# —— 设置持久化与 API Key 落盘 ——————————————————————————————————————————


def test_settings_save_restore_roundtrip(tmp_path):
    s = make_settings(rate="1.25", volume="66", ssml_enabled=True,
                      episode_name="第01章")
    path = tmp_path / "gui_settings.json"
    s.save(path)
    loaded = GuiSettings.load(path)
    assert loaded is not None
    assert loaded.rate == "1.25" and loaded.volume == "66"
    assert loaded.ssml_enabled is True
    assert loaded.default_role == "Lily"
    assert [r.role for r in loaded.roles] == ["Lily", "Leo"]
    assert loaded.roles[0].voice == "longanlingxin"
    # API Key 还原；角色稳定 ID 不变
    assert loaded.api_key == "sk-test"
    assert [r.id for r in loaded.roles] == [r.id for r in s.roles]


def test_settings_file_never_contains_plaintext_key(tmp_path):
    path = tmp_path / "gui_settings.json"
    make_settings(api_key=SECRET_KEY).save(path)
    raw = path.read_text(encoding="utf-8")
    assert SECRET_KEY not in raw
    assert '"api_key"' not in json.loads(raw)  # 只有 api_key_enc
    data = json.loads(raw)
    assert data["api_key_enc"].startswith(("dpapi:", "plain:"))
    assert SECRET_KEY not in data["api_key_enc"]


def test_settings_load_missing_returns_none(tmp_path):
    assert GuiSettings.load(tmp_path / "nope.json") is None


def test_settings_migrates_legacy_plaintext_and_book_fields(tmp_path):
    """旧版明文 api_key 可读；book_speaker/narrator_role 只迁移一次不再写回。"""
    legacy = {
        "api_key": SECRET_KEY,
        "provider": "qwen_audio",
        "model": "qwen-audio-3.0-tts-plus",
        "book_speaker": "Lily",
        "narrator_role": "Leo",
        "roles": [
            {"role": "Lily", "voice": "longanlingxin"},
            {"role": "Leo", "voice": "longanlufeng"},
        ],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    loaded = GuiSettings.load(path)
    assert loaded.api_key == SECRET_KEY
    assert loaded.default_role == "Lily"
    assert not hasattr(loaded, "book_speaker")
    # 角色无旧 id 时补稳定 id
    assert all(r.id for r in loaded.roles)

    # 再次保存：明文 Key 与读书字段全部消失（迁移后不再写回）
    loaded.save(path)
    raw = path.read_text(encoding="utf-8")
    assert SECRET_KEY not in raw
    data = json.loads(raw)
    assert "book_speaker" not in data and "narrator_role" not in data
    assert "api_key" not in data and data["api_key_enc"]


def test_settings_ignores_unknown_json_fields(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({
        "api_key_enc": "", "model": "qwen-audio-3.0-tts-plus",
        "roles": [{"role": "A", "voice": "longanlingxin",
                   "future_field": 123}],
        "unknown_top": "x",
    }), encoding="utf-8")
    loaded = GuiSettings.load(path)
    assert loaded.roles[0].role == "A"


# —— 通用角色管理 ——————————————————————————————————————————————————————


def test_role_add_remove_rename_and_stable_id():
    s = make_settings()
    rid = s.add_role("Nar", voice="longanhuan_v3.6")
    assert rid.id and len(s.roles) == 3

    with pytest.raises(ConfigError):
        s.add_role("Nar")            # 重名
    with pytest.raises(ConfigError):
        s.add_role("   ")            # 空名

    rid.role = "旁白"                # 改名：id 不变
    assert s.role_by_id(rid.id).role == "旁白"

    s.default_role = "旁白"
    s.remove_role(rid.id)
    assert len(s.roles) == 2
    assert s.default_role == ""      # 删除默认角色连带清空
    with pytest.raises(ConfigError):
        s.remove_role(rid.id)        # 再删报错


def test_distinct_voice_quota_for_plus():
    s = make_settings()  # plus 仅 2 个系统音色，已分配 2 个
    ok, msg = s.can_add_non_custom_role()
    assert ok is False and "2 个系统音色" in msg
    # custom 不占系统音色名额
    s.add_role("Clone", voice="my-clone-voice", custom=True)
    # 关闭独立音色限制后允许继续加系统音色角色
    s.distinct_voices = False
    ok2, _ = s.can_add_non_custom_role()
    assert ok2 is True


def test_distinct_quota_flash_has_twelve():
    s = make_settings(model="qwen-audio-3.0-tts-flash")
    # 原 plus 音色在 flash 下失效，不计入已占用名额
    ok, _ = s.can_add_non_custom_role()
    assert ok is True


def test_counts_separates_roles_voices_and_concurrency():
    s = make_settings()
    counts = s.counts()
    assert counts["roles"] == 2
    assert counts["distinct_voices"] == 2   # 已选不同音色
    assert counts["system_voices"] == 2     # plus 系统音色总数
    # 并发数不在 settings 层（cfg.max_concurrency），角色数量不受其限制
    s.add_role("Clone", voice="c1", custom=True)
    assert s.counts()["roles"] == 3
    assert s.counts()["distinct_voices"] == 3


def test_validate_rejects_duplicate_voice_when_distinct_enabled():
    s = make_settings()
    s.roles[1].voice = "longanlingxin"  # 与 Lily 重复
    errors = s.validate()
    assert any("不同音色" in e for e in errors)
    s.distinct_voices = False
    assert s.validate() == []


def test_model_switch_marks_voice_invalid_but_keeps_it():
    s = make_settings(model="qwen-audio-3.0-tts-flash")
    lily = s.roles[0]
    # 切换后旧音色被保留（不静默替换成第一个），但状态判定为 invalid
    assert lily.voice == "longanlingxin"
    assert s.role_voice_status(lily) == "invalid"
    errors = s.validate()
    assert any("longanlingxin" in e and "重新选择" in e for e in errors)

    # 3.1 flash 同样不兼容 plus 音色
    s.model = "qwen-audio-3.1-tts-flash"
    assert s.role_voice_status(lily) == "invalid"

    # empty 必须显式补选
    s.model = "qwen-audio-3.0-tts-plus"
    lily.voice = ""
    assert s.role_voice_status(lily) == "empty"
    assert any("尚未选择音色" in e for e in s.validate())


def test_custom_voice_bypasses_system_whitelist():
    s = make_settings()
    s.roles[0].voice = "any-cloud-voice-id"
    s.roles[0].custom = True
    assert s.role_voice_status(s.roles[0]) == "custom"
    assert s.validate() == []


def test_system_voice_counts_and_sources():
    from podcast_tts import capabilities as caps

    assert caps.system_voice_count("qwen-audio-3.0-tts-plus") == 2
    assert caps.system_voice_count("qwen-audio-3.0-tts-flash") == 12
    assert caps.system_voice_count("qwen-audio-3.1-tts-flash") == 28
    assert caps.system_voice_count("qwen-audio-9.9-tts") is None
    for model in ("qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash",
                  "qwen-audio-3.1-tts-flash"):
        assert "help.aliyun.com" in caps.MATRIX[model].source
        assert "2026-09-29" in caps.MATRIX[model].source


# —— 参数/配置校验 ——————————————————————————————————————————————————————


def test_validate_ok():
    assert make_settings().validate() == []


def test_validate_missing_key():
    assert any("API Key" in e for e in make_settings(api_key="").validate())


@pytest.mark.parametrize("field,value", [
    ("rate", "3"), ("pitch", "x"), ("volume", "150"), ("seed", "-2"),
])
def test_validate_bad_numbers(field, value):
    assert any(field in e for e in make_settings(**{field: value}).validate())


def test_validate_cosyvoice_refused():
    errors = make_settings(model="cosyvoice-v3-flash",
                           roles=[RoleVoice("Lily", "longanlingxin")]).validate()
    assert any("尚未实现" in e for e in errors)


def test_validate_cross_family_voice():
    errors = make_settings(
        provider="qwen", model="qwen3-tts-flash",
        roles=[RoleVoice("Lily", "longanlingxin")]).validate()
    assert any("系统音色" in e or "音色" in e for e in errors)


def test_validate_ssml_only_verified_model():
    assert not [e for e in make_settings(ssml_enabled=True).validate() if "SSML" in e]
    bad = make_settings(model="qwen-audio-9.9-tts", ssml_enabled=True,
                        roles=[RoleVoice("Lily", "custom-v", custom=True)])
    assert any("SSML" in e for e in bad.validate())


def test_to_app_config_overrides_only_gui_fields(isolated_root):
    s = make_settings(rate="1.4", format="mp3", sample_rate=48000,
                      direction="项目指令",
                      projects_dir=str(isolated_root / "out"))
    cfg = s.to_app_config()
    assert cfg.provider == "qwen_audio" and cfg.api_key == "sk-test"
    assert cfg.audio_params.rate == 1.4
    assert cfg.audio_params.format == "mp3" and cfg.audio_params.sample_rate == 48000
    # voices 以稳定 ID 为键，显示名与项目级指令独立保存
    lily = next(v for v in cfg.voices.values() if v.display_name == "Lily")
    assert lily.voice == "longanlingxin"
    assert cfg.default_instruction == "项目指令"
    assert cfg.projects_dir == isolated_root / "out"
    # GUI 默认角色显示名可映射回稳定 ID
    assert s.default_role_id() in cfg.voices


# —— 脚本组装 ——————————————————————————————————————————————————————————


def test_build_script_wraps_plain_text_with_default_role():
    c = GuiController()
    s = make_settings(direction="温柔地")
    script = c.build_script(s, "你好世界")
    # 项目级指令不再被塞进角色头（否则会被当成行内指令错误覆盖）
    assert script == "Lily：\n你好世界\n"
    # 指令通过项目级配置下发
    assert s.to_app_config().default_instruction == "温柔地"


def test_build_script_keeps_existing_header_without_default_role():
    c = GuiController()
    s = make_settings(default_role="")
    raw = "Leo(兴奋地):\n现成台词。\n"
    assert c.build_script(s, raw) == raw


def test_build_script_plain_text_without_default_role_is_error():
    c = GuiController()
    s = make_settings(default_role="")
    with pytest.raises(ConfigError, match="默认角色"):
        c.build_script(s, "没有角色头的普通文本。")


# —— 控制器：自动拼接 ———————————————————————————————————————————————————


def _drain(q: queue.Queue, timeout: float = 10.0):
    events = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            events.append(q.get(timeout=0.1))
        except queue.Empty:
            if any(e[0] == "finished" for e in events):
                break
    return events


def _start_controller(isolated_root, *, provider=None, merge=None):
    q: queue.Queue = queue.Queue()
    c = GuiController(
        q,
        provider_factory=lambda cfg: provider or FakeProvider(),
        merge_func=merge if merge is not None else FakeMerge(),
    )
    return c, q, merge


def test_text_generation_auto_merges_in_script_order(isolated_root):
    merge = FakeMerge()
    c, q, _ = _start_controller(isolated_root, merge=merge)
    c.start_text(make_settings(episode_name="ep1"), THREE_SPEECH)
    c.join(10)
    assert not c.busy

    events = _drain(q)
    assert merge.calls == [("ep1", False)]          # 自动拼接，非 force
    # 拼接严格按 manifest（脚本）顺序，不按请求完成顺序
    assert merge.speech_order == [["Lily", "Leo", "Lily"]]
    statuses = [e[1] for e in events if e[0] == "status"]
    assert "running" in statuses and "merging" in statuses  # 正在生成片段→正在拼接
    assert any(e[0] == "merged" for e in events)
    fin = [e for e in events if e[0] == "finished"][-1][1]
    assert fin["merged"] is True and fin["output"].endswith("ep1.mp3")
    assert c.state == "done"
    assert Path(fin["output"]).is_file()


def test_book_generation_auto_merges(isolated_root, tmp_path):
    book = tmp_path / "chapter1.md"
    book.write_text("# 第一章\n- 角色: Lily\n你好，世界。\n", encoding="utf-8")
    merge = FakeMerge()
    c, q, _ = _start_controller(isolated_root, merge=merge)
    c.start_book(make_settings(episode_name="ignored"), str(book))
    c.join(10)
    events = _drain(q)
    assert merge.calls == [("chapter1", False)]      # episode 取章节文件名
    assert [e for e in events if e[0] == "finished"][-1][1]["merged"] is True


def test_failed_segments_do_not_merge_then_retry_merges(isolated_root):
    # 第一轮：全部失败 → 不拼接，保留片段记录
    merge1 = FakeMerge()
    c1 = GuiController(queue.Queue(),
                       provider_factory=lambda cfg: FailProvider(),
                       merge_func=merge1)
    c1.start_text(make_settings(episode_name="ep-retry"), THREE_SPEECH)
    c1.join(15)
    events1 = _drain(c1.events)
    fin1 = [e for e in events1 if e[0] == "finished"][-1][1]
    assert fin1["failed"] == 3 and fin1["merged"] is False
    assert merge1.calls == []
    assert any("未进行拼接" in e[1] for e in events1 if e[0] == "log")

    # 第二轮：重试失败片段成功 → 自动拼接（无需用户手动合并）
    merge2 = FakeMerge()
    provider2 = FakeProvider()
    c2 = GuiController(queue.Queue(),
                       provider_factory=lambda cfg: provider2,
                       merge_func=merge2)
    c2.episode = "ep-retry"
    c2.set_settings(make_settings(episode_name="ep-retry"))
    c2.retry_failed()
    c2.join(15)
    events2 = _drain(c2.events)
    assert len(provider2.calls) == 3
    assert merge2.calls == [("ep-retry", False)]
    assert [e for e in events2 if e[0] == "finished"][-1][1]["merged"] is True


def test_regenerate_auto_merges(isolated_root):
    merge = FakeMerge()
    provider = FakeProvider()
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: provider,
                      merge_func=merge)
    c.start_text(make_settings(episode_name="ep-regen"), THREE_SPEECH)
    c.join(10)
    _drain(c.events)
    assert len(provider.calls) == 3

    c.regenerate([2])   # 只重做第 2 段（Leo）
    c.join(10)
    events = _drain(c.events)
    assert len(provider.calls) == 4
    assert merge.calls == [("ep-regen", False), ("ep-regen", False)]
    assert [e for e in events if e[0] == "finished"][-1][1]["merged"] is True


def test_merge_failure_keeps_segments_and_merge_only_recovers(isolated_root):
    merge_bad = FakeMerge(fail=True)
    provider = FakeProvider()
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: provider,
                      merge_func=merge_bad)
    c.start_text(make_settings(episode_name="ep-mf"), THREE_SPEECH)
    c.join(10)
    events = _drain(c.events)

    mf = [e for e in events if e[0] == "merge_failed"]
    assert mf and mf[0][1]["error"]
    assert mf[0][1]["output"].endswith("ep-mf.mp3")     # 给出最终输出路径
    assert c.state == "merge_failed"
    fin = [e for e in events if e[0] == "finished"][-1][1]
    assert fin["merged"] is False
    # 片段全部保留
    seg_dir = isolated_root / "projects" / "ep-mf" / "segments"
    assert len(list(seg_dir.glob("*.wav"))) == 3

    # 仅重新合并：不调用任何 API，force=True，成功后状态完成
    merge_good = FakeMerge()

    def provider_forbidden(cfg):
        raise AssertionError("仅重新合并不得调用 Provider/API")

    c2 = GuiController(queue.Queue(),
                       provider_factory=provider_forbidden,
                       merge_func=merge_good)
    c2.episode = "ep-mf"
    c2.set_settings(make_settings(episode_name="ep-mf"))
    c2.merge_only()
    c2.join(10)
    events2 = _drain(c2.events)
    assert merge_good.calls == [("ep-mf", True)]
    assert [e for e in events2 if e[0] == "finished"][-1][1]["merged"] is True
    assert c2.state == "done"


def test_merge_reported_success_without_output_file_is_failure(isolated_root):
    merge = FakeMerge(lazy=True)  # 声称成功但没产出文件
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: FakeProvider(),
                      merge_func=merge)
    c.start_text(make_settings(episode_name="ep-lazy"), "Lily:\n文本\n")
    c.join(10)
    events = _drain(c.events)
    assert any(e[0] == "merge_failed" for e in events)
    assert c.state == "merge_failed"


def test_cancel_does_not_merge(isolated_root):
    class CancelProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.first_done = threading.Event()
            self.may_finish = threading.Event()

        def generate(self, request):
            with self.lock:
                idx = len(self.calls)
            if idx == 0:
                result = super().generate(request)
                self.first_done.set()
                self.may_finish.wait(timeout=10)
                return result
            self.may_finish.wait(timeout=10)
            return super().generate(request)

    merge = FakeMerge()
    provider = CancelProvider()
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: provider,
                      merge_func=merge)
    c.start_text(make_settings(episode_name="ep-cancel"), THREE_SPEECH)
    assert provider.first_done.wait(timeout=5)
    c.cancel()
    provider.may_finish.set()
    c.join(10)
    events = _drain(c.events)
    fin = [e for e in events if e[0] == "finished"][-1][1]
    assert fin["cancelled"] is True and fin["merged"] is False
    assert merge.calls == []


def test_controller_pause_blocks_new_submissions(isolated_root):
    class BlockingProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def generate(self, request):
            with self.lock:
                idx = len(self.calls)
                self.calls.append(request)
            if idx == 0:
                self.entered.set()
            self.release.wait(timeout=10)
            return super().generate(request)

    provider = BlockingProvider()
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: provider,
                      merge_func=FakeMerge())
    c.start_text(make_settings(episode_name="ep-pause"), THREE_SPEECH)
    assert provider.entered.wait(timeout=5)
    c.pause()
    time.sleep(0.4)
    assert len(provider.calls) == 1
    c.resume()
    provider.release.set()
    c.join(15)
    events = _drain(c.events)
    assert [e for e in events if e[0] == "finished"][-1][1]["succeeded"] == 3


def test_controller_audition_emits_path(isolated_root):
    provider = FakeProvider()
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: provider)
    c.audition(make_settings(episode_name="ep-aud"), "试听文本")
    c.join(10)
    events = _drain(c.events)
    assert any(e[0] == "audition_done" for e in events)
    assert Path(provider.calls[0].output_path).name.startswith("audition_")


def test_controller_fatal_event_on_bad_settings(isolated_root):
    c = GuiController(queue.Queue(),
                      provider_factory=lambda cfg: FakeProvider(),
                      merge_func=FakeMerge())
    c.start_text(make_settings(api_key=""), "Lily:\n文本\n")
    c.join(10)
    events = _drain(c.events)
    assert any(e[0] == "fatal" for e in events)


# —— Tk 界面 ————————————————————————————————————————————————————————————


@pytest.fixture
def app_factory(isolated_root, monkeypatch):
    monkeypatch.setattr(
        "podcast_tts.gui.app.default_settings_path",
        lambda: isolated_root / "gui_settings.json")
    # settings 模块内部也会直接引用该函数，必须同时隔离，防止写入真实 %APPDATA%
    monkeypatch.setattr(
        "podcast_tts.gui.settings.default_settings_path",
        lambda: isolated_root / "gui_settings.json")
    # 所有弹窗在测试中转为静默/记录
    shown = {"errors": [], "questions": []}
    import podcast_tts.gui.app as app_mod

    monkeypatch.setattr(app_mod.messagebox, "showerror",
                        lambda title, msg, **k: shown["errors"].append(msg))
    monkeypatch.setattr(app_mod.messagebox, "showinfo",
                        lambda title, msg, **k: shown["errors"].append(msg))
    monkeypatch.setattr(app_mod.messagebox, "showwarning",
                        lambda title, msg, **k: shown["errors"].append(msg))
    monkeypatch.setattr(app_mod.messagebox, "askokcancel", lambda *a, **k: True)

    tk = pytest.importorskip("tkinter")
    # 显式指定 Tcl/Tk 库路径，避免同一进程多次创建 Tk 根时偶发找不到 tk.tcl
    import sys
    import os
    tcl_root = os.path.join(sys.prefix, "tcl")
    for name in ("tcl8.6", "tk8.6"):
        cand = os.path.join(tcl_root, name)
        if os.path.isdir(cand):
            env_name = "TCL_LIBRARY" if name.startswith("tcl") else "TK_LIBRARY"
            os.environ.setdefault(env_name, cand)
    try:
        import tkinter.simpledialog  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    created = []

    def make(controller=None, ask_answers=None):
        from podcast_tts.gui.app import PodviaApp
        answers = list(ask_answers or [])
        monkeypatch.setattr(app_mod.simpledialog, "askstring",
                            lambda *a, **k: answers.pop(0) if answers else None)
        monkeypatch.setattr(app_mod.messagebox, "askyesno",
                            lambda *a, **k: answers.pop(0) if answers else False)
        # Windows 上连续创建多个 Tk 根时，Tcl 库文件读取可能偶发失败（杀软扫描/句柄竞争）
        app = None
        last_exc = None
        for attempt in range(4):
            try:
                app = PodviaApp(controller=controller)
                break
            except tk.TclError as exc:  # noqa: PERF203
                last_exc = exc
                time.sleep(0.4 * (attempt + 1))
        if app is None:
            pytest.skip(f"Tk 初始化多次失败，跳过界面测试：{last_exc}")
        if not app.var_key.get():
            app.var_key.set("sk-test")  # 角色类测试不关心 Key；需要空 Key 的测试自行清空
        created.append(app)
        return app, shown

    yield make

    for app in created:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def _pump(app, controller, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline and controller.busy:
        app.update()
        time.sleep(0.02)
    app.update()


def _all_widget_text(widget) -> str:
    """递归收集界面所有 Label/Button/文本框可见文字。"""
    chunks = []
    try:
        chunks.append(str(widget.cget("text")))
    except Exception:  # noqa: BLE001
        pass
    if widget.winfo_class() in ("Text",):
        chunks.append(widget.get("1.0", "end"))
    for child in widget.winfo_children():
        chunks.append(_all_widget_text(child))
    return "\n".join(chunks)


def test_tk_ui_smoke_general_positioning(app_factory):
    app, _ = app_factory()
    app.update_idletasks()
    assert "通用语音合成" in app.title()
    visible = _all_widget_text(app)
    assert "读书默认角色" not in visible
    assert "qwen-audio-3.0-tts-plus" in app.cmb_model["values"]
    s = app.collect_settings()
    assert len(s.roles) == 2 and all(r.id for r in s.roles)
    # 不静默选默认角色（必须用户显式选择）
    assert app.var_default_role.get() == ""
    # 初始未运行：暂停按钮禁用
    assert str(app.btn_pause["state"]) == "disabled"


def test_tk_add_role_blocked_when_system_voices_exhausted(app_factory):
    app, shown = app_factory(ask_answers=["Nar", False])  # 名字 + 非 custom
    app._on_add_role()
    app.update_idletasks()
    assert len(app.settings.roles) == 2
    assert shown["errors"] and "系统音色" in shown["errors"][-1]


def test_tk_add_custom_role_allowed(app_factory):
    app, _ = app_factory(ask_answers=["Clone", True])  # 第三个角色走 custom
    app._on_add_role()
    app.update_idletasks()
    assert len(app.settings.roles) == 3
    new_rv = app.settings.roles[-1]
    assert new_rv.custom is True and new_rv.voice == ""
    app._role_rows[new_rv.id]["voice"].set("my-clone-001")
    s = app.collect_settings()
    assert s.validate() == []


def test_tk_remove_role_clears_default_role(app_factory):
    app, _ = app_factory()
    lily_id = app.settings.roles[0].id
    app.var_default_role.set("Lily")
    app._on_remove_role(lily_id)
    app.update_idletasks()
    assert len(app.settings.roles) == 1
    assert app.var_default_role.get() == ""


def test_tk_model_switch_keeps_invalid_voice_until_user_reselects(app_factory):
    app, shown = app_factory()
    lily_id = app.settings.roles[0].id
    app.var_model.set("qwen-audio-3.0-tts-flash")
    app._on_model_change()
    app.update_idletasks()

    row = app._role_rows[lily_id]
    # 旧音色保留显示，绝不自动切到列表第一个
    assert row["voice"].get() == "longanlingxin"
    flash_voices = row["voice_widget"]["values"]
    assert "longanfengyue" in flash_voices and "longanlingxin" not in flash_voices
    assert "需重新选择" in row["status"].cget("text")
    s = app.collect_settings()
    assert any("重新选择" in e for e in s.validate())

    # 用户手动改选后恢复有效
    row["voice"].set("longanfengyue")
    app._role_rows[app.settings.roles[1].id]["voice"].set("longanyuanfei")
    assert app.collect_settings().validate() == []
    # 期间没有弹错误（校验只在开始时弹）
    assert not shown["errors"]


def test_tk_distinct_voices_toggle(app_factory):
    app, _ = app_factory()
    lily, leo = app.settings.roles
    app._role_rows[leo.id]["voice"].set("longanlingxin")
    app.var_distinct.set(False)
    assert app.collect_settings().validate() == []
    app.var_distinct.set(True)
    errors = app.collect_settings().validate()
    assert any("不同音色" in e for e in errors)


def test_tk_reused_voice_hint_label(app_factory):
    """取消"不同音色"后界面必须明确提示：多个角色将使用同一个音色。"""
    app, _ = app_factory()
    app.var_distinct.set(False)
    app._refresh_counts()
    assert "多个角色将使用同一个音色" in app.lbl_reuse.cget("text")
    app.var_distinct.set(True)
    app._refresh_counts()
    assert app.lbl_reuse.cget("text") == ""


def test_tk_start_autosaves_encrypted_key_and_restores(app_factory, isolated_root):
    merge = FakeMerge()
    controller = GuiController(
        provider_factory=lambda cfg: FakeProvider(), merge_func=merge)
    app, shown = app_factory(controller=controller)
    app.var_key.set(SECRET_KEY)
    app.var_default_role.set("Lily")
    app.txt_input.delete("1.0", "end")
    app.txt_input.insert("1.0", "这是一段用于冒烟测试的普通文本。")
    settings_file = isolated_root / "gui_settings.json"

    # 开始生成现在先弹"生成预览"；测试中自动点击"确认无误，开始生成"
    def _auto_confirm_preview(s, ws, rows, errors):
        assert rows and not errors  # 占位文本也能规划出片段且无错误
        controller.start_working(s, ws)

    app._show_preview = _auto_confirm_preview
    app._on_start()
    _pump(app, controller)

    # 点开始前已自动保存；磁盘只有密文
    assert settings_file.is_file()
    raw = settings_file.read_text(encoding="utf-8")
    assert SECRET_KEY not in raw and "api_key_enc" in raw
    # 界面日志也不允许出现完整 Key
    log_text = app.txt_log.get("1.0", "end")
    assert SECRET_KEY not in log_text
    assert merge.calls and controller.state == "done"

    # 模拟重启：新窗口从 %APPDATA%（隔离路径）恢复 Key 与角色
    app2, _ = app_factory()
    assert app2.var_key.get() == SECRET_KEY
    assert [r.role for r in app2.settings.roles] == ["Lily", "Leo"]
    assert app2.var_default_role.get() == "Lily"


def test_tk_close_autosaves_nonsensitive_config(app_factory, isolated_root):
    app, _ = app_factory()
    app.var_key.set("")                    # 未填 Key 也要保存非敏感配置
    app.var_rate.set("1.1")
    app._on_close()                       # 不忙，无确认弹窗
    data = json.loads((isolated_root / "gui_settings.json").read_text("utf-8"))
    assert data["rate"] == "1.1"
    assert data.get("api_key_enc", "") == ""
