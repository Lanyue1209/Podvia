"""GUI 设置模型：通用角色管理、校验与持久化。

关键纪律：
- 参数合法规则只有一个事实来源——podcast_tts.capabilities 与 config.load_config，
  GUI 不复制一套校验。
- 角色由用户在 GUI 动态管理（增/删/改名/选音色），稳定 role id 随配置持久化；
  config.yaml 的 voices 仅作为首次启动的种子，之后以 gui_settings.json 为准。
- API Key 经 DPAPI 加密存储（secret_store），明文只存在于内存；绝不进入
  .env/config.yaml/manifest/日志/打包目录。
- "读书"只是输入方式之一：旧版 book_speaker 字段可被读取一次，但不再保存。
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from podcast_tts import capabilities as caps
from podcast_tts.config import (
    ConfigError,
    QwenAudioParams,
    VoiceSetting,
    load_config,
)
from podcast_tts.gui import secret_store


def default_settings_path() -> Path:
    """用户级配置目录（打包后安装包目录只读，必须放用户目录）。"""
    appdata = os.getenv("APPDATA") or str(Path.home())
    return Path(appdata) / "Podvia" / "gui_settings.json"


@dataclass
class RoleVoice:
    """一个用户自定义角色及其音色绑定。

    id：稳定标识（模型切换、改名都不变），GUI 控件与持久化按它索引；
    role：界面显示名（可编辑，可重复名校验在 settings 层做）；
    direction：角色默认导演指令（试听/快速文本补头使用，可选）。
    """

    role: str
    voice: str = ""
    language_hint: str | None = None   # 16 语言码；与 language_type 互斥
    language_type: str | None = None
    ssml: bool | None = None           # None=跟随全局
    custom: bool = False
    direction: str = ""                # 角色默认导演指令（可选）
    # 角色默认语音参数（字符串，空=继承项目默认；行内片段可再覆盖）
    rate: str = ""
    pitch: str = ""
    volume: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex


@dataclass
class GuiSettings:
    # 连接（明文仅内存；落盘走 api_key_enc）
    api_key: str = ""
    provider: str = "qwen_audio"
    model: str = "qwen-audio-3.0-tts-plus"
    # 角色（用户动态管理）
    roles: list[RoleVoice] = field(default_factory=list)
    distinct_voices: bool = True       # 默认"不同角色尽量使用不同音色"
    # 音频参数（留空字符串表示"缺省不发送"）
    rate: str = ""
    pitch: str = ""
    volume: str = ""
    seed: str = ""
    sample_rate: int = 24000
    format: str = "wav"
    direction: str = ""                # 项目级默认导演指令（对未单独设置指令的片段生效）
    default_role: str = ""             # 通用"默认角色（可选）"显示名：无角色头文本使用
    ssml_enabled: bool = False
    # 输入输出
    episode_name: str = "episode"
    projects_dir: str = ""             # 留空=沿用 config.yaml

    # —— 角色管理 ——————————————————————————————————————————————————————

    def role_by_id(self, role_id: str) -> RoleVoice | None:
        return next((r for r in self.roles if r.id == role_id), None)

    def role_by_name(self, name: str) -> RoleVoice | None:
        return next((r for r in self.roles if r.role == name), None)

    def speaker_name_to_id(self) -> dict[str, str]:
        """脚本解析用映射：{角色显示名: 稳定ID}。"""
        return {rv.role: rv.id for rv in self.roles}

    def default_role_id(self) -> str:
        """默认角色显示名 → 稳定 ID（未选择时返回 ""）。"""
        rv = self.role_by_name((self.default_role or "").strip())
        return rv.id if rv else ""

    def add_role(self, name: str, *, voice: str = "", custom: bool = False) -> RoleVoice:
        """新增角色。超过当前模型可分配音色且开启独立音色时，由调用方先行拦截；
        这里只负责重名/空名拦截（模型无关的基本规则）。"""
        name = (name or "").strip()
        if not name:
            raise ConfigError("角色名称不能为空。")
        if any(r.role == name for r in self.roles):
            raise ConfigError(f"角色 {name!r} 已存在，请使用其他名称。")
        rv = RoleVoice(role=name, voice=voice, custom=custom)
        self.roles.append(rv)
        return rv

    def remove_role(self, role_id: str) -> None:
        before = len(self.roles)
        self.roles = [r for r in self.roles if r.id != role_id]
        if len(self.roles) == before:
            raise ConfigError("要删除的角色不存在。")
        if self.default_role and not any(r.role == self.default_role for r in self.roles):
            self.default_role = ""

    def system_voices_for_current(self) -> list[str]:
        """当前模型的系统音色（未知/qwen provider 返回空列表）。"""
        if self.provider != "qwen_audio":
            return []
        c = caps.get_capabilities(self.model)
        return sorted(c.system_voices) if c else []

    def role_voice_status(self, rv: RoleVoice) -> str:
        """单个角色在当前 provider/模型下的音色状态（见 caps.voice_status）。"""
        return caps.voice_status(self.provider, self.model, rv.voice, rv.custom)

    def can_add_non_custom_role(self) -> tuple[bool, str]:
        """独立音色模式下，能否再加一个"系统音色"角色。

        注意三条数量界限互不相关：
        - 角色数量：用户创建多少个说话者；
        - 系统音色数量：当前模型官方列表长度（custom 不占此名额）；
        - API 并发数：generate.max_concurrency，与角色/音色数量无关。
        """
        voices = self.system_voices_for_current()
        if not self.distinct_voices or not voices:
            return True, ""
        used = {r.voice for r in self.roles
                if not r.custom and self.role_voice_status(r) in ("ok",)}
        if len(used) >= len(voices):
            return False, (
                f"当前模型只有 {len(voices)} 个系统音色，且已全部被分配。"
                "可以取消勾选「每个角色使用不同音色」，或为新角色勾选 custom"
                "（复刻/设计音色，云端是否存在无法离线验证）。"
            )
        return True, ""

    def counts(self) -> dict:
        """界面统计：角色数 / 已选不同音色数 / 当前模型系统音色数。"""
        voices = self.system_voices_for_current()
        picked = {r.voice for r in self.roles if r.voice.strip()}
        return {
            "roles": len(self.roles),
            "distinct_voices": len(picked),
            "system_voices": len(voices),
            "model_known": bool(voices) or self.provider != "qwen_audio",
        }

    # —— 校验：直接调用 capabilities，不另立规则 ——————————————————————

    def validate(self) -> list[str]:
        """返回错误信息列表（空列表=通过）。启动/生成/试听前调用。"""
        errors: list[str] = []
        if not self.api_key.strip():
            errors.append("API Key 未配置（DASHSCOPE_API_KEY）。")

        try:
            caps.validate_provider_model(self.provider, self.model)
        except ConfigError as exc:
            errors.append(str(exc))

        if not self.roles:
            errors.append("至少需要一个角色（可在「角色 / 音色」区点「添加角色」）。")

        names = [r.role.strip() for r in self.roles]
        if any(not n for n in names):
            errors.append("存在名称为空的角色，请填写或删除。")
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            errors.append(f"角色名称不能重复：{ '、'.join(dup) }。")

        for rv in self.roles:
            status = self.role_voice_status(rv)
            if status == "empty":
                errors.append(f"角色「{rv.role}」尚未选择音色（模型切换后失效的音色"
                              "不会被自动替换，需要手动重新选择）。")
            elif status == "invalid":
                if self.provider == "qwen":
                    errors.append(
                        f"角色「{rv.role}」的音色 {rv.voice!r} 不属于模型 {self.model} "
                        "的音色体系，请重新选择。"
                    )
                else:
                    errors.append(
                        f"角色「{rv.role}」的音色 {rv.voice!r} 不属于模型 "
                        f"{self.model} 的系统音色，请重新选择（或勾选 custom）。"
                    )
            if rv.language_hint and rv.language_type:
                errors.append(f"角色「{rv.role}」不能同时设置 language_hint 与 language_type。")
            if rv.language_hint:
                try:
                    caps.validate_language_hint(rv.language_hint)
                except ConfigError as exc:
                    errors.append(str(exc))
            # 角色默认语音参数（空字符串=继承项目默认）
            for field_name, validator, caster in (
                ("rate", caps.validate_rate, float),
                ("pitch", caps.validate_pitch, float),
                ("volume", caps.validate_volume, int),
            ):
                raw = getattr(rv, field_name)
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    validator(caster(str(raw).strip()))
                except (ConfigError, ValueError):
                    errors.append(
                        f"角色「{rv.role}」的{field_name} 参数值 {raw!r} 非法。"
                    )

        # 独立音色：系统音色不得被多个非 custom 角色复用
        if self.distinct_voices:
            seen: dict[str, str] = {}
            for rv in self.roles:
                if rv.custom or self.role_voice_status(rv) != "ok":
                    continue
                if rv.voice in seen:
                    errors.append(
                        f"音色 {rv.voice} 同时分配给了角色「{seen[rv.voice]}」和"
                        f"「{rv.role}」。已开启「每个角色使用不同音色」，请改选其他音色，"
                        "或取消该勾选 / 为角色使用 custom 音色。"
                    )
                else:
                    seen[rv.voice] = rv.role

        if self.default_role and self.default_role not in names:
            errors.append(f"默认角色 {self.default_role!r} 不在当前角色列表中，请重新选择。")

        self._validate_num(errors, "rate", self.rate, caps.validate_rate, float)
        self._validate_num(errors, "pitch", self.pitch, caps.validate_pitch, float)
        self._validate_num(errors, "volume", self.volume, caps.validate_volume, int)
        self._validate_num(errors, "seed", self.seed, caps.validate_seed, int)
        try:
            caps.validate_format(self.format)
            caps.validate_sample_rate(self.sample_rate, self.format)
        except ConfigError as exc:
            errors.append(str(exc))
        if self.ssml_enabled:
            profile = next(
                (p for p in caps.list_model_profiles()
                 if p.get("provider") == self.provider and p.get("model") == self.model),
                None,
            )
            if profile is None or not profile.get("supports", {}).get("ssml"):
                errors.append(f"模型 {self.model} 的 SSML 支持未经官方核实，不能开启。")
        return errors

    @staticmethod
    def _validate_num(errors, name, raw, validator, caster):
        if raw is None or str(raw).strip() == "":
            return
        try:
            validator(caster(str(raw).strip()))
        except (ConfigError, ValueError):
            errors.append(f"{name} 参数值 {raw!r} 非法。")

    def warnings(self) -> list[str]:
        warns: list[str] = []
        if self.provider == "qwen_audio" and caps.get_capabilities(self.model) is None:
            warns.append(
                f"模型 {self.model} 状态为“待确认”：只发送基线字段，"
                "rate/pitch/volume/seed/hot_fix/language_hints 不会发送。"
            )
        return warns

    # —— 转换为 pipeline 使用的 AppConfig ——————————————————————————————

    def to_app_config(self):
        """以 config.yaml 为基线，仅覆盖 GUI 显式管理的字段（其余目录/合并/重试沿用）。"""
        errors = self.validate()
        if errors:
            raise ConfigError("GUI 设置校验未通过：\n - " + "\n - ".join(errors))

        base = load_config(key_required=False)

        def _num(raw, caster):
            raw = str(raw if raw is not None else "").strip()
            return caster(raw) if raw else None

        # voices 以稳定 ID 为键；显示名/角色默认指令/角色默认参数一并下发
        voices = {
            rv.id: VoiceSetting(
                voice=rv.voice,
                display_name=rv.role,
                instruction=(rv.direction or "").strip() or None,
                rate=_num(rv.rate, float),
                pitch=_num(rv.pitch, float),
                volume=_num(rv.volume, int),
                language_type=rv.language_type,
                language_hint=rv.language_hint or None,
                custom=rv.custom,
                ssml=rv.ssml,
            )
            for rv in self.roles
        }
        audio = QwenAudioParams(
            rate=float(self.rate) if str(self.rate).strip() else None,
            pitch=float(self.pitch) if str(self.pitch).strip() else None,
            volume=int(self.volume) if str(self.volume).strip() else None,
            seed=int(self.seed) if str(self.seed).strip() else None,
            format=self.format,
            sample_rate=int(self.sample_rate),
            enable_ssml=self.ssml_enabled,
        )
        base.provider = self.provider
        base.model = self.model
        base.voices = voices
        base.audio_params = audio
        # GUI 的"默认导演指令"就是项目级指令（最低优先级，供无细粒度指令的片段继承）
        base.default_instruction = (self.direction or "").strip() or None
        base.api_key = self.api_key.strip()
        if self.projects_dir.strip():
            base.projects_dir = Path(self.projects_dir.strip())
        return base

    # —— JSON 持久化（API Key 走 DPAPI） ——————————————————————————————

    def to_json(self) -> str:
        data = asdict(self)
        key = (data.pop("api_key", "") or "").strip()
        data["api_key_enc"] = secret_store.protect(key) if key else ""
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "GuiSettings":
        data = json.loads(text)

        # API Key：优先 DPAPI 密文；兼容历史明文（加载进内存，下次保存即加密迁移）
        enc = data.pop("api_key_enc", "")
        legacy_plain_key = data.pop("api_key", "")
        api_key = ""
        if enc:
            api_key = secret_store.unprotect(enc)
        elif legacy_plain_key:
            api_key = legacy_plain_key  # 迁移：不再写回明文

        raw_roles = data.pop("roles", [])
        roles = []
        for i, r in enumerate(raw_roles):
            r.setdefault("id", "")  # 旧文件无 id → RoleVoice 自动补稳定 id
            known = {f for f in RoleVoice.__dataclass_fields__}  # type: ignore[attr-defined]
            roles.append(RoleVoice(**{k: v for k, v in r.items() if k in known}))

        # 旧版"读书默认角色"：读取一次作为通用默认角色，之后不再保存该字段
        legacy_book_speaker = data.pop("book_speaker", "")
        data.pop("narrator_role", None)  # 旧字段名统一到 default_role
        if "default_role" not in data and legacy_book_speaker:
            data["default_role"] = legacy_book_speaker

        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        settings = cls(**{k: v for k, v in data.items() if k in known})
        settings.api_key = api_key
        settings.roles = roles
        return settings

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path else default_settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        os.replace(tmp, target)
        secret_store.restrict_file_permissions(target)
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "GuiSettings | None":
        target = Path(path) if path else default_settings_path()
        if not target.exists():
            return None
        return cls.from_json(target.read_text(encoding="utf-8"))


def settings_from_config() -> GuiSettings:
    """从当前 config.yaml 生成 GUI 设置（仅首次启动、无用户设置时做种子）。"""
    cfg = load_config(key_required=False)
    roles = [
        RoleVoice(
            role=setting.display_name or name,
            voice=setting.voice,
            language_hint=setting.language_hint,
            language_type=setting.language_type,
            ssml=setting.ssml,
            custom=setting.custom,
            direction=setting.instruction or "",
            rate=str(setting.rate) if setting.rate is not None else "",
            pitch=str(setting.pitch) if setting.pitch is not None else "",
            volume=str(setting.volume) if setting.volume is not None else "",
        )
        for name, setting in cfg.voices.items()
    ]
    ap = cfg.audio_params
    return GuiSettings(
        api_key=cfg.api_key or "",
        provider=cfg.provider,
        model=cfg.model,
        roles=roles,
        rate=str(ap.rate) if ap.rate is not None else "",
        pitch=str(ap.pitch) if ap.pitch is not None else "",
        volume=str(ap.volume) if ap.volume is not None else "",
        seed=str(ap.seed) if ap.seed is not None else "",
        sample_rate=ap.sample_rate,
        format=ap.format,
        direction=cfg.default_instruction or "",
        ssml_enabled=ap.enable_ssml,
        projects_dir=str(cfg.projects_dir),
    )
