"""配置加载与校验：.env（密钥）+ config.yaml（参数）。

校验失败抛 ConfigError，错误信息必须能直接指导用户修复。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from podcast_tts import capabilities as caps
from podcast_tts import log
from podcast_tts.runtime import (
    bundle_dir,
    bundled_ffmpeg_dir,
    ensure_user_root,
    is_frozen,
    user_data_dir,
)

# 仓库根目录（podcast_tts/ 的上一级）；冻结后改用 %APPDATA%\Podvia（可写、外部配置）
if is_frozen():
    ROOT_DIR = user_data_dir()
    ensure_user_root(ROOT_DIR, bundle_dir() / "config.yaml")
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

_QWEN_DEFAULT_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
# Qwen-Audio-TTS 系列专用端点（与 Qwen-TTS 不通用，文档：非实时语音合成 HTTP API）
_QWEN_AUDIO_DEFAULT_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
)


class ConfigError(Exception):
    """配置错误。消息面向用户，需说明问题与修复方法。"""


@dataclass(frozen=True)
class VoiceSetting:
    """一个角色的声音绑定与角色级默认参数。"""

    voice: str                # Voice ID，如 "Cherry"
    language_type: str | None  # "English" / "Chinese" / None(=Auto)，旧配置字段
    language_hint: str | None = None  # 新字段：官方 16 语言短码（API 字段 language_hints）
    custom: bool = False      # True=复刻/设计/基础音色，仅跳过系统音色白名单
    ssml: bool | None = None  # 本角色 SSML 开关；None=跟随 qwen_audio.enable_ssml
    # 角色显示名（GUI 可为中文）；CLI 下与角色键相同
    display_name: str | None = None
    # 角色默认导演指令（层级：行内 > 章节 > 角色 > 项目）
    instruction: str | None = None
    # 角色级默认语音参数（None=继承项目默认；行内片段可再覆盖）
    rate: float | None = None
    pitch: float | None = None
    volume: int | None = None


@dataclass(frozen=True)
class QwenAudioParams:
    """qwen_audio provider 的全局音频参数（缺省即不发送可选键）。

    范围/枚举见 podcast_tts.capabilities（官方 HTTP API 页 2026-09-28）。
    format/sample_rate 缺省 wav/24000，以锁定旧版本实际请求行为。
    """

    format: str = "wav"
    sample_rate: int = 24000
    rate: float | None = None
    pitch: float | None = None
    volume: int | None = None
    seed: int | None = None
    language_hint: str | None = None  # Podvia 配置名；API 字段名仍是 language_hints
    hot_fix: dict | None = None
    enable_ssml: bool = False  # 显式开启才发送 input.enable_ssml=true；不自动改写文本


@dataclass
class AppConfig:
    """校验后的完整运行配置。"""

    provider: str
    model: str
    voices: dict[str, VoiceSetting]
    max_concurrency: int
    request_timeout: int
    max_retries: int
    retry_base_delay: float
    optimize_instructions: bool
    qwen_base_url: str
    qwen_audio_base_url: str
    audio_params: QwenAudioParams
    merge_format: str
    bitrate: str
    scripts_dir: Path
    projects_dir: Path
    logs_dir: Path
    ffmpeg_dir: Path
    api_key: str | None
    # 项目级默认导演指令（最低优先级；GUI 的"默认导演指令"写入这里）
    default_instruction: str | None = None


def load_config(key_required: bool = True) -> AppConfig:
    """加载并校验配置。key_required=False 供不触网的命令（如 parse/status）使用。"""
    load_dotenv(ROOT_DIR / ".env")
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if key_required and not api_key:
        raise ConfigError(
            "未配置 DASHSCOPE_API_KEY。请在项目根目录的 .env 文件中填入：\n"
            "       DASHSCOPE_API_KEY=你的百炼APIKey\n"
            "       （Key 获取：https://help.aliyun.com/zh/model-studio/get-api-key）"
        )

    cfg_path = ROOT_DIR / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError("未找到 config.yaml。请在项目根目录准备配置文件。")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml 不是合法的 YAML：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml 内容为空或结构不正确。")

    for required in ("provider", "model", "voices"):
        if required not in raw:
            raise ConfigError(f"config.yaml 缺少必填字段：{required}")

    provider = str(raw["provider"]).strip()
    if provider not in ("qwen", "qwen_audio"):
        raise ConfigError(f"provider 暂只支持 qwen / qwen_audio（当前为：{provider}）。")

    model = str(raw["model"]).strip()
    if not model:
        raise ConfigError("model 不能为空。")

    # 阶段 4：provider/模型必须匹配同一端点；CosyVoice/MiniMax 显式拒绝（待实现）
    caps.validate_provider_model(provider, model)

    voices_raw = raw["voices"]
    if not isinstance(voices_raw, dict) or not voices_raw:
        raise ConfigError("voices 不能为空，至少为一个角色配置音色。")
    voices: dict[str, VoiceSetting] = {}
    for name, entry in voices_raw.items():
        if not isinstance(entry, dict) or not str(entry.get("voice", "")).strip():
            raise ConfigError(f"voices.{name} 缺少有效的 voice 值。")
        unknown_voice_keys = set(entry) - {
            "voice", "language_type", "language_hint", "custom", "ssml",
            "display_name", "instruction", "rate", "pitch", "volume",
        }
        if unknown_voice_keys:
            raise ConfigError(
                f"voices.{name} 存在未知配置项：{sorted(unknown_voice_keys)}"
                "（允许：voice/language_type/language_hint/custom/ssml/"
                "display_name/instruction/rate/pitch/volume）。"
            )
        language_type = entry.get("language_type")
        language_hint = entry.get("language_hint")
        if language_type and language_hint:
            raise ConfigError(
                f"voices.{name} 不能同时配置 language_type 与 language_hint（两者语义重叠，"
                "会引起歧义）：旧配置用 language_type: English/Chinese；"
                "精确指定请改用 language_hint: <16 语言代码之一>。"
            )
        hint_value = None
        if language_hint:
            hint_value = caps.validate_language_hint(language_hint)
        custom_flag = entry.get("custom", False)
        if not isinstance(custom_flag, bool):
            raise ConfigError(f"voices.{name}.custom 必须是 true/false（当前为：{custom_flag!r}）。")
        ssml_flag = entry.get("ssml")
        if ssml_flag is not None and not isinstance(ssml_flag, bool):
            raise ConfigError(f"voices.{name}.ssml 必须是 true/false（当前为：{ssml_flag!r}）。")
        instruction_value = entry.get("instruction")
        if instruction_value is not None and not str(instruction_value).strip():
            instruction_value = None
        role_rate = caps.validate_rate(entry["rate"]) if entry.get("rate") is not None else None
        role_pitch = caps.validate_pitch(entry["pitch"]) if entry.get("pitch") is not None else None
        role_volume = caps.validate_volume(entry["volume"]) if entry.get("volume") is not None else None
        voices[str(name).strip()] = VoiceSetting(
            voice=str(entry["voice"]).strip(),
            language_type=str(language_type).strip() if language_type else None,
            language_hint=hint_value,
            custom=custom_flag,
            ssml=ssml_flag,
            display_name=str(
                entry.get("display_name") or str(name).strip()
            ).strip(),
            instruction=str(instruction_value).strip() if instruction_value else None,
            rate=role_rate,
            pitch=role_pitch,
            volume=role_volume,
        )

    # 音色必须属于当前模型（仅 qwen_audio；官方禁止跨模型混用）
    if provider == "qwen_audio":
        for name, setting in voices.items():
            caps.validate_voice(model, setting.voice, setting.custom)
    else:
        # provider=qwen：跨族音色硬拒绝（longan*/_v3.1/_v3.6 是 qwen_audio 音色）
        for name, setting in voices.items():
            caps.validate_qwen_voice(model, setting.voice)

    gen = _section(raw, "generate")
    max_concurrency = _positive_int(gen, "max_concurrency", 3, maximum=20)
    request_timeout = _positive_int(gen, "request_timeout", 60)
    max_retries = _non_negative_int(gen, "max_retries", 3)
    retry_base_delay = _positive_float(gen, "retry_base_delay", 2.0)

    qwen = _section(raw, "qwen")
    optimize_instructions = bool(qwen.get("optimize_instructions", True))
    qwen_base_url = str(qwen.get("base_url") or _QWEN_DEFAULT_BASE_URL).strip()
    qwen_audio_base_url = str(
        qwen.get("audio_base_url") or _QWEN_AUDIO_DEFAULT_BASE_URL
    ).strip()

    audio_params = _parse_qwen_audio_section(raw, provider, model, voices)

    qa_section = raw.get("qwen_audio") if isinstance(raw.get("qwen_audio"), dict) else {}
    default_instruction = qa_section.get("instruction")
    if default_instruction is not None:
        default_instruction = str(default_instruction).strip() or None

    merge = _section(raw, "merge")
    merge_format = str(merge.get("format", "mp3")).strip().lower()
    if merge_format not in ("mp3", "wav"):
        raise ConfigError("merge.format 仅支持 mp3 或 wav。")
    bitrate = str(merge.get("bitrate", "192k")).strip()

    paths = _section(raw, "paths")
    scripts_dir = ROOT_DIR / str(paths.get("scripts_dir", "scripts"))
    projects_dir = ROOT_DIR / str(paths.get("projects_dir", "projects"))
    logs_dir = ROOT_DIR / str(paths.get("logs_dir", "logs"))
    # FFmpeg 目录：随包 > 用户目录配置 > PATH（音频模块负责最终回退）
    ffmpeg_dir = ROOT_DIR / str(paths.get("ffmpeg_dir", "tools/bin"))
    if is_frozen() and bundled_ffmpeg_dir() is not None:
        ffmpeg_dir = bundled_ffmpeg_dir()

    return AppConfig(
        provider=provider,
        model=model,
        voices=voices,
        max_concurrency=max_concurrency,
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        optimize_instructions=optimize_instructions,
        qwen_base_url=qwen_base_url,
        qwen_audio_base_url=qwen_audio_base_url,
        audio_params=audio_params,
        merge_format=merge_format,
        bitrate=bitrate,
        scripts_dir=scripts_dir,
        projects_dir=projects_dir,
        logs_dir=logs_dir,
        ffmpeg_dir=ffmpeg_dir,
        api_key=api_key or None,
        default_instruction=default_instruction,
    )


def _parse_qwen_audio_section(
    raw: dict, provider: str, model: str, voices: dict[str, VoiceSetting]
) -> QwenAudioParams:
    """解析 qwen_audio 段（仅 qwen_audio provider 允许非空配置）。"""
    section = raw.get("qwen_audio")
    if section is None:
        return QwenAudioParams()
    if not isinstance(section, dict):
        raise ConfigError("config.yaml 中 qwen_audio 应为映射结构。")

    enable_ssml = bool(section.get("enable_ssml", False))
    if not isinstance(section.get("enable_ssml", False), bool):
        raise ConfigError("qwen_audio.enable_ssml 必须是 true/false。")

    if provider == "qwen":
        non_empty = [k for k, v in section.items() if v not in (None, False)]
        if non_empty:
            raise ConfigError(
                "qwen_audio 段是 qwen_audio provider（Qwen-Audio-TTS 端点）的专属参数，"
                "与 provider: qwen（multimodal-generation 端点）不通用；当前 provider 为 qwen，"
                f"请删除该段或切换 provider（发现生效项：{sorted(non_empty)}）。"
            )

    allowed = {
        "format", "sample_rate", "rate", "pitch", "volume",
        "seed", "language_hint", "hot_fix", "enable_ssml", "instruction",
    }
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(
            f"qwen_audio 段存在未知配置项：{sorted(unknown)}（仅允许 {sorted(allowed)}）。"
        )

    fmt = caps.validate_format(section.get("format", "wav"))
    sample_rate = caps.validate_sample_rate(section.get("sample_rate", 24000), fmt)
    rate = caps.validate_rate(section["rate"]) if section.get("rate") is not None else None
    pitch = caps.validate_pitch(section["pitch"]) if section.get("pitch") is not None else None
    volume = caps.validate_volume(section["volume"]) if section.get("volume") is not None else None
    seed = caps.validate_seed(section["seed"]) if section.get("seed") is not None else None
    language_hint = (
        caps.validate_language_hint(section["language_hint"])
        if section.get("language_hint") is not None else None
    )
    hot_fix = (
        caps.validate_hot_fix(section["hot_fix"])
        if section.get("hot_fix") is not None else None
    )

    if enable_ssml:
        # 官方 2026-09-18 页仅确认 3 个 qwen-audio 模型支持 SSML；
        # 未知模型不发 enable_ssml，避免未核实行为
        if caps.get_capabilities(model) is None:
            raise ConfigError(
                f"enable_ssml=true 仅在已核实支持 SSML 的模型上开放（当前模型 {model!r} "
                "不在官方确认列表）。请改用 qwen-audio-3.0-tts-plus/flash 或 3.1-tts-flash。"
            )
        for name, setting in voices.items():
            if not setting.custom:
                log.warning(
                    f"角色 {name} 使用系统音色 {setting.voice!r} 且开启了 SSML：官方说明"
                    " SSML 限复刻音色及标注支持 SSML 的系统音色，该音色是否支持需进一步确认，"
                    "若 API 报错请改用 custom: true 的复刻音色。"
                )

    return QwenAudioParams(
        format=fmt, sample_rate=sample_rate, rate=rate, pitch=pitch,
        volume=volume, seed=seed, language_hint=language_hint, hot_fix=hot_fix,
        enable_ssml=enable_ssml,
    )


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config.yaml 中 {name} 应为映射结构。")
    return value


def _positive_int(section: dict, key: str, default: int, maximum: int | None = None) -> int:
    value = section.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} 必须是整数（当前为：{value!r}）。") from None
    if value <= 0:
        raise ConfigError(f"{key} 必须大于 0。")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} 不能超过 {maximum}。")
    return value


def _non_negative_int(section: dict, key: str, default: int) -> int:
    value = section.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} 必须是整数（当前为：{value!r}）。") from None
    if value < 0:
        raise ConfigError(f"{key} 不能为负数。")
    return value


def _positive_float(section: dict, key: str, default: float) -> float:
    value = section.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} 必须是数字（当前为：{value!r}）。") from None
    if value <= 0:
        raise ConfigError(f"{key} 必须大于 0。")
    return value
