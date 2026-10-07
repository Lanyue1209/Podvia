"""Qwen-Audio-TTS 模型能力矩阵与参数校验（阶段 2）。

数据一律来自阿里云官方文档（2026-10-01 核查），不凭名称猜测：
- HTTP API 参数：https://help.aliyun.com/zh/model-studio/qwen-audio-tts-http-api
  （官方更新时间 2026-09-28；仅华北2-北京地域）
- 音色列表：https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list
  （官方更新时间 2026-09-29；原文：音色必须属于当前模型，不能跨模型混用，
  不匹配时服务返回 InvalidParameter）

证据分级：
- 【官方】官方页面明确列出
- 【仓库现状】仓库既有实现/实测记录
- 【推断】接口级文档未逐模型区分，按统一接口行为推定，待真实调用确认
- 【需进一步确认】官方未给出，禁止猜测
"""
from __future__ import annotations

from dataclasses import dataclass, field

from podcast_tts import log


def _config_error(msg: str):
    """延迟导入避免 config ↔ capabilities 循环依赖。"""
    from podcast_tts.config import ConfigError
    return ConfigError(msg)

# —— 官方常量（HTTP API 页 2026-09-28） ————————————————————————————————

# language_hints 合法值（官方明确列出 16 个；官方说明数组仅处理第一个元素）
LANGUAGE_HINTS: frozenset[str] = frozenset({
    "zh", "en", "fr", "de", "ja", "ko", "ru", "pt",
    "th", "id", "vi", "es", "it", "ms", "fil", "ar",
})

# 官方 sample_rate 枚举
SAMPLE_RATES: frozenset[int] = frozenset({
    8000, 12000, 16000, 22050, 24000, 44100, 48000,
})

# API 支持 mp3/pcm/wav/opus；Podvia 本阶段产品限制只放行 wav/mp3
# （pcm 为无头裸流、opus 容器支持均未在合并管线验证；不代表 API 不支持）
PODVIA_FORMATS: frozenset[str] = frozenset({"wav", "mp3"})

# 旧配置 language_type → language_hints 短码（仓库现状行为，必须保持）
LEGACY_LANGUAGE_MAP: dict[str, str] = {"English": "en", "Chinese": "zh"}

# 可选数值参数名（未知模型一律不发送）
_OPTIONAL_NUMERIC_KEYS = ("rate", "pitch", "volume", "seed")
_OPTIONAL_KEYS = _OPTIONAL_NUMERIC_KEYS + ("hot_fix", "language_hints")

# 旧 hash 版本：v1 无参数、v2 记录项目级音频参数；
# v3：参数含角色默认级/行内片段级覆盖，且 instruction 按四级层级取最终一条
HASH_VERSION = "v3"


@dataclass(frozen=True)
class ModelCapabilities:
    """一个 Qwen-Audio-TTS 模型的已核实能力。"""

    model: str
    # 【官方】系统音色白名单（声音复刻/声音设计/基础音色不在其中，用 custom: true）
    system_voices: frozenset[str]
    # 音色代际绑定说明（人工核对用）
    voice_generation: str
    # 【官方】接口统一支持的可选参数（三个模型同一份接口文档 → 逐模型差异【推断】）
    supports_optional: frozenset[str] = field(
        default_factory=lambda: frozenset(_OPTIONAL_KEYS)
    )
    supports_instruction: bool = True
    # 【官方】情绪与人声拟声标签（text 内嵌）：仅官方明确列出的模型为 True，
    # 见 podcast_tts.emotags（标签表/来源/核对日期 2026-10-03）
    supports_emotags: bool = False
    # 最大输入长度（字符）；None=官方未注明
    max_text_len: int | None = None
    max_text_evidence: str = ""  # 证据分级说明
    source: str = ""


# 【官方】音色页 2026-09-29 逐字录入
_VOICES_30_PLUS = frozenset({
    "longanlingxin",  # 龙安灵心，女，25，知心温暖音（旗舰，仅2个系统音色）
    "longanlufeng",   # 龙安鲁风，男，25，明亮开朗音
})

_VOICES_30_FLASH = frozenset({
    # 社交陪伴（精品中文）5 个
    "longanfengyue",      # 龙安风悦 女30 自然亲切音（无后缀）
    "longanyuanfei",      # 龙安元妃 女30 高傲妃子音（无后缀）
    "longanlingxi",       # 龙安灵希 女25 可爱甜美音（无后缀）
    "longanxiaoxin",      # 龙安小昕 女22 亲切活泼音（无后缀）
    "longanhuan_v3.6",    # 龙安欢 女25
    # 儿童/角色
    "longjielidou_v3.6",  # 龙杰力豆 男童
    "longpaopao_v3.6",    # 龙泡泡
    "longhuohuo_v3.6",    # 龙火火 男8 顽皮少年音
    "longchuanshu_v3.6",  # 龙川叔 男40 川普大叔音
    # 精品英文 3 个
    "loongmary",          # loongmary 女20 温暖英音（无后缀）
    "loongeva_v3.6",      # loongeva 女28 高智美音
    "loongjohn",          # loongJohn 男28 沉稳亲切美音（实际值小写 john）
})

_VOICES_31_FLASH = frozenset({
    "longanhuan_v3.1", "longanlingxin_v3.1", "longanfengyue_v3.1",
    "longanyuanfei_v3.1", "longjielidou_v3.1", "longanlingxi_v3.1",
    "longhuohuo_v3.1", "longyingtao_v3.1", "longanya_v3.1",
    "longwan_v3.1", "longxing_v3.1", "longhua_v3.1",
    "longhan_v3.1", "longanzhi_v3.1", "longzhe_v3.1",
    "longanyang_v3.1", "longling_v3.1", "longniuniu_v3.1",
    "longshanshan_v3.1", "longpaopao_v3.1", "longyuan_v3.1",
    "longmiao_v3.1", "longsanshu_v3.1", "longanli_v3.1",
    "longanwen_v3.1", "longanlang_v3.1", "longxiaoxia_v3.1",
    "longanchong_v3.1",
})

_API_SRC = "HTTP API 页 help.aliyun.com/zh/model-studio/qwen-audio-tts-http-api（官方更新 2026-09-28）"
_VOICE_SRC = "音色列表页 help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list（官方更新 2026-09-29）"
_EMOTAG_SRC = "情绪/拟声标签：非实时语音合成指南「情感与富语言标签」节（官方更新 2026-09-28，2026-10-03 核对）"

MATRIX: dict[str, ModelCapabilities] = {
    "qwen-audio-3.0-tts-plus": ModelCapabilities(
        model="qwen-audio-3.0-tts-plus",
        system_voices=_VOICES_30_PLUS,
        voice_generation="仅无后缀的 2 个旗舰系统音色；与 flash/3.1 不通用【官方】",
        supports_emotags=True,
        max_text_len=600,
        max_text_evidence="【仓库现状】实测 API 报错 Range of input length should be [0, 600]",
        source=_VOICE_SRC + "；" + _API_SRC + "；" + _EMOTAG_SRC,
    ),
    "qwen-audio-3.0-tts-flash": ModelCapabilities(
        model="qwen-audio-3.0-tts-flash",
        system_voices=_VOICES_30_FLASH,
        voice_generation="无后缀或 _v3.6 后缀的 12 个系统音色；与 plus/3.1 不通用【官方】",
        supports_emotags=True,
        max_text_len=600,
        max_text_evidence="【仓库现状】同接口 600 字符上限；未对 flash 单独实测【需进一步确认】",
        source=_VOICE_SRC + "；" + _API_SRC + "；" + _EMOTAG_SRC,
    ),
    "qwen-audio-3.1-tts-flash": ModelCapabilities(
        model="qwen-audio-3.1-tts-flash",
        system_voices=_VOICES_31_FLASH,
        voice_generation="仅 _v3.1 后缀的 28 个系统音色；与 3.0 代际不通用【官方】",
        supports_emotags=True,
        max_text_len=None,
        max_text_evidence="【需进一步确认】HTTP API 页未注明该模型字符上限"
                          "（注意 3000 字符属于另一模型 qwen-audio-3.1-tts-next，不可混用）",
        source=_VOICE_SRC + "；" + _API_SRC + "；" + _EMOTAG_SRC,
    ),
}


def get_capabilities(model: str) -> ModelCapabilities | None:
    """返回已核实模型能力；未知模型返回 None（调用方走基线降级）。"""
    return MATRIX.get(model)


# —— Provider/模型注册表（阶段 4：配置切换 + GUI 动态控件的唯一事实来源） ——————

# Qwen-TTS（multimodal-generation 端点）模型，证据来自仓库既有实现与技术文档；
# QwenProvider 仅在模型名含 "instruct" 时发送 instructions 字段。
QWEN_TTS_MODELS: frozenset[str] = frozenset({
    "qwen3-tts-instruct-flash",  # 【仓库现状】已实现，支持导演 instructions
    "qwen3-tts-flash",           # 【仓库现状】已实现，不支持 instructions
    "qwen-tts",                  # 【仓库现状】旧模型名，保留兼容
})

# 百炼已上架但 Podvia 尚未实现适配器的模型：显式拒绝，绝不静默尝试。
# CosyVoice 虽与 qwen-audio 共用 SpeechSynthesizer 端点，但音色体系/参数/响应
# 未在本仓库核实；MiniMax 在百炼走另一请求结构。二者 GUI/配置中只能显示"待实现"。
UNIMPLEMENTED_MODEL_PREFIXES: tuple[str, ...] = (
    "cosyvoice-", "minimax/", "speech-",
)


def is_unimplemented_model(model: str) -> bool:
    name = model.lower()
    return any(name.startswith(p) for p in UNIMPLEMENTED_MODEL_PREFIXES)


def validate_provider_model(provider: str, model: str) -> None:
    """启动期校验 provider 与模型匹配，跨端点组合一律 ConfigError。"""
    if is_unimplemented_model(model):
        raise _config_error(
            f"模型 {model!r} 属于百炼已上架但 Podvia 尚未实现的系列"
            "（CosyVoice/MiniMax），当前版本不可选择；在 GUI 中其状态为"
            "“待实现”，请勿通过手改 config 绕过。"
        )
    if provider == "qwen_audio":
        if model in QWEN_TTS_MODELS:
            raise _config_error(
                f"模型 {model!r} 属于 Qwen-TTS（multimodal-generation 端点），"
                "不能与 provider: qwen_audio 组合使用；请改为 provider: qwen。"
            )
        # 矩阵内=已核实；矩阵外=未知模型，允许启动但运行期基线降级
    elif provider == "qwen":
        if model in MATRIX:
            raise _config_error(
                f"模型 {model!r} 属于 Qwen-Audio-TTS（SpeechSynthesizer 端点），"
                "不能与 provider: qwen 组合使用；请改为 provider: qwen_audio。"
            )
    else:
        raise _config_error(f"未知 provider {provider!r}。")


def describe_model(model: str) -> str:
    """GUI 对单个模型名的状态标签。"""
    if model in MATRIX or model in QWEN_TTS_MODELS:
        return "已实现"
    if is_unimplemented_model(model):
        return "待实现"
    return "待确认"  # 未知模型：允许基线尝试，但能力未核实


def validate_qwen_voice(model: str, voice: str) -> None:
    """provider=qwen（Qwen-TTS 端点）的音色防护。

    Qwen3-TTS 官方音色为 Cherry/Ethan/Serena 等名字（完整清单以官方
    qwen-tts-api 页为准，GUI 标注待确认）；本函数只做"跨族硬拒绝"：
    qwen_audio 体系的 longan* / *_v3.1 / *_v3.6 音色绝不允许带到 qwen provider，
    防止模型切换后沿用错误音色导致云端报错或错音。
    """
    v = voice.lower()
    if v.startswith("longan") or v.endswith("_v3.1") or v.endswith("_v3.6"):
        raise _config_error(
            f"音色 {voice!r} 属于 Qwen-Audio-TTS 模型的系统音色，"
            f"不能用于 Qwen-TTS 模型 {model!r}；模型切换后必须重新选择"
            "该模型支持的音色（如 Cherry/Ethan/Serena，完整清单见官方文档）。"
        )


def list_model_profiles() -> list[dict]:
    """GUI/文档使用的模型清单（唯一事实来源）。

    status: implemented（已实现可选择）/ baseline（未知新模型，仅基线参数）
            / planned（待实现，禁止选择）。
    """
    profiles: list[dict] = []
    optional = sorted(_OPTIONAL_KEYS)
    for model, caps_obj in MATRIX.items():
        profiles.append({
            "provider": "qwen_audio",
            "model": model,
            "status": "implemented",
            "voices": sorted(caps_obj.system_voices),
            "voice_generation": caps_obj.voice_generation,
            "supports": {
                "instruction": caps_obj.supports_instruction,
                "rate": "rate" in caps_obj.supports_optional,
                "pitch": "pitch" in caps_obj.supports_optional,
                "volume": "volume" in caps_obj.supports_optional,
                "seed": "seed" in caps_obj.supports_optional,
                "hot_fix": "hot_fix" in caps_obj.supports_optional,
                "language_hints": "language_hints" in caps_obj.supports_optional,
                "ssml": True,  # 官方 2026-09-18 页确认三个模型均支持
                "emotags": caps_obj.supports_emotags,  # 【官方】见 emotags 模块来源
            },
            "optional_params": optional,
            "max_text_len": caps_obj.max_text_len,
            "max_text_evidence": caps_obj.max_text_evidence,
            "source": caps_obj.source,
        })
    for model in sorted(QWEN_TTS_MODELS):
        profiles.append({
            "provider": "qwen",
            "model": model,
            "status": "implemented",
            "voices": [],  # Qwen-TTS 音色体系不同，由 qwen 文档/配置自行指定
            "voice_generation": "Qwen-TTS 音色（如 Cherry/Ethan/Serena），不做代际白名单【仓库现状】",
            "supports": {
                "instruction": "instruct" in model,
                "rate": False, "pitch": False, "volume": False,
                "seed": False, "hot_fix": False, "language_hints": False,
                "ssml": False, "emotags": False,  # 官方标签仅 Qwen-Audio-TTS 三模型
            },
            "optional_params": [],
            "max_text_len": 512,
            "max_text_evidence": "【官方】Qwen-TTS 为 512 Token（qwen-tts-api 页）",
            "source": "help.aliyun.com/zh/model-studio/qwen-tts-api",
        })
    profiles.append({
        "provider": None,
        "model": "cosyvoice-* (v2/v3/v3.5 flash/plus)",
        "status": "planned",
        "note": "待实现：百炼 CosyVoice 系列（SpeechSynthesizer 端点，音色体系独立）。",
    })
    profiles.append({
        "provider": None,
        "model": "MiniMax/speech-2.8-hd 等",
        "status": "planned",
        "note": "待实现：百炼 MiniMax 系列（请求结构不同）。",
    })
    return profiles


def validate_voice(model: str, voice: str, custom: bool) -> None:
    """启动期音色校验（仅 qwen_audio provider）。

    - custom=True 只跳过系统音色白名单；voice 非空等基础校验已在 config 层完成，
      本函数不负责云端验证该 ID 是否真实存在（离线无法验证，不存在时 API 返回
      InvalidParameter，阶段 1 已将其映射为 Fatal）。
    - 未知模型：无法建立白名单，跳过校验（不阻断新模型使用），由基线策略兜底。
    """
    status = voice_status("qwen_audio", model, voice, custom)
    if status == "invalid":
        raise _config_error(
            f"音色 {voice!r} 不属于模型 {model!r} 的系统音色列表（官方禁止跨模型混用，"
            "否则返回 InvalidParameter）。请改用该模型的音色，或在该角色配置中显式声明 "
            "custom: true（声音复刻/声音设计/基础音色）。"
        )


def system_voice_count(model: str) -> int | None:
    """已核实模型的系统音色数量（GUI 分配角色用）；未知模型返回 None。

    注意：该数字只反映官方音色页（见 MATRIX[model].source 记录的核查日期），
    不是永久事实；custom voice 不占系统音色名额，但云端是否存在无法离线验证。
    """
    caps = MATRIX.get(model)
    return None if caps is None else len(caps.system_voices)


def voice_status(provider: str, model: str, voice: str, custom: bool) -> str:
    """GUI 模型切换/开始前的逐角色音色判定（不弹窗、不改写音色）：

    - "empty"：未选择（必须补选，绝不静默替成第一个音色）
    - "ok"：属于当前模型系统音色
    - "custom"：用户声明 custom，跳过白名单（云端存在性无法离线验证）
    - "invalid"：确定不兼容（跨模型系统音色/跨族带到 qwen provider）→ 需重新选择
    - "unknown"：模型/音色体系未离线核实，无法判定，允许使用
    """
    if not (voice or "").strip():
        return "empty"
    if provider == "qwen_audio":
        caps = MATRIX.get(model)
        if caps is None:
            return "unknown"
        if custom:
            return "custom"
        return "ok" if voice in caps.system_voices else "invalid"
    # provider == "qwen"：只掌握"跨族硬拒绝"规则，其余音色无法离线核实
    try:
        validate_qwen_voice(model, voice)
    except Exception:  # noqa: BLE001 —— 仅做状态归类
        return "invalid"
    return "unknown"


# —— 配置值校验（范围/类型全部以官方文档为准） ————————————————————————————


def _bounded_float(name: str, value, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise _config_error(f"qwen_audio.{name} 必须是数字（当前为：{value!r}）。") from None
    if not (low <= v <= high):
        raise _config_error(f"qwen_audio.{name} 取值范围为 [{low}, {high}]（当前为：{v}）。")
    return v


def _bounded_int(name: str, value, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _config_error(f"qwen_audio.{name} 必须是整数（当前为：{value!r}）。")
    if not (low <= value <= high):
        raise _config_error(f"qwen_audio.{name} 取值范围为 [{low}, {high}]（当前为：{value}）。")
    return value


def validate_rate(value) -> float:
    return _bounded_float("rate", value, 0.5, 2.0)


def validate_pitch(value) -> float:
    return _bounded_float("pitch", value, 0.5, 2.0)


def validate_volume(value) -> int:
    return _bounded_int("volume", value, 0, 100)


def validate_seed(value) -> int:
    return _bounded_int("seed", value, 0, 65535)


def validate_format(value: str) -> str:
    fmt = str(value).strip().lower()
    if fmt in PODVIA_FORMATS:
        return fmt
    if fmt in ("pcm", "opus"):
        raise _config_error(
            f"qwen_audio.format={fmt} 虽为百炼 API 支持的格式，但 Podvia 当前阶段的"
            "下载探测与合并管线仅验证过 wav/mp3，暂不开放（产品限制，不代表 API 不支持）。"
        )
    raise _config_error(f"qwen_audio.format 仅支持 wav / mp3（当前为：{value!r}）。")


def validate_sample_rate(value, fmt: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _config_error(f"qwen_audio.sample_rate 必须是整数（当前为：{value!r}）。")
    if value not in SAMPLE_RATES:
        rates = "/".join(str(r) for r in sorted(SAMPLE_RATES))
        raise _config_error(f"qwen_audio.sample_rate 仅支持 {rates}（当前为：{value}）。")
    return value


def validate_language_hint(value) -> str:
    hint = str(value).strip().lower()
    if hint not in LANGUAGE_HINTS:
        raise _config_error(
            f"qwen_audio.language_hint 必须是官方 16 个语言代码之一（当前为：{value!r}）。"
        )
    return hint


def validate_hot_fix(value) -> dict:
    """校验 hot_fix 严格同构官方示例（HTTP API 页 2026-09-28）：

    官方 JSON：
        "hot_fix": {
          "pronunciation": [ {"天气": "tian1 qi4"} ],
          "replace":       [ {"今天": "金天"} ]
        }
    规则：仅允许 pronunciation/replace 两键；值必须为非空数组；
    每个元素必须是恰好 1 个键值对、键与值均为非空字符串的对象。
    """
    if not isinstance(value, dict) or not value:
        raise _config_error(
            "qwen_audio.hot_fix 必须是非空映射，结构同官方示例："
            "{pronunciation: [{词: 拼音}], replace: [{词: 替换词}]}。"
        )
    unknown = set(value) - {"pronunciation", "replace"}
    if unknown:
        raise _config_error(
            f"qwen_audio.hot_fix 仅允许 pronunciation / replace 键（发现：{sorted(unknown)}）。"
        )
    normalized: dict = {}
    for key in ("pronunciation", "replace"):
        if key not in value:
            continue
        items = value[key]
        if not isinstance(items, list) or not items:
            raise _config_error(f"qwen_audio.hot_fix.{key} 必须是非空数组。")
        clean_items: list[dict[str, str]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or len(item) != 1:
                raise _config_error(
                    f"qwen_audio.hot_fix.{key}[{i}] 必须是恰好含 1 个键值对的对象"
                    "（同官方示例 {词语: 目标字符串}）。"
                )
            word, target = next(iter(item.items()))
            if not isinstance(word, str) or not word.strip():
                raise _config_error(f"qwen_audio.hot_fix.{key}[{i}] 的词必须是非空字符串。")
            if not isinstance(target, str) or not target.strip():
                raise _config_error(f"qwen_audio.hot_fix.{key}[{i}] 的目标值必须是非空字符串。")
            clean_items.append({word: target})
        normalized[key] = clean_items
    return normalized


# —— 运行期：解析每个片段实际发送的 input 附加参数 ————————————————————————


def resolve_input_extras(
    audio_params,
    model: str,
    *,
    language_type: str | None,
    language_hint: str | None,
) -> dict:
    """计算除 text/voice/instruction 外要放入 input 的字段（不含时键不存在）。

    - 已核实模型：发送 format/sample_rate + 已配置的可选参数 + language_hints。
    - 未知模型：仅 format/sample_rate 基线字段，可选参数（含 language_hints）
      一律不发送，并 WARNING（不静默假装完整支持）。
    - language 优先级：voice 级 language_hint > 全局 audio_params.language_hint
      > 旧 language_type 映射；language_type 未识别时省略并 WARNING（沿用仓库现状）。

    注意命名：language_hint 是 Podvia 配置名（单值短码），API 字段仍是 language_hints。
    """
    caps = MATRIX.get(model)
    extras: dict = {
        "format": audio_params.format,
        "sample_rate": audio_params.sample_rate,
    }

    if caps is None:
        log.warning(
            f"模型 {model!r} 不在已核实的 Qwen-Audio-TTS 能力矩阵中，"
            "仅发送基线参数（format/sample_rate/instruction），"
            "rate/pitch/volume/seed/hot_fix/language_hints 均不发送。"
        )
        return extras

    for key in ("rate", "pitch", "volume", "seed"):
        val = getattr(audio_params, key)
        if val is not None:
            extras[key] = val
    if audio_params.hot_fix:
        extras["hot_fix"] = audio_params.hot_fix

    # language_hints（API 数组字段；官方仅处理首元素，故只传单元素数组）
    # 优先级：角色级 language_hint > 全局 qwen_audio.language_hint > 旧 language_type 映射
    effective_hint = language_hint or audio_params.language_hint
    if effective_hint:
        extras["language_hints"] = [effective_hint]
    elif language_type:
        mapped = LEGACY_LANGUAGE_MAP.get(language_type)
        if mapped:
            extras["language_hints"] = [mapped]
        else:
            log.warning(f"未知 language_type={language_type!r}，已省略 language_hints。")
    return extras

