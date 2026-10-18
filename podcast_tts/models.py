"""核心数据结构：脚本片段、manifest 记录、TTS 请求/结果。

约定：parser 产出 Segment；pipeline 维护 SegmentRecord 并序列化进 manifest.json；
providers 只感知 TTSRequest/TTSResult（与脚本概念解耦）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    """Parser 产出：脚本顺序中的一个最小单元（台词或停顿）。"""

    index: int                          # 1-based 显示序号（含 PAUSE），插入内容后会重排
    kind: str                           # "speech" | "pause"
    seg_id: str | None = None           # 稳定片段 ID（reconcile 时分配/沿用，不随序号变化）
    speaker: str | None = None          # 仅 speech：角色稳定 ID（CLI 下即角色名）
    speaker_name: str | None = None     # 角色显示名（日志/GUI 用，可为中文）
    direction: str | None = None        # 行内导演指令原文（最高优先级，不改写）
    chapter_direction: str | None = None  # 章节级导演指令（章节导入）
    # 行内片段覆盖参数（写在角色括号内，如 Lily(轻快地; rate=1.2):）
    rate: float | None = None
    pitch: float | None = None
    volume: int | None = None
    text: str | None = None             # 清洗后的纯文本（仅去除 Markdown 格式符）
    raw_text: str | None = None         # 清洗前原文（仅存档核对用）
    # 长台词自动分段时，为保持情绪连续而在本片段开头补入的控制类标签（如 "[excited] "）；
    # 该前缀已包含在 text 中并参与缓存 hash，此字段仅用于预览中与原文标签区分显示
    injected_prefix: str | None = None
    pause_seconds: float | None = None  # 仅 pause：停顿时长（秒）
    # 章节导入：原文追溯坐标（相对章节源文件的字符偏移，切片即原文）
    chapter: str | None = None          # 章节标题
    source_file: str | None = None      # 来源文件（相对项目根或绝对路径）
    char_start: int | None = None       # 原文起始偏移（含）
    char_end: int | None = None         # 原文结束偏移（不含）
    # 片段来源形态：None=脚本对白；"dialogue"=章节内显式角色对白；
    # "prose"=章节普通正文（由章节默认角色朗读）。仅用于预览/日志展示
    origin: str | None = None
    # 文本模式（片段级）：None="auto" 按内容判断；"plain" 强制普通文本；
    # "ssml" 强制 SSML（校验失败该片段报错）。见 ssml.resolve_text_mode
    text_mode: str | None = None


@dataclass
class SegmentRecord:
    """manifest.json 中每个片段的生成记录（断点续生成/缓存/排错的唯一依据）。"""

    index: int
    kind: str                           # "speech" | "pause"
    seg_id: str | None = None           # 稳定片段 ID（与 Segment.seg_id 对应，跨重解析不变）
    speaker: str | None = None          # 角色稳定 ID
    speaker_name: str | None = None     # 角色显示名
    direction: str | None = None        # 行内导演指令
    chapter_direction: str | None = None  # 章节级导演指令
    rate: float | None = None           # 行内覆盖参数快照
    pitch: float | None = None
    volume: int | None = None
    text: str | None = None             # 发送给 TTS 的最终文本
    pause_seconds: float | None = None
    status: str = "pending"             # pending | success | failed
    file: str | None = None             # 相对 episode 目录，如 "segments/0001_Lily.wav"
    content_hash: str | None = None     # SHA256(v2 公式，见 manifest.content_hash)
    audio_params: dict = field(default_factory=dict)  # 实际发送的音频参数快照（v2）
    attempts: int = 0                   # 累计尝试次数（含重试）
    request_id: str | None = None       # DashScope request_id，便于排查
    error: str | None = None            # 最近一次失败原因
    # 阶段 5：原文追溯（旧 manifest 缺失这些键时按 None 加载）
    chapter: str | None = None
    source_file: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    # 片段级文本模式（auto/plain/ssml；None=auto。旧 manifest 缺失按 None 加载）
    text_mode: str | None = None


@dataclass
class TTSRequest:
    """发给 Provider 的统一请求（Provider 抽象层的语言，与脚本解耦）。"""

    text: str
    voice: str                          # Voice ID，如 "Cherry"
    language_type: str | None           # "English" | "Chinese" | None(=Auto)
    direction: str | None               # 导演指令原文
    output_path: str                    # 音频落盘路径
    # 实际发送给 API 的音频附加参数快照（format/sample_rate/rate/.../language_hints）；
    # 由 pipeline 经 capabilities.resolve_input_extras 统一解析，与 content_hash 同源
    audio_params: dict = field(default_factory=dict)


@dataclass
class TTSResult:
    """Provider 的统一返回。失败时 Provider 抛 ProviderError，不返回本对象。"""

    output_path: str
    request_id: str | None
    provider_data: dict = field(default_factory=dict)  # 厂商补充信息（usage 等）
