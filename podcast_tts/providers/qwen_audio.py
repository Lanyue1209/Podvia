"""Qwen-Audio-TTS Provider：阿里云百炼 qwen-audio-3.x-tts 系列（非实时 HTTP）。

与 Qwen-TTS（qwen.py）的差异（官方《非实时语音合成 HTTP API》，更新 2026-09-28）：
- 端点：/api/v1/services/audio/tts/SpeechSynthesizer（与 multimodal-generation 不通用）
- 请求体：text/voice/format/sample_rate/rate/pitch/volume/seed/hot_fix/
  language_hints/instruction 全部位于 input 内
- 音频参数由 capabilities.resolve_input_extras 统一解析（能力矩阵 + 启动期校验），
  经 TTSRequest.audio_params 传入；缺省可选键不发送
- 导演指令：该系列原生支持，原文传入 input.instruction（无数值化、无白名单概念）
- 未知模型：audio_params 中只有基线字段（format/sample_rate），不静默发送可选参数
- 响应结构与 Qwen-TTS 一致（request_id / output.audio.url / usage），直接复用基类

成功/下载/错误分类逻辑全部继承 QwenProvider（架构约束：新增 Provider 只加适配器）。
"""
from __future__ import annotations

from podcast_tts.config import AppConfig
from podcast_tts.capabilities import resolve_input_extras
from podcast_tts.models import TTSRequest
from podcast_tts.providers.qwen import QwenProvider


class QwenAudioProvider(QwenProvider):
    """百炼 Qwen-Audio-TTS 适配器。"""

    name = "qwen_audio"

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg)
        self._base_url = cfg.qwen_audio_base_url  # 覆盖为专用端点
        self._audio_params = cfg.audio_params

    def _build_payload(self, request: TTSRequest) -> dict:
        """构造 Qwen-Audio-TTS 请求体。指令与文本均原文传入，绝不改写。"""
        if request.audio_params:
            # pipeline 路径：参数已按能力矩阵解析并与 content_hash 同源
            extras = dict(request.audio_params)
        else:
            # 直接构造请求（如单元测试）：按配置现场解析，保证与 pipeline 同一逻辑
            extras = resolve_input_extras(
                self._audio_params, self.model,
                language_type=request.language_type, language_hint=None,
            )

        input_data: dict = {"text": request.text, "voice": request.voice}
        input_data.update(extras)
        if request.direction:
            input_data["instruction"] = request.direction  # 原文传入，非数值参数
        return {"model": self.model, "input": input_data}
