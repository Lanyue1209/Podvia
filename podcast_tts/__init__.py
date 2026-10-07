"""Podvia 通用语音合成工具。

模块布局：
- log.py      日志初始化与便捷函数
- config.py   配置加载与校验
- models.py   Segment / SegmentRecord / TTSRequest / TTSResult
- parser.py   脚本解析器
- book.py     章节文件导入（可选输入方式）与安全分段
- providers/  TTS Provider 抽象与 Qwen 实现
- pipeline.py 批量生成编排
- audio.py    静音/探测/合并
"""
