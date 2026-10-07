"""Qwen TTS Provider：阿里云百炼 qwen3-tts 系列（非流式 HTTP）。

职责链：调 API → 下载音频 → 原子落盘。
- 不做重试（pipeline 统一调度）；异常的 retryable 属性是重试判断的唯一依据
- 导演指令原文传入 input.instructions，绝不改写、绝不拼进文本
- 仅 instruct 系列模型支持 instructions；其他模型忽略指令并记录 WARNING
- 音频 URL 有效期 24 小时，响应后立即下载；先写 .part 再原子改名（防半截文件）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn

import requests

from podcast_tts import log
from podcast_tts.config import AppConfig
from podcast_tts.models import TTSRequest, TTSResult
from podcast_tts.providers.base import (
    FatalProviderError,
    RetryableError,
    TTSProvider,
)

_DOWNLOAD_TIMEOUT = 120  # 音频下载超时（秒）；URL 有效期 24h，但 CDN 偶发较慢
_CHUNK_SIZE = 64 * 1024


class QwenProvider(TTSProvider):
    """百炼 Qwen-TTS 适配器（端点与模型名均来自 config.yaml）。"""

    name = "qwen"

    def __init__(self, cfg: AppConfig) -> None:
        if not cfg.api_key:
            raise FatalProviderError("缺少 DASHSCOPE_API_KEY，无法调用百炼 API。")
        self.model = cfg.model
        self._api_key = cfg.api_key
        self._base_url = cfg.qwen_base_url
        self._timeout = cfg.request_timeout
        self._optimize_instructions = cfg.optimize_instructions
        # instruct 系列模型才支持 instructions（文档第 10 节）
        self._supports_instructions = "instruct" in cfg.model.lower()

    # —— 对外主流程 ——————————————————————————————————————————————

    def generate(self, request: TTSRequest) -> TTSResult:
        if not request.text or not request.text.strip():
            raise FatalProviderError("台词文本为空，拒绝调用 API。")

        try:
            resp = requests.post(
                self._base_url,
                json=self._build_payload(request),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise RetryableError(f"网络请求失败：{exc}") from exc

        body = _safe_json(resp)
        request_id = body.get("request_id") if isinstance(body, dict) else None

        if resp.status_code != 200:
            self._raise_error(
                status=resp.status_code,
                code=str((body or {}).get("code") or ""),
                message=str((body or {}).get("message") or resp.text[:200]),
                request_id=request_id,
                retry_after=resp.headers.get("Retry-After"),
            )

        if not isinstance(body, dict):
            raise RetryableError(
                "响应异常格式：body 不是合法 JSON 对象。",
                request_id=request_id,
                http_status=200,
            )

        # HTTP 200 但 body 含 code/message → 业务错误（DashScope 风格），按映射表分类
        if body.get("code"):
            self._raise_error(
                status=200,
                code=str(body["code"]),
                message=str(body.get("message") or ""),
                request_id=request_id,
            )

        audio = body.get("output") or {}
        audio_url = (audio.get("audio") or {}).get("url")
        if not audio_url:
            raise RetryableError(
                "响应异常格式：缺少 output.audio.url。",
                request_id=request_id,
                http_status=200,
            )

        self._download(audio_url, Path(request.output_path), request_id)
        log.debug(f"API 响应 request_id={request_id}")
        return TTSResult(
            output_path=request.output_path,
            request_id=request_id,
            provider_data={"usage": body.get("usage"), "audio_url": audio_url},
        )

    # —— 请求构造 ——————————————————————————————————————————————————

    def _build_payload(self, request: TTSRequest) -> dict:
        """构造请求体。导演指令原文传入，绝不改写；可选字段缺省时不发送。"""
        input_data: dict = {"text": request.text, "voice": request.voice}
        if request.language_type:
            input_data["language_type"] = request.language_type
        if request.direction:
            if self._supports_instructions:
                input_data["instructions"] = request.direction  # 原文传入
                if self._optimize_instructions:
                    input_data["optimize_instructions"] = True
            else:
                log.warning(
                    f"模型 {self.model} 不支持导演指令（instructions），"
                    f"已忽略该指令：{request.direction}"
                )
        return {"model": self.model, "input": input_data}

    # —— 错误分类（文档 8.4 映射表） ————————————————————————————————

    def _raise_error(
        self,
        *,
        status: int,
        code: str,
        message: str,
        request_id: str | None,
        retry_after: str | None = None,
    ) -> NoReturn:
        """HTTP 层与 body 层（HTTP 200 但含 code）错误共用此映射。

        判断顺序：先 body 业务错误码、后 HTTP 状态码。这样在状态码与业务
        语义冲突时（如官方错误表中 Arrearage 欠费以 HTTP 400 返回、个别
        限流也可能挂 400）仍能按业务语义正确分类。

        业务码依据百炼官方错误码表（2026-10 核实）：
        - 限流：Throttling / RateLimit*（429，官方限流页）
        - 欠费：Arrearage（官方错误码表中常以 HTTP 400 返回）
        - 免费额度耗尽：403 AllocationQuota.FreeTierOnly（用完即停开启时）
        - 鉴权：401 InvalidApiKey
        - 权限/未开通：403 AccessDenied 系列（Unpurchased=服务未开通，非欠费）
        - 参数：400 InvalidParameter*
        - 模型/空间不存在：404 ModelNotFound / model_not_found / WorkSpaceNotFound
        注意：不对 message 做关键词猜测分类（如限流提示 "Allocated quota
        exceeded" 含 quota 字样但属 429 可重试），避免误判。未知 404（无官方
        业务码）不扩大匹配，保守落入可重试兜底。
        """
        code_norm = code.strip().lower()
        label = f"HTTP {status}" + (f"，code={code}" if code else "")

        # —— 1) 业务错误码优先于 HTTP 状态码 ——
        if "throttling" in code_norm or "ratelimit" in code_norm:
            raise _throttle_error(
                label, message, retry_after,
                request_id=request_id, http_status=status,
            )
        if code_norm == "arrearage":
            raise FatalProviderError(
                f"账号欠费或余额不足（{label}）：{message}。"
                "请登录阿里云百炼控制台「费用与成本」检查账户状态并充值，"
                "充值后等待数分钟生效。",
                request_id=request_id, http_status=status,
            )
        if code_norm == "allocationquota.freetieronly":
            raise FatalProviderError(
                f"免费额度已耗尽（{label}）：{message}。"
                "可在百炼控制台转为按量付费（关闭「免费额度用完即停」），"
                "或更换为仍有额度的模型。",
                request_id=request_id, http_status=status,
            )
        if "invalidapikey" in code_norm or "invalid_api_key" in code_norm:
            raise FatalProviderError(
                f"API Key 无效（{label}）：{message}",
                request_id=request_id, http_status=status,
            )
        if "accessdenied" in code_norm:
            if "unpurchased" in code_norm:
                raise FatalProviderError(
                    f"模型服务未开通或当前账号无使用资格（{label}）：{message}。"
                    "请前往百炼控制台模型广场开通对应模型服务。",
                    request_id=request_id, http_status=status,
                )
            raise FatalProviderError(
                f"无访问权限（{label}）：{message}。"
                "请检查 API Key 所属账号、业务空间与模型授权。",
                request_id=request_id, http_status=status,
            )
        if "invalidparameter" in code_norm:
            raise FatalProviderError(
                f"请求参数错误（{label}）：{message}",
                request_id=request_id, http_status=status,
            )
        if code_norm in ("modelnotfound", "model_not_found") or "workspacenotfound" in code_norm:
            # 官方 404：模型名/业务空间不存在——属配置错误，重试无意义
            raise FatalProviderError(
                f"模型或业务空间不存在（{label}）：{message}。"
                "请核对 config.yaml 的 model 名称（对照百炼模型列表）与 API Key 所属业务空间。",
                request_id=request_id, http_status=status,
            )

        # —— 2) 无业务码或业务码未知：按 HTTP 状态码兜底 ——
        if status == 429:
            raise _throttle_error(
                label, message, retry_after,
                request_id=request_id, http_status=status,
            )
        if status in (401, 403):
            # 403 可能是鉴权、权限、配额等多种原因；无明确业务码时不臆断为欠费
            raise FatalProviderError(
                f"鉴权失败或访问被拒绝（{label}）：{message}。"
                "请检查 API Key 是否有效、账号是否有该模型权限；"
                "若刚充值或刚开通额度，请等待数分钟后再试。",
                request_id=request_id, http_status=status,
            )
        if status == 400:
            raise FatalProviderError(
                f"请求参数错误（{label}）：{message}",
                request_id=request_id, http_status=status,
            )

        # 3) 5xx 及未知错误：保守按可重试处理（pipeline 有最大重试上限兜底）
        raise RetryableError(
            f"请求失败（{label}）：{message}",
            request_id=request_id, http_status=status,
        )

    # —— 音频下载 ——————————————————————————————————————————————————

    def _download(self, url: str, output_path: Path, request_id: str | None) -> None:
        """下载音频：先写 .part 临时文件，校验非空后原子改名（防半截文件）。

        任何失败路径都清理 .part，绝不在 segments/ 留下半截临时文件。
        """
        part_path = output_path.with_name(output_path.name + ".part")
        try:
            with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
                if resp.status_code != 200:
                    raise RetryableError(
                        f"下载音频失败（HTTP {resp.status_code}）。",
                        request_id=request_id,
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with part_path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as exc:
            part_path.unlink(missing_ok=True)
            raise RetryableError(
                f"下载音频失败：{exc}", request_id=request_id
            ) from exc
        except OSError as exc:
            part_path.unlink(missing_ok=True)
            raise RetryableError(
                f"写入音频文件失败：{exc}", request_id=request_id
            ) from exc

        try:
            size = part_path.stat().st_size
        except OSError:
            size = 0
        if size <= 0:
            part_path.unlink(missing_ok=True)
            raise RetryableError(
                "下载的音频文件为空。", request_id=request_id
            )

        try:
            os.replace(part_path, output_path)
        except OSError as exc:
            part_path.unlink(missing_ok=True)
            raise RetryableError(
                f"保存音频文件失败：{exc}", request_id=request_id
            ) from exc
        log.debug(f"音频已下载：{output_path}（{size} 字节）")


def _safe_json(resp: requests.Response) -> dict | None:
    """解析响应 JSON；非法或非对象时返回 None（由调用方按异常格式处理）。"""
    try:
        data = resp.json()
    except (ValueError, AttributeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_retry_after(value: str | None) -> float | None:
    """解析 Retry-After 头（支持秒数；HTTP-date 格式不支持时返回 None）。"""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _throttle_error(
    label: str,
    message: str,
    retry_after: str | None,
    *,
    request_id: str | None,
    http_status: int,
) -> RetryableError:
    """构造限流类可重试异常（业务码与 HTTP 429 两条路径共用）。"""
    seconds = _parse_retry_after(retry_after)
    tail = f"（Retry-After: {retry_after}s）" if retry_after else ""
    return RetryableError(
        f"限流{tail}（{label}）：{message}",
        request_id=request_id, http_status=http_status, retry_after=seconds,
    )
