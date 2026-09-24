"""模型调用客户端。

要求（技术栈手册工程底线）：
- 超时、失败、有限重试必须处理
- SDK 初始化失败要有兜底
- 模型输出必须结构校验，校验失败重试
- 日志不打印密钥
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .prompt import build_messages
from .schema import DetectionResult


@dataclass
class CallRecord:
    """一次模型调用的可追溯记录（不含任何密钥）。"""

    photo: str
    sample_index: int
    model: str
    ok: bool
    attempts: int
    elapsed_seconds: float
    error: str = ""
    raw_text: str = ""


class ModelCallError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """从模型返回里取出 JSON 对象。

    2025 年的 VLM 经常不老实：会包 markdown 代码块、会在 JSON 前后加解释。
    这里做三层兜底，尽量不因为格式问题浪费一次调用。
    """
    if not text or not text.strip():
        raise ModelCallError("模型返回为空")

    cleaned = text.strip()

    # 兜底 1：去掉 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # 兜底 2：直接解析
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 兜底 3：截取第一个 { 到最后一个 } 之间的内容
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            raise ModelCallError(f"JSON 解析失败：{exc}") from exc

    raise ModelCallError("返回内容中找不到 JSON 对象")


class QwenVLClient:
    """百炼（DashScope）Qwen-VL 客户端。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sdk_ready = False
        self._sdk_error = ""
        if not settings.dry_run:
            self._init_sdk()

    def _init_sdk(self) -> None:
        """SDK 初始化失败的兜底：不抛异常中断整个批次，改为标记不可用。"""
        try:
            import dashscope  # noqa: F401

            self._sdk_ready = True
        except Exception as exc:  # pragma: no cover - 依赖缺失时的兜底
            self._sdk_error = f"dashscope SDK 不可用：{exc}"
            self._sdk_ready = False

    # ------------------------------------------------------------------
    def detect(self, photo: Path, sample_index: int = 0) -> tuple[DetectionResult, CallRecord]:
        """识别一张照片。失败会重试，重试耗尽则抛出 ModelCallError。"""
        started = time.time()
        attempts = 0
        last_error = ""
        raw_text = ""

        while attempts < max(1, self.settings.max_retries):
            attempts += 1
            try:
                raw_text = self._call_once(photo)
                payload = _extract_json(raw_text)
                result = DetectionResult.model_validate(payload)
                return result, CallRecord(
                    photo=photo.name,
                    sample_index=sample_index,
                    model=self.settings.model,
                    ok=True,
                    attempts=attempts,
                    elapsed_seconds=round(time.time() - started, 2),
                    raw_text=raw_text,
                )
            except (ValidationError, ModelCallError) as exc:
                # 结构不对或解析失败 → 重试
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # 网络/限流等
                last_error = f"{type(exc).__name__}: {exc}"

            if attempts < self.settings.max_retries:
                time.sleep(min(2 ** attempts, 8))  # 退避

        raise ModelCallError(
            f"{photo.name} 识别失败（重试 {attempts} 次）：{last_error}"
        )

    # ------------------------------------------------------------------
    def _call_once(self, photo: Path) -> str:
        if self.settings.dry_run:
            return _fake_response(photo)

        if not self._sdk_ready:
            raise ModelCallError(self._sdk_error or "SDK 未就绪")

        from dashscope import MultiModalConversation

        messages = build_messages(
            photo_url=photo.resolve().as_uri(),
            photo_name=photo.name,
            store_id=_guess_store_id(photo),
            task_id="experiment",
            min_consecutive_gaps=self.settings.min_consecutive_gaps,
        )

        response = MultiModalConversation.call(
            api_key=self.settings.api_key,
            model=self.settings.model,
            messages=messages,
            timeout=self.settings.timeout_seconds,
            result_format="message",
        )

        status = getattr(response, "status_code", None)
        if status != 200:
            code = getattr(response, "code", "")
            message = getattr(response, "message", "")
            raise ModelCallError(f"接口返回 {status} {code} {message}")

        return _response_text(response)

    def probe(self, photo: Path, model: str) -> str:
        """用指定型号试调一次，返回可读结果。用于排查 Key/型号可用性。

        与 detect 的区别：不重试、不校验结构，把接口的原始答复直接暴露出来，
        因为排查阶段最需要看到的就是服务端的真实拒绝原因。
        """
        if self.settings.dry_run:
            return "dry-run 模式，未发起真实调用"

        if not self._sdk_ready:
            return f"SDK 不可用：{self._sdk_error}"

        from dashscope import MultiModalConversation

        messages = build_messages(
            photo_url=photo.resolve().as_uri(),
            photo_name=photo.name,
            store_id="probe",
            task_id="probe",
            min_consecutive_gaps=self.settings.min_consecutive_gaps,
        )
        response = MultiModalConversation.call(
            api_key=self.settings.api_key,
            model=model,
            messages=messages,
            timeout=self.settings.timeout_seconds,
            result_format="message",
        )
        status = getattr(response, "status_code", None)
        if status != 200:
            code = getattr(response, "code", "")
            message = getattr(response, "message", "")
            return f"✗ {status} {code} — {message}"
        try:
            text = _response_text(response)
        except Exception as exc:
            return f"✓ 200 但返回结构异常：{exc}"
        return f"✓ 200 — {text.strip()[:120]}"


def _response_text(response) -> str:
    """从 DashScope 返回结构中取出文本内容。"""
    try:
        content = response.output.choices[0].message.content
    except Exception as exc:  # pragma: no cover
        raise ModelCallError(f"返回结构异常：{exc}") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    raise ModelCallError(f"无法解析返回内容类型：{type(content)}")


def _guess_store_id(photo: Path) -> str:
    """从文件名里猜门店编号；猜不到就返回 unknown。

    约定命名：{门店编号}_{序号}.jpg，如 A012_03.jpg
    """
    stem = photo.stem
    if "_" in stem:
        head = stem.split("_")[0]
        if head:
            return head
    return "unknown"


# ----------------------------------------------------------------------
# dry-run 用：不调接口也能把整条流水线跑通
def _fake_response(photo: Path) -> str:
    """按文件名做确定性模拟，用于在没有 Key/照片时验证流程。"""
    name = photo.stem.lower()
    if "empty" in name or "oos" in name:
        payload = {
            "photo_usable": True,
            "gaps": [
                {
                    "shelf_section": "中",
                    "layer": "2",
                    "consecutive_gaps": 5,
                    "evidence": "[dry-run] 该层连续 5 个位置无商品且无价签",
                    "confidence": "high",
                }
            ],
            "uncertain": [],
            "notes": "",
        }
    elif "dark" in name or "blur" in name:
        payload = {
            "photo_usable": False,
            "gaps": [],
            "uncertain": [{"reason": "[dry-run] 画面过暗，无法判断该层是否为空"}],
            "notes": "",
        }
    else:
        payload = {
            "photo_usable": True,
            "gaps": [],
            "uncertain": [],
            "notes": "",
        }
    return json.dumps(payload, ensure_ascii=False)
