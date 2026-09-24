"""模型返回结果的结构定义与校验。

原则：模型输出必须结构校验；校验不过一律重试，不允许把脏数据写进结果
（见技术栈手册工程底线第 4 条）。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Gap(BaseModel):
    """一条排面缺货建议。"""

    shelf_section: str = Field(..., min_length=1, description="图中的货架区域")
    layer: str = Field(..., min_length=1, description="层号或堆头")
    consecutive_gaps: int = Field(..., ge=1, description="连续空位个数")
    evidence: str = Field(..., min_length=1, description="判定依据")
    confidence: Confidence = Confidence.medium

    @field_validator("layer", "shelf_section", "evidence", mode="before")
    @classmethod
    def _stringify(cls, v):
        # 模型有时把层号返回成数字，统一转成字符串，避免因类型差异整条重试
        return str(v).strip() if v is not None else ""

    @field_validator("consecutive_gaps", mode="before")
    @classmethod
    def _to_int(cls, v):
        if isinstance(v, str):
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                return int(digits)
        return v


class UncertainItem(BaseModel):
    """无法判定、必须交人工的情况。"""

    reason: str = Field(..., min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def _stringify(cls, v):
        return str(v).strip() if v is not None else ""


class DetectionResult(BaseModel):
    """单张照片的一次识别结果。"""

    photo_usable: bool = True
    gaps: list[Gap] = Field(default_factory=list)
    uncertain: list[UncertainItem] = Field(default_factory=list)
    notes: str = ""

    @field_validator("gaps", "uncertain", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        # 模型有时对空结果返回 null 或省略，统一成空列表
        return v if isinstance(v, list) else []

    @field_validator("notes", mode="before")
    @classmethod
    def _notes_to_str(cls, v):
        return str(v).strip() if v is not None else ""

    @field_validator("gaps")
    @classmethod
    def _rules(cls, gaps: list[Gap]) -> list[Gap]:
        return gaps

    def below_threshold(self, min_gaps: int) -> list[Gap]:
        """按门槛过滤：低于门槛的不报（对应 PRD 决策 F2）。"""
        return [g for g in self.gaps if g.consecutive_gaps >= min_gaps]
