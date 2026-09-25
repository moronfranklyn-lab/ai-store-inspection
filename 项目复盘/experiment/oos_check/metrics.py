"""指标计算：一致性、准确率、误报、漏检、耗时。

对应《效果评估方案》第 5 节。所有指标都由真实标注对比得出，
不预设"准确率 95%"这类数字。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .schema import DetectionResult


# ----------------------------------------------------------------------
# 人工标注
@dataclass
class Review:
    """专员对一张照片的标注。

    has_oos：这张照片有没有排面缺货
    layers ：缺货所在的层号集合（如 {"2","3"}）；留空表示未标层号，则不参与层级比对
    """

    photo: str
    has_oos: bool | None = None  # None = 标注为"不确定"
    layers: frozenset[str] = frozenset()
    note: str = ""

    @property
    def is_uncertain(self) -> bool:
        return self.has_oos is None


_TRUE = {"1", "true", "yes", "y", "是", "有"}
_FALSE = {"0", "false", "no", "n", "否", "无", "没有"}


def _norm_layer(raw: str) -> str:
    """把「第5层」「5」「第 5 层」统一成「5」。"""
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits


def load_reviews(path: Path) -> dict[str, Review]:
    """读取人工标注 CSV。

    期望列：photo, has_oos, layers(可选，顿号/逗号分隔，如「2、3」), note(可选)
    - has_oos 填 是/否 或 1/0；**留空或填「不确定」表示标注为不确定**
    - 不确定的样本不参与准确率计算，单独统计
    不提供标注文件时返回空字典，此时只算一致性、不算准确率。
    """
    if not path.is_file():
        return {}

    reviews: dict[str, Review] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("photo") or "").strip()
            if not name:
                continue
            raw = (row.get("has_oos") or "").strip().lower()
            if raw in _TRUE:
                has_oos: bool | None = True
            elif raw in _FALSE:
                has_oos = False
            else:
                has_oos = None  # 空、"不确定"、"?" 一律视为不确定

            layers_raw = (row.get("layers") or "").strip()
            layers = frozenset(
                _norm_layer(part)
                for part in layers_raw.replace("、", ",").replace("，", ",").split(",")
                if _norm_layer(part)
            )
            reviews[name] = Review(
                photo=name,
                has_oos=has_oos,
                layers=layers,
                note=(row.get("note") or "").strip(),
            )
    return reviews


# ----------------------------------------------------------------------
# 多次采样取多数（2025 年 VLM 结论不可复现的产品对策）
@dataclass
class PhotoOutcome:
    photo: str
    samples: list[DetectionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def usable_samples(self) -> list[DetectionResult]:
        return [s for s in self.samples if s.photo_usable]

    @property
    def consensus_has_oos(self) -> bool:
        """多数表决：过半可用样本认为有缺货，才认定有缺货。

        宁可偏保守——误报比漏报更伤专员信任（见 PRD 决策 F2）。
        """
        usable = self.usable_samples
        if not usable:
            return False
        votes = sum(1 for s in usable if s.gaps)
        return votes * 2 > len(usable)

    @property
    def consensus_gaps(self) -> list:
        """多数表决通过的缺货条目（按 区域+层号 归并）。"""
        if not self.consensus_has_oos:
            return []
        counter: dict[tuple[str, str], int] = {}
        for s in self.usable_samples:
            for g in s.gaps:
                counter[(g.shelf_section, g.layer)] = counter.get((g.shelf_section, g.layer), 0) + 1
        total = len(self.usable_samples)
        return [
            {"shelf_section": k[0], "layer": k[1], "votes": v, "samples": total}
            for k, v in sorted(counter.items(), key=lambda kv: -kv[1])
            if v * 2 > total
        ]

    @property
    def unanimous(self) -> bool | None:
        """可用样本之间**结论**是否一致（只比"有没有缺货"）；不足 2 个可用样本返回 None。"""
        usable = self.usable_samples
        if len(usable) < 2:
            return None
        verdicts = {bool(s.gaps) for s in usable}
        return len(verdicts) == 1

    @property
    def location_signature(self) -> frozenset[tuple[str, str]] | None:
        """本次采样的"缺货位置集合"指纹；无可比对样本返回 None。"""
        usable = self.usable_samples
        if len(usable) < 2:
            return None
        sigs = {
            frozenset((g.shelf_section, g.layer) for g in s.gaps)
            for s in usable
        }
        return sigs.pop() if len(sigs) == 1 else frozenset()  # 空集代表"位置不一致"

    @property
    def location_consistent(self) -> bool | None:
        """**严格**位置一致：要求每一次采样的缺货位置集合完全相同。

        这个口径很苛刻——只要有一次采样多报或少报一层，就算不一致。
        保留它是因为它衡量的是"单次调用能不能直接采信"，答案是基本不能。
        """
        usable = self.usable_samples
        if len(usable) < 2:
            return None
        if not any(s.gaps for s in usable):
            return None
        return self.location_signature not in (None, frozenset())

    @property
    def consensus_layer_stable(self) -> bool | None:
        """**多数表决后**的位置是否稳定（即最终给出的层号是否有过半支持）。

        这是产品实际使用的口径：专员看到的是多数表决结果，不是单次采样。
        """
        if self.location_consistent is None:
            return None
        total = len(self.usable_samples)
        if total == 0:
            return None
        for g in self.consensus_gaps:
            if g["votes"] * 2 > total:
                return True
        return False

    @property
    def sample_layer_noise(self) -> float | None:
        """单次采样的层号噪声率：与"共识位置"不一致的采样占比。

        用来回答"模型是不是经常多报/少报一层"——这个数字决定要不要多次采样。
        """
        if self.location_consistent is None:
            return None
        usable = self.usable_samples
        total = len(usable)
        if total == 0:
            return None
        # 以多数表决结果（含被过滤的条目）为参照，统计各次采样偏离程度
        consensus = {
            (_norm_section(g["shelf_section"]), _norm_layer(g["layer"]))
            for g in self.consensus_gaps
        }
        noisy = 0
        for s in usable:
            layers = {(_norm_section(g.shelf_section), _norm_layer(g.layer)) for g in s.gaps}
            if layers != consensus:
                noisy += 1
        return noisy / total


    @property
    def layer_disagreement(self) -> str:
        """把各次采样报出的层号列出来，便于人工看分歧在哪。"""
        return " / ".join(
            ",".join(f"{g.shelf_section}{g.layer}" for g in s.gaps) or "无"
            for s in self.usable_samples
        )

    @property
    def usable_flip_rate(self) -> float | None:
        """photo_usable 标记本身是否稳定——不稳说明照片质量判断也不可靠。"""
        if not self.samples:
            return None
        flags = {s.photo_usable for s in self.samples}
        return 0.0 if len(flags) == 1 else 1.0


# ----------------------------------------------------------------------
# 汇总
@dataclass
class Metrics:
    photo_count: int = 0
    samples_per_photo: int = 0
    total_model_calls: int = 0
    failed_calls: int = 0
    photos_with_review: int = 0
    # 一致性
    consistency_rate: float | None = None
    usable_flag_stable_rate: float | None = None
    # 位置相关
    location_consistency_rate: float | None = None   # 严格：每次采样位置完全相同
    location_checked: int = 0
    consensus_layer_stable_rate: float | None = None  # 多数表决后位置有稳定多数
    sample_layer_noise_rate: float | None = None      # 单次采样层号噪声率
    # 有标注时的准确率
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    precision: float | None = None
    recall: float | None = None
    accuracy: float | None = None
    unusable_rate: float | None = None
    # 标注为"不确定"、未参与准确率计算的样本数
    uncertain_reviews: int = 0
    # 层级准确率：AI 报出的层号与人工标注层号的命中情况
    layer_checked: int = 0
    layer_hit: int = 0
    layer_accuracy: float | None = None
    layer_misses: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "照片数": self.photo_count,
            "每张采样次数": self.samples_per_photo,
            "模型调用次数": self.total_model_calls,
            "调用失败次数": self.failed_calls,
            "有标注的照片数": self.photos_with_review,
            "结论一致性": _fmt_pct(self.consistency_rate),
            "位置严格一致(每次采样都相同)": _fmt_pct(self.location_consistency_rate)
            + (f"（{self.location_checked} 张可比）" if self.location_checked else ""),
            "位置多数稳定(表决后有稳定多数)": _fmt_pct(self.consensus_layer_stable_rate),
            "单次采样层号噪声率": _fmt_pct(self.sample_layer_noise_rate),
            "可用性标记稳定率": _fmt_pct(self.usable_flag_stable_rate),
            "准确率": _fmt_pct(self.accuracy),
            "精确率(误报相关)": _fmt_pct(self.precision),
            "召回率(漏检相关)": _fmt_pct(self.recall),
            "层级准确率(报对层号)": _fmt_pct(self.layer_accuracy)
            + (f"（{self.layer_hit}/{self.layer_checked} 张）" if self.layer_checked else ""),
            "人工标注为不确定": self.uncertain_reviews,
            "照片不可用占比": _fmt_pct(self.unusable_rate),
            "混淆矩阵": {
                "AI报有_人工说有(对)": self.true_positive,
                "AI报有_人工说无(误报)": self.false_positive,
                "AI报无_人工说无(对)": self.true_negative,
                "AI报无_人工说有(漏检)": self.false_negative,
            },
        }


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _safe_div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def _norm_section(raw: str) -> str:
    """区域名归一：去掉空白，把常见同义写法合并，避免"中"与"中间"被当成两个位置。"""
    s = str(raw or "").strip()
    for alias, canon in (("中间", "中"), ("中部", "中"), ("左侧", "左"), ("右侧", "右"),
                         ("左边", "左"), ("右边", "右"), ("顶层", "上"), ("底层", "下")):
        if s == alias:
            return canon
    return s


def compute(outcomes: list[PhotoOutcome], reviews: dict[str, Review]) -> Metrics:
    m = Metrics()
    m.photo_count = len(outcomes)
    if outcomes:
        m.samples_per_photo = max((len(o.samples) for o in outcomes), default=0)
    m.total_model_calls = sum(len(o.samples) + len(o.errors) for o in outcomes)
    m.failed_calls = sum(len(o.errors) for o in outcomes)

    # 一致性
    flags = [o.unanimous for o in outcomes if o.unanimous is not None]
    if flags:
        m.consistency_rate = sum(1 for f in flags if f) / len(flags)

    usable_flags = [o.usable_flip_rate for o in outcomes if o.usable_flip_rate is not None]
    if usable_flags:
        m.usable_flag_stable_rate = sum(1 for f in usable_flags if f == 0.0) / len(usable_flags)

    # 位置稳定性：只看多次采样都判"有缺货"的照片
    loc_flags = [o.location_consistent for o in outcomes if o.location_consistent is not None]
    m.location_checked = len(loc_flags)
    if loc_flags:
        m.location_consistency_rate = sum(1 for f in loc_flags if f) / len(loc_flags)

    stable_flags = [o.consensus_layer_stable for o in outcomes if o.consensus_layer_stable is not None]
    if stable_flags:
        m.consensus_layer_stable_rate = sum(1 for f in stable_flags if f) / len(stable_flags)

    noise = [o.sample_layer_noise for o in outcomes if o.sample_layer_noise is not None]
    if noise:
        m.sample_layer_noise_rate = sum(noise) / len(noise)

    # 不可用占比
    all_samples = [s for o in outcomes for s in o.samples]
    if all_samples:
        m.unusable_rate = sum(1 for s in all_samples if not s.photo_usable) / len(all_samples)

    # 有标注时算混淆矩阵（标注为"不确定"的样本排除在外）
    matched = [
        (o, reviews[o.photo])
        for o in outcomes
        if o.photo in reviews and not reviews[o.photo].is_uncertain
    ]
    m.uncertain_reviews = sum(
        1 for o in outcomes if o.photo in reviews and reviews[o.photo].is_uncertain
    )
    m.photos_with_review = len(matched)
    if matched:
        for o, r in matched:
            ai, human = o.consensus_has_oos, bool(r.has_oos)
            if ai and human:
                m.true_positive += 1
            elif ai and not human:
                m.false_positive += 1
            elif not ai and not human:
                m.true_negative += 1
            else:
                m.false_negative += 1

            # 层级准确率：人工标了层号、且人工认为有缺货时才比
            if human and r.layers:
                m.layer_checked += 1
                ai_layers = {
                    _norm_layer(str(g.get("layer", ""))) for g in o.consensus_gaps
                }
                ai_layers.discard("")
                if ai_layers & r.layers:
                    m.layer_hit += 1
                else:
                    m.layer_misses.append(
                        f"{o.photo}: AI报{sorted(ai_layers) or '无'} / 人工标{sorted(r.layers)}"
                    )

        m.precision = _safe_div(m.true_positive, m.true_positive + m.false_positive)
        m.recall = _safe_div(m.true_positive, m.true_positive + m.false_negative)
        m.accuracy = _safe_div(m.true_positive + m.true_negative, len(matched))
        m.layer_accuracy = _safe_div(m.layer_hit, m.layer_checked)
    return m
