"""流程自检：不需要 API Key、不需要真实照片也能验证整条链路。

覆盖三件事：
1. 模型返回的脏格式（markdown 围栏、前后带解释）能不能被正确解析
2. 结构校验能不能挡住不合格的返回
3. 一致性、误报、漏检的计算是否正确
"""

from __future__ import annotations

import json

import pytest

from oos_check.client import ModelCallError, _extract_json
from oos_check.metrics import PhotoOutcome, Review, compute
from oos_check.schema import DetectionResult


# ----------------------------------------------------------------------
# 1. 脏格式解析
def test_extract_plain_json():
    payload = _extract_json('{"photo_usable": true, "gaps": []}')
    assert payload["photo_usable"] is True


def test_extract_json_in_markdown_fence():
    text = '```json\n{"photo_usable": true, "gaps": []}\n```'
    assert _extract_json(text)["gaps"] == []


def test_extract_json_with_leading_explanation():
    text = '好的，我判断如下：\n{"photo_usable": false, "gaps": [], "uncertain": [{"reason": "过暗"}]}\n以上。'
    payload = _extract_json(text)
    assert payload["photo_usable"] is False
    assert payload["uncertain"][0]["reason"] == "过暗"


def test_extract_json_empty_raises():
    with pytest.raises(ModelCallError):
        _extract_json("   ")


def test_extract_json_no_object_raises():
    with pytest.raises(ModelCallError):
        _extract_json("这张照片看起来没有问题。")


# ----------------------------------------------------------------------
# 2. 结构校验
def test_schema_accepts_valid_payload():
    r = DetectionResult.model_validate({
        "photo_usable": True,
        "gaps": [{
            "shelf_section": "中",
            "layer": 2,
            "consecutive_gaps": "5 个",
            "evidence": "连续 5 个空位且无价签",
            "confidence": "high",
        }],
        "uncertain": [],
        "notes": "",
    })
    # 数字型层号与带单位数量都要被容错转换，不该因为类型差异整条重试
    assert r.gaps[0].layer == "2"
    assert r.gaps[0].consecutive_gaps == 5


def test_schema_rejects_missing_required_field():
    with pytest.raises(Exception):
        DetectionResult.model_validate({"gaps": [{"layer": "2"}]})


def test_schema_null_lists_become_empty():
    r = DetectionResult.model_validate({"photo_usable": True, "gaps": None, "uncertain": None})
    assert r.gaps == [] and r.uncertain == []


def test_threshold_filters_below_minimum():
    r = DetectionResult.model_validate({
        "photo_usable": True,
        "gaps": [
            {"shelf_section": "左", "layer": "1", "consecutive_gaps": 2, "evidence": "x"},
            {"shelf_section": "中", "layer": "2", "consecutive_gaps": 3, "evidence": "y"},
            {"shelf_section": "右", "layer": "3", "consecutive_gaps": 6, "evidence": "z"},
        ],
    })
    kept = r.below_threshold(3)
    assert [g.layer for g in kept] == ["2", "3"]


# ----------------------------------------------------------------------
# 3. 一致性与指标
def _r(has_gaps: bool, usable: bool = True) -> DetectionResult:
    gaps = [{"shelf_section": "中", "layer": "2", "consecutive_gaps": 5, "evidence": "e"}] if has_gaps else []
    return DetectionResult.model_validate({"photo_usable": usable, "gaps": gaps})


def test_consensus_majority_vote():
    o = PhotoOutcome(photo="a.jpg", samples=[_r(True), _r(True), _r(False)])
    assert o.consensus_has_oos is True
    assert o.unanimous is False


def test_consensus_ignores_unusable_samples():
    o = PhotoOutcome(photo="b.jpg", samples=[_r(True), _r(False, usable=False), _r(False, usable=False)])
    # 只有 1 个可用样本，且它说有 → 认定有
    assert o.consensus_has_oos is True
    assert o.unanimous is None  # 可用样本不足 2 个，无法判断一致性


def test_consensus_gaps_need_majority_votes():
    same = _r(True)
    other = DetectionResult.model_validate({
        "photo_usable": True,
        "gaps": [{"shelf_section": "左", "layer": "1", "consecutive_gaps": 4, "evidence": "e"}],
    })
    o = PhotoOutcome(photo="c.jpg", samples=[same, same, other])
    assert o.consensus_gaps == [{"shelf_section": "中", "layer": "2", "votes": 2, "samples": 3}]


def test_metrics_confusion_matrix_and_rates():
    outcomes = [
        PhotoOutcome(photo="p1", samples=[_r(True), _r(True), _r(True)]),    # 报有 / 人工说有 → TP
        PhotoOutcome(photo="p2", samples=[_r(True), _r(True), _r(False)]),  # 报有 / 人工说无 → FP
        PhotoOutcome(photo="p3", samples=[_r(False), _r(False), _r(False)]),# 报无 / 人工说无 → TN
        PhotoOutcome(photo="p4", samples=[_r(False), _r(False), _r(False)]),# 报无 / 人工说有 → FN
    ]
    reviews = {
        "p1": Review("p1", True),
        "p2": Review("p2", False),
        "p3": Review("p3", False),
        "p4": Review("p4", True),
    }
    m = compute(outcomes, reviews)
    assert (m.true_positive, m.false_positive, m.true_negative, m.false_negative) == (1, 1, 1, 1)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.accuracy == 0.5
    # p1、p3、p4 三次采样结论一致，p2 不一致 → 75%
    assert m.consistency_rate == pytest.approx(0.75)


def test_metrics_without_reviews_only_consistency():
    outcomes = [PhotoOutcome(photo="p1", samples=[_r(True), _r(True)])]
    m = compute(outcomes, {})
    assert m.photos_with_review == 0
    assert m.accuracy is None
    assert m.consistency_rate == 1.0


def test_failed_calls_are_counted():
    o = PhotoOutcome(photo="p1", samples=[_r(True)], errors=["超时"])
    m = compute([o], {})
    assert m.failed_calls == 1
    assert m.total_model_calls == 2


def test_report_is_json_serializable():
    outcomes = [PhotoOutcome(photo="p1", samples=[_r(True), _r(True)])]
    m = compute(outcomes, {"p1": Review("p1", True)})
    json.dumps(m.as_dict(), ensure_ascii=False)  # 不抛异常即通过


# ----------------------------------------------------------------------
# 4. 位置一致性（结论一致但位置不一致的情况）
def test_location_consistent_when_same_layer():
    o = PhotoOutcome(photo="a", samples=[_r(True), _r(True), _r(True)])
    assert o.location_consistent is True
    assert o.consensus_has_oos is True


def test_location_inconsistent_when_layer_differs():
    l2 = DetectionResult.model_validate({
        "photo_usable": True,
        "gaps": [{"shelf_section": "中", "layer": "2", "consecutive_gaps": 3, "evidence": "e"}],
    })
    l3 = DetectionResult.model_validate({
        "photo_usable": True,
        "gaps": [{"shelf_section": "中", "layer": "3", "consecutive_gaps": 3, "evidence": "e"}],
    })
    o = PhotoOutcome(photo="b", samples=[l2, l2, l3])
    # 结论一致（都有缺货），但位置不一致 —— 对专员来说仍不可用
    assert o.unanimous is True
    assert o.location_consistent is False
    assert o.location_signature == frozenset()


def test_location_none_when_no_gaps():
    o = PhotoOutcome(photo="c", samples=[_r(False), _r(False)])
    assert o.location_consistent is None


def test_metrics_report_location_consistency():
    o = PhotoOutcome(photo="a", samples=[_r(True), _r(True), _r(True)])
    m = compute([o], {})
    assert m.location_checked == 1
    assert m.location_consistency_rate == 1.0
