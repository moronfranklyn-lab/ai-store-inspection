"""排面缺货识别 · 效果验证实验

目的：在写任何产品代码之前，先回答一个问题——
      2025 年的多模态模型，能不能稳定识别出门店照片里的"排面缺货"？

用法：
    # 1) 没有 Key 时先用模拟数据验证整条流程
    python -m oos_check.run --dry-run

    # 2) 有 Key 和真实照片后跑实测
    cp .env.example .env          # 把 Key 填进 .env，不要贴在对话里
    python -m oos_check.run --photos /path/to/photos

产出：
    output/raw_{时间戳}.jsonl    每次模型调用的原始结果（含失败记录）
    output/report_{时间戳}.md    汇总报告（一致性、准确率、误报、漏检）
    output/report_{时间戳}.json  同上，机器可读
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .client import ModelCallError, QwenVLClient
from .config import OUTPUT_DIR, REVIEWS_DIR, Settings
from .metrics import Metrics, PhotoOutcome, compute, load_reviews


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="排面缺货识别效果验证实验")
    p.add_argument("--photos", help="照片目录（默认取 OOS_PHOTOS_DIR 或 data/photos）")
    p.add_argument("--reviews", help="人工标注 CSV（默认取 data/reviews/reviews.csv）")
    p.add_argument("--model", help="覆盖模型名，例如 qwen-vl-plus-latest")
    p.add_argument("--samples", type=int, help="每张照片采样次数（默认 3）")
    p.add_argument("--min-gaps", type=int, help="判定门槛：连续几个空位算缺货（默认 3）")
    p.add_argument("--limit", type=int, help="只跑前 N 张，用于快速试跑")
    p.add_argument("--dry-run", action="store_true", help="用模拟响应验证流程，不调接口")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env(
        photos_dir=args.photos,
        model=args.model,
        samples_per_photo=args.samples,
        min_consecutive_gaps=args.min_gaps,
        dry_run=True if args.dry_run else None,
    )

    photos = settings.list_photos()
    if args.limit:
        photos = photos[: args.limit]

    print("=" * 62)
    print("排面缺货识别 · 效果验证实验")
    print("=" * 62)
    print(f"模型        ：{settings.model}")
    print(f"运行模式    ：{'dry-run（模拟响应）' if settings.dry_run else '真实调用'}")
    print(f"照片目录    ：{settings.photos_dir}")
    print(f"照片数量    ：{len(photos)}")
    print(f"每张采样    ：{settings.samples_per_photo} 次")
    print(f"判定门槛    ：连续 {settings.min_consecutive_gaps} 个空位")
    for note in settings.notes:
        print(f"提示        ：{note}")
    print("-" * 62)

    if not photos:
        print("没有找到照片。把照片放进 data/photos/，或用 --photos 指定目录。")
        print("（dry-run 也需要至少一个占位文件才能验证流程）")
        return 2

    client = QwenVLClient(settings)
    outcomes: list[PhotoOutcome] = []

    for idx, photo in enumerate(photos, start=1):
        outcome = PhotoOutcome(photo=photo.name)
        started = time.time()
        for sample_index in range(settings.samples_per_photo):
            try:
                result, record = client.detect(photo, sample_index)
                outcome.samples.append(result)
            except ModelCallError as exc:
                outcome.errors.append(str(exc))
        outcome.elapsed_seconds = round(time.time() - started, 2)
        outcomes.append(outcome)

        status = "有缺货" if outcome.consensus_has_oos else "无缺货"
        if outcome.errors:
            status += f"（失败 {len(outcome.errors)} 次）"
        print(f"[{idx}/{len(photos)}] {photo.name:32} → {status}  {outcome.elapsed_seconds}s")

    # 人工标注
    review_path = Path(args.reviews) if args.reviews else REVIEWS_DIR / "reviews.csv"
    reviews = load_reviews(review_path)
    if not reviews:
        print("-" * 62)
        print(f"未找到人工标注（{review_path}），本次只算一致性与可用性，不算准确率。")

    metrics = compute(outcomes, reviews)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = OUTPUT_DIR / f"raw_{stamp}.jsonl"
    report_json = OUTPUT_DIR / f"report_{stamp}.json"
    report_md = OUTPUT_DIR / f"report_{stamp}.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps({
                "photo": o.photo,
                "elapsed_seconds": o.elapsed_seconds,
                "consensus_has_oos": o.consensus_has_oos,
                "consensus_gaps": o.consensus_gaps,
                "consistent": o.unanimous,
                "location_consistent": o.location_consistent,
                "layer_disagreement": o.layer_disagreement,
                "samples": [s.model_dump() for s in o.samples],
                "errors": o.errors,
            }, ensure_ascii=False) + "\n")

    payload = {
        "model": settings.model,
        "dry_run": settings.dry_run,
        "min_consecutive_gaps": settings.min_consecutive_gaps,
        "samples_per_photo": settings.samples_per_photo,
        "metrics": metrics.as_dict(),
        "photos": [
            {
                "photo": o.photo,
                "consensus_has_oos": o.consensus_has_oos,
                "consensus_gaps": o.consensus_gaps,
                "consistent": o.unanimous,
                "location_consistent": o.location_consistent,
                "layer_detail": o.layer_disagreement,
                "error_count": len(o.errors),
            }
            for o in outcomes
        ],
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_render_markdown(payload, metrics, outcomes), encoding="utf-8")

    print("-" * 62)
    print(_render_console(metrics))
    print("-" * 62)
    print(f"原始结果：{jsonl_path}")
    print(f"汇总报告：{report_md}")
    return 0


def _render_console(m: Metrics) -> str:
    lines = [
        f"照片数            ：{m.photo_count}",
        f"模型调用          ：{m.total_model_calls} 次（失败 {m.failed_calls}）",
        f"结论一致性        ：{_pct(m.consistency_rate)}",
        f"位置严格一致      ：{_pct(m.location_consistency_rate)}"
        + (f"（{m.location_checked} 张可比）" if m.location_checked else ""),
        f"位置多数稳定      ：{_pct(m.consensus_layer_stable_rate)}",
        f"单次采样层号噪声  ：{_pct(m.sample_layer_noise_rate)}",
        f"可用性标记稳定率  ：{_pct(m.usable_flag_stable_rate)}",
    ]
    if m.photos_with_review:
        lines += [
            f"有标注照片        ：{m.photos_with_review}",
            f"准确率            ：{_pct(m.accuracy)}",
            f"精确率（误报）    ：{_pct(m.precision)}",
            f"召回率（漏检）    ：{_pct(m.recall)}",
            f"误报 {m.false_positive} 次 / 漏检 {m.false_negative} 次",
        ]
    return "\n".join(lines)


def _pct(v: float | None) -> str:
    return "—（数据不足）" if v is None else f"{v * 100:.1f}%"


def _fmt_gap(g: dict) -> str:
    """格式化一条缺货条目，兼容模型把层号写成「第5层」或「5」两种形式。"""
    section = str(g.get("shelf_section", "")).strip()
    layer = str(g.get("layer", "")).strip()
    if layer and not layer.startswith("第"):
        layer = f"第{layer}层" if layer.isdigit() else layer
    return f"{section}{layer}({g['votes']}/{g['samples']})"


def _render_markdown(payload: dict, m: Metrics, outcomes: list[PhotoOutcome]) -> str:
    lines = [
        "# 排面缺货识别 · 效果验证实验报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：{payload['model']}",
        f"- 运行模式：{'dry-run（模拟响应，结果不可用于效果结论）' if payload['dry_run'] else '真实调用'}",
        f"- 判定门槛：连续 {payload['min_consecutive_gaps']} 个空位",
        f"- 每张采样：{payload['samples_per_photo']} 次",
        "",
        "## 一、汇总指标",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
    ]
    for k, v in m.as_dict().items():
        if k == "混淆矩阵":
            continue
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## 二、混淆矩阵（以照片为单位）",
        "",
        "| AI 结论 / 人工标注 | 人工说有 | 人工说无 |",
        "| --- | --- | --- |",
        f"| AI 报有 | {m.true_positive}（对） | {m.false_positive}（误报） |",
        f"| AI 报无 | {m.false_negative}（漏检） | {m.true_negative}（对） |",
        "",
        "## 三、逐张结果",
        "",
        "| 照片 | 结论 | 结论一致性 | 位置一致性 | 各次采样的层号 | 缺货条目 | 失败 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for o in outcomes:
        gaps = "；".join(
            _fmt_gap(g) for g in o.consensus_gaps
        ) or "—"
        consistent = "—" if o.unanimous is None else ("一致" if o.unanimous else "**不一致**")
        loc = (
            "—" if o.location_consistent is None
            else ("一致" if o.location_consistent else "**位置不一致**")
        )
        lines.append(
            f"| {o.photo} | {'有缺货' if o.consensus_has_oos else '无缺货'} | "
            f"{consistent} | {loc} | {o.layer_disagreement or '—'} | {gaps} | {len(o.errors)} |"
        )

    lines += [
        "",
        "## 四、怎么读这份报告",
        "",
        "1. **结论一致性**回答「这个方案稳不稳」：同一张照片跑多次结论是否一样。"
        "低于 80% 说明模型对这类判断本身不稳定，规模放大会变成大量申诉与返工。",
        "2. **精确率**回答「专员会不会被误报烦到弃用」：这是本功能能不能落地的第一道门。",
        "3. **召回率**回答「会不会漏掉真实缺货」：漏检影响业务价值，但不直接伤信任。",
        "4. **误报与漏检的具体条目**要看 raw jsonl，判断集中在哪些门店形态/光线条件。",
        "5. 数字不达标时，先调判定门槛与提示词重跑，再考虑换模型；",
        "   不要为了让数字好看去改标注。",
        "",
        "## 五、下一步",
        "",
        "- 一致性或精确率不达标 → 收紧判定门槛 / 补充「不算缺货」的反例说明 / 换模型实测",
        "- 特定光照条件显著更差 → 把这些情况排除出识别范围或强制标「不确定」",
        "- 指标达标 → 用这批数据确定验收标准，再进入 MVP 开发",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
