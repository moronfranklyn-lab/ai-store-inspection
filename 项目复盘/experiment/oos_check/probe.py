"""排面缺货识别 · 模型可用性探测

用途：在正式跑实验前，先确认这个账号 + 这个 Key 到底能用哪个多模态型号。
不重试、不校验结构，直接把服务端答复打出来——排查阶段要看的就是真实拒绝原因。

用法：
    python -m oos_check.probe
    python -m oos_check.probe --models qwen-vl-plus qwen-vl-max
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .client import QwenVLClient
from .config import Settings

# 百炼上 2025 年可用的多模态候选，按"便宜好用 → 更强更贵"排列
CANDIDATE_MODELS = [
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
    "qwen-vl-max",
    "qwen-vl-max-latest",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-72b-instruct",
    "qwen-vl-max-2025-04-08",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="探测百炼上哪些多模态型号可用")
    p.add_argument("--models", nargs="*", help="指定要试的型号；不给则用内置候选列表")
    p.add_argument("--photo", help="用哪张照片试（默认取高价值样本里第一张）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()

    if settings.dry_run:
        print("未读到 API Key，无法探测。请先把 Key 填进 experiment/.env")
        return 2

    print(f"Key 前缀：{settings.api_key[:3]}…  长度 {len(settings.api_key)}")
    print("=" * 62)

    photo = _pick_photo(args.photo)
    if photo is None:
        print("找不到可用于探测的照片")
        return 2
    print(f"探测用照片：{photo.name}（{photo.stat().st_size // 1024} KB）")
    print("=" * 62)

    client = QwenVLClient(settings)
    models = args.models or CANDIDATE_MODELS
    ok_models: list[str] = []

    for model in models:
        try:
            result = client.probe(photo, model)
        except Exception as exc:  # 网络等异常也不该中断整轮探测
            result = f"✗ 调用异常：{type(exc).__name__}: {exc}"
        print(f"{model:28} {result}")
        if result.startswith("✓"):
            ok_models.append(model)

    print("=" * 62)
    if ok_models:
        print(f"可用型号（{len(ok_models)} 个）：{', '.join(ok_models)}")
        print(f"建议：export OOS_MODEL={ok_models[0]}  然后重跑实验")
    else:
        print("没有可用型号。常见原因：")
        print("  1. 百炼控制台未开通「模型服务」，或该账号需要先实名认证")
        print("  2. Key 归属的账号与开通模型服务的账号不是同一个")
        print("  3. Key 被禁用/删除，或复制时缺字符")
        print("  4. 所在区域与模型可用区域不一致（百炼主要在华北2-北京）")
    return 0 if ok_models else 1


def _pick_photo(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for candidate in [
        Path("data/photos/高价值样本"),
        Path("data/photos/S01_全量"),
    ]:
        if candidate.is_dir():
            files = sorted(
                f for f in candidate.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if files:
                return files[0]
    return None


if __name__ == "__main__":
    sys.exit(main())
