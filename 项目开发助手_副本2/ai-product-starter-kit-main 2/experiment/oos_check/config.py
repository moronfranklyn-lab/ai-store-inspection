"""实验配置：全部通过环境变量注入，密钥不写进代码、不进仓库。

用法：
    1. 复制 .env.example 为 .env
    2. 把 API Key 填进 .env（不要把 Key 贴到对话、日志或截图里）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# config.py 位于 experiment/oos_check/ 下，向上两级才是实验根目录
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = EXPERIMENT_DIR / ".env"


def _load_env_file() -> None:
    """把 .env 载入环境变量。

    只做最小实现，不引入额外依赖。已存在的系统环境变量优先，不被 .env 覆盖，
    这样临时用 export 覆盖参数时不会失效。
    .env 不进仓库（见 .gitignore），密钥只从环境变量读取。
    """
    if not ENV_FILE.is_file():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# 数据目录
PHOTOS_DIR = EXPERIMENT_DIR / "data" / "photos"
REVIEWS_DIR = EXPERIMENT_DIR / "data" / "reviews"      # 人工标注文件（专员判定）
OUTPUT_DIR = EXPERIMENT_DIR / "output"                  # 模型原始输出与汇总

# 可选：直接给出图片目录（优先于 PHOTOS_DIR）
PHOTOS_PATH_ENV = "OOS_PHOTOS_DIR"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# 默认模型：百炼上的 Qwen-VL。型号不写死，可用环境变量覆盖，
# 因为"选哪个型号"本身就是要用真实样例实测决定的（见 PRD 第 6 节）。
#
# 注意：`-latest` 后缀型号与 qwen2.5-vl-* 在部分账号下会返回 403 AccessDenied
#（型号未开通/未列入可用列表）。这里默认用实测确认可用的型号；
# 换账号后先用 `python -m oos_check.probe` 探测一遍再跑。
DEFAULT_MODEL = "qwen-vl-max"
FALLBACK_MODEL = "qwen-vl-plus"


@dataclass
class Settings:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout_seconds: int = 60
    max_retries: int = 3
    # 同一张照片采样次数：2025 年 VLM 结论不可复现，用多次采样取多数来稳结果
    samples_per_photo: int = 3
    # 判定门槛（连续几个空位算缺货）—— 这是待评估校准的参数，不是定论
    min_consecutive_gaps: int = 3
    photos_dir: str = ""
    dry_run: bool = False
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        _load_env_file()
        notes: list[str] = []
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            notes.append("未检测到 DASHSCOPE_API_KEY：将进入 dry-run（用模拟响应验证流程）")

        model = os.environ.get("OOS_MODEL", "").strip() or DEFAULT_MODEL
        photos_dir = os.environ.get(PHOTOS_PATH_ENV, "").strip()
        if not photos_dir:
            photos_dir = str(PHOTOS_DIR)
            if not Path(photos_dir).is_dir():
                notes.append(f"照片目录不存在：{photos_dir}")

        s = cls(
            api_key=api_key,
            model=model,
            timeout_seconds=int(os.environ.get("OOS_TIMEOUT", "60")),
            max_retries=int(os.environ.get("OOS_MAX_RETRIES", "3")),
            samples_per_photo=int(os.environ.get("OOS_SAMPLES", "3")),
            min_consecutive_gaps=int(os.environ.get("OOS_MIN_GAPS", "3")),
            photos_dir=photos_dir,
            dry_run=not bool(api_key),
            notes=notes,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        return s

    def list_photos(self) -> list[Path]:
        d = Path(self.photos_dir)
        if not d.is_dir():
            return []
        return sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )


# 模型返回的严格结构：不合格的返回一律重试，不允许把脏数据写进结果
DETECTION_SCHEMA_DOC = """
{
  "photo_usable": true,
  "gaps": [
    {
      "shelf_section": "图中货架的哪个区域，如 左/中/右/入口堆头",
      "layer": "货架第几层，从下往上或从上往下都要说明口径；非层状货架填 堆头",
      "consecutive_gaps": 3,
      "evidence": "一句话说明判断依据，必须描述图里能看到的东西",
      "confidence": "high | medium | low"
    }
  ],
  "uncertain": [
    {"reason": "无法判断的原因（如画面过暗、被顾客遮挡、角度看不到该层）"}
  ],
  "notes": "其他需要人工注意的情况，可为空字符串"
}
"""
