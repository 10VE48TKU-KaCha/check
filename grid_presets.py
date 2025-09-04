# grid_presets.py
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class GridPreset:
    name: str
    num_questions: int
    grid_rows: int
    grid_cols: int
    choices: List[str]                      # ["ก","ข","ค","ง","จ"]
    roi: Tuple[float, float, float, float]  # x,y,w,h in [0..1] of warped A4
    column_major: bool = True               # ✅ เรียงลงตามคอลัมน์ก่อน (1..10 ซ้าย, 11..20 ขวา)

A4_20Q_5C = GridPreset(
    name="A4_20Q_5C",
    num_questions=20,
    grid_rows=10, grid_cols=2,              # 2 คอลัมน์ × 10 แถว
    choices=["ก","ข","ค","ง","จ"],
    roi=(0.10, 0.28, 0.80, 0.60),           # ครอบบริเวณกริดคำตอบ
    column_major=True                       # ✅ สำคัญ
)

A4_40Q_5C = GridPreset(
    name="A4_40Q_5C",
    num_questions=40,
    grid_rows=10, grid_cols=4,              # 4 คอลัมน์ × 10 แถว
    choices=["ก","ข","ค","ง","จ"],
    roi=(0.06, 0.26, 0.88, 0.64),
    column_major=True
)

A4_60Q_5C = GridPreset(
    name="A4_60Q_5C",
    num_questions=60,
    grid_rows=10, grid_cols=6,              # 6 คอลัมน์ × 10 แถว
    choices=["ก","ข","ค","ง","จ"],
    roi=(0.05, 0.24, 0.90, 0.66),
    column_major=True
)

PRESETS: Dict[str, GridPreset] = {
    "A4_20Q_5C": A4_20Q_5C,
    "A4_40Q_5C": A4_40Q_5C,
    "A4_60Q_5C": A4_60Q_5C,
}
