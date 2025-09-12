# grid_presets.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class AnswerZone:
    roi: Tuple[float, float, float, float]  # (x,y,w,h) 0..1
    cols: int                                # จำนวนคอลัมน์ในโซนนี้
    rows_per_col: int                        # กี่แถวต่อคอลัมน์ (10 สำหรับแบบนี้)

@dataclass
class GridPreset:
    name: str
    choices: List[str]
    grid_rows: int
    grid_cols: int
    roi: Tuple[float, float, float, float]             # bounding รวมของทุกโซน (เผื่อไว้)
    id_roi: Tuple[float, float, float, float]
    id_cols: int
    id_rows: int
    id_digits: str
    column_major: bool = True
    zones: List[AnswerZone] = field(default_factory=list)  # << เพิ่มโซน

# ====== ฟอร์มตามรูป (บน: 2 คอลัมน์ 1–20, ล่าง: 4 คอลัมน์ 21–60) + รหัสซ้ายบน ======
A5_60Q_5C_ID = GridPreset(
    name="A5_60Q_5C_ID",
    choices=["ก", "ข", "ค", "ง", "จ"],
    grid_rows=60,
    grid_cols=6,
    # bounding รวมบริเวณสีแดงทั้งหมด (กันพลาด)
    roi=(0.15, 0.14, 0.78, 0.75),
    # กรอบน้ำเงินรหัสนักเรียน (ซ้ายบน) – ปรับเผื่อขอบเล็กน้อย
    id_roi=(0.08, 0.16, 0.26, 0.44),
    id_cols=5,                # ถ้าฟอร์มของจริงมีหลักมาก/น้อยกว่านี้ เปลี่ยนตัวเลขนี้ได้
    id_rows=10,
    id_digits="0123456789",
    column_major=True,
    zones=[
        # โซนบน (1–20) ด้านขวาบน 2 คอลัมน์
        AnswerZone(roi=(0.38, 0.15, 0.55, 0.33), cols=2, rows_per_col=10),
        # โซนล่าง (21–60) แถวล่าง 4 คอลัมน์
        AnswerZone(roi=(0.15, 0.53, 0.78, 0.36), cols=4, rows_per_col=10),
    ]
)

PRESETS: Dict[str, GridPreset] = {"A5_60Q_5C_ID": A5_60Q_5C_ID}
