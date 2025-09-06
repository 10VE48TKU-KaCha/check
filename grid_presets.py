# grid_presets.py
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass(frozen=True)
class GridPreset:
    name: str
    grid_rows: int     # แถวต่อคอลัมน์
    grid_cols: int     # จำนวนคอลัมน์ (กระดาษ 20/40/60 ข้อ = 2 คอลัมน์)
    choices: List[str] # ตัวเลือก
    roi: Tuple[float, float, float, float]  # (x,y,w,h) 0..1 บริเวณกริดคำตอบ
    column_major: bool = True               # ไล่ข้อแบบ คอลัมน์ก่อน (11..20 ต่อจาก 1..10)

    @property
    def num_questions(self) -> int:
        return self.grid_rows * self.grid_cols

# ROI กลางกระดาษ A4 ตามฟอร์มที่ส่งมา (แถบฟองอยู่กึ่งกลางค่อนไปด้านล่าง)
# ถ้าแบบฟอร์มต่างเล็กน้อย ตัวอ่านจะหา 2 คอลัมน์จริงอัตโนมัติอยู่แล้ว
ROI_A4 = (0.24, 0.42, 0.52, 0.42)

PRESETS: Dict[str, GridPreset] = {
    "A4_20Q_5C": GridPreset(
        name="A4_20Q_5C", grid_rows=10, grid_cols=2,
        choices=["ก", "ข", "ค", "ง", "จ"], roi=ROI_A4, column_major=True
    ),
    "A4_40Q_5C": GridPreset(
        name="A4_40Q_5C", grid_rows=20, grid_cols=2,
        choices=["ก", "ข", "ค", "ง", "จ"], roi=ROI_A4, column_major=True
    ),
    "A4_60Q_5C": GridPreset(
        name="A4_60Q_5C", grid_rows=30, grid_cols=2,
        choices=["ก", "ข", "ค", "ง", "จ"], roi=ROI_A4, column_major=True
    ),
}
