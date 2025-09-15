# main.py — OMR (A5/TH) for test60
# รองรับฟอร์ม A5 แนวตั้ง "ตัวเต็ม" : ตีกรอบจากสี่เหลี่ยมมุม 4 จุด (ในกระดาษจริงมีแค่นี้)
# โซนภายในกรอบ: น้ำเงิน=StudentID, เขียว=คำตอบบนขวา(2×10), แดง=คำตอบล่าง(4×10)
import io

import os, json, time, uuid, shutil
import traceback  # <--- เพิ่มบรรทัดนี้
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from grid_presets import PRESETS, GridPreset  # ต้องมี A5_60Q_5C_ID

# ================= Layout mapping (สัมพัทธ์ภายใน "กรอบแดง" =================
# ปรับตัวเลขเหล่านี้ให้ตรงฟอร์มของคุณได้ (0..1 ของกรอบแดงที่ warp แล้ว)
# --- Final Version: Fine-tuned for both portrait and rotated layouts ---
ID_FRAC_IN_FRAME   = (0.04, 0.07, 0.32, 0.42)  # กรอบรหัสนักเรียน
ANS_TR_FRAC        = (0.39, 0.07, 0.58, 0.42)  # กรอบข้อ 1-20
ANS_BOTTOM_FRAC    = (0.04, 0.51, 0.93, 0.45)  # กรอบข้อ 21-60
BOUNDING_ANS_ROI   = (0.04, 0.07, 0.93, 0.89)  # fallback เมื่อหา frame ไม่เจอ


# ================= เตรียมโฟลเดอร์ =================
for d in ["data/uploads", "data/keys_json", "data/results", "data/debug"]:
    os.makedirs(d, exist_ok=True)

SAVE_DEBUG = os.getenv("SAVE_DEBUG", "1") == "1"

app = FastAPI(title="OMR Checker – TH (A5 60Q + StudentID)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ================= Utils (ป้องกันรูปว่าง/ช่องสีผิด) =================
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _is_small(img: Optional[np.ndarray], min_w: int = 8, min_h: int = 8) -> bool:
    return (img is None) or (img.size == 0) or (img.ndim < 2) or (img.shape[1] < min_w) or (img.shape[0] < min_h)

def safe_blur(img: np.ndarray, k: Tuple[int, int]=(3,3), sigma: float=0) -> np.ndarray:
    if _is_small(img):  # หลีกเลี่ยง _src.empty()
        return img if isinstance(img, np.ndarray) else np.zeros((10,10), dtype=np.uint8)
    kx = max(1, int(k[0])); ky = max(1, int(k[1]))
    if kx % 2 == 0: kx += 1
    if ky % 2 == 0: ky += 1
    try:
        return cv.GaussianBlur(img, (kx, ky), sigma)
    except Exception:
        return img

def to_bgr(img: np.ndarray) -> np.ndarray:
    if _is_small(img): raise HTTPException(400, "อ่านไฟล์ภาพไม่สำเร็จ")
    if img.ndim == 2: return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 3: return img
        if c == 4: return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        if c == 1: return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        return img[:, :, :3].copy()
    m = np.squeeze(img)
    if m.ndim == 2: return cv.cvtColor(m, cv.COLOR_GRAY2BGR)
    return m[:, :, :3].copy()

def to_gray(img: np.ndarray) -> np.ndarray:
    if _is_small(img): return np.zeros((10,10), dtype=np.uint8)
    if img.ndim == 2: return img
    if img.ndim == 3:
        c = img.shape[2]
        if c == 1: return img[:, :, 0]
        if c >= 3:
            try:    return cv.cvtColor(img[:, :, :3], cv.COLOR_BGR2GRAY)
            except: return cv.cvtColor(to_bgr(img), cv.COLOR_BGR2GRAY)
    return cv.cvtColor(to_bgr(img), cv.COLOR_BGR2GRAY)

def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    try:
        # อ่านผ่าน Pillow เพื่อใช้ EXIF orientation
        img_pil = Image.open(io.BytesIO(data))
        img_pil = ImageOps.exif_transpose(img_pil)   # หมุนตาม EXIF
        img_pil = img_pil.convert("RGB")
        img = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    except Exception:
        # เผื่อไฟล์ที่ Pillow อ่านไม่ได้ ค่อย fallback ไป OpenCV
        arr = np.frombuffer(data, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise HTTPException(400, "อ่านไฟล์ภาพไม่สำเร็จ")
        img = to_bgr(img)
    upload.file.seek(0)   # รีเซ็ต pointer เผื่อใช้ซ้ำ
    return img

def save_upload(upload: UploadFile, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    fname = f"{int(time.time())}_{uuid.uuid4().hex}_{upload.filename}"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, out_w=1480, out_h=2100) -> np.ndarray:
    rect = order_points(pts)
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(image, M, (out_w, out_h))

# ================= Image quality =================
def _blur_metric(gray: np.ndarray) -> float:
    lap = cv.Laplacian(gray, cv.CV_64F).var()
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    ten = float(np.mean(gx*gx + gy*gy))
    h, w = gray.shape[:2]
    return float(0.6 * lap + 0.4 * (ten / (h*w)))

def image_quality_report(img_bgr: np.ndarray) -> Dict:
    gray = safe_blur(to_gray(img_bgr), (3,3), 0)
    blur_var = _blur_metric(gray)
    overexp_pct = float((gray >= 245).sum()) / gray.size * 100.0
    underexp_pct = float((gray <= 15).sum()) / gray.size * 100.0
    bg = safe_blur(gray, (49,49), 0)
    rel = cv.absdiff(gray, bg)
    illum_score = float(np.mean(rel))
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi/180, 180)
    skew_deg = 0.0
    if lines is not None and len(lines) > 0:
        angs = []
        for l in lines[:80]:
            _, theta = l[0]
            a = (theta * 180.0 / np.pi)
            a = min(abs(a % 180), abs(180 - (a % 180)))
            if a <= 45: angs.append(a if a <= 45 else 90-a)
        if angs: skew_deg = float(np.median(angs))
    warnings = []
    if blur_var < 60.0: warnings.append("ภาพเบลอ/สั่น (แนะนำถ่ายใหม่ให้ชัด)")
    if overexp_pct > 8.0: warnings.append("สว่างจัด (ส่วนสว่างอิ่มตัวมาก)")
    if underexp_pct > 8.0: warnings.append("มืดจัด (ส่วนมืดทึบมาก)")
    if illum_score > 14.0: warnings.append("แสงไม่สม่ำเสมอ (มีเงาหนัก)")
    return {
        "blur_var": round(blur_var,2),
        "overexp_pct": round(overexp_pct,2),
        "underexp_pct": round(underexp_pct,2),
        "illum_score": round(illum_score,2),
        "skew_deg": round(skew_deg,2),
        "warnings": warnings
    }

# ================= หา 4 มุม (สี่เหลี่ยมทึบ) =================
def _find_corners_strict(image: np.ndarray) -> np.ndarray:
    g0 = to_gray(image)
    h0, w0 = g0.shape[:2]
    diag = (w0**2 + h0**2) ** 0.5
    scales = [1.0, 0.85, 0.7, 0.5]
    cand = []
    for sc in scales:
        g = g0 if sc == 1.0 else cv.resize(g0, (int(w0*sc), int(h0*sc)), interpolation=cv.INTER_AREA)
        th = cv.threshold(safe_blur(g, (5,5), 0), 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
        th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
        th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
        cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        H, W = g.shape[:2]
        for c in cnts:
            a = cv.contourArea(c)
            if a < W*H*0.00025 or a > W*H*0.04: continue
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.035*peri, True)
            if len(approx) != 4 or not cv.isContourConvex(approx): continue
            x,y,w,h = cv.boundingRect(approx)
            ar = w / (h + 1e-6)
            if ar < 0.65 or ar > 1.35: continue
            fill = a / (w*h + 1e-6)
            if fill < 0.75: continue
            mask = np.zeros_like(g, np.uint8); cv.drawContours(mask,[approx],-1,255,-1)
            if cv.mean(g, mask=mask)[0] > 145: continue
            pts = approx.reshape(-1,2).astype(np.float32) / sc
            cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
            ds = [(cx-0)**2+(cy-0)**2, (cx-w0)**2+(cy-0)**2, (cx-w0)**2+(cy-h0)**2, (cx-0)**2+(cy-h0)**2]
            which = int(np.argmin(ds)); dmin = (ds[which]) ** 0.5
            if dmin > diag*0.60: continue
            cand.append((which, dmin, np.array([cx,cy], np.float32)))
    best = {0:(1e18,None), 1:(1e18,None), 2:(1e18,None), 3:(1e18,None)}
    for which, d, pt in cand:
        if d < best[which][0]: best[which] = (d, pt)
    if sum(best[k][1] is not None for k in best) >= 3:
        for k in best:
            if best[k][1] is None:
                others = [best[j][1] for j in best if best[j][1] is not None]
                best[k] = (0, np.mean(others, axis=0))
    if any(best[k][1] is None for k in best): raise RuntimeError("strict-not-found")
    return np.vstack([best[0][1], best[1][1], best[2][1], best[3][1]]).astype(np.float32)

def _find_corners_fallback(image: np.ndarray) -> np.ndarray:
    g = safe_blur(to_gray(image), (5,5), 0)
    th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 10)
    cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    h, w = th.shape[:2]; boxes=[]
    for c in cnts:
        a = cv.contourArea(c)
        if a < w*h*0.00015: continue
        approx = cv.approxPolyDP(c, 0.04*cv.arcLength(c, True), True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            x,y,ww,hh = cv.boundingRect(approx)
            fill = a / (ww*hh + 1e-6)
            if fill > 0.55:
                cx, cy = x+ww/2, y+hh/2
                d = min((cx-0)**2+(cy-0)**2,(cx-w)**2+(cy-0)**2,(cx-w)**2+(cy-h)**2,(cx-0)**2+(cy-h)**2)
                boxes.append((d, approx.reshape(-1,2).astype(np.float32)))
    if len(boxes) < 4: raise HTTPException(422, "ตรวจไม่พบสี่เหลี่ยมทึบครบ 4 มุม")
    boxes.sort(key=lambda x: x[0])
    return np.array([b[1].mean(axis=0) for b in boxes[:4]], dtype=np.float32)

def find_corner_markers(image: np.ndarray) -> np.ndarray:
    try:
        return _find_corners_strict(image)
    except Exception:
        return _find_corners_fallback(image)

# ================= ROI helpers =================
@dataclass
class Zone:
    roi: Tuple[float, float, float, float]  # x,y,w,h (0..1) ภายในหน้า warp แล้ว
    cols: int
    rows_per_col: int

def _crop_from_roi(warp: np.ndarray, roi: Tuple[float,float,float,float]) -> np.ndarray:
    h, w = warp.shape[:2]
    x, y, ww, hh = roi
    x = max(0.0, min(1.0, x)); y = max(0.0, min(1.0, y))
    ww = max(0.01, min(1.0 - x, ww)); hh = max(0.01, min(1.0 - y, hh))
    x1, y1 = int(x*w), int(y*h)
    x2, y2 = int((x+ww)*w), int((y+hh)*h)
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w)); y2 = max(y1+1, min(y2, h))
    return warp[y1:y2, x1:x2].copy()

def _crop_from_roi_with_pad(warp: np.ndarray, roi: Tuple[float,float,float,float],
                            pad_x: float=0.015, pad_y: float=0.02) -> np.ndarray:
    x,y,w,h = roi
    w = max(0.02, w - pad_x*2)
    h = max(0.02, h - pad_y*2)
    x = min(0.98, max(0.0, x + pad_x))
    y = min(0.98, max(0.0, y + pad_y))
    return _crop_from_roi(warp, (x,y,w,h))

def _find_red_frame_in_warp(warp: np.ndarray) -> Tuple[float,float,float,float]:
    # หา boundary (ครอบโดยสี่เหลี่ยมมุม) — ใช้รวมเพื่ออ้างอิงโซนต่างๆ
    H, W = warp.shape[:2]
    g = to_gray(warp)
    th = cv.threshold(g, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    th = cv.medianBlur(th, 5)
    cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w_, h_ = cv.boundingRect(c)
        area = w_*h_
        if area < 0.0002*W*H or area > 0.02*W*H: continue
        ar = w_/(h_+1e-6)
        if not (0.5 < ar < 1.5): continue
        fill = cv.contourArea(c)/(area+1e-6)
        if fill < 0.5: continue
        cx, cy = x+w_/2.0, y+h_/2.0
        if not ((cx < 0.25*W and cy < 0.25*H) or (cx > 0.75*W and cy < 0.25*H) or
                (cx < 0.25*W and cy > 0.75*H) or (cx > 0.75*W and cy > 0.75*H)):
            continue
        boxes.append((x,y,w_,h_))
    if len(boxes) >= 4:
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[0]+b[2] for b in boxes); y2 = max(b[1]+b[3] for b in boxes)
        pad = int(0.01 * min(W, H))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
        if x2 - x1 > 10 and y2 - y1 > 10:
            return (x1/W, y1/H, (x2-x1)/W, (y2-y1)/H)
    return (0.02, 0.06, 0.96, 0.90)  # fallback

def _map_frac(frame_xywh: Tuple[float,float,float,float],
              inner_frac: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    fx, fy, fw, fh = frame_xywh
    rx, ry, rw, rh = inner_frac
    return (fx + rx*fw, fy + ry*fh, rw*fw, rh*fh)

# --- NEW: helper กลับซ้าย-ขวาแบบสัมพัทธ์ภายในกรอบ ---
def _flip_frac_x(frac: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    x, y, w, h = frac
    return (1.0 - x - w, y, w, h)

def _build_dynamic_layout(warp: np.ndarray, flip_h: bool=False):
    frame = _find_red_frame_in_warp(warp)

    def F(fr):
        return _map_frac(frame, _flip_frac_x(fr) if flip_h else fr)

    id_roi = F(ID_FRAC_IN_FRAME)
    zones = [
        Zone(roi=F(ANS_TR_FRAC),     cols=2, rows_per_col=10),
        Zone(roi=F(ANS_BOTTOM_FRAC), cols=4, rows_per_col=10),
    ]
    return frame, id_roi, zones

# ================= Auto locate (สำรอง) =================
def auto_locate_grid(warp: np.ndarray, preset: GridPreset) -> List[Tuple[int,int,int,int]]:
    g = to_gray(warp)
    n = _normalize_illum(g)
    H, W = n.shape[:2]
    x1 = int(W*0.10); x2 = int(W*0.96)
    bandY = (255 - n[:, x1:x2]).sum(axis=1).astype(np.float32)
    bandY = safe_blur(bandY.reshape(-1,1), (1, 101), 0).ravel()
    bandY = cv.medianBlur(bandY.astype(np.uint8), 11).astype(np.float32)
    if bandY.max() <= 1e-6: return []
    y_idx = np.where(bandY > 0.15*bandY.max())[0]
    if len(y_idx) == 0: return []
    y_top, y_bot = int(y_idx[0]), int(y_idx[-1])
    y_top = max(0, y_top - int(0.03*H))
    y_bot = min(H, y_bot + int(0.03*H))
    ry = int(BOUNDING_ANS_ROI[1]*H); rh = int(BOUNDING_ANS_ROI[3]*H)
    y_top = min(y_top, ry); y_bot = max(y_bot, ry+rh)
    band = (255 - n[y_top:y_bot, :]).sum(axis=0).astype(np.float32)
    band = safe_blur(band.reshape(1,-1), (151,1), 0).ravel()
    if band.max() <= 1e-6: return []
    need = 6
    idx = np.argpartition(band, -need)[-need:]; idx = np.sort(idx)
    def grow(center:int) -> Tuple[int,int]:
        thr = 0.35 * band[center]
        L = center
        while L-1 >= 0 and band[L-1] > thr: L -= 1
        R = center
        while R+1 < band.size and band[R+1] > thr: R += 1
        return L, R
    boxes = []; pad = int(0.02*W)
    for p in idx:
        L, R = grow(int(p))
        xL = max(0, L-pad); xR = min(W-1, R+pad)
        boxes.append((xL, y_top, xR-xL+1, y_bot-y_top))
    boxes.sort(key=lambda b: b[0])
    merged = []
    for b in boxes:
        if not merged: merged.append(list(b)); continue
        px,py,pw,ph = merged[-1]; x,y,w,h = b
        if x < px + int(pw*0.6):
            nx = min(px,x); ny = min(py,y)
            nx2 = max(px+pw, x+w); ny2 = max(py+ph, y+h)
            merged[-1] = [nx,ny, nx2-nx, ny2-ny]
        else:
            merged.append(list(b))
    if len(merged) > 6: merged = merged[:6]
    if SAVE_DEBUG:
        dbg = warp.copy()
        for (x,y,w,h) in merged: cv.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 3)
        cv.imwrite(os.path.join("data/debug", f"cols_auto_{int(time.time())}.jpg"), dbg)
    return [tuple(m) for m in merged]

# ================= Illumination helpers =================
def _normalize_illum(gray: np.ndarray) -> np.ndarray:
    bg = safe_blur(gray, (31,31), 0)
    bg = np.clip(bg, 8, 255).astype(np.float32)
    norm = (gray.astype(np.float32) / bg) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    g = safe_blur(gray, (3,3), 0)
    try:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
    except Exception:
        pass
    return _normalize_illum(g)

# ================= Scoring =================
def _score_vector(cell: np.ndarray, centers: List[int], r_in: float) -> List[float]:
    h2, w2 = cell.shape[:2]
    base = float(np.percentile(cell, 80))
    minv = float(np.percentile(cell, 10))
    norm_dark = lambda v: (base - v) / (base - minv + 1e-6)
    th = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    r_ring = r_in * 1.35
    scores = []
    for cx in centers:
        cy = h2 * 0.5
        Y, X = np.ogrid[:h2, :w2]
        m_in   = ((X-cx)**2 + (Y-cy)**2) <= (r_in**2)
        m_ring = (((X-cx)**2 + (Y-cy)**2) <= (r_ring**2)) & (~m_in)
        if m_in.sum() == 0:
            scores.append(0.0); continue
        dark_in   = max(0.0, norm_dark(float(cell[m_in].mean())))
        dark_ring = max(0.0, norm_dark(float(cell[m_ring].mean()))) if m_ring.sum() else 0.0
        fill_in   = float(th[m_in].mean()) / 255.0
        score = 0.65*dark_in + 0.25*fill_in - 0.20*dark_ring
        scores.append(score)
    return scores

def _find_centers_from_band(roi: np.ndarray, n_choices: int) -> List[int]:
    HH, WW = roi.shape[:2]

    # --- ADD THIS SANITY CHECK ---
    if WW < n_choices:
        # If the ROI is narrower than the number of choices, we can't find distinct peaks.
        # Return evenly spaced points as a robust fallback.
        slice_w = WW / max(1, n_choices)
        return [int(i * slice_w + slice_w * 0.5) for i in range(n_choices)]
    # -----------------------------

    band = roi[int(HH*0.20):int(HH*0.80), :]
    xprof = (255 - band).sum(axis=0).astype(np.float32)
    xprof = safe_blur(xprof.reshape(1,-1), (151,1), 0).ravel()
    
    # Add another check for profile length after processing
    if xprof.size < n_choices:
        slice_w = WW / max(1, n_choices)
        return [int(i * slice_w + slice_w * 0.5) for i in range(n_choices)]

    idx = np.argpartition(xprof, -n_choices)[-n_choices:]; idx = np.sort(idx)
    centers = []
    if len(idx) == n_choices:
        win = max(3, int(0.04*WW))
        for p in idx:
            L, R = max(0, int(p)-win), min(WW-1, int(p)+win)
            xs = np.arange(L, R+1)
            w  = xprof[L:R+1] - xprof[L:R+1].min()
            centers.append(int(np.round((xs*w).sum()/max(w.sum(),1e-6))))
    else:
        slice_w = WW / n_choices
        centers = [int(i*slice_w + slice_w*0.5) for i in range(n_choices)]
    return centers

def _split_columns_by_profile(g: np.ndarray, cols: int) -> List[Tuple[int,int]]:
    H, W = g.shape[:2]
    xprof = (255 - g).sum(axis=0).astype(np.float32)
    xprof = safe_blur(xprof.reshape(1,-1), (151,1), 0).ravel()
    idx = np.argpartition(xprof, -cols)[-cols:]; idx = np.sort(idx)
    centers = []
    win = max(3, int(0.05*W))
    for p in idx:
        L, R = max(0, int(p)-win), min(W-1, int(p)+win)
        xs = np.arange(L, R+1)
        w  = xprof[L:R+1] - xprof[L:R+1].min()
        centers.append(int(np.round((xs*w).sum()/max(w.sum(),1e-6))))
    centers = sorted(centers)
    bounds = [0]
    for i in range(len(centers)-1):
        bounds.append(int((centers[i]+centers[i+1])//2))
    bounds.append(W)
    out = []
    for i in range(cols):
        x1, x2 = bounds[i], bounds[i+1]
        pad = int(0.04*(x2-x1))
        out.append((x1+pad, x2-pad))
    return out

def _read_column_answers(col_img: np.ndarray, choices: List[str], rows: int):
    if _is_small(col_img, 24, 24): return ([""]*rows, [0.0]*rows, [False]*rows)
    g0 = to_gray(col_img); g  = _enhance_gray(g0)
    H, W = g.shape[:2]
    prof_v = (255 - g).sum(axis=1).astype(np.float32)
    prof_v = safe_blur(prof_v.reshape(-1,1), (1, 31), 0).ravel()
    thr_v  = 0.05 * float(prof_v.max())
    ys     = np.where(prof_v > thr_v)[0]
    if len(ys) > 0: y_top, y_bot = int(ys[0]), int(ys[-1])
    else:           y_top, y_bot = int(H*0.15), int(H*0.96)
    # y_top = max(0, y_top - int(0.03*H)); y_bot = min(H, y_bot + int(0.02*H))
    roi = g[y_top:y_bot, :]
    if _is_small(roi, 24, 24): return ([""]*rows, [0.0]*rows, [False]*rows)
    HH, WW = roi.shape[:2]
    centers = _find_centers_from_band(roi, len(choices))
    row_h = HH / rows
    if len(centers) >= 2: avg_gap = float(np.mean(np.diff(sorted(centers))))
    else:                 avg_gap = WW / max(5.0, float(len(choices)))
    r_small = max(5.0, 0.32 * min(avg_gap, HH/rows))
    r_large = r_small * 1.25
    answers: List[str] = []; confidences: List[float] = []; multimarks: List[bool] = []
    for r in range(rows):
        yy1 = int(r * row_h + row_h * 0.05)
        yy2 = int((r + 1) * row_h - row_h * 0.05)
        if yy2 <= yy1 or yy2-yy1 < 6:
            answers.append(""); confidences.append(0.0); multimarks.append(False); continue
        cell = roi[yy1:yy2, :]
        s1 = _score_vector(cell, centers, r_small)
        best_i = int(np.argmax(s1)); best = float(s1[best_i])
        second = float(sorted(s1, reverse=True)[1]) if len(s1) > 1 else 0.0
        conf1 = (best - second) / max(best, 1e-6)
        multi1 = second > 0.70 * best
        pass1 = (best > max(0.20, np.percentile(s1, 75)*0.90)) and ((best-second) > max(0.06, 0.35*np.std(s1) if len(s1)>1 else 0.06))
        if pass1 and not multi1:
            answers.append(choices[best_i]); confidences.append(conf1); multimarks.append(False); continue
        s2 = _score_vector(cell, centers, r_large)
        best_i2 = int(np.argmax(s2)); best2 = float(s2[best_i2])
        second2 = float(sorted(s2, reverse=True)[1]) if len(s2) > 1 else 0.0
        conf2 = (best2 - second2) / max(best2, 1e-6)
        multi2 = second2 > 0.70 * best2
        pass2 = (best2 > max(0.18, np.percentile(s2, 75)*0.85)) and ((best2-second2) > max(0.05, 0.30*np.std(s2) if len(s2)>1 else 0.05))
        if pass2 and not multi2:
            answers.append(choices[best_i2]); confidences.append(conf2); multimarks.append(False)
        else:
            answers.append(""); confidences.append(max(conf1, conf2)); multimarks.append(multi2 or multi1)
    return answers, confidences, multimarks

def _read_grid_rect(grid_img: np.ndarray, cols: int, rows_per_col: int, choices: List[str]):
    if _is_small(grid_img, 40, 40):
        n = cols*rows_per_col
        return ([""]*n, [0.0]*n, [False]*n)
    gray = to_gray(grid_img); g = _enhance_gray(gray)
    H, W = g.shape[:2]
    prof_v = (255 - g).sum(axis=1).astype(np.float32)
    prof_v = safe_blur(prof_v.reshape(-1,1), (1, 31), 0).ravel()
    thr_v  = 0.05 * float(prof_v.max())
    ys     = np.where(prof_v > thr_v)[0]
    if len(ys) > 0: y_top, y_bot = int(ys[0]), int(ys[-1])
    else:           y_top, y_bot = int(H*0.15), int(H*0.96)
    y_top = max(0, y_top - int(0.03*H)); y_bot = min(H, y_bot + int(0.02*H))
    band = g[y_top:y_bot, :]
    if _is_small(band, 24, 24):
        n = cols*rows_per_col
        return ([""]*n, [0.0]*n, [False]*n)
    col_bounds = _split_columns_by_profile(band, cols)
    answers: List[str] = []; confs: List[float] = []; multis: List[bool] = []
    for (x1,x2) in col_bounds:
        col_img = band[:, x1:x2]
        a, c, m = _read_column_answers(col_img, choices, rows_per_col)
        answers.extend(a); confs.extend(c); multis.extend(m)
    return answers, confs, multis

# ================= Answer pipeline =================
def extract_answer_grid_robust(warp: np.ndarray, preset: GridPreset) -> np.ndarray:
    base = BOUNDING_ANS_ROI
    best_img = _crop_from_roi(warp, base)
    if SAVE_DEBUG:
        cv.imwrite(os.path.join("data/debug", f"grid_fallback_{int(time.time())}.jpg"), best_img)
    return best_img

def detect_marks_gray(grid_img: np.ndarray, preset: GridPreset):
    if _is_small(grid_img, 40, 40):
        n = 60
        return ([""]*n, [0.0]*n, [False]*n)
    gray = to_gray(grid_img); g = _enhance_gray(gray)
    H,W=g.shape[:2]
    rows, cols = 10, 6
    nC=len(preset.choices); margin_y,margin_x=0.12,0.06
    cell_h,cell_w=H/rows,W/cols
    answers=[]; confs=[]; multis=[]
    for c in range(cols):
        for r in range(rows):
            y1=int(r*cell_h+cell_h*margin_y); y2=int((r+1)*cell_h-cell_h*margin_y)
            x1=int(c*cell_w+cell_w*margin_x); x2=int((c+1)*cell_w-cell_w*margin_x)
            if y2-y1<6 or x2-x1<6:
                answers.append(""); confs.append(0.0); multis.append(False); continue
            cell=g[y1:y2,x1:x2]; h2,w2=cell.shape[:2]
            slice_w=w2/nC
            r_small=max(4.0,min(slice_w,h2)*0.22); r_large=r_small*1.25
            def score_cell(rmask: float) -> List[float]:
                base=float(np.percentile(cell,80)); minv=float(np.percentile(cell,10))
                norm_dark=lambda v:(base-v)/(base-minv+1e-6)
                th=cv.threshold(cell,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
                th=cv.morphologyEx(th,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
                edge=cv.Sobel(cell,cv.CV_32F,1,1,ksize=3); edge=np.minimum(1.0,np.abs(edge)/40.0)
                scores=[]
                for i in range(nC):
                    cx=i*slice_w+slice_w*0.5; cy=h2*0.5
                    Y,X=np.ogrid[:h2,:w2]; m=((X-cx)**2+(Y-cy)**2)<=(rmask**2)
                    if m.sum()==0: scores.append(0.0); continue
                    darkness=max(0.0,norm_dark(float(cell[m].mean())))
                    fill=float(th[m].mean())/255.0
                    e=float(edge[m].mean())
                    scores.append(0.55*darkness+0.35*fill+0.10*e)
                return scores
            s1=score_cell(r_small)
            b1,i1=(max(s1),int(np.argmax(s1))) if s1 else (0.0,0)
            second1=float(sorted(s1,reverse=True)[1]) if len(s1)>1 else 0.0
            conf1=(b1-second1)/max(b1,1e-6)
            multi1 = second1 > 0.70 * b1
            pass1 = (b1>max(0.22,np.percentile(s1,75)*0.90)) and ((b1-second1)>max(0.06,0.35*np.std(s1) if len(s1)>1 else 0.06))
            if pass1 and not multi1:
                answers.append(preset.choices[i1]); confs.append(conf1); multis.append(False); continue
            s2=score_cell(r_large)
            b2,i2=(max(s2),int(np.argmax(s2))) if s2 else (0.0,0)
            second2=float(sorted(s2,reverse=True)[1]) if len(s2)>1 else 0.0
            conf2=(b2-second2)/max(b2,1e-6)
            multi2 = second2 > 0.70 * b2
            pass2 = (b2>max(0.18,np.percentile(s2,75)*0.85)) and ((b2-second2)>max(0.05,0.30*np.std(s2) if len(s2)>1 else 0.05))
            if pass2 and not multi2:
                answers.append(preset.choices[i2]); confs.append(conf2); multis.append(False)
            else:
                answers.append(""); confs.append(max(conf1,conf2)); multis.append(multi2 or multi1)
    return answers, confs, multis

def _combine_dual_results(col, grid, low_conf_thr=0.15):
    a1, c1, m1 = col; a2, c2, m2 = grid
    n = max(len(a1), len(a2))
    out_a=[]; out_c=[]; out_m=[]
    for i in range(n):
        A1 = a1[i] if i < len(a1) else ""; C1 = c1[i] if i < len(c1) else 0.0; M1 = m1[i] if i < len(m1) else False
        A2 = a2[i] if i < len(a2) else ""; C2 = c2[i] if i < len(c2) else 0.0; M2 = m2[i] if i < len(m2) else False
        if A1 == A2:
            out_a.append(A1); out_c.append(max(C1,C2)); out_m.append(M1 or M2); continue
        if A1 and not A2:
            out_a.append(A1 if C1 >= low_conf_thr and not M1 else ""); out_c.append(C1); out_m.append(M1); continue
        if A2 and not A1:
            out_a.append(A2 if C2 >= low_conf_thr and not M2 else ""); out_c.append(C2); out_m.append(M2); continue
        if abs(C1 - C2) > 0.05:
            pick = (A1,C1,M1) if C1 > C2 else (A2,C2,M2)
            if pick[1] >= low_conf_thr and not pick[2]:
                out_a.append(pick[0]); out_c.append(pick[1]); out_m.append(pick[2])
            else:
                out_a.append(""); out_c.append(pick[1]); out_m.append(pick[2])
        else:
            out_a.append(""); out_c.append(max(C1,C2)); out_m.append(M1 or M2)
    return out_a, out_c, out_m

def _locate_answer_quads_by_inner_squares(_warp: np.ndarray, _preset: GridPreset):
    return []  # ปิดไว้ก่อน (ไม่จำเป็นกับเลย์เอาต์นี้)

def get_answers_from_warp(warp: np.ndarray, preset: GridPreset, flip_h: bool=False):
    _frame, _id_roi_dyn, ZONES_DYN = _build_dynamic_layout(warp, flip_h)
    all_a: List[str]=[]; all_c: List[float]=[]; all_m: List[bool]=[]
    for z in ZONES_DYN:
        zimg = _crop_from_roi_with_pad(warp, z.roi, pad_x=0.012, pad_y=0.016)
        a,c,m = _read_grid_rect(zimg, z.cols, z.rows_per_col, preset.choices)
        all_a.extend(a); all_c.extend(c); all_m.extend(m)
    if len(all_a) == 60 and sum(1 for x in all_a if x) >= 6:
        return all_a, all_c, all_m
    # สำรอง: อ่านทั้งกรอบรวม + auto locate
    grid_img = extract_answer_grid_robust(warp, preset)
    grid_res = detect_marks_gray(grid_img, preset)
    boxes = auto_locate_grid(warp, preset)
    if len(boxes) == 6:
        rows_per_col = 10
        cols_ans = []; cols_conf=[]; cols_multi=[]
        for (x,y,w,h) in boxes:
            a,c,m = _read_column_answers(warp[y:y+h, x:x+w].copy(), preset.choices, rows_per_col)
            cols_ans.append(a); cols_conf.append(c); cols_multi.append(m)
        col_a=[]; col_c=[]; col_m=[]
        for ci in range(6):
            col_a.extend(cols_ans[ci]); col_c.extend(cols_conf[ci]); col_m.extend(cols_multi[ci])
        ans, conf, multi = _combine_dual_results((col_a,col_c,col_m), grid_res)
    else:
        ans, conf, multi = grid_res
    return ans, conf, multi

def find_and_warp(img_bgr: np.ndarray) -> np.ndarray:
    corners = find_corner_markers(img_bgr)
    return four_point_transform(img_bgr, corners)

def ensure_portrait_upright(warp: np.ndarray, preset: GridPreset) -> np.ndarray:
    """บังคับแนวตั้ง (H > W) เท่านั้น ไม่ตัดสิน 0°/180° ที่นี่"""
    img = warp
    if img.shape[0] < img.shape[1]:
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    return img

# ================= Student ID =================
def read_student_id(warp: np.ndarray, preset: GridPreset, flip_h: bool=False) -> Tuple[str, List[float], List[bool]]:
    _frame, id_roi_dyn, _zones = _build_dynamic_layout(warp, flip_h)
    roi = _crop_from_roi_with_pad(warp, id_roi_dyn, pad_x=0.01, pad_y=0.015)
    if _is_small(roi, 30, 30): return "", [0.0]*preset.id_cols, [False]*preset.id_cols
    g = _enhance_gray(to_gray(roi))
    H, W = g.shape[:2]
    cols, rows = preset.id_cols, preset.id_rows
    cell_w, cell_h = W/cols, H/rows
    digits = list(preset.id_digits)
    out_digits=[]; confs=[]; multis=[]
    for c in range(cols):
        x1 = int(c*cell_w + cell_w*0.15)
        x2 = int((c+1)*cell_w - cell_w*0.15)
        col_img = g[:, x1:x2]
        h2, w2 = col_img.shape[:2]
        cx = w2*0.5
        r = max(4.0, min(w2, cell_h)*0.28)
        scores = []
        for r_idx in range(rows):
            y1 = int(r_idx*cell_h + cell_h*0.15)
            y2 = int((r_idx+1)*cell_h - cell_h*0.15)
            if y2-y1 < 6: scores.append(0.0); continue
            cell = col_img[y1:y2, :]
            svec = _score_vector(cell, [int(cx)], r)
            scores.append(svec[0] if svec else 0.0)
        if not scores or max(scores) <= 0:
            out_digits.append("?"); confs.append(0.0); multis.append(False); continue
        best_i = int(np.argmax(scores)); best = float(scores[best_i])
        second = float(sorted(scores, reverse=True)[1]) if len(scores) > 1 else 0.0
        conf = (best - second) / max(best, 1e-6)
        multi = second > 0.70*best
        pass_min = best > max(0.18, np.percentile(scores,75)*0.85)
        pass_gap = (best-second) > max(0.05, 0.30*np.std(scores) if len(scores)>1 else 0.05)
        out_digits.append(digits[best_i] if (pass_min and pass_gap and not multi) else "?")
        confs.append(conf); multis.append(multi)
    return "".join(out_digits), confs, multis

# ================= Overlay =================
def _draw_cross(img, center, size, color, thick=2):
    x,y = center
    cv.line(img, (x-size, y-size), (x+size, y+size), color, thick)
    cv.line(img, (x-size, y+size), (x+size, y-size), color, thick)

def draw_overlay_with_key(warp: np.ndarray, preset: GridPreset,
                          answers: List[str], key: List[str], flip_h: bool=False) -> np.ndarray:
    out = warp.copy()
    H, W = out.shape[:2]
    frame, id_roi, zones = _build_dynamic_layout(warp, flip_h)
    fx, fy, fw, fh = frame
    rx1, ry1 = int(fx*W), int(fy*H); rx2, ry2 = int((fx+fw)*W), int((fy+fh)*H)
    cv.rectangle(out, (rx1,ry1), (rx2,ry2), (0,0,255), 3)             # กรอบแดง (อ้างอิง)
    ix, iy, iw_, ih_ = id_roi
    ix1, iy1 = int(ix*W), int(iy*H); ix2, iy2 = int((ix+iw_)*W), int((iy+ih_)*H)
    cv.rectangle(out, (ix1,iy1), (ix2,iy2), (60,120,255), 2)          # น้ำเงิน (StudentID)
    offset = 0
    for z in zones:
        zx,zy,zw,zh = z.roi
        ax1, ay1 = int(zx*W), int(zy*H); ax2, ay2 = int((zx+zw)*W), int((zy+zh)*H)
        cv.rectangle(out, (ax1,ay1), (ax2,ay2), (0,200,0), 2)         # เขียว/แดง โซนคำตอบ
        zcrop = _crop_from_roi_with_pad(out, z.roi, pad_x=0.012, pad_y=0.016)
        if _is_small(zcrop, 40, 40):
            offset += z.cols*z.rows_per_col; continue
        gz = _enhance_gray(to_gray(zcrop))
        HZ, WZ = gz.shape[:2]
        prof_v = (255 - gz).sum(axis=1).astype(np.float32)
        prof_v = safe_blur(prof_v.reshape(-1,1), (1, 31), 0).ravel()
        thr_v  = 0.05 * float(prof_v.max())
        ys     = np.where(prof_v > thr_v)[0]
        if len(ys) > 0: y_top, y_bot = int(ys[0]), int(ys[-1])
        else:           y_top, y_bot = int(HZ*0.15), int(HZ*0.96)
        y_top = max(0, y_top - int(0.03*HZ)); y_bot = min(HZ, y_bot + int(0.02*HZ))
        band = gz[y_top:y_bot, :]
        bounds = _split_columns_by_profile(band, z.cols)
        row_h  = (y_bot-y_top) / z.rows_per_col
        nC = len(preset.choices)
        pad_x = int(0.012*zw*W); pad_y = int(0.016*zh*H)
        zx0, zy0 = ax1 + pad_x, ay1 + pad_y
        for ci, (bx1,bx2) in enumerate(bounds):
            col_roi = band[:, bx1:bx2]
            centers = _find_centers_from_band(col_roi, nC)
            for r in range(z.rows_per_col):
                gidx = offset + ci*z.rows_per_col + r
                if gidx >= len(answers) or gidx >= len(key): break
                yy1 = zy0 + y_top + int(r*row_h); yy2 = zy0 + y_top + int((r+1)*row_h)
                cy  = int((yy1+yy2)//2)
                for j,ch in enumerate(preset.choices):
                    if key[gidx] == ch:
                        cx = zx0 + bx1 + int(centers[j]); cv.circle(out, (cx,cy), 12, (255,180,0), 2)
                if answers[gidx]:
                    j = preset.choices.index(answers[gidx]); cx = zx0 + bx1 + int(centers[j])
                    if key[gidx] and answers[gidx] == key[gidx]:
                        cv.circle(out, (cx,cy), 10, (0,200,0), 2)
                    else:
                        cv.circle(out, (cx,cy), 10, (0,0,255), 2); _draw_cross(out, (cx,cy), 7, (0,0,255), 2)
        offset += z.cols*z.rows_per_col
    return out


# ================= Keys / Compare =================
def compute_correctness(student_ans: List[str], key_ans: List[str]) -> List[bool]:
    return [(sa != "" and ka != "" and sa == ka) for sa, ka in zip(student_ans, key_ans)]

def key_json_path(subject: str, version: str | None = None) -> str:
    return os.path.join("data/keys_json", f"{subject}__{'latest' if version is None else version}.json")

def key_exists(subject: str) -> bool:
    return os.path.exists(key_json_path(subject))

def load_latest_key(subject: str) -> Dict:
    p = key_json_path(subject)
    if not os.path.exists(p): raise HTTPException(404, "ยังไม่มีเฉลยของวิชานี้")
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def save_key(subject: str, preset_name: str, key_list: List[str]) -> Dict:
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"subject": subject, "preset": preset_name, "version": version, "created_at": now_ts(), "answers": key_list}
    with open(key_json_path(subject, version), "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(key_json_path(subject), "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

# ================= UI =================
DARK_CSS = """
:root{--bg:#0b1220;--card:#0f172a;--muted:#94a3b8;--txt:#e2e8f0;--accent:#3b82f6;--ok:#22c55e;--bd:#1e293b}
*{box-sizing:border-box} body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--txt)}
header{background:#0a0f1d;border-bottom:1px solid var(--bd);padding:14px 18px}
main{max-width:1100px;margin:20px auto;padding:0 16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:18px;margin:12px 0}
.row{display:flex;gap:10px;flex-wrap:wrap}
.label{font-size:13px;color:var(--muted);margin-bottom:6px}
.input,.select{width:100%;padding:10px 12px;border:1px solid var(--bd);background:#0b1220;color:#e2e8f0;border-radius:10px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:10px;border:1px solid var(--bd);background:var(--accent);color:#fff;cursor:pointer;font-weight:600}
.btn.sub{background:#0b1220} .btn:disabled{opacity:.6;cursor:not-allowed}
.badge{padding:4px 10px;border-radius:999px;border:1px solid var(--bd);font-size:12px;color:var(--muted)}
.ok{background:#0f2a1d;color:#a7f3d0;border-color:#14532d}
.table{width:100%;border-collapse:collapse;margin-top:10px}
.table th,.table td{border-bottom:1px solid var(--bd);padding:8px 10px;text-align:left}
.donut{--p:0;width:72px;height:72px;border-radius:50%;background:conic-gradient(var(--ok) calc(var(--p)*1%), #243044 0);display:grid;place-items:center}
"""

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – A5 60Q</title><style>__CSS__</style></head>
<body><header><h3 style="margin:0">📊 OMR A5 60 ข้อ + รหัส นศ.</h3></header>
<main>
  <div class="card"><h3>ตั้ง/แก้ไขเฉลย (60 ข้อ)</h3><a class="btn" href="/keys_a5">เปิดหน้า /keys_a5</a></div>
  <div class="card"><h3>ตรวจจากรูปเดี่ยว (A5)</h3><a class="btn" href="/check_a5">เปิดหน้า /check_a5</a></div>
  <div class="card"><h3>สแกนด้วยกล้อง (A5)</h3><a class="btn" href="/scan_a5">เปิดหน้า /scan_a5</a></div>
  <div class="card"><h3>อ่านอย่างเดียว (เก็บรหัส+คำตอบ)</h3><a class="btn" href="/read_a5">เปิดหน้า /read_a5</a></div>
  <div class="card"><h3>API Docs</h3><a class="btn sub" href="/docs">/docs</a></div>
</main></body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

# -------- Keys page (ครบ) --------
@app.get("/keys_a5", response_class=HTMLResponse)
def keys_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ตั้งเฉลย A5</title><style>__CSS__</style>
<style>
.qrow{display:grid;grid-template-columns:84px repeat(5,1fr);gap:10px;align-items:center;margin:6px 0}
.choice{display:flex;align-items:center;gap:6px}
.radio{appearance:none;width:18px;height:18px;border-radius:50%;border:2px solid #334155;display:inline-block}
.radio:checked{border-color:var(--accent);background:var(--accent)}
.active{outline:2px solid var(--accent);outline-offset:3px;border-radius:10px}
.quick{display:flex;align-items:center;gap:10px;margin:8px 0;padding:10px;border:1px dashed var(--bd);border-radius:12px}
.qbtn{padding:8px 12px;border-radius:10px;border:1px solid var(--bd);background:#0b1220;color:#e2e8f0;cursor:pointer;font-weight:600}
.qbtn:hover{filter:brightness(1.1)}
</style></head>
<body>
<header><h3 style="margin:0">📝 ตั้ง Subject และเฉลย 60 ข้อ (ก/ข/ค/ง/จ)</h3></header>
<main>
  <div class="card">
    <div class="row">
      <div style="flex:1 1 420px"><div class="label">Subject</div>
        <input id="sub" class="input" placeholder="เช่น MATH101_A5">
      </div>
      <div style="align-self:end"><button class="btn sub" onclick="loadKey()">โหลดเฉลยล่าสุด</button></div>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>

    <div id="status" style="margin:10px 0"><span class="badge">ยังไม่ได้ตรวจสอบเฉลย</span></div>

    <div class="quick">
      <div class="label" style="min-width:160px">กรอกตามลำดับ: <b>#<span id="cur">1</span></b> / <span>60</span></div>
      <div class="row" style="gap:8px">
        <button class="qbtn" data-v="ก">ก (1)</button>
        <button class="qbtn" data-v="ข">ข (2)</button>
        <button class="qbtn" data-v="ค">ค (3)</button>
        <button class="qbtn" data-v="ง">ง (4)</button>
        <button class="qbtn" data-v="จ">จ (5)</button>
        <button class="qbtn" id="btnSkip">ข้าม</button>
        <button class="qbtn" id="btnBack">ย้อน</button>
      </div>
    </div>

    <div id="grid"></div>

    <div class="row" style="margin-top:12px">
      <button id="btnCreate" class="btn" onclick="saveKey(false)">บันทึกครั้งแรก</button>
      <label style="display:flex;align-items:center;gap:6px"><input id="okUpd" type="checkbox"> อัปเดตเฉลย</label>
      <button id="btnUpdate" class="btn sub" onclick="saveKey(true)" disabled>อัปเดต</button>
    </div>
  </div>
</main>

<script>
const CHOICES=["ก","ข","ค","ง","จ"];
const $=q=>document.querySelector(q);
const N=60; let idx=1;

function render(){
  let h='';
  for(let i=1;i<=N;i++){
    h+=`<div class="qrow" id="row${i}">
      <div>ข้อ ${i}</div>
      ${CHOICES.map(c=>`
        <label class="choice">
          <input class="radio" type="radio" name="q${i}" value="${c}">
          <span>${c}</span>
        </label>`).join('')}
    </div>`;
  }
  $('#grid').innerHTML=h;
  for(let i=1;i<=N;i++){ $('#row'+i).addEventListener('click',()=>{idx=i;highlight();}); }
  highlight();
}
function highlight(){
  $('#cur').textContent=idx;
  for(let i=1;i<=N;i++){ const r=$('#row'+i); if(!r) continue; r.classList.toggle('active', i===idx); }
  const el=$('#row'+idx); if(el) el.scrollIntoView({block:'nearest'});
}
function setAnswer(i, val){
  const el=document.querySelector(`input[name="q${i}"][value="${val}"]`);
  if(el){ el.checked=true; }
}
function next(){ if(idx<N){ idx++; highlight(); } }
function back(){ if(idx>1){ idx--; highlight(); } }
function collect(){
  const arr=[]; for(let i=1;i<=N;i++){ const el=document.querySelector(`input[name="q${i}"]:checked`); arr.push(el?el.value:""); } return arr;
}
async function checkStatus(sub){
  const r=await fetch('/api/keys/status?subject='+encodeURIComponent(sub)); const js=await r.json();
  if(js.exists){
    $('#status').innerHTML=`<span class="badge ok">มีเฉลยแล้ว · ${js.version} · ${js.num_questions} ข้อ · preset ${js.preset}</span>`;
    $('#btnCreate').disabled=true;
  }else{
    $('#status').innerHTML='<span class="badge">ยังไม่มีเฉลย</span>';
    $('#btnCreate').disabled=false;
  }
  return js.exists;
}
async function loadKey(){
  const sub=$('#sub').value.trim(); if(!sub){ alert('กรอก Subject ก่อน'); return; }
  const ok=await checkStatus(sub); if(!ok){ alert('ยังไม่มีเฉลยของวิชานี้'); return; }
  const r=await fetch('/api/keys/export?subject='+encodeURIComponent(sub)); const js=await r.json();
  js.answers.forEach((a,i)=>{ if(a){ setAnswer(i+1,a); } });
}
async function saveKey(isUpdate){
  const sub=$('#sub').value.trim(); if(!sub){ alert('กรอก Subject'); return; }
  const answers=collect();
  const fd=new FormData();
  fd.append('subject',sub);
  fd.append('template_count',String(60));
  fd.append('answers',JSON.stringify(answers));
  if(isUpdate) fd.append('confirm','1');
  const r=await fetch('/api/keys/set_manual',{method:'POST',body:fd}); const js=await r.json();
  if(!r.ok){ alert(js.detail||'บันทึกไม่สำเร็จ'); return; }
  alert(isUpdate?'อัปเดตสำเร็จ':'บันทึกสำเร็จ'); await checkStatus(sub);
}
document.addEventListener('click',e=>{
  const t=e.target.closest('.qbtn'); if(!t) return;
  if(t.id==='btnSkip'){ next(); return; }
  if(t.id==='btnBack'){ back(); return; }
  if(t.dataset.v){ setAnswer(idx, t.dataset.v); next(); }
});
document.addEventListener('keydown',e=>{
  if(e.key>='1' && e.key<='5'){ const v = CHOICES[parseInt(e.key)-1]; setAnswer(idx,v); next(); }
  if(e.key==='ArrowRight'){ next(); }
  if(e.key==='ArrowLeft'){ back(); }
  if(e.key===' '){ next(); }
});
render();
document.getElementById('sub').addEventListener('input',()=>{const s=$('#sub').value.trim(); if(s) checkStatus(s);});
document.getElementById('okUpd').addEventListener('change',e=>$('#btnUpdate').disabled=!e.target.checked);
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

# -------- Check page --------
@app.get("/check_a5", response_class=HTMLResponse)
def check_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – ตรวจ A5</title><style>__CSS__</style></head>
<body>
<header><h3 style="margin:0">🧪 ตรวจข้อสอบ A5 (60 ข้อ + รหัส นศ.)</h3></header>
<main>
  <div class="card">
    <div class="row">
      <div style="flex:1 1 360px"><div class="label">Subject (ต้องตั้งเฉลยแล้ว)</div><input id="subG" class="input" placeholder="เช่น MATH101_A5"></div>
      <div style="align-self:end"><input id="img" type="file" accept="image/*" class="input"></div>
      <div style="align-self:end"><button id="btn" class="btn">ตรวจ</button></div>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>

    <div id="sum" style="display:none;margin-top:14px;gap:16px;align-items:center">
      <div class="donut" id="donut" style="--p:0"><b id="pct">0%</b></div>
      <div>
        <div class="label">ผลรวม</div>
        <div style="font-size:20px"><b id="score">0 / 0</b></div>
        <div class="label">เวอร์ชันเฉลย: <span id="ver">-</span></div>
        <div class="label">รหัส นศ.: <b id="sid">-</b></div>
      </div>
    </div>
    <table class="table" id="tbl" style="margin-top:10px;display:none"></table>
    <div id="overlayWrap" style="margin-top:10px;display:none">
      <div class="label">รูปประกอบ (วงเฉลย + คำตอบ):</div>
      <img id="ovimg" style="max-width:100%;border:1px solid #1e293b;border-radius:10px"/>
    </div>
  </div>
</main>

<script>
var $ = function(q){ return document.querySelector(q); };

function renderResult(d){
  var pct = Math.round(100 * d.total_correct / d.num_questions);
  $('#sum').style.display = 'flex';
  $('#tbl').style.display = 'table';
  $('#donut').style.setProperty('--p', pct);
  $('#pct').textContent = String(pct) + '%';
  $('#score').textContent = d.total_correct + ' / ' + d.num_questions;
  $('#ver').textContent = d.version;
  $('#sid').textContent = d.student_id || '-';

  var rows = d.answers_marked.map(function(a, i){
    var mark = d.correctness[i] ? '✅' : '❌';
    var ans = a && a.length ? a : '-';
    var key = d.key_answers && d.key_answers[i] ? d.key_answers[i] : '-';
    return '<tr><td>' + (i+1) + '</td><td>' + ans + '</td><td>' + key + '</td><td>' + mark + '</td></tr>';
  }).join('');
  $('#tbl').innerHTML =
    '<thead><tr><th>ข้อ</th><th>คำตอบที่เลือก</th><th>เฉลย</th><th>ถูก/ผิด</th></tr></thead>' +
    '<tbody>' + rows + '</tbody>';

  if (d.overlay_path){
    $('#overlayWrap').style.display='block';
    $('#ovimg').src = '/' + d.overlay_path;
  } else {
    $('#overlayWrap').style.display='none';
  }

  if (d.quality && d.quality.warnings && d.quality.warnings.length){
    alert('คำเตือนคุณภาพภาพ:\\n- ' + d.quality.warnings.join('\\n- '));
  }
}

document.getElementById('btn').addEventListener('click', function(){
  var sub = $('#subG').value.trim();
  var f = $('#img').files[0];
  if(!sub || !f){ alert('กรอก Subject และเลือกไฟล์รูป'); return; }

  var fd = new FormData();
  fd.append('subject', sub);
  fd.append('image', f);

  fetch('/api/grade_a5', { method:'POST', body: fd })
    .then(function(r){ return r.json().then(function(js){ return {ok:r.ok, js:js}; }).catch(function(){ return {ok:false, js:{detail:'ตอบกลับไม่ใช่ JSON'}}; }); })
    .then(function(res){
      if(!res.ok){ alert(res.js.detail || 'ตรวจไม่สำเร็จ'); return; }
      renderResult(res.js);
    })
    .catch(function(err){ console.error(err); alert('เกิดข้อผิดพลาดในการเชื่อมต่อ'); });
});
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

# -------- Scan page (ครบ) --------
@app.get("/scan_a5", response_class=HTMLResponse)
def scan_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – สแกน A5</title><style>__CSS__</style>
<style>
.cam{position:relative;max-width:980px;margin:12px auto;border:1px solid var(--bd);border-radius:16px;overflow:hidden}
.cam video{width:100%;display:block;background:#000}
.tip{font-size:13px;color:var(--muted)}
.rowc{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
</style></head>
<body>
<header><h3 style="margin:0">📷 สแกนด้วยกล้อง (A5 60 ข้อ)</h3></header>
<main>
  <div class="card">
    <div class="rowc">
      <div style="flex:1 1 360px"><div class="label">Subject</div><input id="sub" class="input" placeholder="ต้องมีเฉลยอยู่แล้ว"></div>
      <button id="btnStart" class="btn">เริ่มกล้อง</button>
      <button id="btnShot" class="btn sub" disabled>ถ่ายภาพ</button>
      <button id="btnStop" class="btn sub" disabled>หยุด</button>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>
    <div class="tip" style="margin:8px 0">ให้มาร์กสี่เหลี่ยม 4 มุมอยู่ครบ ระบบจะวอร์ปและตรวจเมื่อกดถ่าย</div>
    <div class="cam" id="cam">
      <video id="video" playsinline autoplay muted></video>
      <canvas id="shot" style="display:none"></canvas>
    </div>
  </div>
</main>

<script>
const $=q=>document.querySelector(q);
let stream=null;
async function startCam(){
  if(stream){ stopCam(); }
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment', width:{ideal:1280}, height:{ideal:720}}, audio:false});
  const v=$('#video'); v.srcObject=stream;
  v.onloadedmetadata=()=>{ v.play(); $('#btnShot').disabled=false; $('#btnStop').disabled=false; };
}
function stopCam(){ if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null; $('#btnShot').disabled=true; $('#btnStop').disabled=true; } }
async function capture(){
  const v=$('#video'), c=$('#shot'); const sub=$('#sub').value.trim();
  if(!sub){ alert('กรอก Subject ก่อน'); return; }
  c.width=v.videoWidth; c.height=v.videoHeight;
  const ctx=c.getContext('2d'); ctx.drawImage(v,0,0,c.width,c.height);
  c.toBlob(async (blob)=>{
    const fd=new FormData(); fd.append('subject',sub); fd.append('image',blob,'capture.jpg');
    const r=await fetch('/api/grade_a5',{method:'POST',body:fd}); let js=null;
    try{ js=await r.json(); }catch{ alert('ตอบกลับไม่ใช่ JSON'); return; }
    if(!r.ok){ alert(js.detail||'ตรวจไม่สำเร็จ'); return; }
    alert(`รหัส นศ.: ${js.student_id}\\nคะแนน: ${js.total_correct}/${js.num_questions}`);
    if (js.overlay_path){ const w=window.open(); w.document.write('<img style="max-width:100%" src="/'+js.overlay_path+'">'); }
  }, 'image/jpeg', 0.92);
}
$('#btnStart').onclick=startCam; $('#btnStop').onclick=stopCam; $('#btnShot').onclick=capture;
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

# ================= API MODELS =================
class Flags(BaseModel):
    low_conf: List[int]
    multi_mark: List[int]
    id_multi: List[int] = []

class QualityReport(BaseModel):
    blur_var: float
    overexp_pct: float
    underexp_pct: float
    illum_score: float
    skew_deg: float
    warnings: List[str]

class GradeA5Response(BaseModel):
    subject: str
    version: str
    timestamp: str
    num_questions: int
    answers_marked: List[str]
    key_answers: List[str]
    correctness: List[bool]
    total_correct: int
    student_id: str
    confidences: List[float]
    flags: Flags
    quality: QualityReport
    overlay_path: Optional[str] = None
    engine: str = "a5_smart_v2"

# ================= Keys API =================
@app.get("/api/keys/status")
async def key_status(subject: str):
    exists = key_exists(subject)
    resp = {"subject": subject, "exists": exists}
    if exists:
        m = load_latest_key(subject)
        resp.update(preset=m["preset"], version=m["version"], created_at=m["created_at"], num_questions=len(m["answers"]))
    return resp

@app.get("/api/keys/export")
async def key_export(subject: str):
    m = load_latest_key(subject)
    return {"subject": subject, "preset": m["preset"], "version": m["version"], "answers": m["answers"]}

@app.post("/api/keys/set_manual")
async def set_manual_key(
    subject: str = Form(...),
    template_count: int = Form(...),
    answers: str = Form(...),
    confirm: str = Form("0")
):
    if template_count != 60: raise HTTPException(400, "ฟอร์มนี้รองรับเฉพาะ 60 ข้อ")
    try:
        key_list = json.loads(answers)
    except Exception:
        raise HTTPException(400, "answers ต้องเป็น JSON array")
    if len(key_list) != 60: raise HTTPException(400, "จำนวนคำตอบไม่ตรง 60")
    preset_name = "A5_60Q_5C_ID"
    if preset_name not in PRESETS: raise HTTPException(500, "preset ไม่ถูกต้อง")
    if key_exists(subject) and confirm not in ("1","true","True","YES","yes"):
        raise HTTPException(409, f"วิชา '{subject}' มีเฉลยอยู่แล้ว ต้องส่ง confirm=1 เพื่ออัปเดต")
    valid = PRESETS[preset_name].choices
    key_list = [a if a in valid else "" for a in key_list]
    meta = save_key(subject, preset_name, key_list)
    return {"message":"บันทึกเฉลยสำเร็จ","subject":subject,"version":meta["version"],"num_questions":len(key_list)}

# ================= Grade API =================
@app.post("/api/grade_a5", response_model=GradeA5Response)
async def grade_a5(subject: str = Form(...), image: UploadFile = File(...)):
    meta = load_latest_key(subject)
    preset = PRESETS["A5_60Q_5C_ID"]
    key_list = meta["answers"]

    img_bgr = read_image(image)
    image.file.seek(0)
    save_upload(image, "data/uploads")

    quality = image_quality_report(img_bgr)

    try:
        # 1. Warp the image. This 'warped_image' is now our clean, upright canvas.
        warped_image = find_and_warp(img_bgr)

        best_pack = None
        best_score = -1

        # 2. Create orientations to test for reading marks
        orientations = [(warped_image, 0)]
        orientations.append((cv.rotate(warped_image, cv.ROTATE_180), 180))

        for warped_test, angle in orientations:
             for flip_h in (False, True):
                ans, conf, multi = get_answers_from_warp(warped_test, preset, flip_h)
                sid, sid_conf, sid_multi = read_student_id(warped_test, preset, flip_h)

                non_empty = sum(1 for a in ans if a)
                ok_digits = sum(1 for ch in sid if ch != "?")
                score = non_empty + ok_digits

                if score > best_score:
                    best_score = score
                    # Store the results, but not the warped image itself yet
                    best_pack = (ans, conf, multi, sid, sid_conf, sid_multi, flip_h)

        if best_pack is None:
            raise HTTPException(422, "Could not read marks from the image.")

        student_ans, confidences, multimarks, student_id, sid_conf, sid_multi, final_flip_h = best_pack

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Failed to process image: {e}")
    
    correctness = compute_correctness(student_ans, key_list)
    total_correct = int(sum(correctness))

    # --- FINAL FIX: Always draw on the original, upright warped_image ---
    overlay = draw_overlay_with_key(warped_image, preset, student_ans, key_list, flip_h=final_flip_h)
    
    rid = f"{subject}__{int(time.time())}__{uuid.uuid4().hex}"
    overlay_path = os.path.join("data/debug", f"{rid}__overlay.jpg")
    if SAVE_DEBUG:
        # Save the final, correctly-drawn overlay
        cv.imwrite(overlay_path, overlay)
        # Always save the upright warped image for consistency
        cv.imwrite(os.path.join("data/debug", f"{rid}__warped.jpg"), warped_image)

    low_conf_idx = [i for i,(a,c) in enumerate(zip(student_ans,confidences)) if a and c < 0.15]
    multi_idx    = [i for i,m in enumerate(multimarks) if m]
    id_multi_idx = [i for i,m in enumerate(sid_multi) if m]

    return GradeA5Response(
        subject=subject,
        version=meta["version"],
        timestamp=now_ts(),
        num_questions=len(student_ans),
        answers_marked=student_ans,
        key_answers=key_list,
        correctness=correctness,
        total_correct=total_correct,
        student_id=student_id,
        confidences=confidences,
        flags=Flags(low_conf=low_conf_idx, multi_mark=multi_idx, id_multi=id_multi_idx),
        quality=QualityReport(**quality),
        overlay_path=overlay_path,
        engine="a5_portrait_stable"
    )

# --------- peek ----------
@app.post("/api/peek_a5")
async def peek_a5(image: UploadFile = File(...)):
    preset = PRESETS["A5_60Q_5C_ID"]
    img = read_image(image)
    ans, conf, multi, sid, sid_conf, sid_multi, warped, _rot, flip_h = get_all_any_orientation(img, preset)
    return {"answers": ans, "confidences": conf, "multimarks": multi, "student_id": sid}

# ให้ static เสิร์ฟไฟล์ overlay
from fastapi.staticfiles import StaticFiles
app.mount("/data", StaticFiles(directory="data"), name="data")

# ====================== เพิ่มฟังก์ชัน overlay (วาดคำตอบอย่างเดียว) ======================
def draw_overlay_answers_only(warp: np.ndarray, preset: GridPreset, answers: List[str], flip_h: bool=False) -> np.ndarray:
    out = warp.copy()
    H, W = out.shape[:2]

    frame, id_roi, zones = _build_dynamic_layout(warp, flip_h)
    fx, fy, fw, fh = frame
    rx1, ry1 = int(fx*W), int(fy*H)
    rx2, ry2 = int((fx+fw)*W), int((fy+fh)*H)
    cv.rectangle(out, (rx1,ry1), (rx2,ry2), (0,0,255), 2)   # กรอบแดงใหญ่

    # กรอบรหัส นศ.
    ix, iy, iw_, ih_ = id_roi
    ix1, iy1 = int(ix*W), int(iy*H)
    ix2, iy2 = int((ix+iw_)*W), int((iy+ih_)*H)
    cv.rectangle(out, (ix1,iy1), (ix2,iy2), (60,120,255), 2)

    offset = 0
    for z in zones:
        zx,zy,zw,zh = z.roi
        ax1, ay1 = int(zx*W), int(zy*H)
        ax2, ay2 = int((zx+zw)*W), int((zy+zh)*H)
        cv.rectangle(out, (ax1,ay1), (ax2,ay2), (0,200,0), 2)  # โซนคำตอบ

        zcrop = _crop_from_roi_with_pad(out, z.roi, pad_x=0.012, pad_y=0.016)
        gz = _enhance_gray(to_gray(zcrop))
        HZ, WZ = gz.shape[:2]

        # หา band แนวตั้งของวงกลม
        prof_v = (255 - gz).sum(axis=1).astype(np.float32)
        prof_v = safe_blur(prof_v.reshape(-1,1), (1,31), 0).ravel()
        thr_v = 0.05 * float(prof_v.max())
        ys = np.where(prof_v > thr_v)[0]
        if len(ys) > 0:
            y_top, y_bot = int(ys[0]), int(ys[-1])
        else:
            y_top, y_bot = int(HZ*0.15), int(HZ*0.96)
        y_top = max(0, y_top - int(0.03*HZ))
        y_bot = min(HZ, y_bot + int(0.02*HZ))

        band = gz[y_top:y_bot, :]
        col_bounds = _split_columns_by_profile(band, z.cols)
        row_h = (y_bot - y_top) / z.rows_per_col
        nC = len(preset.choices)

        pad_x = int(0.012*zw*W)
        pad_y = int(0.016*zh*H)
        zx0, zy0 = ax1 + pad_x, ay1 + pad_y

        for ci, (bx1, bx2) in enumerate(col_bounds):
            col_roi = band[:, bx1:bx2]
            centers = _find_centers_from_band(col_roi, nC)

            for r in range(z.rows_per_col):
                gidx = offset + ci*z.rows_per_col + r
                if gidx >= len(answers): break
                if not answers[gidx]:    continue
                yy1 = zy0 + y_top + int(r*row_h)
                yy2 = zy0 + y_top + int((r+1)*row_h)
                cy  = int((yy1 + yy2)//2)

                j = preset.choices.index(answers[gidx])
                cx = zx0 + bx1 + int(centers[j])
                cv.circle(out, (cx,cy), 10, (0,200,0), 2)  # วาดตำแหน่งคำตอบผู้สอบ

        offset += z.cols*z.rows_per_col

    return out


# ====================== CSV helper (บันทึกเฉพาะ รหัส + คำตอบ) ======================
CSV_PATH = os.path.join("data", "results", "a5_min.csv")
def _ensure_csv_header():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            f.write("timestamp,student_id," + ",".join([f"Q{i}" for i in range(1,61)]) + "\n")

def _append_csv(student_id: str, answers: List[str]):
    _ensure_csv_header()
    row = [now_ts(), student_id] + [(a or "") for a in answers[:60]]
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        f.write(",".join([str(x) for x in row]) + "\n")


# ====================== หน้า UI ใหม่: /read_a5 ======================
@app.get("/read_a5", response_class=HTMLResponse)
def read_a5_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>อ่าน OMR A5 (แนวตั้ง) – เฉพาะรหัส+คำตอบ</title>
<style>__CSS__</style>
<style>
.wrap{max-width:880px;margin:20px auto}
.card{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:18px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
.small{font-size:12px;color:var(--muted)}
.kv{display:grid;grid-template-columns:110px 1fr;gap:8px}
.table{width:100%;border-collapse:collapse;margin-top:10px}
.table th,.table td{border-bottom:1px solid var(--bd);padding:6px 8px}
.code{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace}
</style>
</head>
<body>
<header><h3 style="margin:0">📥 อ่าน A5 แนวตั้ง (เก็บรหัส + คำตอบ 60 ข้อ)</h3></header>
<main class="wrap">
  <div class="card">
    <div class="row">
      <div style="flex:1 1 360px">
        <div class="label">เลือกรูป (ถ่ายแนวตั้ง/แนวนอนก็ได้ ระบบหมุนให้เอง)</div>
        <input id="img" type="file" accept="image/*" class="input">
      </div>
      <button id="btn" class="btn">อ่าน</button>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>
    <div id="sum" style="display:none;margin-top:14px;gap:16px;align-items:center">
      <div class="kv">
        <div class="label">รหัสนักเรียน</div>
        <div class="code"><b id="sid">-</b> <button id="copy" class="btn sub" style="padding:6px 10px;margin-left:8px">คัดลอก</button></div>
        <div class="label">บันทึกไปที่ไฟล์</div>
        <div><span class="small code">data/results/a5_min.csv</span></div>
      </div>
    </div>
    <table id="tbl" class="table" style="display:none"></table>

    <div id="preview" style="display:none;margin-top:10px">
      <div class="label">รูปประกอบ (เฉพาะคำตอบของผู้สอบ):</div>
      <img id="ovimg" style="max-width:100%;border:1px solid var(--bd);border-radius:10px"/>
    </div>
  </div>
</main>

<script>
const $=q=>document.querySelector(q);

function show(d){
  $('#sum').style.display='block';
  $('#sid').textContent=d.student_id || '-';

  const rows = d.answers.map((a,i)=>'<tr><td>'+(i+1)+'</td><td>'+(a||'-')+'</td></tr>').join('');
  $('#tbl').innerHTML = '<thead><tr><th>ข้อ</th><th>คำตอบ</th></tr></thead><tbody>'+rows+'</tbody>';
  $('#tbl').style.display = 'table';

  if(d.overlay_path){
    $('#preview').style.display='block';
    $('#ovimg').src='/' + d.overlay_path;
  }else{
    $('#preview').style.display='none';
  }
}
$('#copy').onclick = ()=>{ navigator.clipboard.writeText($('#sid').textContent||''); };

document.getElementById('btn').addEventListener('click', async ()=>{
  const f = $('#img').files[0];
  if(!f){ alert('เลือกไฟล์ภาพก่อน'); return; }
  const fd=new FormData(); fd.append('image', f);
  const r=await fetch('/api/read_a5', {method:'POST', body:fd});
  const js=await r.json();
  if(!r.ok){ alert(js.detail || 'อ่านไม่สำเร็จ'); return; }
  show(js);
});
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)


# ====================== API ใหม่: /api/read_a5 ======================
class ReadA5Response(BaseModel):
    student_id: str
    answers: List[str]
    overlay_path: Optional[str] = None

@app.post("/api/read_a5", response_model=ReadA5Response)
async def api_read_a5(image: UploadFile = File(...)):
    # อ่านภาพ + ประเมิน
    img_bgr = read_image(image)
    preset = PRESETS["A5_60Q_5C_ID"]

    ans, conf, multi, sid, sid_conf, sid_multi, warped, _rot, flip_h = get_all_any_orientation(img_bgr, preset)

    # บันทึก CSV (รหัส + 60 คำตอบ)
    _append_csv(sid, ans)

    # ทำ overlay เฉพาะคำตอบ (ไม่ใช้เฉลย)
    rid = f"readA5__{int(time.time())}__{uuid.uuid4().hex}"
    overlay = draw_overlay_answers_only(warped, preset, ans, flip_h=flip_h)
    overlay_path = os.path.join("data", "debug", f"{rid}__answers_only.jpg")
    if SAVE_DEBUG:
        cv.imwrite(overlay_path, overlay)

    return ReadA5Response(student_id=sid, answers=ans, overlay_path=overlay_path if SAVE_DEBUG else None)
