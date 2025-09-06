# main.py — OMR (TH) robust: auto grid band + gray scoring + full pages (no GUI, no Pillow)
import os, json, time, uuid, shutil
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import cv2 as cv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from grid_presets import PRESETS, GridPreset  # A4_20Q_5C / 40 / 60

# ---------- Folders ----------
for d in ["data/uploads", "data/keys", "data/keys_json", "data/results", "data/debug"]:
    os.makedirs(d, exist_ok=True)

SAVE_DEBUG = os.getenv("SAVE_DEBUG", "1") == "1"

app = FastAPI(title="OMR Checker – TH")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------- Utils ----------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "อ่านไฟล์ภาพไม่สำเร็จ")
    upload.file.seek(0)
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

def four_point_transform(image: np.ndarray, pts: np.ndarray, out_w=2100, out_h=2970) -> np.ndarray:
    rect = order_points(pts)
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(image, M, (out_w, out_h))

# ---------- Corner detection (strict + fallback) ----------
def _find_corners_strict(image: np.ndarray) -> np.ndarray:
    g0 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h0, w0 = g0.shape[:2]
    diag = (w0**2 + h0**2) ** 0.5

    scales = [1.0, 0.85, 0.7, 0.5]
    cand = []
    for sc in scales:
        g = g0 if sc == 1.0 else cv.resize(g0, (int(w0*sc), int(h0*sc)), interpolation=cv.INTER_AREA)
        th = cv.threshold(cv.GaussianBlur(g, (5,5), 0), 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
        th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
        th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
        cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        H, W = g.shape[:2]
        for c in cnts:
            a = cv.contourArea(c)
            if a < W*H*0.00025 or a > W*H*0.04: 
                continue
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.035*peri, True)
            if len(approx) != 4 or not cv.isContourConvex(approx): 
                continue
            x,y,w,h = cv.boundingRect(approx)
            ar = w / (h + 1e-6)
            if ar < 0.65 or ar > 1.35: 
                continue
            fill = a / (w*h + 1e-6)
            if fill < 0.75: 
                continue
            mask = np.zeros_like(g, np.uint8); cv.drawContours(mask,[approx],-1,255,-1)
            if cv.mean(g, mask=mask)[0] > 145:   # ผ่อนให้จับมาร์กซีด
                continue
            pts = approx.reshape(-1,2).astype(np.float32) / sc
            cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
            ds = [(cx-0)**2+(cy-0)**2, (cx-w0)**2+(cy-0)**2, (cx-w0)**2+(cy-h0)**2, (cx-0)**2+(cy-h0)**2]
            which = int(np.argmin(ds)); dmin = (ds[which]) ** 0.5
            if dmin > diag*0.60: 
                continue
            cand.append((which, dmin, np.array([cx,cy], np.float32)))

    best = {0:(1e18,None), 1:(1e18,None), 2:(1e18,None), 3:(1e18,None)}
    for which, d, pt in cand:
        if d < best[which][0]:
            best[which] = (d, pt)

    # เติมมุมที่หายด้วย centroid ของมุมอื่น (ถ้าพบอย่างน้อย 3)
    if sum(best[k][1] is not None for k in best) >= 3:
        for k in best:
            if best[k][1] is None:
                others = [best[j][1] for j in best if best[j][1] is not None]
                best[k] = (0, np.mean(others, axis=0))

    if any(best[k][1] is None for k in best):
        raise RuntimeError("strict-not-found")

    return np.vstack([best[0][1], best[1][1], best[2][1], best[3][1]]).astype(np.float32)

def _find_corners_fallback(image: np.ndarray) -> np.ndarray:
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY); g = cv.GaussianBlur(g, (5,5), 0)
    th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 10)
    cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    h, w = th.shape[:2]; boxes=[]
    for c in cnts:
        a = cv.contourArea(c)
        if a < w*h*0.00015: 
            continue
        approx = cv.approxPolyDP(c, 0.04*cv.arcLength(c, True), True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            x,y,ww,hh = cv.boundingRect(approx)
            fill = a / (ww*hh + 1e-6)
            if fill > 0.55:
                cx, cy = x+ww/2, y+hh/2
                d = min((cx-0)**2+(cy-0)**2,(cx-w)**2+(cy-0)**2,(cx-w)**2+(cy-h)**2,(cx-0)**2+(cy-h)**2)
                boxes.append((d, approx.reshape(-1,2).astype(np.float32)))
    if len(boxes) < 4: 
        raise HTTPException(422, "ตรวจไม่พบสี่เหลี่ยมทึบครบ 4 มุม")
    boxes.sort(key=lambda x: x[0])
    return np.array([b[1].mean(axis=0) for b in boxes[:4]], dtype=np.float32)

def find_corner_markers(image: np.ndarray) -> np.ndarray:
    try:
        return _find_corners_strict(image)
    except Exception:
        return _find_corners_fallback(image)

# ---------- Illumination / masks ----------
def _normalize_illum(gray: np.ndarray) -> np.ndarray:
    bg = cv.GaussianBlur(gray, (31, 31), 0)
    bg = np.clip(bg, 8, 255).astype(np.float32)
    norm = (gray.astype(np.float32) / bg) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def _circle_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= (r ** 2)

# ---------- Auto locate 2 answer columns ----------
def auto_locate_grid(warp: np.ndarray, preset: GridPreset) -> List[Tuple[int,int,int,int]]:
    g = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
    n = _normalize_illum(g)
    H, W = n.shape[:2]

    # โปรไฟล์แนวตั้งในช่วง X กลาง
    x1 = int(W*0.22); x2 = int(W*0.78)
    bandY = (255 - n[:, x1:x2]).sum(axis=1).astype(np.float32)
    bandY = cv.GaussianBlur(bandY.reshape(-1,1), (1, 101), 0).ravel()
    bandY = cv.medianBlur(bandY.astype(np.uint8), 11).astype(np.float32)
    if bandY.max() <= 1e-6: 
        return []

    y_idx = np.where(bandY > 0.15*bandY.max())[0]
    if len(y_idx) == 0: 
        return []
    y_top, y_bot = int(y_idx[0]), int(y_idx[-1])
    y_top = max(0, y_top - int(0.03*H))
    y_bot = min(H, y_bot + int(0.03*H))
    if y_bot - y_top < int(0.18*H):
        ry = max(0, int((preset.roi[1]-0.03)*H))
        rh = min(H, int((preset.roi[3]+0.06)*H))
        y_top = min(y_top, ry)
        y_bot = max(y_bot, ry+rh)

    # โปรไฟล์แนวนอนใน band
    band = (255 - n[y_top:y_bot, :]).sum(axis=0).astype(np.float32)
    band = cv.GaussianBlur(band.reshape(1,-1), (151,1), 0).ravel()
    if band.max() <= 1e-6: 
        return []

    need = max(2, preset.grid_cols)
    idx = np.argpartition(band, -need)[-need:]
    idx = np.sort(idx)

    def grow(center:int) -> Tuple[int,int]:
        thr = 0.35 * band[center]
        L = center
        while L-1 >= 0 and band[L-1] > thr: L -= 1
        R = center
        while R+1 < band.size and band[R+1] > thr: R += 1
        return L, R

    boxes = []
    pad = int(0.02*W)
    for p in idx:
        L, R = grow(int(p))
        xL = max(0, L-pad); xR = min(W-1, R+pad)
        boxes.append((xL, y_top, xR-xL+1, y_bot-y_top))

    boxes.sort(key=lambda b: b[0])
    merged = []
    for b in boxes:
        if not merged: merged.append(list(b)); continue
        px,py,pw,ph = merged[-1]
        x,y,w,h = b
        if x < px + int(pw*0.6):
            nx = min(px,x); ny = min(py,y)
            nx2 = max(px+pw, x+w); ny2 = max(py+ph, y+h)
            merged[-1] = [nx,ny, nx2-nx, ny2-ny]
        else:
            merged.append(list(b))
    if len(merged) > preset.grid_cols:
        merged = merged[:preset.grid_cols]

    if SAVE_DEBUG:
        dbg = warp.copy()
        for (x,y,w,h) in merged:
            cv.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 3)
        cv.imwrite(os.path.join("data/debug", f"cols_auto_{int(time.time())}.jpg"), dbg)

    return [tuple(m) for m in merged]

# ---------- Read a column with gray-based scoring ----------
def _read_column_answers(col_img: np.ndarray, choices: List[str], rows: int) -> List[str]:
    g0 = cv.cvtColor(col_img, cv.COLOR_BGR2GRAY)
    g  = _normalize_illum(g0)
    H, W = g.shape[:2]

    # 1) จำกัดช่วงแนวตั้งที่มีฟองจริง
    prof_v = (255 - g).sum(axis=1).astype(np.float32)
    prof_v = cv.GaussianBlur(prof_v.reshape(-1,1), (1, 31), 0).ravel()
    thr_v  = 0.05 * float(prof_v.max())
    ys     = np.where(prof_v > thr_v)[0]
    if len(ys) > 0:
        y_top, y_bot = int(ys[0]), int(ys[-1])
    else:
        y_top, y_bot = int(H*0.15), int(H*0.96)
    y_top = max(0, y_top - int(0.03*H))
    y_bot = min(H, y_bot + int(0.02*H))

    roi = g[y_top:y_bot, :]
    HH, WW = roi.shape[:2]

    # 2) หา center ของตัวเลือกในแนวนอน
    mid1, mid2 = int(HH*0.20), int(HH*0.80)
    band = roi[mid1:mid2, :]
    xprof = (255 - band).sum(axis=0).astype(np.float32)
    xprof = cv.GaussianBlur(xprof.reshape(1,-1), (151,1), 0).ravel()

    nC = len(choices)
    idx = np.argpartition(xprof, -nC)[-nC:]
    idx = np.sort(idx)

    centers = []
    if len(idx) == nC:
        win = max(3, int(0.04*WW))
        for p in idx:
            L, R = max(0, p-win), min(WW-1, p+win)
            xs = np.arange(L, R+1)
            w  = xprof[L:R+1] - xprof[L:R+1].min()
            centers.append(int(np.round((xs*w).sum()/max(w.sum(),1e-6))))
    else:
        slice_w = WW / nC
        centers = [int(i*slice_w + slice_w*0.5) for i in range(nC)]

    # 3) วัดคะแนนแต่ละแถว
    if len(centers) >= 2:
        avg_gap = float(np.mean(np.diff(sorted(centers))))
    else:
        avg_gap = WW / max(5.0, float(nC))
    r_base = 0.32 * min(avg_gap, HH/rows)
    rmask  = max(5.0, r_base)

    row_h = HH / rows
    answers: List[str] = []
    for r in range(rows):
        yy1 = int(r * row_h + row_h * 0.05)
        yy2 = int((r + 1) * row_h - row_h * 0.05)
        if yy2 <= yy1 or yy2-yy1 < 6:
            answers.append("")
            continue
        cell = roi[yy1:yy2, :]
        h2, w2 = cell.shape[:2]
        base = float(np.percentile(cell, 80))
        minval = float(np.percentile(cell, 10))
        norm_dark = lambda v: (base-v)/(base-minval+1e-6)

        scores = []
        for cx in centers:
            cy = h2 * 0.5
            m  = _circle_mask(h2, w2, float(cx), float(cy), rmask)
            mean_in  = float(cell[m].mean())
            darkness = max(0.0, norm_dark(mean_in))
            scores.append(darkness)

        best_i = int(np.argmax(scores))
        best   = float(scores[best_i])
        second = float(sorted(scores, reverse=True)[1]) if nC > 1 else 0.0

        pass_min = best > 0.22
        pass_gap = (best-second) > 0.08
        answers.append(choices[best_i] if (pass_min and pass_gap) else "")

    if SAVE_DEBUG:
        dbg = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
        for cx in centers:
            cv.line(dbg, (int(cx), 0), (int(cx), HH-1), (0,255,255), 1)
        cv.imwrite(os.path.join("data/debug", f"centers_{int(time.time())}.jpg"), dbg)

    return answers

# ---------- Smart pipeline ----------
def _crop_from_roi(warp: np.ndarray, roi: Tuple[float,float,float,float]) -> np.ndarray:
    h, w = warp.shape[:2]
    x, y, ww, hh = roi
    x = max(0.0, min(1.0, x)); y = max(0.0, min(1.0, y))
    ww = max(0.01, min(1.0 - x, ww)); hh = max(0.01, min(1.0 - y, hh))
    x1, y1 = int(x*w), int(y*h)
    x2, y2 = int((x+ww)*w), int((y+hh)*h)
    return warp[y1:y2, x1:x2].copy()

def _score_grid_alignment(grid_img: np.ndarray, preset: GridPreset) -> float:
    try: gray = cv.cvtColor(grid_img, cv.COLOR_BGR2GRAY)
    except Exception: return 0.0
    norm = _normalize_illum(gray)
    th = cv.threshold(norm, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    th = cv.medianBlur(th, 3)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    H, W = th.shape[:2]
    rows, cols = preset.grid_rows, preset.grid_cols
    nC = len(preset.choices)
    if rows*cols <= 0 or nC <= 0: return 0.0
    margin_y, margin_x = 0.12, 0.06
    cell_h, cell_w = H/rows, W/cols
    pairs=[]
    if getattr(preset,"column_major",True):
        for c in range(cols):
            for r in range(rows): pairs.append((r,c))
    else:
        for r in range(rows):
            for c in range(cols): pairs.append((r,c))
    total_gap=0.0
    for (r,c) in pairs:
        y1=int(r*cell_h+cell_h*margin_y); y2=int((r+1)*cell_h-cell_h*margin_y)
        x1=int(c*cell_w+cell_w*margin_x); x2=int((c+1)*cell_w-cell_w*margin_x)
        if y2-y1<6 or x2-x1<6: continue
        cell_th=th[y1:y2,x1:x2]; cell_g=norm[y1:y2,x1:x2]
        h2,w2=cell_th.shape[:2]; slice_w=w2/nC; rmask=max(3.0,min(slice_w,h2)*0.28)
        scores=[]
        for i in range(nC):
            cx=i*slice_w+slice_w*0.5; cy=h2*0.5
            Y,X=np.ogrid[:h2,:w2]; m=((X-cx)**2+(Y-cy)**2)<=(rmask**2)
            if m.sum()==0: scores.append(0.0); continue
            filled=float((cell_th[m]>0).sum())/float(m.sum())
            dark=1.0-float(cell_g[m].mean())/255.0
            scores.append(0.6*filled+0.4*dark)
        sc=sorted(scores,reverse=True)
        total_gap+= (sc[0]-sc[1]) if len(sc)>=2 else (sc[0] if sc else 0.0)
    return float(total_gap)/max(1,rows*cols)

def extract_answer_grid_robust(warp: np.ndarray, preset: GridPreset) -> np.ndarray:
    base = preset.roi
    best_img = _crop_from_roi(warp, base)
    best_sc  = _score_grid_alignment(best_img, preset)
    shifts = [-0.05,-0.03,-0.02,0.0,0.02,0.03,0.05]
    scales = [0.96,1.00,1.04]
    bx,by,bw,bh=base
    for dx in shifts:
        for dy in shifts:
            for s in scales:
                roi=(bx+dx, by+dy, bw*s, bh*s)
                img=_crop_from_roi(warp, roi)
                sc=_score_grid_alignment(img, preset)
                if sc>best_sc: best_sc, best_img=sc, img
    if SAVE_DEBUG:
        cv.imwrite(os.path.join("data/debug", f"grid_fallback_{int(time.time())}.jpg"), best_img)
    return best_img

def detect_marks_gray(grid_img: np.ndarray, preset: GridPreset) -> List[str]:
    gray = cv.cvtColor(grid_img, cv.COLOR_BGR2GRAY)
    g    = _normalize_illum(gray)
    H,W=g.shape[:2]; rows,cols=preset.grid_rows,preset.grid_cols; nC=len(preset.choices)
    margin_y,margin_x=0.12,0.06
    cell_h,cell_w=H/rows,W/cols
    pairs=[]
    if getattr(preset,"column_major",True):
        for c in range(cols):
            for r in range(rows): pairs.append((r,c))
    else:
        for r in range(rows):
            for c in range(cols): pairs.append((r,c))
    answers=[]
    heat = grid_img.copy()
    for (r,c) in pairs:
        y1=int(r*cell_h+cell_h*margin_y); y2=int((r+1)*cell_h-cell_h*margin_y)
        x1=int(c*cell_w+cell_w*margin_x); x2=int((c+1)*cell_w-cell_w*margin_x)
        if y2-y1<6 or x2-x1<6: answers.append(""); continue
        cell=g[y1:y2,x1:x2]; h2,w2=cell.shape[:2]
        slice_w=w2/nC; rmask=max(4.0,min(slice_w,h2)*0.22)
        base=float(np.percentile(cell,80)); scores=[]
        for i in range(nC):
            cx=i*slice_w+slice_w*0.5; cy=h2*0.5
            m=_circle_mask(h2,w2,cx,cy,rmask)
            mean_in=float(cell[m].mean()); scores.append(max(0.0, base-mean_in))
            if SAVE_DEBUG:
                cv.circle(heat, (x1+int(cx), y1+int(cy)), int(rmask), (0,255,255), 1)
        best_i=int(np.argmax(scores)); best=float(scores[best_i])
        second=float(sorted(scores,reverse=True)[1]) if nC>1 else 0.0
        if SAVE_DEBUG:
            cv.putText(heat, f"{best:.1f}", (x1+5, y1+14), cv.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,0), 1, cv.LINE_AA)
        pass_min=best>2.2; pass_gap=(best-second)>0.7
        answers.append(preset.choices[best_i] if (pass_min and pass_gap) else "")
    if SAVE_DEBUG:
        cv.imwrite(os.path.join("data/debug", f"heat_{int(time.time())}.jpg"), heat)
    return answers

def detect_answers_smart(warp: np.ndarray, preset: GridPreset) -> List[str]:
    boxes = auto_locate_grid(warp, preset)
    if len(boxes) == preset.grid_cols:
        cols_ans: List[List[str]] = []
        for (x, y, w, h) in boxes:
            col_img = warp[y:y+h, x:x+w].copy()
            cols_ans.append(_read_column_answers(col_img, preset.choices, preset.grid_rows))
        answers: List[str] = []
        for ci in range(preset.grid_cols):
            answers.extend(cols_ans[ci])
        return answers

    # Fallback: ใช้ ROI แบบลองขยับหาอันที่ดีสุด
    grid_img = extract_answer_grid_robust(warp, preset)
    return detect_marks_gray(grid_img, preset)

# ---------- Compare ----------
def compute_correctness(student_ans: List[str], key_ans: List[str]) -> List[bool]:
    return [(sa != "" and ka != "" and sa == ka) for sa, ka in zip(student_ans, key_ans)]

# ---------- Key helpers ----------
def key_json_path(subject: str, version: str | None = None) -> str:
    return os.path.join("data/keys_json", f"{subject}__{'latest' if version is None else version}.json")

def key_exists(subject: str) -> bool:
    return os.path.exists(key_json_path(subject))

def load_latest_key(subject: str) -> Dict:
    p = key_json_path(subject)
    if not os.path.exists(p):
        raise HTTPException(404, "ยังไม่มีเฉลยของวิชานี้")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_key(subject: str, preset_name: str, key_list: List[str]) -> Dict:
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "subject": subject, "preset": preset_name, "version": version,
        "created_at": now_ts(), "answers": key_list
    }
    with open(key_json_path(subject, version), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(key_json_path(subject), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

def get_answers_from_image(img_bgr: np.ndarray, preset: GridPreset) -> Tuple[List[str], np.ndarray]:
    corners = find_corner_markers(img_bgr)
    warped = four_point_transform(img_bgr, corners)

    # DEBUG: วง ROI ให้ดู
    if SAVE_DEBUG:
        h, w = warped.shape[:2]
        rx, ry, rw, rh = preset.roi
        x1, y1 = int(rx*w), int(ry*h)
        x2, y2 = int((rx+rw)*w), int((ry+rh)*h)
        warped_dbg = warped.copy()
        cv.rectangle(warped_dbg, (x1, y1), (x2, y2), (0,0,255), 4)
        cv.imwrite(os.path.join("data/debug", f"warped_{int(time.time())}.jpg"), warped_dbg)

    # ใช้ตัวอ่านแบบ “smart” (หา 2 คอลัมน์จริง) — สำคัญ!
    answers = detect_answers_smart(warped, preset)
    return answers, warped

# ---------- CSS ----------
DARK_CSS = """
:root{--bg:#0b1220;--card:#0f172a;--muted:#94a3b8;--txt:#e2e8f0;--accent:#3b82f6;--ok:#22c55e;--bd:#1e293b}
*{box-sizing:border-box} body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--txt)}
header{background:#0a0f1d;border-bottom:1px solid var(--bd);padding:14px 18px}
main{max-width:1100px;margin:20px auto;padding:0 16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:18px;margin:12px 0}
.row{display:flex;gap:10px;flex-wrap:wrap}
.label{font-size:13px;color:var(--muted);margin-bottom:6px}
.input,.select{width:100%;padding:10px 12px;border:1px solid var(--bd);background:#0b1220;color:var(--txt);border-radius:10px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:10px;border:1px solid var(--bd);background:var(--accent);color:#fff;cursor:pointer;font-weight:600}
.btn.sub{background:#0b1220} .btn:disabled{opacity:.6;cursor:not-allowed}
.badge{padding:4px 10px;border-radius:999px;border:1px solid var(--bd);font-size:12px;color:var(--muted)}
.ok{background:#0f2a1d;color:#a7f3d0;border-color:#14532d}
.table{width:100%;border-collapse:collapse;margin-top:10px}
.table th,.table td{border-bottom:1px solid var(--bd);padding:8px 10px;text-align:left}
.donut{--p:0;width:72px;height:72px;border-radius:50%;background:conic-gradient(var(--ok) calc(var(--p)*1%), #243044 0);display:grid;place-items:center}
.radio{appearance:none;width:18px;height:18px;border-radius:50%;border:2px solid #334155;display:inline-block;position:relative}
.radio:checked{border-color:var(--accent);background:var(--accent)}
"""

# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – Dashboard</title><style>__CSS__</style></head>
<body><header><h3 style="margin:0">📊 OMR Dashboard</h3></header>
<main>
  <div class="card"><h3>สแกนด้วยกล้อง (เหมือน ZipGrade)</h3><p>จับสี่เหลี่ยมทึบ 4 มุมแบบเรียลไทม์ แล้วถ่ายอัตโนมัติเมื่อเสถียร</p><a class="btn" href="/scan">เปิดหน้า /scan</a></div>
  <div class="card"><h3>ตั้ง/แก้ไขเฉลย</h3><a class="btn" href="/keys">เปิดหน้า /keys</a></div>
  <div class="card"><h3>ตรวจจากรูปเดี่ยว</h3><a class="btn" href="/check">เปิดหน้า /check</a></div>
  <div class="card"><h3>API Docs</h3><a class="btn sub" href="/docs">/docs</a></div>
</main></body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

@app.get("/keys", response_class=HTMLResponse)
def keys_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ตั้งเฉลย</title><style>__CSS__</style>
<style>
.qrow{display:grid;grid-template-columns:84px repeat(5,1fr);gap:10px;align-items:center;margin:6px 0}
.choice{display:flex;align-items:center;gap:6px}
.radio{appearance:none;width:18px;height:18px;border-radius:50%;border:2px solid #334155;display:inline-block}
.radio:checked{border-color:var(--accent);background:var(--accent)}
.active{outline:2px solid var(--accent);outline-offset:3px;border-radius:10px}
.quick{display:flex;align-items:center;gap:10px;margin:8px 0;padding:10px;border:1px dashed var(--bd);border-radius:12px}
.qbtn{padding:8px 12px;border-radius:10px;border:1px solid var(--bd);background:#0b1220;color:var(--txt);cursor:pointer;font-weight:600}
.qbtn:hover{filter:brightness(1.1)}
</style></head>
<body>
<header><h3 style="margin:0">📝 ตั้งชื่อและเลือกคำตอบ ก / ข / ค / ง / จ (เรียงตามข้อ)</h3></header>
<main>
  <div class="card">
    <div class="row">
      <div style="flex:1 1 420px"><div class="label">Subject</div>
        <input id="sub" class="input" placeholder="เช่น MATH101_Midterm">
      </div>
      <div style="width:200px"><div class="label">จำนวนข้อ</div>
        <select id="cnt" class="select"><option value="20">20</option><option value="40">40</option><option value="60">60</option></select>
      </div>
      <div style="align-self:end"><button class="btn sub" onclick="loadKey()">โหลดเฉลยล่าสุด</button></div>
    </div>

    <div id="status" style="margin:10px 0"><span class="badge">ยังไม่ได้ตรวจสอบเฉลย</span></div>

    <div class="quick">
      <div class="label" style="min-width:160px">กรอกตามลำดับ: <b>#<span id="cur">1</span></b> / <span id="tot">20</span></div>
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
      <label class="choice"><input id="okUpd" type="checkbox"> อัปเดตเฉลย</label>
      <button id="btnUpdate" class="btn sub" onclick="saveKey(true)" disabled>อัปเดต</button>
      <button class="btn sub" onclick="clearAll()">ล้างทั้งหมด</button>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>
  </div>
</main>

<script>
const CHOICES=["ก","ข","ค","ง","จ"];
const $=q=>document.querySelector(q);
let N=20, idx=1;

function render(n){
  N=n; $('#tot').textContent=N; idx=1; highlight();
  let h='';
  for(let i=1;i<=n;i++){
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
  for(let i=1;i<=n;i++){ $('#row'+i).addEventListener('click',()=>{idx=i;highlight();}); }
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
function clearAll(){
  for(let i=1;i<=N;i++){ const el=document.querySelector(`input[name="q${i}"]:checked`); if(el) el.checked=false; }
  idx=1; highlight();
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
  const n=js.answers.length; const cnt=n<=20?20:(n<=40?40:60); $('#cnt').value=cnt; render(cnt);
  js.answers.forEach((a,i)=>{ if(a){ setAnswer(i+1,a); } });
}
async function saveKey(isUpdate){
  const sub=$('#sub').value.trim(); if(!sub){ alert('กรอก Subject'); return; }
  const answers=collect();
  const fd=new FormData();
  fd.append('subject',sub);
  fd.append('template_count',String(N));
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
render(parseInt($('#cnt').value));
$('#cnt').addEventListener('change',()=>render(parseInt($('#cnt').value)));
$('#okUpd').addEventListener('change',e=>$('#btnUpdate').disabled=!e.target.checked);
document.getElementById('sub').addEventListener('input',()=>{const s=$('#sub').value.trim(); if(s) checkStatus(s);});
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

@app.get("/check", response_class=HTMLResponse)
def check_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – ตรวจข้อสอบ</title><style>__CSS__</style></head>
<body>
<header><h3 style="margin:0">🧪 ตรวจข้อสอบ</h3></header>
<main>
  <div class="card">
    <div class="row">
      <div style="flex:1 1 360px"><div class="label">Subject</div><input id="subG" class="input" placeholder="ต้องมีเฉลยอยู่แล้ว"></div>
      <div style="align-self:end"><input id="img" type="file" accept="image/*" class="input"></div>
      <div style="align-self:end"><button id="btn" class="btn">ตรวจ</button></div>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>

    <div id="sum" style="display:none;margin-top:14px;display:flex;gap:16px;align-items:center">
      <div class="donut" id="donut" style="--p:0"><b id="pct">0%</b></div>
      <div><div class="label">ผลรวม</div><div style="font-size:20px"><b id="score">0 / 0</b></div><div class="label">เวอร์ชันเฉลย: <span id="ver">-</span></div></div>
    </div>
    <table class="table" id="tbl" style="margin-top:10px;display:none"></table>
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

  var rows = d.answers_marked.map(function(a, i){
    var mark = d.correctness[i] ? '✅' : '❌';
    var ans = a && a.length ? a : '-';
    return '<tr><td>' + (i+1) + '</td><td>' + ans + '</td><td>' + mark + '</td></tr>';
  }).join('');
  $('#tbl').innerHTML =
    '<thead><tr><th>ข้อ</th><th>คำตอบที่เลือก</th><th>ถูก/ผิด</th></tr></thead>' +
    '<tbody>' + rows + '</tbody>';
}

document.getElementById('btn').addEventListener('click', function(){
  var sub = $('#subG').value.trim();
  var f = $('#img').files[0];
  if(!sub || !f){ alert('กรอก Subject และเลือกไฟล์รูป'); return; }

  var fd = new FormData();
  fd.append('subject', sub);
  fd.append('image', f);

  fetch('/api/grade', { method:'POST', body: fd })
    .then(function(r){ return r.json().then(function(js){ return {ok:r.ok, js:js}; }); })
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

@app.get("/scan", response_class=HTMLResponse)
def scan_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – สแกนด้วยกล้อง</title><style>__CSS__</style>
<style>
.cam{position:relative;max-width:980px;margin:12px auto;border:1px solid var(--bd);border-radius:16px;overflow:hidden}
.cam video{width:100%;display:block;background:#000}
.cam canvas{position:absolute;inset:0;pointer-events:none}
.tip{font-size:13px;color:var(--muted)}
.rowc{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
</style></head>
<body>
<header><h3 style="margin:0">📷 สแกนด้วยกล้อง (จับสี่เหลี่ยม 4 มุม)</h3></header>
<main>
  <div class="card">
    <div class="rowc">
      <div style="flex:1 1 360px"><div class="label">Subject</div><input id="sub" class="input" placeholder="ต้องมีเฉลยอยู่แล้ว"></div>
      <button id="btnStart" class="btn">เริ่มสแกน</button>
      <button id="btnStop" class="btn sub" disabled>หยุด</button>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>
    <div class="tip" style="margin:8px 0">ชูแผ่นให้มาร์คสี่เหลี่ยมอยู่ครบทั้ง 4 มุมและชัดเจน ระบบจะถ่ายอัตโนมัติเมื่อจับได้</div>
    <div class="cam" id="cam">
      <video id="video" playsinline autoplay muted></video>
      <canvas id="ov"></canvas>
    </div>
  </div>
</main>

<script>
var $=function(q){return document.querySelector(q);};
var stream=null;

function sizeCanvas(){
  var v=$('#video'), c=$('#ov');
  c.width=v.clientWidth||v.videoWidth||640;
  c.height=v.clientHeight||v.videoHeight||480;
}
async function startCam(){
  if(stream){ stream.getTracks().forEach(function(t){t.stop();}); }
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment', width:{ideal:1280}, height:{ideal:720}}, audio:false});
  var v=$('#video'); v.srcObject=stream; v.onloadedmetadata=function(){ v.play(); sizeCanvas(); };
}
function stopCam(){ if(stream){stream.getTracks().forEach(function(t){t.stop();}); stream=null; } }
document.getElementById('btnStart').addEventListener('click', async function(){ await startCam(); this.disabled=true; $('#btnStop').disabled=false; });
document.getElementById('btnStop').addEventListener('click', function(){ stopCam(); $('#btnStart').disabled=false; this.disabled=true; });
window.addEventListener('resize', sizeCanvas);
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)

# ---------- API ----------
class GradeResponse(BaseModel):
    subject: str
    version: str
    timestamp: str
    num_questions: int
    answers_marked: List[str]
    correctness: List[bool]
    total_correct: int

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
    if template_count not in (20, 40, 60):
        raise HTTPException(400, "template_count ต้องเป็น 20, 40 หรือ 60")
    try:
        key_list = json.loads(answers)
    except Exception:
        raise HTTPException(400, "answers ต้องเป็น JSON array")
    if len(key_list) != template_count:
        raise HTTPException(400, "จำนวนคำตอบไม่ตรงกับจำนวนข้อ")

    preset_name = {20:"A4_20Q_5C",40:"A4_40Q_5C",60:"A4_60Q_5C"}[template_count]
    if preset_name not in PRESETS:
        raise HTTPException(500, "preset ไม่ถูกต้อง")
    if key_exists(subject) and confirm not in ("1","true","True","YES","yes"):
        raise HTTPException(409, f"วิชา '{subject}' มีเฉลยอยู่แล้ว ต้องส่ง confirm=1 เพื่ออัปเดต")

    meta = save_key(subject, preset_name, key_list)
    return {"message":"บันทึกเฉลยสำเร็จ","subject":subject,"version":meta["version"],"num_questions":len(key_list)}

@app.post("/api/grade", response_model=GradeResponse)
async def grade_sheet(subject: str = Form(...), image: UploadFile = File(...)):
    meta = load_latest_key(subject)
    preset = PRESETS[meta["preset"]]
    key_list = meta["answers"]

    img_bgr = read_image(image)
    image.file.seek(0)
    save_upload(image, "data/uploads")

    student_ans, warped = get_answers_from_image(img_bgr, preset)
    correctness = compute_correctness(student_ans, key_list)
    total_correct = int(sum(correctness))

    rid = f"{subject}__{int(time.time())}__{uuid.uuid4().hex}"
    result = {
        "result_id": rid, "subject": subject, "version": meta["version"], "timestamp": now_ts(),
        "num_questions": len(student_ans), "answers_marked": student_ans,
        "correctness": correctness, "total_correct": total_correct
    }
    with open(os.path.join("data/results", f"{rid}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if SAVE_DEBUG:
        cv.imwrite(os.path.join("data/debug", f"{rid}__warped.jpg"), warped)

    return GradeResponse(**result)

@app.post("/api/probe_markers")
async def probe_markers(image: UploadFile = File(...)):
    try:
        img = read_image(image)
        h, w = img.shape[:2]
        corners = find_corner_markers(img)
        norm = [[float(x)/w, float(y)/h] for (x, y) in corners.tolist()]
        return {"ok": True, "w": w, "h": h, "corners_norm": norm}
    except Exception:
        return {"ok": False}

# ---------- shortcuts สำหรับทดสอบนับข้อ ----------
@app.post("/api/grade_20")
async def grade_20(image: UploadFile = File(...)):
    preset = PRESETS["A4_20Q_5C"]
    img = read_image(image)
    answers, _ = get_answers_from_image(img, preset)
    return {"answers": answers}

@app.post("/api/grade_40")
async def grade_40(image: UploadFile = File(...)):
    preset = PRESETS["A4_40Q_5C"]
    img = read_image(image)
    answers, _ = get_answers_from_image(img, preset)
    return {"answers": answers}

@app.post("/api/grade_60")
async def grade_60(image: UploadFile = File(...)):
    preset = PRESETS["A4_60Q_5C"]
    img = read_image(image)
    answers, _ = get_answers_from_image(img, preset)
    return {"answers": answers}
