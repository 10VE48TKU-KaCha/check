# main.py — /keys (กรอก/แก้เฉลย) และ /check (ตรวจจากรูป) — FIX: no f-strings in HTML
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

SAVE_DEBUG = os.getenv("SAVE_DEBUG", "0") == "1"

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

def find_corner_markers(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 10)
    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    h, w = th.shape[:2]
    boxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < (w*h)*0.0002:
            continue
        approx = cv.approxPolyDP(cnt, 0.04*cv.arcLength(cnt, True), True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            x, y, ww, hh = cv.boundingRect(approx)
            fill_ratio = area / (ww*hh + 1e-6)
            if fill_ratio > 0.6:
                cx, cy = x+ww/2, y+hh/2
                d = min((cx-0)**2+(cy-0)**2,
                        (cx-w)**2+(cy-0)**2,
                        (cx-w)**2+(cy-h)**2,
                        (cx-0)**2+(cy-h)**2)
                boxes.append((d, approx.reshape(-1, 2).astype(np.float32)))
    if len(boxes) < 4:
        raise HTTPException(422, "ตรวจไม่พบสี่เหลี่ยมทึบครบ 4 มุม")
    boxes.sort(key=lambda x: x[0])
    centers = np.array([b[1].mean(axis=0) for b in boxes[:4]], dtype=np.float32)
    return centers

def extract_answer_grid(warp: np.ndarray, preset: GridPreset) -> np.ndarray:
    h, w = warp.shape[:2]
    x, y, ww, hh = preset.roi
    x1, y1 = int(x*w), int(y*h)
    x2, y2 = int((x+ww)*w), int((y+hh)*h)
    return warp[y1:y2, x1:x2].copy()

def detect_marks(grid_img: np.ndarray, preset: GridPreset) -> List[str]:
    """
    อ่านคำตอบทีละข้อจากภาพกริด โดยรองรับลำดับหมายเลขแบบ column-major
    (เช่น 1..10 คอลัมน์ซ้าย แล้ว 11..20 คอลัมน์ขวา)
    """
    # ปรับภาพให้ทนการฝนเบา
    gray = cv.cvtColor(grid_img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    th = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 8
    )
    th = cv.medianBlur(th, 3)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    H, W = th.shape[:2]
    rows, cols = preset.grid_rows, preset.grid_cols
    total_cells = rows * cols
    if preset.num_questions != total_cells:
        raise HTTPException(500, f"preset.num_questions ({preset.num_questions}) != rows*cols ({total_cells})")

    nC = len(preset.choices)
    margin_y, margin_x = 0.12, 0.06   # ตัดขอบเพื่อไม่ให้เส้นกรอบ/ตัวเลขรบกวน
    cell_h, cell_w = H / rows, W / cols

    # ⬇️ ลำดับตำแหน่งที่ต้องอ่านต่อข้อ (สำคัญ!)
    pairs = []
    if getattr(preset, "column_major", False):
        # 1..ลงตามคอลัมน์ซ้าย → ขวา
        for c in range(cols):
            for r in range(rows):
                pairs.append((r, c))
    else:
        # เดิม: แถวก่อนคอลัมน์
        for r in range(rows):
            for c in range(cols):
                pairs.append((r, c))

    answers: List[str] = []

    for (r, c) in pairs:
        y1 = int(r * cell_h + cell_h * margin_y)
        y2 = int((r + 1) * cell_h - cell_h * margin_y)
        x1 = int(c * cell_w + cell_w * margin_x)
        x2 = int((c + 1) * cell_w - cell_w * margin_x)

        cell = th[y1:y2, x1:x2]
        h_cell, w_cell = cell.shape[:2]

        # แบ่งแนวนอน 5 ส่วน (ก ข ค ง จ)
        slice_w = w_cell / nC
        scores = []
        for i in range(nC):
            sx1 = int(i * slice_w + slice_w * 0.25)
            sx2 = int((i + 1) * slice_w - slice_w * 0.25)
            sy1 = int(h_cell * 0.25)
            sy2 = int(h_cell * 0.75)
            if sx2 <= sx1 or sy2 <= sy1:
                scores.append(0.0);  continue
            roi = cell[sy1:sy2, sx1:sx2]
            filled = float((roi > 0).sum()) / (roi.size + 1e-6)
            scores.append(filled)

        best_i = int(np.argmax(scores))
        best_score = scores[best_i]
        answers.append(preset.choices[best_i] if best_score > 0.10 else "")

    return answers

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
    if SAVE_DEBUG:
        cv.imwrite(os.path.join("data/debug", f"warped_{int(time.time())}.jpg"), warped)
    grid = extract_answer_grid(warped, preset)
    return detect_marks(grid, preset), warped

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

# ---------- Pages (NO f-strings; use placeholder for CSS) ----------
@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OMR – Dashboard</title><style>__CSS__</style></head>
<body><header><h3 style="margin:0">📊 OMR Dashboard</h3></header>
<main>
  <div class="card">
    <h3>สแกนด้วยกล้อง (เหมือน ZipGrade)</h3>
    <p>จับสี่เหลี่ยมทึบ 4 มุมแบบเรียลไทม์ แล้วถ่ายอัตโนมัติเมื่อเสถียร</p>
    <a class="btn" href="/scan">เปิดหน้า /scan</a>
  </div>
  <div class="card">
    <h3>ตั้ง/แก้ไขเฉลย</h3>
    <a class="btn" href="/keys">เปิดหน้า /keys</a>
  </div>
  <div class="card">
    <h3>ตรวจจากรูปเดี่ยว</h3>
    <a class="btn" href="/check">เปิดหน้า /check</a>
  </div>
  <div class="card">
    <h3>API Docs</h3>
    <a class="btn sub" href="/docs">/docs</a>
  </div>
</main></body></html>
    """
    return html.replace("__CSS__", DARK_CSS)


@app.get("/keys", response_class=HTMLResponse)
def keys_page():
    html = """
<!doctype html><html lang="th"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ตั้งเฉลยตามลำดับ</title><style>__CSS__</style>
<style>
.qrow{display:grid;grid-template-columns:84px repeat(5,1fr);gap:10px;align-items:center;margin:6px 0}
.choice{display:flex;align-items:center;gap:6px}
.radio{appearance:none;width:18px;height:18px;border-radius:50%;border:2px solid #334155;display:inline-block;position:relative}
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
        <select id="cnt" class="select">
          <option value="20">20</option><option value="40">40</option><option value="60">60</option>
        </select>
      </div>
      <div style="align-self:end"><button class="btn sub" onclick="loadKey()">โหลดเฉลยล่าสุด</button></div>
    </div>

    <div id="status" style="margin:10px 0"><span class="badge">ยังไม่ได้ตรวจสอบเฉลย</span></div>

    <!-- โหมดกรอกเร็วตามลำดับ -->
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
let N=20, idx=1;   // จำนวนข้อ & ข้อที่กำลังกรอก

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

// ปุ่มกรอกเร็ว
document.addEventListener('click',e=>{
  const t=e.target.closest('.qbtn'); if(!t) return;
  if(t.id==='btnSkip'){ next(); return; }
  if(t.id==='btnBack'){ back(); return; }
  if(t.dataset.v){ setAnswer(idx, t.dataset.v); next(); }
});

// คีย์ลัด 1..5 = ก/ข/ค/ง/จ | ArrowLeft/Right = ย้อน/ถัดไป | Space = ข้าม
document.addEventListener('keydown',e=>{
  if(e.key>='1' && e.key<='5'){ const v = CHOICES[parseInt(e.key)-1]; setAnswer(idx,v); next(); }
  if(e.key==='ArrowRight'){ next(); }
  if(e.key==='ArrowLeft'){ back(); }
  if(e.key===' '){ next(); }
});

// init
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
        resp.update(
            preset=m["preset"], version=m["version"],
            created_at=m["created_at"], num_questions=len(m["answers"])
        )
    return resp

@app.get("/api/keys/export")
async def key_export(subject: str):
    m = load_latest_key(subject)
    return {"subject": subject, "preset": m["preset"], "version": m["version"], "answers": m["answers"]}

@app.post("/api/keys/set_manual")
async def set_manual_key(
    subject: str = Form(...),
    template_count: int = Form(...),     # 20 / 40 / 60
    answers: str = Form(...),            # JSON array ["ก","","ค",...]
    confirm: str = Form("0")             # "1" เพื่ออัปเดตแทนที่ของเดิม
):
    if template_count not in (20, 40, 60):
        raise HTTPException(400, "template_count ต้องเป็น 20, 40 หรือ 60")
    try:
        key_list = json.loads(answers)
    except Exception:
        raise HTTPException(400, "answers ต้องเป็น JSON array")
    if len(key_list) != template_count:
        raise HTTPException(400, "จำนวนคำตอบไม่ตรงกับจำนวนข้อ")

    preset_name = {20: "A4_20Q_5C", 40: "A4_40Q_5C", 60: "A4_60Q_5C"}[template_count]
    if preset_name not in PRESETS:
        raise HTTPException(500, "preset ไม่ถูกต้อง")

    # สร้างได้ครั้งเดียว หากจะอัปเดตต้องส่ง confirm=1
    if key_exists(subject) and confirm not in ("1", "true", "True", "YES", "yes"):
        raise HTTPException(409, f"วิชา '{subject}' มีเฉลยอยู่แล้ว ต้องส่ง confirm=1 เพื่ออัปเดต")

    meta = save_key(subject, preset_name, key_list)
    return {"message": "บันทึกเฉลยสำเร็จ", "subject": subject, "version": meta["version"], "num_questions": len(key_list)}

@app.post("/api/grade", response_model=GradeResponse)
async def grade_sheet(subject: str = Form(...), image: UploadFile = File(...)):
    meta = load_latest_key(subject)
    preset = PRESETS[meta["preset"]]
    key_list = meta["answers"]

    # อ่านภาพ + เก็บต้นฉบับ
    img_bgr = read_image(image)
    image.file.seek(0)
    save_upload(image, "data/uploads")

    # ตรวจจากภาพ
    student_ans, warped = get_answers_from_image(img_bgr, preset)
    correctness = compute_correctness(student_ans, key_list)
    total_correct = int(sum(correctness))

    # เซฟผลลัพธ์แบบย่อ
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
    """
    ตรวจแบบเร็วว่าภาพมีมาร์กสี่เหลี่ยม 4 มุมแล้วหรือยัง
    ถ้ามี คืนค่าพิกัด normalized เพื่อนำไปวาด overlay
    """
    try:
        img = read_image(image)
        h, w = img.shape[:2]
        corners = find_corner_markers(img)  # Nx2 float
        norm = [[float(x)/w, float(y)/h] for (x, y) in corners.tolist()]
        return {"ok": True, "w": w, "h": h, "corners_norm": norm}
    except Exception:
        return {"ok": False}


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
      <div style="width:220px"><div class="label">กล้อง</div><select id="camera" class="select"></select></div>
      <label class="choice"><input id="flip" type="checkbox"> กลับภาพ (mirror)</label>
      <button id="btnStart" class="btn">เริ่มสแกน</button>
      <button id="btnStop" class="btn sub" disabled>หยุด</button>
      <a class="btn sub" href="/">กลับหน้าแรก</a>
    </div>
    <div class="tip" style="margin:8px 0">ให้มาร์กสี่เหลี่ยมอยู่ครบทั้ง 4 มุม ระบบจะถ่ายอัตโนมัติเมื่อจับได้ 2 เฟรมติดกัน</div>

    <div class="cam">
      <video id="video" playsinline autoplay muted></video>
      <canvas id="ov"></canvas>
    </div>

    <div id="sum" style="display:none;margin-top:14px;display:flex;gap:16px;align-items:center">
      <div class="donut" id="donut" style="--p:0"><b id="pct">0%</b></div>
      <div><div class="label">ผลรวม</div><div style="font-size:20px"><b id="score">0 / 0</b></div><div class="label">เวอร์ชันเฉลย: <span id="ver">-</span></div></div>
    </div>
    <table class="table" id="tbl" style="margin-top:10px;display:none"></table>
  </div>
</main>

<script>
var $=function(q){return document.querySelector(q);};
var stream=null, running=false, busy=false, okCount=0, lastBlob=null;

function fitCanvas(){
  var v=$('#video'), c=$('#ov');
  c.width=v.clientWidth||v.videoWidth||640;
  c.height=v.clientHeight||v.videoHeight||480;
}
function drawCorners(norm){
  var c=$('#ov'), ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!norm) return;
  ctx.lineWidth=2; ctx.strokeStyle='rgba(34,197,94,.9)';
  for(var i=0;i<norm.length;i++){
    var x=norm[i][0]*c.width, y=norm[i][1]*c.height;
    ctx.beginPath(); ctx.arc(x,y,8,0,Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x-14,y); ctx.lineTo(x+14,y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x,y-14); ctx.lineTo(x,y+14); ctx.stroke();
  }
}

async function listCams(){
  try{
    var ds=await navigator.mediaDevices.enumerateDevices();
    var cams=ds.filter(function(d){return d.kind==='videoinput';});
    var sel=$('#camera'); sel.innerHTML='';
    cams.forEach(function(d,i){
      var o=document.createElement('option'); o.value=d.deviceId; o.textContent=d.label||('Camera '+(i+1)); sel.appendChild(o);
    });
  }catch(e){ console.error(e); }
}
async function startCam(){
  var id=$('#camera').value||undefined;
  if(stream){ stream.getTracks().forEach(function(t){t.stop();}); }
  stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:id?id:undefined,width:{ideal:1280},height:{ideal:720},facingMode:'environment'},audio:false});
  var v=$('#video'); v.srcObject=stream; v.onloadedmetadata=function(){ v.play(); fitCanvas(); };
}
function stopCam(){ if(stream){stream.getTracks().forEach(function(t){t.stop();}); stream=null; } }

function canvasToBlob(canvas, cb){
  if(canvas.toBlob){ canvas.toBlob(function(b){ cb(b); }, 'image/jpeg', 0.7); }
  else{ var dataURL=canvas.toDataURL('image/jpeg',0.7); var b64=dataURL.split(',')[1]; var bin=atob(b64); var arr=new Uint8Array(bin.length); for(var i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i); cb(new Blob([arr],{type:'image/jpeg'})); }
}

function resetSummary(){
  $('#sum').style.display='none'; $('#tbl').style.display='none';
  $('#donut').style.setProperty('--p',0); $('#pct').textContent='0%'; $('#score').textContent='0 / 0'; $('#ver').textContent='-';
}

async function tick(){
  if(!running || busy) return;
  busy=true;
  try{
    var v=$('#video'), c=$('#ov'), ctx=c.getContext('2d');
    var flip=$('#flip').checked;
    ctx.save();
    if(flip){ ctx.translate(c.width,0); ctx.scale(-1,1); }
    ctx.drawImage(v,0,0,c.width,c.height);
    ctx.restore();

    await new Promise(function(res){ canvasToBlob(c,res); }).then(async function(blob){
      lastBlob=blob;
      var fd=new FormData(); fd.append('image', blob, 'cam.jpg');
      var r=await fetch('/api/probe_markers',{method:'POST',body:fd});
      var js=await r.json();
      if(js.ok){
        drawCorners(js.corners_norm);
        okCount++;
        if(okCount>=2){
          okCount=0;
          var sub=$('#sub').value.trim(); if(!sub){ alert('กรอก Subject ก่อน'); running=false; $('#btnStart').disabled=false; $('#btnStop').disabled=true; return; }
          var fd2=new FormData(); fd2.append('subject', sub); fd2.append('image', lastBlob, 'snap.jpg');
          var r2=await fetch('/api/grade',{method:'POST',body:fd2}); var js2=await r2.json();
          if(r2.ok){ renderResult(js2); } else { alert(js2.detail||'ตรวจไม่สำเร็จ'); }
        }
      }else{ drawCorners(null); okCount=0; }
    });
  }catch(e){ console.error(e); }
  finally{ busy=false; }
  if(running){ requestAnimationFrame(tick); }
}

function renderResult(d){
  var pct=Math.round(100*d.total_correct/d.num_questions);
  $('#sum').style.display='flex'; $('#tbl').style.display='table';
  $('#donut').style.setProperty('--p',pct); $('#pct').textContent=String(pct)+'%';
  $('#score').textContent=d.total_correct+' / '+d.num_questions; $('#ver').textContent=d.version;
  var rows=d.answers_marked.map(function(a,i){
    var mark=d.correctness[i]?'✅':'❌'; var ans=a&&a.length?a:'-';
    return '<tr><td>'+(i+1)+'</td><td>'+ans+'</td><td>'+mark+'</td></tr>';
  }).join('');
  $('#tbl').innerHTML='<thead><tr><th>ข้อ</th><th>คำตอบที่เลือก</th><th>ถูก/ผิด</th></tr></thead><tbody>'+rows+'</tbody>';
}

document.getElementById('btnStart').addEventListener('click', async function(){
  resetSummary();
  await listCams();
  await startCam();
  running=true; $('#btnStart').disabled=true; $('#btnStop').disabled=false;
  requestAnimationFrame(tick);
});
document.getElementById('btnStop').addEventListener('click', function(){
  running=false; stopCam(); $('#btnStart').disabled=false; $('#btnStop').disabled=true; drawCorners(null);
});
window.addEventListener('resize', fitCanvas);
</script>
</body></html>
    """
    return html.replace("__CSS__", DARK_CSS)
