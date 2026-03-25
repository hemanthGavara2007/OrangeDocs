#!/usr/bin/env python3
"""
OrangeDocs v3.0 — Multi-page session handwriting to polished PDF.
  - /session/add_page    : add one handwriting image per call (builds up a session)
  - /session/add_diagram : add a diagram image (embedded as-is, no OCR)
  - /session/build       : combine all session pages -> download PDF
  - /session/status      : see pages queued in current session
  - /session/clear       : wipe the session
  - CORS enabled         : works from Android Studio WebView / Retrofit

SETUP:
  pip install flask flask-cors ollama pillow reportlab pymupdf numpy
  ollama pull glm-ocr
  python app.py  ->  http://localhost:5000
"""

import base64, io, json, os, re, tempfile, time, uuid, statistics, collections
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import ollama
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

app = Flask(__name__, static_folder=".")
CORS(app)   # allow cross-origin from Android app

MAX_FILE_SIZE   = 100 * 1024 * 1024
OUTPUT_DIR      = tempfile.mkdtemp(prefix="orangedocs_")
ALLOWED_EXT     = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# Layout
MARGIN        = 50.0
LINE_SPACING  = 1.35
MIN_FONT      = 7.0
MAX_FONT      = 36.0
DEFAULT_FONT  = 11.0

OCR_MAX_SIDE  = 1024
OCR_MIN_SIDE  = 64

COL_BG      = (1.00, 1.00, 1.00)
COL_ACCENT  = (1.00, 0.42, 0.00)
COL_TEXT    = (0.08, 0.08, 0.08)
COL_HEADING = (0.04, 0.04, 0.04)
COL_RULE    = (0.88, 0.88, 0.88)
COL_LABEL   = (0.60, 0.60, 0.60)
FONT_BODY   = "Helvetica"
FONT_BOLD   = "Helvetica-Bold"

# ── Session store (in-memory, keyed by session_id) ────────────────────────────
# Each session = list of page dicts:
#   {"type": "handwriting", "image": PIL.Image, "label": str}
#   {"type": "diagram",     "image": PIL.Image, "label": str}
_sessions = {}   # session_id -> {"pages": [...], "created": float}
SESSION_TTL = 3600  # 1 hour

def _get_session(sid):
    now = time.time()
    if sid not in _sessions:
        _sessions[sid] = {"pages": [], "created": now}
    _sessions[sid]["last_used"] = now
    return _sessions[sid]

def _gc_sessions():
    """Remove sessions older than SESSION_TTL."""
    now = time.time()
    dead = [s for s, v in _sessions.items()
            if now - v.get("last_used", v["created"]) > SESSION_TTL]
    for s in dead:
        del _sessions[s]


# ── 1. Rasterisation ──────────────────────────────────────────────────────────

def _adaptive_dpi(raw_len):
    if raw_len > 20 * 1024 * 1024: return 120
    if raw_len > 8  * 1024 * 1024: return 150
    return 200

def _cap_image(img, max_side):
    w, h = img.size
    if max(w, h) <= max_side: return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def rasterize_pdf(raw, dpi=None):
    if dpi is None: dpi = _adaptive_dpi(len(raw))
    try:
        import fitz
        doc  = fitz.open(stream=raw, filetype="pdf")
        zoom = dpi / 72.0
        mat  = fitz.Matrix(zoom, zoom)
        imgs = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(_cap_image(img, 4000))
        doc.close()
        return imgs
    except ImportError:
        pass
    try:
        from pdf2image import convert_from_bytes
        return [_cap_image(p.convert("RGB"), 4000)
                for p in convert_from_bytes(raw, dpi=dpi)]
    except Exception as e:
        raise RuntimeError(f"Cannot rasterize PDF: {e}")

def load_image(raw, name="image"):
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return _cap_image(img, 4000)
    except Exception as e:
        raise RuntimeError(f"Cannot open image '{name}': {e}")


# ── 2. Paragraph segmentation ─────────────────────────────────────────────────

@dataclass
class Region:
    x: int; y: int; w: int; h: int
    crop: Image.Image = field(repr=False)

def _binarise(img):
    grey = np.array(img.convert("L"), dtype=np.float32)
    blur = np.array(img.convert("L").filter(ImageFilter.GaussianBlur(15)), dtype=np.float32)
    return grey < (blur - 10)

def _dilate(mask, kh, kw):
    H, W = mask.shape
    m    = mask.astype(np.uint8)
    cs   = np.cumsum(m, axis=1)
    cs   = np.hstack([np.zeros((H, kw), np.uint8), cs])
    horiz = (cs[:, kw:] - cs[:, :-kw]) > 0
    cs2  = np.cumsum(horiz.astype(np.uint8), axis=0)
    cs2  = np.vstack([np.zeros((kh, W), np.uint8), cs2])
    return (cs2[kh:, :] - cs2[:-kh, :]) > 0

def _merge_nearby(regs, gap=40):
    if not regs: return regs
    regs = sorted(regs, key=lambda r: r.y)
    merged = [regs[0]]
    for r in regs[1:]:
        prev = merged[-1]
        v_close   = (r.y - (prev.y + prev.h)) < gap
        h_overlap = not (r.x + r.w < prev.x or r.x > prev.x + prev.w)
        if v_close and h_overlap:
            nx  = min(prev.x, r.x); ny  = min(prev.y, r.y)
            nx2 = max(prev.x + prev.w, r.x + r.w)
            ny2 = max(prev.y + prev.h, r.y + r.h)
            merged[-1] = Region(nx, ny, nx2 - nx, ny2 - ny, None)
        else:
            merged.append(r)
    return merged

def segment_regions(img, min_area=500, pad=10):
    H, W = img.size[1], img.size[0]
    ink  = _binarise(img)
    para = _dilate(_dilate(ink, 14, 30), 26, 70)
    raw_regs = []
    try:
        from scipy.ndimage import label as sp_label
        lbl, n = sp_label(para)
        for lid in range(1, n + 1):
            ys, xs = np.where(lbl == lid)
            if not len(xs): continue
            x0,y0,x1,y1 = int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
            if (x1-x0)*(y1-y0) < min_area: continue
            px0,py0 = max(0,x0-pad), max(0,y0-pad)
            px1,py1 = min(W,x1+pad), min(H,y1+pad)
            raw_regs.append(Region(px0,py0,px1-px0,py1-py0,None))
    except ImportError:
        row_ink = para.any(axis=1)
        in_b, y0 = False, 0
        for ri, has in enumerate(row_ink):
            if has and not in_b:   in_b=True; y0=ri
            elif not has and in_b:
                in_b = False
                cols = np.where(para[y0:ri].any(axis=0))[0]
                if not len(cols): continue
                x0c,x1c = int(cols.min()),int(cols.max())
                px0,py0 = max(0,x0c-pad), max(0,y0-pad)
                px1,py1 = min(W,x1c+pad), min(H,ri+pad)
                if (px1-px0)*(py1-py0) >= min_area:
                    raw_regs.append(Region(px0,py0,px1-px0,py1-py0,None))
    raw_regs = _merge_nearby(raw_regs, gap=40)
    regs = []
    for r in raw_regs:
        px0,py0 = max(0,r.x),max(0,r.y)
        px1,py1 = min(W,r.x+r.w),min(H,r.y+r.h)
        if (px1-px0)<OCR_MIN_SIDE or (py1-py0)<OCR_MIN_SIDE: continue
        crop = img.crop((px0,py0,px1,py1))
        regs.append(Region(px0,py0,px1-px0,py1-py0,crop))
    regs.sort(key=lambda r: (r.y, r.x))
    return regs


# ── 3. OCR per region ─────────────────────────────────────────────────────────

OCR_SYSTEM = """You are a precise handwriting OCR engine.
You receive a cropped image of ONE handwritten paragraph or heading.
Your ONLY job is to transcribe every word exactly as written.
Do NOT fix spelling, grammar, or punctuation.
Mark any illegible word as [illegible].

Respond with ONLY valid JSON — no markdown fences, no extra text:
{"text":"<full transcription>","font_size_est":<number>,"is_heading":<true|false>}

font_size_est: estimate the apparent body font size (body text ~11-13, headings ~16-24).
is_heading: true if this looks like a title or heading.
If the image contains NO text at all, return: {"text":"","font_size_est":11,"is_heading":false}"""

OCR_USER = "Transcribe all handwritten text in this image. Return only the JSON object."

# glm-ocr outputs plain text directly — no JSON wrapping needed
OCR_USER_GLM = "Transcribe all handwritten text in this image exactly as written. Mark illegible words as [illegible]."

def _prepare_crop(crop):
    w, h = crop.size
    if max(w, h) < 256:
        scale = 256 / max(w, h)
        crop = crop.resize((int(w*scale),int(h*scale)), Image.LANCZOS)
        w, h = crop.size
    if max(w, h) > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / max(w, h)
        crop = crop.resize((int(w*scale),int(h*scale)), Image.LANCZOS)
    return ImageEnhance.Contrast(crop).enhance(1.3)

def _parse_json_robust(raw):
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?","",raw).strip().replace("```","").strip()
    depth,start,end = 0,-1,-1
    for i,ch in enumerate(raw):
        if ch == '{':
            if depth==0: start=i
            depth+=1
        elif ch == '}':
            depth-=1
            if depth==0: end=i+1; break
    if start!=-1 and end!=-1:
        try: return json.loads(raw[start:end])
        except: pass
    tm = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    fm = re.search(r'"font_size_est"\s*:\s*(\d+(?:\.\d+)?)', raw)
    hm = re.search(r'"is_heading"\s*:\s*(true|false)', raw)
    return {
        "text":          tm.group(1) if tm else "",
        "font_size_est": float(fm.group(1)) if fm else DEFAULT_FONT,
        "is_heading":    (hm.group(1)=="true") if hm else False,
    }

def _f(v, fb=0.0):
    try:    return float(v)
    except: return float(fb)

def _is_glm(model):
    return model and model.lower().startswith("glm-ocr")

def ocr_region(region, model):
    crop = _prepare_crop(region.crop)
    buf  = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    b64  = base64.b64encode(buf.getvalue()).decode()

    # glm-ocr is a dedicated OCR model: outputs plain text, no JSON, no system role
    if _is_glm(model):
        messages = [{"role": "user", "content": OCR_USER_GLM, "images": [b64]}]
    else:
        user_prompt = OCR_SYSTEM + "\n\n" + OCR_USER
        messages = [{"role": "user", "content": user_prompt, "images": [b64]}]

    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0, "num_predict": 1024, "num_ctx": 4096},
        )
        raw = resp["message"]["content"]
    except Exception as e:
        return {"text": "", "font_size_est": DEFAULT_FONT, "is_heading": False, "_error": str(e)}

    # glm-ocr returns plain text — wrap it directly without JSON parsing
    if _is_glm(model):
        text = raw.strip()
        # Estimate heading: short line (<=6 words) that starts with a capital
        words = text.split()
        is_heading = len(words) <= 6 and bool(words) and words[0][0].isupper()
        return {
            "text":          text,
            "font_size_est": DEFAULT_FONT,
            "is_heading":    is_heading,
        }

    result = _parse_json_robust(raw)
    result["text"]          = str(result.get("text", "")).strip()
    result["font_size_est"] = max(MIN_FONT, min(_f(result.get("font_size_est", DEFAULT_FONT)), MAX_FONT))
    result["is_heading"]    = bool(result.get("is_heading", False))
    return result


# ── 4. PDF builder ────────────────────────────────────────────────────────────

@dataclass
class PlacedBlock:
    pdf_x:float; pdf_top:float; pdf_w:float; font:str; fsize:float
    is_heading:bool; lines:List[str]; line_h:float; total_h:float

def _resolve_blocks(regions_data, img_w, img_h, pdf_w, pdf_h):
    content_w = pdf_w - 2*MARGIN
    content_h = pdf_h - MARGIN - 24 - 36
    sx = content_w / max(img_w,1)
    sy = content_h / max(img_h,1)
    blocks = []
    for rd in regions_data:
        text = rd.get("text","").strip()
        if not text: continue
        is_h  = bool(rd.get("is_heading",False))
        fsize = max(MIN_FONT, min(_f(rd.get("font_size_est",DEFAULT_FONT)), MAX_FONT))
        if is_h: fsize = max(fsize, DEFAULT_FONT*1.25)
        font    = FONT_BOLD if is_h else FONT_BODY
        pdf_x   = MARGIN + rd["region_x"]*sx
        pdf_top = (pdf_h-24-MARGIN) - rd["region_y"]*sy
        pdf_bw  = max(rd["region_w"]*sx, 60.0)
        lines   = simpleSplit(text, font, fsize, pdf_bw)
        line_h  = fsize*LINE_SPACING
        total_h = len(lines)*line_h + (8 if is_h else 4)
        blocks.append(PlacedBlock(
            pdf_x=pdf_x, pdf_top=pdf_top, pdf_w=pdf_bw,
            font=font, fsize=fsize, is_heading=is_h,
            lines=lines, line_h=line_h, total_h=total_h,
        ))
    blocks.sort(key=lambda b: -b.pdf_top)
    GAP = 8.0
    for i in range(len(blocks)):
        for j in range(i+1, len(blocks)):
            a,b = blocks[i], blocks[j]
            if a.pdf_x+a.pdf_w <= b.pdf_x or b.pdf_x+b.pdf_w <= a.pdf_x: continue
            a_bot = a.pdf_top-a.total_h
            if b.pdf_top > a_bot-GAP: b.pdf_top = a_bot-GAP
    return blocks

def _chrome(c, pdf_w, pdf_h, page_num, total):
    c.setFillColorRGB(*COL_ACCENT)
    c.rect(0, pdf_h-5, pdf_w, 5, stroke=0, fill=1)
    c.setFont(FONT_BOLD, 7.5)
    c.setFillColorRGB(*COL_LABEL)
    c.drawString(MARGIN, pdf_h-18, "OrangeDocs")
    c.setStrokeColorRGB(*COL_RULE)
    c.setLineWidth(0.4)
    c.line(MARGIN, 34, pdf_w-MARGIN, 34)
    c.setFont(FONT_BODY, 7)
    c.setFillColorRGB(*COL_LABEL)
    c.drawRightString(pdf_w-MARGIN, 22, f"Page {page_num} of {total}")

def _draw_diagram_page(c, img, pdf_w, pdf_h, page_num, total, label):
    """Embed diagram image centered on the page, preserving aspect ratio."""
    c.setPageSize((pdf_w, pdf_h))
    c.setFillColorRGB(*COL_BG)
    c.rect(0,0,pdf_w,pdf_h, stroke=0, fill=1)
    _chrome(c, pdf_w, pdf_h, page_num, total)

    # Label bar
    c.setFont(FONT_BOLD, 8)
    c.setFillColorRGB(*COL_ACCENT)
    diag_label = f"[DIAGRAM] {label}" if label else "[DIAGRAM]"
    c.drawString(MARGIN, pdf_h-32, diag_label)

    usable_w = pdf_w - 2*MARGIN
    usable_h = pdf_h - MARGIN - 24 - 50 - 36   # top chrome + label + bottom
    img_w, img_h = img.size
    scale = min(usable_w/img_w, usable_h/img_h, 1.0)
    draw_w = img_w*scale
    draw_h = img_h*scale
    x = MARGIN + (usable_w - draw_w)/2
    y = 36 + (usable_h - draw_h)/2

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(buf), x, y, width=draw_w, height=draw_h)

def build_pdf(pages, out_path):
    """
    pages: list of dicts
      handwriting: {type:"handwriting", img_w, img_h, pdf_w, pdf_h, regions_data, label}
      diagram:     {type:"diagram",     image:PIL.Image, label:str}
    """
    total = len(pages)
    c = canvas.Canvas(out_path)

    for pnum, page in enumerate(pages, 1):
        if page["type"] == "diagram":
            pdf_w, pdf_h = 612.0, 792.0
            _draw_diagram_page(c, page["image"], pdf_w, pdf_h, pnum, total, page.get("label",""))
            c.showPage()
            continue

        # handwriting page
        img_w = int(_f(page["img_w"],1)) or 1
        img_h = int(_f(page["img_h"],1)) or 1
        pdf_w = _f(page["pdf_w"], 612)
        pdf_h = _f(page["pdf_h"], 792)
        c.setPageSize((pdf_w, pdf_h))
        c.setFillColorRGB(*COL_BG)
        c.rect(0,0,pdf_w,pdf_h, stroke=0, fill=1)
        _chrome(c, pdf_w, pdf_h, pnum, total)

        # Optional label above content
        if page.get("label"):
            c.setFont(FONT_BOLD, 8)
            c.setFillColorRGB(*COL_ACCENT)
            c.drawString(MARGIN, pdf_h-32, page["label"])

        blocks = _resolve_blocks(page.get("regions_data",[]), img_w, img_h, pdf_w, pdf_h)
        for blk in blocks:
            if blk.is_heading:
                bar_bot = blk.pdf_top-blk.total_h+2
                c.setFillColorRGB(*COL_ACCENT)
                c.rect(blk.pdf_x-8, bar_bot, 2.5, blk.pdf_top-bar_bot, stroke=0, fill=1)
            for i,line in enumerate(blk.lines):
                y = blk.pdf_top-(i+1)*blk.line_h+(blk.line_h-blk.fsize)
                if y<38 or y>pdf_h-28: continue
                c.setFont(blk.font, blk.fsize)
                c.setFillColorRGB(*(COL_HEADING if blk.is_heading else COL_TEXT))
                c.drawString(blk.pdf_x, y, line)
            if blk.is_heading and blk.lines:
                uy = blk.pdf_top-len(blk.lines)*blk.line_h-2
                if uy>38:
                    c.setStrokeColorRGB(*COL_ACCENT)
                    c.setLineWidth(0.7)
                    c.line(blk.pdf_x, uy, blk.pdf_x+min(blk.pdf_w*0.55,200), uy)
        c.showPage()

    c.save()


# ── 5. Model cache ────────────────────────────────────────────────────────────

VISION_MODEL = "glm-ocr"
_cache = {"ollama_ok": None, "ollama_err": None, "models": [], "vision_model": None, "last_checked": 0}
CACHE_TTL = 30

def _refresh():
    now = time.time()
    if now - _cache["last_checked"] < CACHE_TTL:
        return
    try:
        models = [m.model for m in ollama.list().models]
        matched = next((m for m in models if m == VISION_MODEL or m.startswith(VISION_MODEL + ":")), None)
        _cache.update(ollama_ok=True, ollama_err=None, models=models,
                      vision_model=matched, last_checked=now)
    except Exception as e:
        _cache.update(ollama_ok=False, ollama_err=str(e), models=[],
                      vision_model=None, last_checked=now)


# ── 6. OCR helper for a single image ─────────────────────────────────────────

def _ocr_image(img, model):
    """Run segmentation + OCR on a PIL image. Returns (regions_data, warnings)."""
    img_w, img_h = img.size
    pdf_w = 612.0
    pdf_h = pdf_w*img_h/img_w
    warnings = []
    try:
        regions = segment_regions(img)
    except Exception as e:
        warnings.append(f"Segmentation failed: {e}")
        regions = []
    if not regions:
        regions = [Region(0,0,img_w,img_h,img)]

    regions_data = []
    for ridx,region in enumerate(regions):
        try:
            result = ocr_region(region, model)
        except Exception as e:
            warnings.append(f"R{ridx}: OCR error ({e})")
            result = {"text":"","font_size_est":DEFAULT_FONT,"is_heading":False}
        text = result.get("text","").strip()
        if not text: continue
        regions_data.append({
            "region_x":     region.x,
            "region_y":     region.y,
            "region_w":     region.w,
            "region_h":     region.h,
            "text":         text,
            "font_size_est":result.get("font_size_est",DEFAULT_FONT),
            "is_heading":   result.get("is_heading",False),
        })
    return img_w, img_h, pdf_w, pdf_h, regions_data, warnings



# ── 7a. Data Pipeline with Stage Logging ──────────────────────────────────────

class PipelineLogger:
    """
    Records timing and metadata for each named stage of the OCR pipeline.
    Attach to a request so the caller gets a structured pipeline_trace.
    """
    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self._t0 = time.time()

    def start(self, name: str, meta: dict = None):
        self.stages.append({
            "stage":   name,
            "status":  "running",
            "start_ms": round((time.time() - self._t0) * 1000),
            "meta":    meta or {},
        })

    def end(self, name: str, status: str = "ok", meta: dict = None):
        for s in reversed(self.stages):
            if s["stage"] == name and s["status"] == "running":
                s["status"]   = status
                s["end_ms"]   = round((time.time() - self._t0) * 1000)
                s["duration_ms"] = s["end_ms"] - s["start_ms"]
                if meta:
                    s["meta"].update(meta)
                break

    def fail(self, name: str, error: str):
        self.end(name, status="error", meta={"error": error})

    def summary(self) -> Dict[str, Any]:
        total = round((time.time() - self._t0) * 1000)
        return {
            "total_ms": total,
            "stages":   self.stages,
            "ok":       all(s["status"] != "error" for s in self.stages),
        }


def _ocr_image_with_pipeline(img, model) -> tuple:
    """
    Full pipeline: load → preprocess → segment → OCR → validate.
    Returns (img_w, img_h, pdf_w, pdf_h, regions_data, warnings, pipeline_trace).
    """
    log = PipelineLogger()

    # Stage 1: image analysis
    log.start("image_analysis", {"size": f"{img.size[0]}x{img.size[1]}"})
    img_w, img_h = img.size
    pdf_w = 612.0
    pdf_h = pdf_w * img_h / img_w
    log.end("image_analysis", meta={"pdf_size": f"{pdf_w:.0f}x{pdf_h:.0f}"})

    # Stage 2: preprocessing
    log.start("preprocessing")
    try:
        enhanced = ImageEnhance.Contrast(img).enhance(1.2)
        log.end("preprocessing", meta={"contrast_boost": 1.2})
    except Exception as e:
        log.fail("preprocessing", str(e))
        enhanced = img

    # Stage 3: segmentation
    log.start("segmentation")
    warnings = []
    try:
        regions = segment_regions(enhanced)
        log.end("segmentation", meta={"regions_found": len(regions)})
    except Exception as e:
        log.fail("segmentation", str(e))
        warnings.append(f"Segmentation failed: {e}")
        regions = []
    if not regions:
        regions = [Region(0, 0, img_w, img_h, img)]

    # Stage 4: OCR
    log.start("ocr", {"model": model, "region_count": len(regions)})
    regions_data = []
    ocr_errors = 0
    for ridx, region in enumerate(regions):
        try:
            result = ocr_region(region, model)
        except Exception as e:
            warnings.append(f"R{ridx}: OCR error ({e})")
            result = {"text": "", "font_size_est": DEFAULT_FONT, "is_heading": False}
            ocr_errors += 1
        text = result.get("text", "").strip()
        if not text:
            continue
        regions_data.append({
            "region_x":      region.x,
            "region_y":      region.y,
            "region_w":      region.w,
            "region_h":      region.h,
            "text":          text,
            "font_size_est": result.get("font_size_est", DEFAULT_FONT),
            "is_heading":    result.get("is_heading", False),
        })
    log.end("ocr", meta={"regions_extracted": len(regions_data), "ocr_errors": ocr_errors})

    # Stage 5: statistical anomaly detection
    log.start("anomaly_detection")
    stat_report = statistical_anomaly_scan(regions_data)
    log.end("anomaly_detection", meta={
        "anomalies_found": stat_report["anomaly_count"],
        "method": "zscore+iqr",
    })

    # Stage 6: validation
    log.start("validation")
    val = validate_regions(regions_data)
    log.end("validation", meta={
        "passed":     val["passed"],
        "confidence": val["confidence"],
        "errors":     val["error_count"],
    })

    return img_w, img_h, pdf_w, pdf_h, regions_data, warnings, log.summary(), stat_report, val


# ── 7b. Statistical Anomaly Detection (z-score + IQR) ────────────────────────

def _text_features(rd: dict) -> dict:
    """Extract numeric features from one OCR region for statistical analysis."""
    text = rd.get("text", "")
    words = text.split()
    chars = len(text)
    return {
        "char_count":      chars,
        "word_count":      len(words),
        "avg_word_len":    (sum(len(w) for w in words) / max(len(words), 1)),
        "digit_ratio":     sum(c.isdigit() for c in text) / max(chars, 1),
        "upper_ratio":     sum(c.isupper() for c in text) / max(chars, 1),
        "space_ratio":     sum(c == " " for c in text) / max(chars, 1),
        "special_ratio":   sum(not c.isalnum() and c != " " for c in text) / max(chars, 1),
        "font_size":       float(rd.get("font_size_est", DEFAULT_FONT)),
    }

def _zscore(values: list) -> list:
    if len(values) < 2:
        return [0.0] * len(values)
    mu  = statistics.mean(values)
    sd  = statistics.stdev(values) or 1e-9
    return [(v - mu) / sd for v in values]

def _iqr_outliers(values: list, threshold: float = 1.5) -> list:
    """Return boolean mask: True = outlier by IQR method."""
    if len(values) < 4:
        return [False] * len(values)
    sv   = sorted(values)
    n    = len(sv)
    # Linear interpolation for Q1 and Q3
    def _percentile(data, pct):
        idx = (len(data) - 1) * pct
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (idx - lo)
    q1   = _percentile(sv, 0.25)
    q3   = _percentile(sv, 0.75)
    iqr  = q3 - q1
    lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
    return [v < lo or v > hi for v in values]

def statistical_anomaly_scan(regions_data: list) -> dict:
    """
    Run z-score and IQR anomaly detection across all OCR regions.
    Returns a stat_report dict with per-region anomaly flags and summary.
    """
    if len(regions_data) < 2:
        return {
            "anomaly_count": 0,
            "method": "insufficient_data",
            "region_scores": [],
            "feature_stats": {},
        }

    features = [_text_features(rd) for rd in regions_data]
    feature_names = list(features[0].keys())
    region_scores = [{"region_index": i, "anomalies": [], "z_scores": {}} for i in range(len(features))]

    feature_stats = {}
    for fname in feature_names:
        vals = [f[fname] for f in features]
        zs   = _zscore(vals)
        iqr_flags = _iqr_outliers(vals)
        mu   = statistics.mean(vals)
        sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
        feature_stats[fname] = {
            "mean":   round(mu, 4),
            "stdev":  round(sd, 4),
            "min":    round(min(vals), 4),
            "max":    round(max(vals), 4),
        }
        for i, (z, iqr_flag) in enumerate(zip(zs, iqr_flags)):
            region_scores[i]["z_scores"][fname] = round(z, 3)
            if abs(z) > 2.5 or iqr_flag:
                region_scores[i]["anomalies"].append({
                    "feature":   fname,
                    "value":     round(vals[i], 4),
                    "z_score":   round(z, 3),
                    "iqr_flag":  iqr_flag,
                    "severity":  "high" if abs(z) > 3.5 else "medium",
                })

    anomaly_count = sum(1 for r in region_scores if r["anomalies"])
    return {
        "anomaly_count":  anomaly_count,
        "total_regions":  len(regions_data),
        "method":         "zscore+iqr",
        "region_scores":  region_scores,
        "feature_stats":  feature_stats,
    }


# ── 7c. Evaluation Metrics (Precision / Recall / F1) ─────────────────────────
# In-memory store: ground truth labels submitted via /metrics/label
# Each entry: {doc_id, region_index, true_category, true_severity}
_ground_truth_store: Dict[str, list] = {}   # doc_id -> list of label dicts
_predictions_store:  Dict[str, list] = {}   # doc_id -> list of validation error dicts

def _confusion(true_labels: list, pred_labels: list, categories: list):
    """Compute per-category TP, FP, FN."""
    result = {}
    for cat in categories:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cat and p == cat)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cat and p == cat)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cat and p != cat)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
        result[cat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn,
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }
    return result

def compute_metrics(doc_id: str) -> dict:
    """Compute precision/recall/F1 for a given doc_id using stored labels + predictions."""
    gt   = _ground_truth_store.get(doc_id, [])
    pred = _predictions_store.get(doc_id, [])
    if not gt:
        return {"error": "No ground truth labels found for this doc_id. Submit via POST /metrics/label"}

    categories = list(ERROR_CATEGORIES.keys()) + ["ok"]

    # Align by region_index
    gt_by_region   = {g["region_index"]: g["true_category"] for g in gt}
    pred_by_region = {p["region_index"]: p["category"] for p in pred}

    all_regions = sorted(set(gt_by_region) | set(pred_by_region))
    true_seq = [gt_by_region.get(r, "ok")   for r in all_regions]
    pred_seq = [pred_by_region.get(r, "ok") for r in all_regions]

    per_cat = _confusion(true_seq, pred_seq, categories)

    # Macro averages
    cats_with_support = [c for c in categories if any(t == c for t in true_seq)]
    macro_p  = statistics.mean([per_cat[c]["precision"] for c in cats_with_support]) if cats_with_support else 0.0
    macro_r  = statistics.mean([per_cat[c]["recall"]    for c in cats_with_support]) if cats_with_support else 0.0
    macro_f1 = statistics.mean([per_cat[c]["f1"]        for c in cats_with_support]) if cats_with_support else 0.0

    correct  = sum(1 for t, p in zip(true_seq, pred_seq) if t == p)
    total    = len(all_regions)
    accuracy = round(correct / max(total, 1), 4)

    return {
        "doc_id":           doc_id,
        "total_regions":    total,
        "per_category":     per_cat,
        "macro_precision":  round(macro_p, 4),
        "macro_recall":     round(macro_r, 4),
        "macro_f1":         round(macro_f1, 4),
        "accuracy":         accuracy,
        "label_count":      len(gt),
        "prediction_count": len(pred),
    }



# ── 7. Validation Engine ──────────────────────────────────────────────────────

# Error category definitions
ERROR_CATEGORIES = {
    "missing_field":    "Required field absent or blank in OCR output",
    "misread_char":     "Character-level OCR error — garbled, [illegible], or implausible char sequence",
    "format_violation": "Value does not match expected pattern (date, number, email, etc.)",
}

# Regex patterns for format checks
_RE_DATE     = re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b')
_RE_EMAIL    = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_RE_PHONE    = re.compile(r'\b[\+\(]?[\d\s\-\(\)]{7,15}\d\b')
_RE_AMOUNT   = re.compile(r'\b(?:Rs\.?|INR|USD|\$|€|£)?\s*\d[\d,]*(?:\.\d{1,2})?\b')
_RE_ILLEGIBLE = re.compile(r'\[illegible\]', re.IGNORECASE)
_RE_GARBLED  = re.compile(r'[^\x20-\x7E\u0900-\u097F\n]')  # non-printable outside ASCII + Devanagari

def _count_garbled(text):
    """Count garbled / non-printable chars that are likely misreads."""
    return len(_RE_GARBLED.findall(text))

def _detect_format_violations(text):
    """Return list of format-violation sub-errors found in text."""
    issues = []
    # Broken date-like strings
    broken_date = re.findall(r'\b\d{1,2}[\/\-\.][A-Za-z]{2,}[\/\-\.]\d{2,4}\b', text)
    if broken_date:
        issues.append(f"malformed_date: {broken_date[:2]}")
    # Amount with non-numeric after currency symbol
    bad_amount = re.findall(r'(?:Rs\.?|INR|USD|\$|€|£)\s*[A-Za-z]+', text)
    if bad_amount:
        issues.append(f"malformed_amount: {bad_amount[:2]}")
    return issues

def validate_regions(regions_data):
    """
    Validate a list of OCR region dicts.
    Returns a validation_report dict with:
      - passed: bool
      - confidence: float 0-1
      - error_count: int
      - errors: list of {region_index, category, detail, severity}
      - summary: dict of counts per category
    """
    errors = []
    summary = {k: 0 for k in ERROR_CATEGORIES}

    if not regions_data:
        errors.append({
            "region_index": -1,
            "category":     "missing_field",
            "detail":       "No text regions detected — document may be blank or unsegmented",
            "severity":     "high",
        })
        summary["missing_field"] += 1

    for idx, rd in enumerate(regions_data):
        text = rd.get("text", "").strip()

        # ── missing_field ──────────────────────────────────────────
        if not text:
            errors.append({
                "region_index": idx,
                "category":     "missing_field",
                "detail":       f"Region {idx} produced empty text",
                "severity":     "high",
            })
            summary["missing_field"] += 1
            continue

        word_count = len(text.split())

        # ── misread_char ───────────────────────────────────────────
        illegible_hits = len(_RE_ILLEGIBLE.findall(text))
        if illegible_hits:
            errors.append({
                "region_index": idx,
                "category":     "misread_char",
                "detail":       f"Region {idx}: {illegible_hits} [illegible] marker(s) detected",
                "severity":     "medium" if illegible_hits < 3 else "high",
            })
            summary["misread_char"] += 1

        garbled = _count_garbled(text)
        garbled_ratio = garbled / max(len(text), 1)
        if garbled_ratio > 0.04:
            errors.append({
                "region_index": idx,
                "category":     "misread_char",
                "detail":       f"Region {idx}: {garbled} garbled character(s) ({garbled_ratio:.1%} of text)",
                "severity":     "medium",
            })
            summary["misread_char"] += 1

        # Very short regions that look like misreads
        if word_count < 2 and len(text) < 6 and not rd.get("is_heading", False):
            errors.append({
                "region_index": idx,
                "category":     "misread_char",
                "detail":       f"Region {idx}: suspiciously short output '{text}' — possible fragment",
                "severity":     "low",
            })
            summary["misread_char"] += 1

        # ── format_violation ──────────────────────────────────────
        fmt_issues = _detect_format_violations(text)
        for fi in fmt_issues:
            errors.append({
                "region_index": idx,
                "category":     "format_violation",
                "detail":       f"Region {idx}: {fi}",
                "severity":     "medium",
            })
            summary["format_violation"] += 1

        # Detect expected structured tokens that look wrong
        if any(kw in text.lower() for kw in ("date", "dob", "dated")):
            if not _RE_DATE.search(text):
                errors.append({
                    "region_index": idx,
                    "category":     "format_violation",
                    "detail":       f"Region {idx}: 'date' keyword present but no valid date pattern found",
                    "severity":     "medium",
                })
                summary["format_violation"] += 1

        if any(kw in text.lower() for kw in ("email", "e-mail", "mail")):
            if not _RE_EMAIL.search(text):
                errors.append({
                    "region_index": idx,
                    "category":     "format_violation",
                    "detail":       f"Region {idx}: 'email' keyword present but no valid email found",
                    "severity":     "medium",
                })
                summary["format_violation"] += 1

        if any(kw in text.lower() for kw in ("phone", "mobile", "tel", "contact")):
            if not _RE_PHONE.search(text):
                errors.append({
                    "region_index": idx,
                    "category":     "format_violation",
                    "detail":       f"Region {idx}: phone/contact keyword present but no valid phone number found",
                    "severity":     "low",
                })
                summary["format_violation"] += 1

    # Confidence: penalise per *region* (not per individual error) so one
    # heavily-errored region can't tank the whole document score.
    severity_penalty = {"high": 0.15, "medium": 0.07, "low": 0.03}
    # Group errors by region and take the worst per region
    region_worst: Dict[int, str] = {}
    for e in errors:
        ri = e["region_index"]
        cur = severity_penalty.get(region_worst.get(ri, ""), 0)
        new = severity_penalty.get(e["severity"], 0.05)
        if new > cur:
            region_worst[ri] = e["severity"]
    penalty = sum(severity_penalty.get(sev, 0.05) for sev in region_worst.values())
    confidence = max(0.0, min(1.0, 1.0 - penalty))

    return {
        "passed":      len(errors) == 0,
        "confidence":  round(confidence, 3),
        "error_count": len(errors),
        "errors":      errors,
        "summary":     summary,
        "categories":  ERROR_CATEGORIES,
    }


# ── 7b. Batch Processing ───────────────────────────────────────────────────────

def _process_one_file(raw, filename, model):
    """OCR + validate a single file. Returns a result dict."""
    ext = os.path.splitext(filename.lower())[1]
    try:
        if ext == ".pdf":
            images = rasterize_pdf(raw)
        else:
            images = [load_image(raw, filename)]
    except Exception as e:
        return {"filename": filename, "status": "error", "error": str(e)}

    all_regions, all_warnings = [], []
    for img in images:
        img_w, img_h, pdf_w, pdf_h, regions_data, w = _ocr_image(img, model)
        all_regions.extend(regions_data)
        all_warnings.extend(w)

    validation = validate_regions(all_regions)
    return {
        "filename":   filename,
        "status":     "ok",
        "page_count": len(images),
        "regions":    len(all_regions),
        "warnings":   all_warnings,
        "validation": validation,
    }


# ── 8. Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/status")
def status():
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"ok":False,"error":"Ollama not running — run: ollama serve"})
    if not _cache["vision_model"]:
        return jsonify({"ok":False,"error":"glm-ocr not found — run: ollama pull glm-ocr"})
    return jsonify({"ok":True,"model":_cache["vision_model"],"available":_cache["models"]})


# ── Validation endpoint ────────────────────────────────────────────────────────

@app.route("/validate", methods=["POST"])
def validate_endpoint():
    """
    Validate OCR output for anomalies, categorising errors.

    Accepts EITHER:
      A) JSON body: {"regions": [...], "session_id": "...", "page_index": 0}
         (validates pre-existing session page)
      B) multipart form-data: file=<image/pdf>, session_id, label
         (runs fresh OCR then validates)

    Returns: {ok, validation: {passed, confidence, error_count, errors, summary}}
    """
    _refresh()

    # ── Mode A: validate a session page by index ──────────────────
    if request.is_json:
        data = request.get_json(silent=True) or {}
        sid   = data.get("session_id", "default")
        pidx  = data.get("page_index", 0)
        sess  = _get_session(sid)
        if not sess["pages"]:
            return jsonify({"error": "No pages in session."}), 400
        if not (0 <= pidx < len(sess["pages"])):
            return jsonify({"error": "Invalid page_index."}), 400
        page = sess["pages"][pidx]
        if page["type"] != "handwriting":
            return jsonify({"error": "Validation only applies to handwriting pages."}), 400
        regions_data = page.get("regions_data", [])
        validation   = validate_regions(regions_data)
        return jsonify({"ok": True, "page_index": pidx, "validation": validation})

    # ── Mode B: upload a file and validate on the fly ─────────────
    if not _cache["ollama_ok"]:
        return jsonify({"error": "Ollama not running"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error": "No vision model"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f   = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported type '{ext}'."}), 400
    raw = f.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large (max 100 MB)."}), 400

    try:
        if ext == ".pdf":
            images = rasterize_pdf(raw)
        else:
            images = [load_image(raw, f.filename)]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    all_regions, all_warnings = [], []
    for img in images:
        _, _, _, _, regions_data, w = _ocr_image(img, model)
        all_regions.extend(regions_data)
        all_warnings.extend(w)

    validation = validate_regions(all_regions)
    return jsonify({
        "ok":         True,
        "filename":   f.filename,
        "page_count": len(images),
        "regions":    len(all_regions),
        "warnings":   all_warnings,
        "validation": validation,
    })


# ── Batch processing endpoints ─────────────────────────────────────────────────

@app.route("/batch/validate", methods=["POST"])
def batch_validate():
    """
    Validate-only batch: OCR + anomaly-check multiple files, return JSON report.
    Does NOT build a PDF.

    POST multipart: files[]=<file1>, files[]=<file2>, …  (up to 50 files)
    Returns: {ok, total, results:[{filename, status, validation, ...}]}
    """
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"error": "Ollama not running"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error": "No vision model"}), 503

    uploaded = request.files.getlist("files[]")
    if not uploaded:
        # Fallback: single-key form
        uploaded = [v for k, v in request.files.items()]
    if not uploaded:
        return jsonify({"error": "No files uploaded. Use files[] form field."}), 400
    if len(uploaded) > 50:
        return jsonify({"error": "Batch limit is 50 files per request."}), 400

    results = []
    for f in uploaded:
        ext = os.path.splitext(f.filename.lower())[1]
        if ext not in ALLOWED_EXT:
            results.append({"filename": f.filename, "status": "skipped",
                             "error": f"Unsupported type '{ext}'"})
            continue
        raw = f.read()
        if len(raw) > MAX_FILE_SIZE:
            results.append({"filename": f.filename, "status": "skipped",
                             "error": "File too large (max 100 MB)"})
            continue
        result = _process_one_file(raw, f.filename, model)
        results.append(result)

    total_errors = sum(r.get("validation", {}).get("error_count", 0)
                       for r in results if r.get("status") == "ok")
    passed = sum(1 for r in results
                 if r.get("status") == "ok" and r.get("validation", {}).get("passed"))

    return jsonify({
        "ok":           True,
        "total":        len(uploaded),
        "processed":    len([r for r in results if r.get("status") == "ok"]),
        "passed":       passed,
        "total_errors": total_errors,
        "results":      results,
    })


@app.route("/batch/convert", methods=["POST"])
def batch_convert():
    """
    Full batch: OCR + validate + build one combined PDF from multiple files.

    POST multipart: files[]=<file1>, files[]=<file2>, …, output_name=<str>
    Returns: PDF file download + validation summary in X-OrangeDocs-Validation header.
    """
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"error": "Ollama not running"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error": "No vision model"}), 503

    uploaded = request.files.getlist("files[]")
    if not uploaded:
        uploaded = [v for k, v in request.files.items()]
    if not uploaded:
        return jsonify({"error": "No files uploaded. Use files[] form field."}), 400
    if len(uploaded) > 50:
        return jsonify({"error": "Batch limit is 50 files per request."}), 400

    out_name = request.form.get("output_name", "orangedocs_batch.pdf")
    if not out_name.endswith(".pdf"):
        out_name += ".pdf"

    pages, all_warnings, all_validations = [], [], []

    for f in uploaded:
        ext = os.path.splitext(f.filename.lower())[1]
        if ext not in ALLOWED_EXT:
            all_warnings.append(f"Skipped '{f.filename}': unsupported type '{ext}'")
            continue
        raw = f.read()
        if len(raw) > MAX_FILE_SIZE:
            all_warnings.append(f"Skipped '{f.filename}': file too large")
            continue

        try:
            if ext == ".pdf":
                images = rasterize_pdf(raw)
            else:
                images = [load_image(raw, f.filename)]
        except Exception as e:
            all_warnings.append(f"Skipped '{f.filename}': {e}")
            continue

        for pidx, img in enumerate(images):
            img_w, img_h, pdf_w, pdf_h, regions_data, w = _ocr_image(img, model)
            all_warnings.extend(w)
            label = f"{os.path.splitext(f.filename)[0]} p{pidx+1}" if len(images) > 1 else os.path.splitext(f.filename)[0]
            pages.append({
                "type": "handwriting", "img_w": img_w, "img_h": img_h,
                "pdf_w": pdf_w, "pdf_h": pdf_h,
                "regions_data": regions_data, "label": label,
            })
            validation = validate_regions(regions_data)
            all_validations.append({"file": f.filename, "page": pidx, "validation": validation})

    if not pages:
        return jsonify({"error": "No processable pages found in uploaded files."}), 400

    out_path = os.path.join(OUTPUT_DIR, f"batch_{uuid.uuid4().hex[:8]}.pdf")
    try:
        build_pdf(pages, out_path)
    except Exception as e:
        return jsonify({"error": f"PDF build failed: {e}"}), 500

    # Summarise validation across all pages
    total_errors = sum(v["validation"]["error_count"] for v in all_validations)
    avg_conf     = (sum(v["validation"]["confidence"] for v in all_validations)
                    / max(len(all_validations), 1))
    val_summary  = json.dumps({
        "total_errors": total_errors,
        "avg_confidence": round(avg_conf, 3),
        "pages_checked": len(all_validations),
    })

    resp = send_file(out_path, mimetype="application/pdf",
                     as_attachment=True, download_name=out_name)
    resp.headers["X-OrangeDocs-Validation"] = val_summary
    if all_warnings:
        resp.headers["X-OrangeDocs-Warnings"] = "; ".join(all_warnings[:5])
    return resp




# ── Metrics endpoints ──────────────────────────────────────────────────────────

@app.route("/metrics/label", methods=["POST"])
def metrics_label():
    """
    Submit ground truth labels for a document to enable precision/recall/F1.
    POST JSON:
    {
      "doc_id": "my_doc_001",
      "labels": [
        {"region_index": 0, "true_category": "ok"},
        {"region_index": 1, "true_category": "missing_field"},
        {"region_index": 2, "true_category": "misread_char"}
      ]
    }
    true_category: one of "ok", "missing_field", "misread_char", "format_violation"
    """
    data = request.get_json(silent=False, force=True) if request.content_length else {}
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400
    doc_id = data.get("doc_id")
    labels = data.get("labels", [])
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    if not isinstance(labels, list):
        return jsonify({"error": "'labels' must be a JSON array. Got: " + type(labels).__name__}), 400
    valid_cats = set(ERROR_CATEGORIES.keys()) | {"ok"}
    sanitised = []
    for i, lb in enumerate(labels):
        if not isinstance(lb, dict):
            return jsonify({"error": f"labels[{i}] must be an object, got {type(lb).__name__}"}), 400
        cat = lb.get("true_category", "ok")
        if cat not in valid_cats:
            return jsonify({"error": f"Invalid category '{cat}'. Must be one of {sorted(valid_cats)}"}), 400
        try:
            region_index = int(lb.get("region_index", 0))
        except (TypeError, ValueError):
            return jsonify({"error": f"labels[{i}].region_index must be an integer"}), 400
        sanitised.append({
            "region_index":  region_index,
            "true_category": cat,
            "true_severity": lb.get("true_severity", "medium"),
        })
    try:
        _ground_truth_store[doc_id] = sanitised
    except Exception as e:
        return jsonify({"error": f"Failed to store labels: {e}"}), 500
    return jsonify({"ok": True, "doc_id": doc_id, "label_count": len(sanitised)})


@app.route("/metrics/predict", methods=["POST"])
def metrics_predict():
    """
    Run OCR + validation on a file and store predictions for a doc_id.
    POST multipart: file=<image/pdf>, doc_id=<str>
    Then call GET /metrics/report?doc_id=<str> after labelling to get metrics.
    """
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"error": "Ollama not running"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error": "No vision model"}), 503

    doc_id = request.form.get("doc_id") or request.args.get("doc_id")
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f   = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported type '{ext}'"}), 400
    raw = f.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large"}), 400

    try:
        if ext == ".pdf": images = rasterize_pdf(raw)
        else:             images = [load_image(raw, f.filename)]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    all_regions = []
    for img in images:
        _, _, _, _, rd, _ = _ocr_image(img, model)
        all_regions.extend(rd)

    val = validate_regions(all_regions)
    # Deduplicate: keep the highest-severity error per region_index so that
    # pred_by_region in compute_metrics sees exactly one category per region.
    _SEV_RANK = {"high": 3, "medium": 2, "low": 1}
    seen: dict = {}
    for err in val["errors"]:
        ri = err["region_index"]
        if ri not in seen or _SEV_RANK.get(err["severity"], 0) > _SEV_RANK.get(seen[ri]["severity"], 0):
            seen[ri] = err
    _predictions_store[doc_id] = list(seen.values())

    return jsonify({
        "ok":      True,
        "doc_id":  doc_id,
        "regions": len(all_regions),
        "errors":  val["error_count"],
        "message": f"Predictions stored. Now POST /metrics/label with doc_id='{doc_id}', then GET /metrics/report?doc_id={doc_id}",
    })


@app.route("/metrics/report", methods=["GET"])
def metrics_report():
    """
    GET /metrics/report?doc_id=<str>
    Returns precision, recall, F1 per category + macro averages.
    Requires ground truth from /metrics/label and predictions from /metrics/predict.
    """
    doc_id = request.args.get("doc_id")
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    return jsonify(compute_metrics(doc_id))


@app.route("/metrics/report/all", methods=["GET"])
def metrics_report_all():
    """Return metrics for all doc_ids that have both labels and predictions."""
    reports = {}
    for doc_id in set(_ground_truth_store) | set(_predictions_store):
        reports[doc_id] = compute_metrics(doc_id)
    return jsonify({"ok": True, "reports": reports, "doc_count": len(reports)})


# ── Pipeline trace endpoint ────────────────────────────────────────────────────

@app.route("/pipeline/run", methods=["POST"])
def pipeline_run():
    """
    Run the full instrumented pipeline on a file and return the trace.
    POST multipart: file=<image/pdf>
    Returns: {ok, pipeline_trace, stat_report, validation, regions, warnings}
    """
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"error": "Ollama not running"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error": "No vision model"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f   = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported type '{ext}'"}), 400
    raw = f.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large"}), 400

    try:
        if ext == ".pdf": images = rasterize_pdf(raw)
        else:             images = [load_image(raw, f.filename)]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    all_traces, all_stats, all_regions, all_warnings, all_validations = [], [], [], [], []
    for img in images:
        img_w, img_h, pdf_w, pdf_h, regions_data, warnings, trace, stat, val = _ocr_image_with_pipeline(img, model)
        all_traces.append(trace)
        all_stats.append(stat)
        all_regions.extend(regions_data)
        all_warnings.extend(warnings)
        all_validations.append(val)

    combined_val = validate_regions(all_regions)
    combined_stat = statistical_anomaly_scan(all_regions)

    return jsonify({
        "ok":             True,
        "filename":       f.filename,
        "page_count":     len(images),
        "regions":        len(all_regions),
        "warnings":       all_warnings,
        "pipeline_trace": all_traces,
        "stat_report":    combined_stat,
        "validation":     combined_val,
    })


# ── Dashboard data endpoint ────────────────────────────────────────────────────

@app.route("/dashboard/data", methods=["GET"])
def dashboard_data():
    """
    Return aggregated stats across all active sessions for the dashboard.
    GET /dashboard/data?session_id=<str>  (omit for global)
    """
    sid = request.args.get("session_id")
    sessions_to_scan = ([_sessions[sid]] if sid and sid in _sessions
                        else list(_sessions.values()))

    total_pages = 0
    total_regions = 0
    error_counts: Dict[str, int] = {k: 0 for k in ERROR_CATEGORIES}
    confidence_vals: List[float] = []
    page_types: Dict[str, int] = {"handwriting": 0, "diagram": 0}

    for sess in sessions_to_scan:
        for page in sess.get("pages", []):
            total_pages += 1
            ptype = page.get("type", "handwriting")
            page_types[ptype] = page_types.get(ptype, 0) + 1
            if ptype == "handwriting":
                rds = page.get("regions_data", [])
                total_regions += len(rds)
                val = validate_regions(rds)
                confidence_vals.append(val["confidence"])
                for cat, cnt in val["summary"].items():
                    error_counts[cat] = error_counts.get(cat, 0) + cnt

    avg_conf = round(statistics.mean(confidence_vals), 3) if confidence_vals else 1.0
    return jsonify({
        "ok":            True,
        "total_pages":   total_pages,
        "total_regions": total_regions,
        "page_types":    page_types,
        "error_counts":  error_counts,
        "avg_confidence": avg_conf,
        "confidence_distribution": confidence_vals,
        "session_count": len(sessions_to_scan),
    })



# ── Session endpoints ─────────────────────────────────────────────────────────

@app.route("/session/status", methods=["GET"])
def session_status():
    """Return number and types of pages queued in the session."""
    sid  = request.args.get("session_id","default")
    sess = _get_session(sid)
    pages_info = [
        {"index":i,"type":p["type"],"label":p.get("label","")}
        for i,p in enumerate(sess["pages"])
    ]
    return jsonify({
        "session_id": sid,
        "page_count": len(sess["pages"]),
        "pages": pages_info,
    })

@app.route("/session/clear", methods=["POST"])
def session_clear():
    """Wipe all pages from the session."""
    sid = request.json.get("session_id","default") if request.is_json else request.form.get("session_id","default")
    if sid in _sessions:
        del _sessions[sid]
    return jsonify({"ok":True,"session_id":sid,"message":"Session cleared."})

@app.route("/session/add_page", methods=["POST"])
def session_add_page():
    """
    Add one handwriting image to the session queue.
    POST form-data: file=<image>, session_id=<str>, label=<str optional>
    Returns: {ok, session_id, page_index, page_count, warnings}
    """
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"error":"Ollama not running"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error":"No vision model"}), 503

    if "file" not in request.files:
        return jsonify({"error":"No file uploaded."}), 400
    f   = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"error":f"Unsupported type '{ext}'."}), 400

    raw = f.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error":"File too large (max 100 MB)."}), 400

    sid   = request.form.get("session_id","default")
    label = request.form.get("label","")

    try:
        img = load_image(raw, f.filename)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    img_w, img_h, pdf_w, pdf_h, regions_data, warnings = _ocr_image(img, model)

    sess = _get_session(sid)
    sess["pages"].append({
        "type":         "handwriting",
        "img_w":        img_w,
        "img_h":        img_h,
        "pdf_w":        pdf_w,
        "pdf_h":        pdf_h,
        "regions_data": regions_data,
        "label":        label,
    })
    _gc_sessions()

    return jsonify({
        "ok":         True,
        "session_id": sid,
        "page_index": len(sess["pages"])-1,
        "page_count": len(sess["pages"]),
        "warnings":   warnings,
    })

@app.route("/session/add_diagram", methods=["POST"])
def session_add_diagram():
    """
    Add one diagram image to the session queue (no OCR — embedded as image).
    POST form-data: file=<image>, session_id=<str>, label=<str optional>
    Returns: {ok, session_id, page_index, page_count}
    """
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded."}), 400
    f   = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"error":f"Unsupported type '{ext}'."}), 400

    raw = f.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error":"File too large (max 100 MB)."}), 400

    sid   = request.form.get("session_id","default")
    label = request.form.get("label","Diagram")

    try:
        img = load_image(raw, f.filename)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    sess = _get_session(sid)
    sess["pages"].append({
        "type":  "diagram",
        "image": img,
        "label": label,
    })
    _gc_sessions()

    return jsonify({
        "ok":         True,
        "session_id": sid,
        "page_index": len(sess["pages"])-1,
        "page_count": len(sess["pages"]),
    })

@app.route("/session/remove_page", methods=["POST"])
def session_remove_page():
    """Remove a specific page by index from the session."""
    data  = request.get_json(force=True, silent=True) or {}
    sid   = data.get("session_id","default")
    index = data.get("index")
    sess  = _get_session(sid)
    if index is None or not (0 <= index < len(sess["pages"])):
        return jsonify({"error":"Invalid page index."}), 400
    sess["pages"].pop(index)
    return jsonify({"ok":True,"page_count":len(sess["pages"])})

@app.route("/session/build", methods=["POST"])
def session_build():
    """
    Build the final PDF from all queued pages and return it as a download.
    POST JSON: {"session_id":"...", "output_name":"notes.pdf"}
    or form-data with the same fields.
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict()

    sid      = data.get("session_id","default")
    out_name = data.get("output_name","orangedocs_notes.pdf")
    if not out_name.endswith(".pdf"):
        out_name += ".pdf"

    sess = _get_session(sid)
    if not sess["pages"]:
        return jsonify({"error":"No pages in session. Add pages first."}), 400

    out_path = os.path.join(OUTPUT_DIR, f"build_{uuid.uuid4().hex[:8]}.pdf")
    try:
        build_pdf(sess["pages"], out_path)
    except Exception as e:
        return jsonify({"error":f"PDF build failed: {e}"}), 500

    return send_file(out_path, mimetype="application/pdf",
                     as_attachment=True, download_name=out_name)


# ── Legacy single-file convert (kept for backward compatibility) ──────────────

@app.route("/convert", methods=["POST"])
def convert():
    _refresh()
    if not _cache["ollama_ok"]:
        return jsonify({"error":"Ollama not running — run: ollama serve"}), 503
    model = _cache["vision_model"]
    if not model:
        return jsonify({"error":"glm-ocr not found — run: ollama pull glm-ocr"}), 503

    if "file" not in request.files:
        return jsonify({"error":"No file uploaded."}), 400
    f   = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"error":f"Unsupported type '{ext}'."}), 400
    raw = f.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error":"File too large (max 100 MB)."}), 400

    try:
        if ext==".pdf": images = rasterize_pdf(raw)
        else:           images = [load_image(raw, f.filename)]
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    pages, warnings = [], []
    for pidx, img in enumerate(images, 1):
        img_w, img_h, pdf_w, pdf_h, regions_data, w = _ocr_image(img, model)
        warnings.extend(w)
        pages.append({"type":"handwriting","img_w":img_w,"img_h":img_h,
                       "pdf_w":pdf_w,"pdf_h":pdf_h,"regions_data":regions_data,"label":""})

    out_name = f"orangedocs_{uuid.uuid4().hex[:8]}.pdf"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    try:
        build_pdf(pages, out_path)
    except Exception as e:
        return jsonify({"error":f"PDF build failed: {e}"}), 500

    stem = os.path.splitext(f.filename)[0]
    resp = send_file(out_path, mimetype="application/pdf",
                     as_attachment=True, download_name=f"{stem}_typed.pdf")
    if warnings:
        resp.headers["X-OrangeDocs-Warnings"] = "; ".join(warnings[:5])
    return resp


# ── 8. Startup ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print("\n" + "="*58)
    print("  OrangeDocs — Handwriting to Typed PDF  (v3.0)")
    print("="*58)
    _refresh()
    if _cache["ollama_ok"]:
        vm = _cache["vision_model"]
        if vm:
            print(f"  Model  : {vm}")
        else:
            print("  Model  : glm-ocr not found — run: ollama pull glm-ocr")
    else:
        print("  Ollama : not detected — run: ollama serve")

    print(f"\n  Local  : http://localhost:{PORT}")
    print("\n  API endpoints:")
    print("    GET  /status")
    print("    GET  /session/status?session_id=X")
    print("    POST /session/add_page     (file, session_id, label)")
    print("    POST /session/add_diagram  (file, session_id, label)")
    print("    POST /session/remove_page  {session_id, index}")
    print("    POST /session/build        {session_id, output_name}")
    print("    POST /session/clear        {session_id}")
    print("    POST /validate             (file or JSON {session_id, page_index})")
    print("    POST /batch/validate       (files[] — OCR + validate, JSON report)")
    print("    POST /batch/convert        (files[] — OCR + validate + combined PDF)")
    print("    POST /convert              (legacy single-file)")
    print("="*58 + "\n")

    app.run(debug=False, host="0.0.0.0", port=PORT, threaded=True)