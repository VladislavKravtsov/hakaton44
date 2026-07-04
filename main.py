"""
FastAPI web interface for the ore-grade classification demo.

Run:
    uvicorn main:app --reload --port 8000
Then open http://127.0.0.1:8000
"""
import base64
import csv
import io
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import inference
import report

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Ore Grade Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MAX_UPLOAD_MB = 300  # the real panoramas are 48-220MB JPEGs
MAX_SIDE_PX = 4000  # downscale very large panoramas so a demo request stays fast


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение (поддерживаются PNG/JPEG/TIFF).")
    return img


def _downscale_if_needed(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _encode_png_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Файл больше {MAX_UPLOAD_MB} МБ.")

    t0 = time.time()
    img = _decode_upload(data)
    img, scale = _downscale_if_needed(img)

    result = inference.analyze(img)
    elapsed = time.time() - t0

    image_size = {"width": img.shape[1], "height": img.shape[0]}
    analysis_id = report.append_analysis_log(file.filename, image_size, scale,
                                             result["metrics"], elapsed)
    response = {
        "filename": file.filename,
        "analysis_id": analysis_id,
        "processing_time_sec": round(elapsed, 2),
        "downscaled": scale != 1.0,
        "scale_applied": round(scale, 4),
        "image_size": image_size,
        "metrics": result["metrics"],
        "overlay_png_base64": _encode_png_base64(result["overlay"]),
        "mask_png_base64": _encode_png_base64(result["color_mask"]),
    }
    return response


@app.post("/api/analyze/pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """Full analysis returned as a one-page PDF report."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Файл больше {MAX_UPLOAD_MB} МБ.")
    t0 = time.time()
    img = _decode_upload(data)
    img, scale = _downscale_if_needed(img)
    result = inference.analyze(img)
    elapsed = time.time() - t0

    image_size = {"width": img.shape[1], "height": img.shape[0]}
    analysis_id = report.append_analysis_log(file.filename, image_size, scale,
                                             result["metrics"], elapsed)
    pdf_bytes = report.build_pdf(file.filename, img, result["overlay"],
                                 result["metrics"], analysis_id)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{Path(file.filename).stem}_report.pdf"'},
    )


@app.post("/api/analyze/csv")
async def analyze_csv(file: UploadFile = File(...)):
    """Same analysis, but returns just the metrics as a downloadable CSV row."""
    data = await file.read()
    img = _decode_upload(data)
    img, _ = _downscale_if_needed(img)
    result = inference.analyze(img)
    m = result["metrics"]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "filename", "grade", "pct_ore_total", "pct_ordinary", "pct_thin", "pct_talc",
        "ordinary_share_of_ore", "thin_share_of_ore", "summary",
    ])
    writer.writerow([
        file.filename, m["grade"], m["pct_ore_total"], m["pct_ordinary"], m["pct_thin"], m["pct_talc"],
        m["ordinary_share_of_ore"], m["thin_share_of_ore"], m["summary"],
    ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{Path(file.filename).stem}_metrics.csv"'},
    )


# ---------------------------------------------------------------------------
# Batch mode: upload a whole folder, poll progress, download a zip with
# masks + overlays + per-image statistics.csv
# ---------------------------------------------------------------------------
BATCH_JOBS: dict = {}  # job_id -> {"total", "done", "current", "finished", "error", "dir"}

CSV_FIELDS = [
    "filename", "grade", "pct_ore_total", "pct_ordinary", "pct_thin", "pct_talc",
    "ordinary_share_of_ore", "thin_share_of_ore", "frag_per_mpx",
    "small_grain_ore_share", "median_grain_area",
    "width", "height", "downscaled", "time_sec", "summary",
]


def _imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


def _run_batch(job_id: str, input_dir: Path, out_dir: Path) -> None:
    job = BATCH_JOBS[job_id]
    rows = []
    try:
        files = sorted(input_dir.iterdir())
        for i, path in enumerate(files, start=1):
            job["current"] = path.name
            t0 = time.time()
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                job["done"] = i
                continue
            img, scale = _downscale_if_needed(img)
            result = inference.analyze(img)
            m = result["metrics"]

            stem = Path(path.name).stem
            _imwrite(out_dir / "overlays" / f"{stem}.png", result["overlay"])
            _imwrite(out_dir / "masks_color" / f"{stem}.png", result["color_mask"])
            report.append_analysis_log(path.name, {"width": img.shape[1], "height": img.shape[0]},
                                       scale, m, time.time() - t0)

            rows.append({
                "filename": path.name, **{k: m.get(k, "") for k in CSV_FIELDS if k in m},
                "grade": m["grade"], "summary": m["summary"],
                "width": img.shape[1], "height": img.shape[0],
                "downscaled": scale != 1.0, "time_sec": round(time.time() - t0, 1),
            })
            job["done"] = i

        with open(out_dir / "statistics.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        job["finished"] = True
    except Exception as e:
        job["error"] = str(e)
        job["finished"] = True
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)


@app.post("/api/batch")
async def batch_start(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы.")
    job_id = uuid.uuid4().hex[:12]
    job_root = Path(tempfile.gettempdir()) / f"ore_batch_{job_id}"
    input_dir = job_root / "inputs"
    out_dir = job_root / "results"
    for d in (input_dir, out_dir / "overlays", out_dir / "masks_color"):
        d.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in files:
        # keep only the basename -- browsers send folder-relative paths
        name = Path(f.filename).name
        if not name:
            continue
        with open(input_dir / f"{n:04d}_{name}", "wb") as out:
            shutil.copyfileobj(f.file, out)
        n += 1

    BATCH_JOBS[job_id] = {"total": n, "done": 0, "current": "", "finished": False,
                          "error": None, "dir": str(out_dir)}
    threading.Thread(target=_run_batch, args=(job_id, input_dir, out_dir), daemon=True).start()
    return {"job_id": job_id, "total": n}


@app.get("/api/batch/{job_id}/status")
def batch_status(job_id: str):
    job = BATCH_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return {k: job[k] for k in ("total", "done", "current", "finished", "error")}


# ---------------------------------------------------------------------------
# Manual mask correction + fine-tuning (active learning loop)
# ---------------------------------------------------------------------------
CORRECTIONS_ROOT = BASE_DIR / "corrections_dataset"
RETRAIN_STATE: dict = {"running": False, "log": "", "finished_at": None}

_COLOR_TO_CLASS = {(0, 0, 0): 0, (0, 255, 0): 1, (0, 0, 255): 2, (255, 0, 0): 3}  # BGR


def _color_mask_to_classes(color_bgr: np.ndarray) -> np.ndarray:
    out = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    for (b, g, r), cls in _COLOR_TO_CLASS.items():
        out[(color_bgr[:, :, 0] == b) & (color_bgr[:, :, 1] == g) & (color_bgr[:, :, 2] == r)] = cls
    return out


def _apply_corrections(color_mask_bytes: bytes, corrections_bytes: bytes) -> np.ndarray:
    """color mask + painted RGBA overrides -> corrected class mask."""
    mask_arr = np.frombuffer(color_mask_bytes, dtype=np.uint8)
    color_mask = cv2.imdecode(mask_arr, cv2.IMREAD_COLOR)
    corr_arr = np.frombuffer(corrections_bytes, dtype=np.uint8)
    corr = cv2.imdecode(corr_arr, cv2.IMREAD_UNCHANGED)
    if color_mask is None or corr is None or corr.ndim != 3 or corr.shape[2] != 4:
        raise HTTPException(status_code=400, detail="Некорректные данные коррекции.")
    if corr.shape[:2] != color_mask.shape[:2]:
        corr = cv2.resize(corr, (color_mask.shape[1], color_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    class_mask = _color_mask_to_classes(color_mask)
    painted = corr[:, :, 3] > 127
    corr_classes = _color_mask_to_classes(corr[:, :, :3])
    class_mask[painted] = corr_classes[painted]
    return class_mask, int(painted.sum())


@app.post("/api/recompute")
async def recompute_metrics(mask_png: UploadFile = File(...), corrections_png: UploadFile = File(...)):
    """Re-derive all statistics (and the grade) from the corrected mask,
    without saving anything."""
    class_mask, n_painted = _apply_corrections(await mask_png.read(), await corrections_png.read())
    metrics = inference.compute_metrics(class_mask)
    return {"metrics": metrics, "painted_pixels": n_painted}


@app.post("/api/corrections")
async def save_correction(file: UploadFile = File(...), mask_png: UploadFile = File(...),
                          corrections_png: UploadFile = File(...)):
    """Stores (image, expert-corrected mask) as a training sample.

    mask_png: the color mask the UI received from /api/analyze.
    corrections_png: RGBA image of the same size; painted pixels (alpha>0)
    override the model's class with the painted color's class."""
    img = _decode_upload(await file.read())
    img, _ = _downscale_if_needed(img)

    class_mask, n_painted = _apply_corrections(await mask_png.read(), await corrections_png.read())
    if n_painted == 0:
        raise HTTPException(status_code=400, detail="Нет нарисованных правок.")
    if class_mask.shape[:2] != img.shape[:2]:
        class_mask = cv2.resize(class_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    sample_id = f"corr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    img_dir = CORRECTIONS_ROOT / "train" / "images"
    mask_dir = CORRECTIONS_ROOT / "train" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    _imwrite(img_dir / f"{sample_id}.png", img)
    _imwrite(mask_dir / f"{sample_id}.png", class_mask)

    n_total = len(list(img_dir.glob("*.png")))
    metrics = inference.compute_metrics(class_mask)
    return {"sample_id": sample_id, "painted_pixels": n_painted,
            "corrections_total": n_total, "metrics": metrics}


@app.post("/api/retrain")
def start_retrain(epochs: int = 3):
    """Fine-tunes the current model on everything incl. expert corrections."""
    import subprocess, sys as _sys
    if RETRAIN_STATE["running"]:
        raise HTTPException(status_code=409, detail="Дообучение уже идёт.")
    if not (CORRECTIONS_ROOT / "train" / "images").is_dir() or \
       not any((CORRECTIONS_ROOT / "train" / "images").glob("*.png")):
        raise HTTPException(status_code=400, detail="Нет сохранённых коррекций для дообучения.")

    ckpt = BASE_DIR / "checkpoints" / "best_model.pt"
    cmd = [_sys.executable, "-u", str(BASE_DIR / "train.py"),
           "--data-root", str(BASE_DIR / "own_dataset_prepared"),
           "--data-root", str(BASE_DIR / "lumenstone_prepared"),
           "--data-root", str(CORRECTIONS_ROOT),
           "--arch", "deeplabv3plus", "--init-from", str(ckpt),
           "--batch-size", "4", "--epochs", str(epochs), "--repeat", "2",
           "--patience", str(epochs), "--num-workers", "0",
           "--output", str(BASE_DIR / "checkpoints")]

    log_path = BASE_DIR / "retrain.log"

    def _worker():
        RETRAIN_STATE["running"] = True
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(BASE_DIR))
        finally:
            RETRAIN_STATE["running"] = False
            RETRAIN_STATE["finished_at"] = time.time()
            # next inference call reloads the (possibly updated) checkpoint
            inference._model_cache.update({"tried": False, "model": None, "device": None})

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "epochs": epochs}


@app.get("/api/retrain/status")
def retrain_status():
    log_tail = ""
    log_path = BASE_DIR / "retrain.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        epoch_lines = [l for l in text.splitlines() if l.startswith("Epoch") or "Warm start" in l or "Done." in l]
        log_tail = "\n".join(epoch_lines[-4:])
    return {"running": RETRAIN_STATE["running"], "log_tail": log_tail,
            "finished_at": RETRAIN_STATE["finished_at"]}


@app.get("/api/batch/{job_id}/download")
def batch_download(job_id: str):
    job = BATCH_JOBS.get(job_id)
    if job is None or not job["finished"]:
        raise HTTPException(status_code=404, detail="Результаты не готовы.")
    out_dir = Path(job["dir"])
    zip_path = out_dir.parent / "results.zip"
    if not zip_path.is_file():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out_dir))
    return FileResponse(str(zip_path), media_type="application/zip",
                        filename="ore_batch_results.zip")
