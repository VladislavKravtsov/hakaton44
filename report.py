"""
PDF report generation + reproducibility logging for ore-grade analyses.

- append_analysis_log(): one JSON line per analysis into analysis_log.jsonl
  with everything needed to reproduce the result (model checkpoint identity,
  thresholds, preprocessing parameters, versions).
- build_pdf(): a one-page report: verdict, metrics table, source + overlay
  previews, and the same reproducibility parameters in the footer.
"""
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_THIS_DIR, "analysis_log.jsonl")

# Cyrillic needs a registered TTF; Arial ships with Windows.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

GRADE_LABELS = {
    "рядовая": "РЯДОВАЯ",
    "труднообогатимая": "ТРУДНООБОГАТИМАЯ",
    "оталькованная": "ОТАЛЬКОВАННАЯ",
}


def collect_run_parameters() -> dict:
    """Everything that affects the result, for the log and the PDF footer."""
    import inference

    params = {
        "python": sys.version.split()[0],
        "tile_size": inference.TILE_SIZE,
        "tile_overlap": inference.TILE_OVERLAP,
        "talc_grade_threshold_pct_rule": inference.TALC_GRADE_THRESHOLD_PCT,
    }
    try:
        import torch
        params["torch"] = torch.__version__
        params["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    model = inference._try_load_model()
    if model is not None:
        try:
            import torch
            ckpt = torch.load(inference.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            params["segmentation_model"] = {
                "arch": ckpt.get("arch", "unet"),
                "encoder": ckpt.get("encoder_name", "resnet34"),
                "trained_epoch": ckpt.get("epoch"),
                "val_fg_f1": round(float(ckpt.get("val_fg_f1", float("nan"))), 4),
                "checkpoint_mtime": datetime.fromtimestamp(
                    os.path.getmtime(inference.CHECKPOINT_PATH)).isoformat(timespec="seconds"),
            }
        except Exception:
            params["segmentation_model"] = {"arch": "unknown"}
    else:
        params["segmentation_model"] = {"backend": "cv_heuristic_fallback"}

    bundle = inference._try_load_grade_model()
    if bundle is not None:
        params["grade_classifier"] = {
            "type": "decision_tree_depth3",
            "cv_macro_f1": round(float(bundle.get("cv_macro_f1_mean", float("nan"))), 4),
            "talc_hard_override_pct": bundle.get("talc_hard_override_pct"),
            "features": bundle.get("feature_names"),
        }
    else:
        params["grade_classifier"] = {"type": "threshold_rule"}
    return params


def append_analysis_log(filename: str, image_size: dict, scale: float,
                        metrics: dict, elapsed_sec: float) -> str:
    analysis_id = uuid.uuid4().hex[:10]
    record = {
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "analyzed_size": image_size,
        "downscale_applied": round(scale, 4),
        "elapsed_sec": round(elapsed_sec, 2),
        "metrics": metrics,
        "parameters": collect_run_parameters(),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return analysis_id


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _find_font(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _img_to_reader(img_bgr: np.ndarray, max_side: int = 1200):
    from reportlab.lib.utils import ImageReader
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return ImageReader(io.BytesIO(buf.tobytes())), img_bgr.shape[1] / img_bgr.shape[0]


def build_pdf(filename: str, img_bgr: np.ndarray, overlay_bgr: np.ndarray,
              metrics: dict, analysis_id: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas

    font_path = _find_font(_FONT_CANDIDATES)
    bold_path = _find_font(_FONT_BOLD_CANDIDATES)
    font = "ReportFont"
    bold = "ReportFontBold"
    pdfmetrics.registerFont(TTFont(font, font_path))
    pdfmetrics.registerFont(TTFont(bold, bold_path or font_path))

    buf = io.BytesIO()
    W, H = A4
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    margin = 40
    y = H - margin

    c.setFont(bold, 16)
    c.drawString(margin, y, "Отчёт: классификация геолого-технологического сорта руды")
    y -= 20
    c.setFont(font, 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(margin, y, f"Файл: {filename}   ·   Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}   ·   ID анализа: {analysis_id}")
    c.setFillColorRGB(0, 0, 0)
    y -= 28

    grade = metrics.get("grade", "?")
    c.setFont(bold, 13)
    c.drawString(margin, y, f"Сорт руды: {GRADE_LABELS.get(grade, grade)}")
    y -= 16
    c.setFont(font, 10)
    for chunk in _wrap(metrics.get("summary", ""), 100):
        c.drawString(margin, y, chunk)
        y -= 13
    y -= 8

    rows = [
        ("Доля сульфидов (всего)", f"{metrics.get('pct_ore_total', 0)}%"),
        ("— обычные срастания", f"{metrics.get('pct_ordinary', 0)}%"),
        ("— тонкие срастания", f"{metrics.get('pct_thin', 0)}%"),
        ("Доля талька", f"{metrics.get('pct_talc', 0)}%"),
        ("Обычные / все срастания", f"{metrics.get('ordinary_share_of_ore', 0)}%"),
        ("Тонкие / все срастания", f"{metrics.get('thin_share_of_ore', 0)}%"),
        ("Зёрен руды на Мпкс", f"{metrics.get('frag_per_mpx', '-')}"),
        ("Доля руды в мелких зёрнах", f"{metrics.get('small_grain_ore_share', '-')}%"),
    ]
    c.setFont(font, 10)
    for label, val in rows:
        c.drawString(margin + 4, y, label)
        c.drawRightString(margin + 330, y, str(val))
        c.setStrokeColorRGB(0.85, 0.85, 0.85)
        c.line(margin, y - 3, margin + 330, y - 3)
        y -= 16
    y -= 10

    img_w = (W - 2 * margin - 12) / 2
    reader1, ar1 = _img_to_reader(img_bgr)
    reader2, ar2 = _img_to_reader(overlay_bgr)
    img_h = min(img_w / ar1, img_w / ar2, y - margin - 120)
    c.drawImage(reader1, margin, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='nw')
    c.drawImage(reader2, margin + img_w + 12, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='nw')
    c.setFont(font, 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(margin, y - img_h - 11, "Исходное изображение")
    c.drawString(margin + img_w + 12, y - img_h - 11, "Сегментация: зелёный — обычные, красный — тонкие, синий — тальк")
    y = y - img_h - 30

    params = collect_run_parameters()
    seg = params.get("segmentation_model", {})
    tree = params.get("grade_classifier", {})
    c.setFont(font, 7.5)
    footer = (f"Параметры воспроизводимости: модель {seg.get('arch', '?')}/{seg.get('encoder', '?')} "
              f"(эпоха {seg.get('trained_epoch', '?')}, val F1 {seg.get('val_fg_f1', '?')}); "
              f"классификатор сорта: {tree.get('type', '?')} (CV F1 {tree.get('cv_macro_f1', '?')}); "
              f"тайл {params.get('tile_size')}px, перекрытие {params.get('tile_overlap')}px; "
              f"torch {params.get('torch', '-')}, CUDA {params.get('cuda', '-')}. "
              f"Полные параметры: analysis_log.jsonl, ID {analysis_id}.")
    for chunk in _wrap(footer, 130):
        c.drawString(margin, y, chunk)
        y -= 10

    c.save()
    return buf.getvalue()


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out
