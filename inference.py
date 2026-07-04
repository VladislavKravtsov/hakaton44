"""
Segmentation + geological-grade classification logic used by the web API.

`predict_class_mask()` picks its backend automatically:
  1. If checkpoints/best_model.pt exists (produced by train.py) and torch is
     importable, the trained U-Net is used, with tiled sliding-window
     inference so gigapixel panoramas fit in 6GB VRAM.
  2. Otherwise it falls back to the classic-CV heuristic (Otsu + morphology)
     so the web demo keeps working even without a checkpoint.

Both backends return the same thing: a uint8 array, same H x W as the input,
values 0=background, 1=ordinary intergrowth, 2=thin intergrowth, 3=talc.
"""
import os

import numpy as np
import cv2
from scipy import ndimage as ndi

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_THIS_DIR, "checkpoints", "best_model.pt")

# ---------------------------------------------------------------------------
# Class scheme (must match the training pipeline)
# ---------------------------------------------------------------------------
CLASS_BACKGROUND = 0
CLASS_ORDINARY = 1
CLASS_THIN = 2
CLASS_TALC = 3

CLASS_COLOR_BGR = {
    CLASS_BACKGROUND: (0, 0, 0),
    CLASS_ORDINARY: (0, 255, 0),   # green
    CLASS_THIN: (0, 0, 255),       # red
    CLASS_TALC: (255, 0, 0),       # blue
}

OVERLAY_ALPHA = 0.45

# Heuristic thresholds (placeholder model only -- tune or drop once the real
# model is wired in).
MIN_ORE_COMPONENT_AREA_PX = 25
THIN_MAX_RADIUS_PX = 6
TALC_DARK_PERCENTILE = 20
TALC_MIN_COMPONENT_AREA_PX = 40
TALC_MORPH_KERNEL = 5

# Business rule threshold from the spec
TALC_GRADE_THRESHOLD_PCT = 10.0


# ---------------------------------------------------------------------------
# Preprocessing: normalize illumination/contrast so the same thresholds work
# across images taken under different lighting conditions.
# ---------------------------------------------------------------------------
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    return out


# ---------------------------------------------------------------------------
# Backend 1: trained U-Net (loaded lazily on first request; stays None if
# torch or the checkpoint is unavailable so the heuristic keeps working).
# ---------------------------------------------------------------------------
_model_cache = {"tried": False, "model": None, "device": None}

TILE_SIZE = 512
TILE_OVERLAP = 64
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _try_load_model():
    if _model_cache["tried"]:
        return _model_cache["model"]
    _model_cache["tried"] = True
    if not os.path.isfile(CHECKPOINT_PATH):
        return None
    try:
        import torch
        import segmentation_models_pytorch as smp

        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        arch_cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[ckpt.get("arch", "unet")]
        model = arch_cls(encoder_name=ckpt.get("encoder_name", "resnet34"),
                         encoder_weights=None, in_channels=3,
                         classes=ckpt.get("num_classes", 4))
        model.load_state_dict(ckpt["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        _model_cache["model"] = model
        _model_cache["device"] = device
        print(f"[inference] loaded trained model from {CHECKPOINT_PATH} "
              f"(val fg_f1={ckpt.get('val_fg_f1', float('nan')):.3f}) on {device}")
    except Exception as e:
        print(f"[inference] could not load model checkpoint ({e}), using CV heuristic")
        _model_cache["model"] = None
    return _model_cache["model"]


def _predict_with_model(img_bgr: np.ndarray, model) -> np.ndarray:
    """Tiled sliding-window inference. Tiles are fed exactly as during
    training: raw pixels (no CLAHE), RGB, ImageNet-normalized. Overlapping
    logits are averaged, then argmax."""
    import torch

    device = _model_cache["device"]
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD

    stride = TILE_SIZE - TILE_OVERLAP
    logit_sum = np.zeros((4, h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - TILE_SIZE, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - TILE_SIZE, 0) + 1, stride)) or [0]
    if ys[-1] + TILE_SIZE < h:
        ys.append(h - TILE_SIZE)
    if xs[-1] + TILE_SIZE < w:
        xs.append(w - TILE_SIZE)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y0, x0 = max(y, 0), max(x, 0)
                tile = img_rgb[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
                th, tw = tile.shape[:2]
                if th < TILE_SIZE or tw < TILE_SIZE:  # image smaller than one tile
                    tile = cv2.copyMakeBorder(tile, 0, TILE_SIZE - th, 0, TILE_SIZE - tw,
                                              cv2.BORDER_REFLECT_101)
                t = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                logits = model(t)[0].cpu().numpy()
                logit_sum[:, y0:y0 + th, x0:x0 + tw] += logits[:, :th, :tw]
                weight[y0:y0 + th, x0:x0 + tw] += 1.0

    logit_sum /= np.maximum(weight, 1e-6)
    return logit_sum.argmax(axis=0).astype(np.uint8)


def predict_class_mask(img_bgr: np.ndarray) -> np.ndarray:
    model = _try_load_model()
    if model is not None:
        return _predict_with_model(img_bgr, model)
    return _predict_heuristic(img_bgr)


# ---------------------------------------------------------------------------
# Backend 2 (fallback): Otsu threshold separates bright ore phases from the
# dark/grey silicate matrix; ore components are then split into
# ordinary/thin by grain thickness; talc is detected as dark scattered
# patches inside the non-ore matrix.
# ---------------------------------------------------------------------------
def _predict_heuristic(img_bgr: np.ndarray) -> np.ndarray:
    proc = preprocess(img_bgr)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    _, ore_mask_u8 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ore_mask = ore_mask_u8.astype(bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ore_mask = cv2.morphologyEx(ore_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    class_mask = _classify_ore_components(ore_mask)

    gangue_mask = ~ore_mask
    talc_mask = _detect_talc(proc, gangue_mask)
    class_mask[talc_mask & (class_mask == CLASS_BACKGROUND)] = CLASS_TALC

    return class_mask


def _classify_ore_components(ore_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(ore_mask.shape, dtype=np.uint8)
    if not ore_mask.any():
        return out

    labeled, n = ndi.label(ore_mask)
    dist = cv2.distanceTransform(ore_mask.astype(np.uint8), cv2.DIST_L2, 5)

    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        if int(comp.sum()) < MIN_ORE_COMPONENT_AREA_PX:
            continue
        max_radius = dist[comp].max()
        out[comp] = CLASS_THIN if max_radius < THIN_MAX_RADIUS_PX else CLASS_ORDINARY

    return out


def _detect_talc(img_bgr: np.ndarray, gangue_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(gangue_mask.shape, dtype=bool)
    if not gangue_mask.any():
        return out

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray[gangue_mask], TALC_DARK_PERCENTILE)
    candidate = (gray <= threshold) & gangue_mask

    k = TALC_MORPH_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    labeled, n = ndi.label(candidate)
    sizes = ndi.sum(candidate, labeled, range(1, n + 1))
    for comp_id, size in enumerate(sizes, start=1):
        if size < TALC_MIN_COMPONENT_AREA_PX:
            candidate[labeled == comp_id] = False

    return candidate


# ---------------------------------------------------------------------------
# Metrics + expert business rule (per the spec)
# ---------------------------------------------------------------------------
GRADE_MODEL_PATH = os.path.join(_THIS_DIR, "checkpoints", "grade_tree.joblib")
_grade_model_cache = {"tried": False, "bundle": None}

# Features fed to the grade classifier, in this exact order. Keep in sync
# with fit_grade_classifier.py.
GRADE_FEATURES = ["pct_talc", "pct_ore_total", "thin_share_of_ore",
                  "frag_per_mpx", "small_grain_ore_share", "median_grain_area"]

SMALL_GRAIN_AREA_PX = 500  # ore component below this area counts as "small"


def _try_load_grade_model():
    if _grade_model_cache["tried"]:
        return _grade_model_cache["bundle"]
    _grade_model_cache["tried"] = True
    if os.path.isfile(GRADE_MODEL_PATH):
        try:
            import joblib
            _grade_model_cache["bundle"] = joblib.load(GRADE_MODEL_PATH)
            print(f"[inference] loaded grade classifier from {GRADE_MODEL_PATH}")
        except Exception as e:
            print(f"[inference] could not load grade classifier ({e}), using threshold rule")
    return _grade_model_cache["bundle"]


def compute_morphology_features(class_mask: np.ndarray) -> dict:
    """Grain-structure features that encode the spec's own wording: in a
    hard-to-concentrate ore the sulfides are 'significantly replaced', i.e.
    what's left of each grain is fragmented into many small pieces. Pure
    per-pixel shares miss that (a heavily-replaced grain still yields plenty
    of 'ordinary' pixels), so the grade classifier gets these as well."""
    ore_mask = (class_mask == CLASS_ORDINARY) | (class_mask == CLASS_THIN)
    mpx = class_mask.size / 1e6
    if not ore_mask.any():
        return {"frag_per_mpx": 0.0, "small_grain_ore_share": 0.0, "median_grain_area": 0.0}

    labeled, n = ndi.label(ore_mask)
    areas = ndi.sum(ore_mask, labeled, range(1, n + 1))
    small_ore = float(areas[areas < SMALL_GRAIN_AREA_PX].sum())
    return {
        "frag_per_mpx": round(n / mpx, 2),
        "small_grain_ore_share": round(100.0 * small_ore / float(areas.sum()), 2),
        "median_grain_area": round(float(np.median(areas)), 1),
    }


def compute_metrics(class_mask: np.ndarray) -> dict:
    total = class_mask.size
    n_ordinary = int((class_mask == CLASS_ORDINARY).sum())
    n_thin = int((class_mask == CLASS_THIN).sum())
    n_talc = int((class_mask == CLASS_TALC).sum())
    n_ore = n_ordinary + n_thin

    pct_ordinary = 100.0 * n_ordinary / total
    pct_thin = 100.0 * n_thin / total
    pct_talc = 100.0 * n_talc / total
    pct_ore_total = pct_ordinary + pct_thin

    ordinary_share_of_ore = 100.0 * n_ordinary / n_ore if n_ore else 0.0
    thin_share_of_ore = 100.0 * n_thin / n_ore if n_ore else 0.0

    morph = compute_morphology_features(class_mask)

    bundle = _try_load_grade_model()
    if bundle is not None:
        feats = {
            "pct_talc": pct_talc, "pct_ore_total": pct_ore_total,
            "thin_share_of_ore": thin_share_of_ore, **morph,
        }
        x = np.array([[feats[k] for k in bundle["feature_names"]]])
        grade = bundle["model"].predict(x)[0]
        if pct_talc > bundle.get("talc_hard_override_pct", 1e9):
            grade = "оталькованная"  # spec's own hard rule, with a calibrated estimator threshold
        if grade == "оталькованная":
            summary = (f"Руда классифицирована как оталькованная: содержание талька — {pct_talc:.0f}%, "
                       f"доля тонких срастаний — {thin_share_of_ore:.0f}%.")
        elif grade == "труднообогатимая":
            summary = (f"Руда классифицирована как труднообогатимая: содержание талька — {pct_talc:.0f}%, "
                       f"доля тонких срастаний — {thin_share_of_ore:.0f}%, "
                       f"рудная фаза раздроблена ({morph['frag_per_mpx']:.0f} зёрен/Мпкс).")
        else:
            summary = (f"Руда классифицирована как рядовая: содержание талька — {pct_talc:.0f}%, "
                       f"преобладание обычных срастаний — {ordinary_share_of_ore:.0f}%.")
        return {
            "grade": grade, "summary": summary,
            "pct_ore_total": round(pct_ore_total, 2), "pct_ordinary": round(pct_ordinary, 2),
            "pct_thin": round(pct_thin, 2), "pct_talc": round(pct_talc, 2),
            "ordinary_share_of_ore": round(ordinary_share_of_ore, 2),
            "thin_share_of_ore": round(thin_share_of_ore, 2), **morph,
        }

    if pct_talc > TALC_GRADE_THRESHOLD_PCT:
        grade = "оталькованная"
        dominant_pct = thin_share_of_ore if thin_share_of_ore >= ordinary_share_of_ore else ordinary_share_of_ore
        summary = (
            f"Руда классифицирована как оталькованная: содержание талька — {pct_talc:.0f}%, "
            f"преобладание {'тонких' if thin_share_of_ore >= ordinary_share_of_ore else 'обычных'} "
            f"срастаний — {dominant_pct:.0f}%."
        )
    elif thin_share_of_ore >= ordinary_share_of_ore and n_ore > 0:
        grade = "труднообогатимая"
        summary = (
            f"Руда классифицирована как труднообогатимая: содержание талька — {pct_talc:.0f}%, "
            f"преобладание тонких срастаний — {thin_share_of_ore:.0f}%."
        )
    else:
        grade = "рядовая"
        summary = (
            f"Руда классифицирована как рядовая: содержание талька — {pct_talc:.0f}%, "
            f"преобладание обычных срастаний — {ordinary_share_of_ore:.0f}%."
        )

    return {
        "grade": grade,
        "summary": summary,
        "pct_ore_total": round(pct_ore_total, 2),
        "pct_ordinary": round(pct_ordinary, 2),
        "pct_thin": round(pct_thin, 2),
        "pct_talc": round(pct_talc, 2),
        "ordinary_share_of_ore": round(ordinary_share_of_ore, 2),
        "thin_share_of_ore": round(thin_share_of_ore, 2),
        **morph,
    }


def class_mask_to_color(class_mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for cls, bgr in CLASS_COLOR_BGR.items():
        color[class_mask == cls] = bgr
    return color


def make_overlay(img_bgr: np.ndarray, class_mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    color = class_mask_to_color(class_mask)
    fg = class_mask != CLASS_BACKGROUND
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)
    overlay = img_bgr.copy()
    overlay[fg] = blended[fg]
    return overlay


def analyze(img_bgr: np.ndarray) -> dict:
    """Full pipeline for one image: predict, compute metrics, build overlay."""
    class_mask = predict_class_mask(img_bgr)
    metrics = compute_metrics(class_mask)
    overlay = make_overlay(img_bgr, class_mask)
    color_mask = class_mask_to_color(class_mask)
    return {"metrics": metrics, "class_mask": class_mask, "overlay": overlay, "color_mask": color_mask}
