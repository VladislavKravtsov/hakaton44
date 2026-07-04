r"""
Builds pixel-level green/red/blue masks from your team's real ore-photo
dataset (the one on the shared Yandex.Disk) and writes it out in the same
layout as lumenstone_pipeline.py's output (train/val split, images/masks/
masks_color/overlays + manifest.csv) so the two can simply be merged into
one training set afterward.

SETUP: download the "Фото руд по сортам. ч1" (and optionally "ч2") folders
from Yandex.Disk (select them -> "Скачать") and unzip them so this script's
directory contains, preserving the original folder names exactly:

    webapp/raw_own_dataset/
      Фото руд по сортам. ч1/
        Оталькованные руды/
          Области оталькования/   <- ~42 images with a drawn talc outline
          *.JPG                   <- same filenames, plain version
        Рядовые руды/
        Труднообогатимые руды/
      Фото руд по сортам. ч2/      (optional, only if --include-ch2)
        оталькованные/
        рядовые/
        тонкие/

Supervision strategy per source folder (read before trusting the output
blindly):

  "Оталькованные руды/Области оталькования" (~42 pairs)
      GOLD talc mask, extracted from the geologist's own drawn contour via
      talc_from_contour.py. Ordinary/thin comes from the same CV heuristic
      used in inference.py, applied only outside the talc region.

  "Оталькованные руды/*" without a contour match (~26 images, ч1)
  "Фото руд по сортам. ч2/оталькованные" (~87 images)
      These ARE talc-bearing (that's why they're in this folder) but have no
      drawn outline, so there's no pixel ground truth. We fall back to the
      dark-patch heuristic as a WEAK pseudo-label here -- better than
      nothing since we already know a positive is present somewhere, but
      noisier than the gold set. Ordinary/thin: same heuristic as above.

  "Рядовые руды" / "Труднообогатимые руды" (ч1)
  "рядовые" / "тонкие" (ч2)
      Grade was decided by a geologist WITHOUT talc being the deciding
      factor, i.e. talc content is at most the ~10% grade threshold, usually
      much less for clean examples. We approximate talc=0 for these (a weak
      assumption -- flagged here, not hidden). Ordinary/thin: CV heuristic.

Usage:
    python build_own_dataset.py                              # ч1 only (has all the gold pairs)
    python build_own_dataset.py --include-ch2 --limit-ch2 150 # + up to 150 images per ч2 folder
"""
import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
import inference
import talc_from_contour as tc

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(_THIS_DIR, "raw_own_dataset")

# ч2 photos are ~4000px/12MP vs ~2272px in ч1; full-size processing costs
# ~40s/image (NlMeans denoise + 4 huge PNG writes), which made a 487-image
# batch take hours. Downscaling to ч1's scale keeps grain sizes comparable
# across the training set AND makes prep ~10x faster.
MAX_SIDE_PX = 2600


def _preprocess_fast(img_bgr):
    """CLAHE-only illumination fix. Replaces inference.preprocess (which
    also runs fastNlMeansDenoisingColored -- the single slowest step of the
    whole prep, ~30s on a 12MP photo) for pseudo-labeling: the morphology
    opening in the heuristic already removes speck noise, so the denoiser
    added nothing but time here."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


inference.preprocess = _preprocess_fast
OUTPUT_ROOT = os.path.join(_THIS_DIR, "own_dataset_prepared")
VAL_FRACTION = 0.10
RANDOM_SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CH1_ROOT = os.path.join(RAW_ROOT, "Фото руд по сортам. ч1")
CH2_ROOT = os.path.join(RAW_ROOT, "Фото руд по сортам. ч2")

GOLD_MARKED_DIR = os.path.join(CH1_ROOT, "Оталькованные руды", "Области оталькования")
GOLD_PLAIN_DIR = os.path.join(CH1_ROOT, "Оталькованные руды")
ZERO_DIRS_CH1 = [
    os.path.join(CH1_ROOT, "Рядовые руды"),
    os.path.join(CH1_ROOT, "Труднообогатимые руды"),
]
CH2_WEAK_POSITIVE_DIR = os.path.join(CH2_ROOT, "оталькованные")
CH2_ZERO_DIRS = [
    os.path.join(CH2_ROOT, "рядовые"),
    os.path.join(CH2_ROOT, "тонкие"),
]


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    """cv2.imread silently returns None for paths containing non-ASCII
    characters on Windows -- your raw folder/file names are Cyrillic, so
    this matters everywhere in this script. Read the bytes ourselves and
    let cv2 decode them instead."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _list_images(directory: str) -> dict:
    if not os.path.isdir(directory):
        return {}
    return {
        name: os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in IMG_EXTS and os.path.isfile(os.path.join(directory, name))
    }


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Counterpart to imread_unicode -- cv2.imwrite also silently fails on
    non-ASCII paths on Windows."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def _safe_ascii(name: str) -> str:
    """Source filenames mix Cyrillic and Latin lookalikes (e.g. a Cyrillic
    "х" instead of Latin "x" in "...10х.JPG") inconsistently. Output names
    must stay pure ASCII so every later step (training, cv2 I/O) can't trip
    on the same Unicode-path bug again."""
    ascii_name = name.encode("ascii", errors="ignore").decode("ascii")
    ascii_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in ascii_name)
    return ascii_name or "unnamed"


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def _ordinary_thin_mask(img_bgr: np.ndarray, exclude_mask: np.ndarray = None) -> np.ndarray:
    """Same CV heuristic as inference.predict_class_mask, but skips talc
    detection (handled separately here) and can exclude a region (the gold
    talc area) from being considered ore."""
    proc = inference.preprocess(img_bgr)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    _, ore_mask_u8 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ore_mask = ore_mask_u8.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ore_mask = cv2.morphologyEx(ore_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    if exclude_mask is not None:
        ore_mask &= ~exclude_mask
    return inference._classify_ore_components(ore_mask)


def process_gold_pair(plain_bgr: np.ndarray, marked_bgr: np.ndarray) -> np.ndarray:
    talc_mask = tc.extract_talc_mask(plain_bgr, marked_bgr)
    class_mask = _ordinary_thin_mask(plain_bgr, exclude_mask=talc_mask)
    class_mask[talc_mask] = inference.CLASS_TALC
    return class_mask


def process_weak_positive(img_bgr: np.ndarray) -> np.ndarray:
    """Talc folder without a drawn contour: use the dark-patch heuristic as
    a noisy pseudo-label instead of leaving talc unlabeled."""
    proc = inference.preprocess(img_bgr)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    _, ore_mask_u8 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ore_mask = ore_mask_u8.astype(bool)
    gangue_mask = ~ore_mask
    talc_mask = inference._detect_talc(proc, gangue_mask)
    class_mask = _ordinary_thin_mask(img_bgr, exclude_mask=talc_mask)
    class_mask[talc_mask] = inference.CLASS_TALC
    return class_mask


def process_talc_zero(img_bgr: np.ndarray) -> np.ndarray:
    return _ordinary_thin_mask(img_bgr)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _ensure_split_dirs(split: str) -> dict:
    dirs = {k: os.path.join(OUTPUT_ROOT, split, k) for k in ("images", "masks", "masks_color", "overlays")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _save_sample(name: str, img_bgr: np.ndarray, class_mask: np.ndarray, split: str, dirs: dict) -> dict:
    name = _safe_ascii(name)
    color_mask = inference.class_mask_to_color(class_mask)
    overlay = inference.make_overlay(img_bgr, class_mask)
    imwrite_unicode(os.path.join(dirs["images"], name), img_bgr)
    imwrite_unicode(os.path.join(dirs["masks"], name), class_mask)
    imwrite_unicode(os.path.join(dirs["masks_color"], name), color_mask)
    imwrite_unicode(os.path.join(dirs["overlays"], name), overlay)
    return {"final_split": split, "filename": name}


def _expected_out_name(kind: str, payload) -> str:
    if kind == "gold":
        name = payload[0]
        return _safe_ascii("OWN_gold_" + _stem(name) + ".png")
    name = payload[0]
    tag = "weakpos" if kind == "weak_positive" else "zero"
    return _safe_ascii(f"OWN_{tag}_" + _stem(name) + ".png")


def _already_done(out_name: str, dirs: dict) -> bool:
    """All four output files for this sample already exist -- skip
    reprocessing it. Lets a re-run after a crash/interruption (or after
    tweaking unrelated code) only do the work that's actually missing."""
    return all(os.path.isfile(os.path.join(dirs[k], out_name)) for k in ("images", "masks", "masks_color", "overlays"))


def _process_entry(kind: str, payload, split_name: str, dirs: dict):
    out_name = _expected_out_name(kind, payload)
    if _already_done(out_name, dirs):
        return {"final_split": split_name, "filename": out_name}

    if kind == "gold":
        name, plain_path, marked_path = payload
        plain = imread_unicode(plain_path)
        marked = imread_unicode(marked_path)
        if plain is None or marked is None:
            print(f"  ! could not read {name}, skipping")
            return None
        class_mask = process_gold_pair(plain, marked)
        return _save_sample(out_name, plain, class_mask, split_name, dirs)

    name, path = payload
    img = imread_unicode(path)
    if img is None:
        print(f"  ! could not read {name}, skipping")
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE_PX:
        s = MAX_SIDE_PX / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    if kind == "weak_positive":
        class_mask = process_weak_positive(img)
    else:
        class_mask = process_talc_zero(img)
    return _save_sample(out_name, img, class_mask, split_name, dirs)


def run(include_ch2: bool, limit_ch2: int):
    entries = []  # (kind, payload)

    marked = _list_images(GOLD_MARKED_DIR)
    plain = _list_images(GOLD_PLAIN_DIR)
    shared = sorted(set(marked) & set(plain))
    print(f"Gold talc pairs found: {len(shared)} (looked in {GOLD_MARKED_DIR})")
    if not shared:
        print("  !! 0 pairs found -- check that raw_own_dataset/ is laid out as described in this file's docstring.")
    for name in shared:
        entries.append(("gold", (name, plain[name], marked[name])))

    for d in ZERO_DIRS_CH1:
        imgs = _list_images(d)
        print(f"{d}: {len(imgs)} files")
        for name, path in imgs.items():
            entries.append(("zero", (name, path)))

    if include_ch2:
        imgs = _list_images(CH2_WEAK_POSITIVE_DIR)
        items = list(imgs.items())
        if limit_ch2:
            items = items[:limit_ch2]
        print(f"{CH2_WEAK_POSITIVE_DIR}: using {len(items)} files (weak positive talc)")
        for name, path in items:
            entries.append(("weak_positive", (name, path)))

        for d in CH2_ZERO_DIRS:
            imgs = _list_images(d)
            items = list(imgs.items())
            if limit_ch2:
                items = items[:limit_ch2]
            print(f"{d}: using {len(items)} files")
            for name, path in items:
                entries.append(("zero", (name, path)))

    print(f"\nTotal entries to process: {len(entries)}")
    if not entries:
        print("Nothing to do -- see the docstring at the top of this file for the expected folder layout.")
        return

    train_entries, val_entries = train_test_split(entries, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    dirs_cache = {"train": _ensure_split_dirs("train"), "val": _ensure_split_dirs("val")}

    manifest_rows = []
    for split_name, split_entries in (("train", train_entries), ("val", val_entries)):
        print(f"\n[{split_name}] processing {len(split_entries)} entries...")
        done, skipped = 0, 0
        for i, (kind, payload) in enumerate(split_entries, start=1):
            out_name = _expected_out_name(kind, payload)
            was_cached = _already_done(out_name, dirs_cache[split_name])
            row = _process_entry(kind, payload, split_name, dirs_cache[split_name])
            if row:
                manifest_rows.append(row)
            if was_cached:
                skipped += 1
            else:
                done += 1
            if i % 20 == 0 or i == len(split_entries):
                print(f"  {i}/{len(split_entries)} (processed {done}, skipped {skipped} already-done)")

    manifest_path = os.path.join(OUTPUT_ROOT, "dataset_manifest.csv")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("final_split,filename\n")
        for row in manifest_rows:
            f.write(f"{row['final_split']},{row['filename']}\n")

    print(f"\nDone. {len(manifest_rows)} samples written under {OUTPUT_ROOT}")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-ch2", action="store_true", help="also use ч2 folders (much larger, ~1000 images)")
    parser.add_argument("--limit-ch2", type=int, default=150, help="max files per ч2 folder (0 = no limit)")
    args = parser.parse_args()
    run(include_ch2=args.include_ch2, limit_ch2=args.limit_ch2)
