r"""
Imports LumenStone (S1/S2/S3) via the teammate's already-run Kaggle notebook
output ("nornikel-lumenstone-preprocess", scriptVersionId=332410982) instead
of the raw datasets -- Kaggle exposes finished notebook outputs through a
public, unauthenticated API endpoint with signed download URLs, so this
needs no Kaggle account/API token.

What we trust from that output and what we don't:

  Original images (lumenstone_processed/<DS>/<split>/img/*.png)
      Unmodified copies of the real photos -- used directly, no caveats.

  S3 masks (lumenstone_processed/S3/<split>/masks/*.png)
      Their conversion for S3 maps real per-mineral identity: main sulfides
      (pyrite/arsenopyrite/covelline/bornite/chalcopyrite) -> green,
      magnetite/hematite -> red. That's mineralogically grounded (matches
      the task spec's own definition of a hard-to-concentrate intergrowth),
      so we reuse it -- but refine the green (main-ore) region further with
      our own morphology split (small/thin grains among the main sulfides
      also become "thin"), which their conversion skipped entirely.

  S1/S2 masks -- NOT used.
      Their conversion for S1/S2 falls back to HSV/brightness thresholds on
      `masks_colored` (an arbitrary per-mineral legend palette, not real
      mineral identity), which is ungrounded guessing -- e.g. one S1 test
      image comes out "34% talc" purely from a legend hue landing in a
      blue-ish HSV range. We don't import that. Instead we treat S1/S2
      images exactly like your own dataset's "no gold label" folders: the
      same Otsu+morphology CV heuristic from inference.py, talc forced to 0
      (LumenStone genuinely contains none).

Usage:
    python import_lumenstone_output.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
import inference
import build_own_dataset as bod  # reuse imread/imwrite-unicode, _safe_ascii, _ordinary_thin_mask

KERNEL_OUTPUT_API = "https://www.kaggle.com/api/v1/kernels/output"
KERNEL_USER = "antonoof"
KERNEL_SLUG = "nornikel-lumenstone-preprocess"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_THIS_DIR, "_lumenstone_cache")
OUTPUT_ROOT = os.path.join(_THIS_DIR, "lumenstone_prepared")
VAL_FRACTION = 0.10
RANDOM_SEED = 42
DATASETS = ["S1", "S2", "S3"]

# BGR colors used by the notebook's masks -- confirmed identical to our own
# CLASS_COLOR_BGR scheme (checked against real downloaded samples).
COLOR_TO_CLASS = {
    (0, 0, 0): inference.CLASS_BACKGROUND,
    (0, 255, 0): inference.CLASS_ORDINARY,
    (0, 0, 255): inference.CLASS_THIN,
    (255, 0, 0): inference.CLASS_TALC,
}


def fetch_file_index(max_retries: int = 5) -> dict:
    url = f"{KERNEL_OUTPUT_API}?user_name={KERNEL_USER}&kernel_slug={KERNEL_SLUG}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            files = {f["fileName"]: f["url"] for f in data["files"]}
            if data.get("nextPageToken"):
                print("  warning: output is paginated, only the first page was fetched")
            return files
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  kernel-output API call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def download_cached(url: str, local_path: str, max_retries: int = 4) -> bool:
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(local_path, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            if os.path.isfile(local_path):
                os.remove(local_path)  # never leave a truncated file behind -- it'd look "cached" next run
            if attempt < max_retries - 1:
                wait = 3 * (attempt + 1)
                print(f"  retrying {os.path.basename(local_path)} in {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f"  ! download failed for {local_path}: {e}")
                return False
    return False


def color_mask_to_class_mask(color_bgr: np.ndarray) -> np.ndarray:
    out = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    for (b, g, r), cls in COLOR_TO_CLASS.items():
        match = (color_bgr[:, :, 0] == b) & (color_bgr[:, :, 1] == g) & (color_bgr[:, :, 2] == r)
        out[match] = cls
    return out


def refine_s3_mask(their_class_mask: np.ndarray) -> np.ndarray:
    """Their S3 conversion never splits the main-ore (green) region by
    morphology -- every main-sulfide pixel is green regardless of grain
    size. Re-run our own thin/ordinary split on just that region; keep
    their red (magnetite/hematite) untouched since that's already a
    mineralogically grounded "thin" label."""
    ore_main_mask = their_class_mask == inference.CLASS_ORDINARY
    refined = inference._classify_ore_components(ore_main_mask)
    out = refined.copy()
    out[their_class_mask == inference.CLASS_THIN] = inference.CLASS_THIN
    return out


def _ensure_split_dirs(split: str) -> dict:
    dirs = {k: os.path.join(OUTPUT_ROOT, split, k) for k in ("images", "masks", "masks_color", "overlays")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _save_sample(name: str, img_bgr: np.ndarray, class_mask: np.ndarray, split: str, dirs: dict) -> dict:
    name = bod._safe_ascii(name)
    color_mask = inference.class_mask_to_color(class_mask)
    overlay = inference.make_overlay(img_bgr, class_mask)
    bod.imwrite_unicode(os.path.join(dirs["images"], name), img_bgr)
    bod.imwrite_unicode(os.path.join(dirs["masks"], name), class_mask)
    bod.imwrite_unicode(os.path.join(dirs["masks_color"], name), color_mask)
    bod.imwrite_unicode(os.path.join(dirs["overlays"], name), overlay)
    return {"final_split": split, "filename": name}


def _already_done(out_name: str, dirs: dict) -> bool:
    out_name = bod._safe_ascii(out_name)
    return all(os.path.isfile(os.path.join(dirs[k], out_name)) for k in ("images", "masks", "masks_color", "overlays"))


def run():
    print("Fetching Kaggle kernel output file index (no auth needed)...")
    files = fetch_file_index()
    print(f"  {len(files)} files listed")

    entries = []  # (dataset, split, stem, img_key, mask_key_or_none)
    for ds in DATASETS:
        for split in ("train", "test"):
            img_keys = sorted(k for k in files if k.startswith(f"lumenstone_processed/{ds}/{split}/img/"))
            for img_key in img_keys:
                stem = os.path.splitext(os.path.basename(img_key))[0]
                mask_key = f"lumenstone_processed/{ds}/{split}/masks/{os.path.basename(img_key)}"
                mask_key = mask_key if mask_key in files else None
                entries.append((ds, split, stem, img_key, mask_key))
    print(f"Total images across S1/S2/S3: {len(entries)}")

    # keep LumenStone's own train/test split; further split their "train"
    # 90/10 into our train/val, same convention as build_own_dataset.py.
    lumen_train = [e for e in entries if e[1] == "train"]
    lumen_test = [e for e in entries if e[1] == "test"]
    train_entries, val_entries = train_test_split(lumen_train, test_size=VAL_FRACTION, random_state=RANDOM_SEED)

    dirs_cache = {sp: _ensure_split_dirs(sp) for sp in ("train", "val", "test")}
    manifest_rows = []

    for split_name, split_entries in (("train", train_entries), ("val", val_entries), ("test", lumen_test)):
        print(f"\n[{split_name}] processing {len(split_entries)} entries...")
        done, skipped, failed = 0, 0, 0
        for i, (ds, orig_split, stem, img_key, mask_key) in enumerate(split_entries, start=1):
            out_name = f"LUMEN_{ds}_{orig_split}_{stem}.png"
            if _already_done(out_name, dirs_cache[split_name]):
                manifest_rows.append({"final_split": split_name, "filename": bod._safe_ascii(out_name)})
                skipped += 1
            else:
                img_local = os.path.join(CACHE_DIR, img_key.replace("/", "_"))
                if not download_cached(files[img_key], img_local):
                    failed += 1
                    continue
                img = bod.imread_unicode(img_local)
                if img is None:
                    failed += 1
                    continue

                if ds == "S3" and mask_key is not None:
                    mask_local = os.path.join(CACHE_DIR, mask_key.replace("/", "_"))
                    if download_cached(files[mask_key], mask_local):
                        their_color_mask = bod.imread_unicode(mask_local)
                        their_class_mask = color_mask_to_class_mask(their_color_mask)
                        class_mask = refine_s3_mask(their_class_mask)
                    else:
                        class_mask = bod.process_talc_zero(img)
                else:
                    class_mask = bod.process_talc_zero(img)

                row = _save_sample(out_name, img, class_mask, split_name, dirs_cache[split_name])
                manifest_rows.append(row)
                done += 1

            if i % 20 == 0 or i == len(split_entries):
                print(f"  {i}/{len(split_entries)} (processed {done}, skipped {skipped}, failed {failed})")

    manifest_path = os.path.join(OUTPUT_ROOT, "dataset_manifest.csv")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("final_split,filename\n")
        for row in manifest_rows:
            f.write(f"{row['final_split']},{row['filename']}\n")

    print(f"\nDone. {len(manifest_rows)} samples written under {OUTPUT_ROOT}")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    run()
