r"""
Measures the metric the spec actually asks for: image-level ore-grade
classification F1 (>= 0.9 required), using your raw dataset's folder
structure as expert ground truth:

    raw_own_dataset/Фото руд по сортам. ч1/Рядовые руды            -> рядовая
    raw_own_dataset/Фото руд по сортам. ч1/Труднообогатимые руды   -> труднообогатимая
    raw_own_dataset/Фото руд по сортам. ч1/Оталькованные руды/*.JPG -> оталькованная
        (top-level files only; the "Области оталькования" subfolder is the
        same photos with a drawn outline, not an extra sample)

Each image goes through the full production pipeline (inference.analyze:
model/heuristic mask -> class percentages -> grade rule), then predictions
are compared against the folder truth.

Caveat printed with the results: most of these images were also in the
training set (with heuristic pseudo-labels), so the aggregate number is
optimistic; the val-only rows are the honest subset. Both are reported.

--calibrate additionally grid-searches the two business-rule thresholds
(talc % for "оталькованная", thin-share % for "труднообогатимая") to
maximize macro F1, and prints the best pair. That's legitimate validation-
set tuning of two scalar cutoffs, not model training.

Usage:
    python evaluate_grades.py                # evaluate with current thresholds
    python evaluate_grades.py --calibrate    # also search better thresholds
    python evaluate_grades.py --limit 30     # quick pass on a subset per class
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import inference
from build_own_dataset import imread_unicode, _safe_ascii, _stem, CH1_ROOT, GOLD_PLAIN_DIR, ZERO_DIRS_CH1, _list_images

GRADES = ["рядовая", "труднообогатимая", "оталькованная"]

TRUTH_FOLDERS = [
    (os.path.join(CH1_ROOT, "Рядовые руды"), "рядовая"),
    (os.path.join(CH1_ROOT, "Труднообогатимые руды"), "труднообогатимая"),
    (GOLD_PLAIN_DIR, "оталькованная"),  # top-level plain files only (_list_images is not recursive)
]

VAL_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "own_dataset_prepared", "val", "images")


def _val_stems() -> set:
    """ASCII-safe stems of images that were held out in the val split, so we
    can report the honest (unseen-during-training) subset separately."""
    if not os.path.isdir(VAL_IMAGES_DIR):
        return set()
    stems = set()
    for name in os.listdir(VAL_IMAGES_DIR):
        stem = _stem(name)
        for prefix in ("OWN_gold_", "OWN_weakpos_", "OWN_zero_"):
            if stem.startswith(prefix):
                stems.add(stem[len(prefix):])
    return stems


def classify_with_thresholds(pct_talc: float, thin_share: float, ordinary_share: float,
                             talc_thr: float, thin_thr: float) -> str:
    if pct_talc > talc_thr:
        return "оталькованная"
    if thin_share >= thin_thr and (thin_share + ordinary_share) > 0:
        return "труднообогатимая"
    return "рядовая"


def macro_f1(truths: list, preds: list) -> tuple:
    per_class = {}
    for g in GRADES:
        tp = sum(1 for t, p in zip(truths, preds) if t == g and p == g)
        fp = sum(1 for t, p in zip(truths, preds) if t != g and p == g)
        fn = sum(1 for t, p in zip(truths, preds) if t == g and p != g)
        denom = 2 * tp + fp + fn
        per_class[g] = (2 * tp / denom) if denom else 0.0
    return float(np.mean(list(per_class.values()))), per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="max images per class folder (0 = all)")
    parser.add_argument("--calibrate", action="store_true", help="grid-search talc/thin thresholds for best macro F1")
    parser.add_argument("--output", default="grade_eval_results.json")
    args = parser.parse_args()

    val_stems = _val_stems()
    print(f"Val-holdout stems known: {len(val_stems)}")

    rows = []
    for folder, truth in TRUTH_FOLDERS:
        imgs = _list_images(folder)
        names = sorted(imgs)
        if args.limit:
            names = names[: args.limit]
        print(f"\n[{truth}] {folder}: {len(names)} images")
        for i, name in enumerate(names, start=1):
            img = imread_unicode(imgs[name])
            if img is None:
                print(f"  ! unreadable: {name}")
                continue
            t0 = time.time()
            result = inference.analyze(img)
            m = result["metrics"]
            rows.append({
                "file": name,
                "truth": truth,
                "predicted": m["grade"],
                "pct_talc": m["pct_talc"],
                "thin_share": m["thin_share_of_ore"],
                "ordinary_share": m["ordinary_share_of_ore"],
                "in_val_holdout": _safe_ascii(_stem(name)) in val_stems,
            })
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} (last took {time.time() - t0:.1f}s)")

    truths = [r["truth"] for r in rows]
    preds = [r["predicted"] for r in rows]
    score, per_class = macro_f1(truths, preds)

    print("\n" + "=" * 60)
    print(f"ALL {len(rows)} images -- macro F1 = {score:.3f}")
    for g in GRADES:
        print(f"  {g:20s} F1 = {per_class[g]:.3f}")

    val_rows = [r for r in rows if r["in_val_holdout"]]
    if val_rows:
        v_score, v_per_class = macro_f1([r["truth"] for r in val_rows], [r["predicted"] for r in val_rows])
        print(f"\nVAL-HOLDOUT ONLY ({len(val_rows)} images, never seen in training) -- macro F1 = {v_score:.3f}")
        for g in GRADES:
            print(f"  {g:20s} F1 = {v_per_class[g]:.3f}")
        print("(the honest number for the presentation is the val-holdout one)")

    # confusion matrix
    print("\nConfusion (rows=truth, cols=predicted):")
    header = " " * 20 + "".join(f"{g[:12]:>16s}" for g in GRADES)
    print(header)
    for t in GRADES:
        counts = [sum(1 for r in rows if r["truth"] == t and r["predicted"] == p) for p in GRADES]
        print(f"{t:20s}" + "".join(f"{c:16d}" for c in counts))

    best = None
    if args.calibrate:
        print("\nCalibrating thresholds (talc % x thin-share %)...")
        for talc_thr in np.arange(2, 20.5, 0.5):
            for thin_thr in np.arange(20, 80, 2.5):
                p = [classify_with_thresholds(r["pct_talc"], r["thin_share"], r["ordinary_share"],
                                              talc_thr, thin_thr) for r in rows]
                s, _ = macro_f1(truths, p)
                if best is None or s > best[0]:
                    best = (s, float(talc_thr), float(thin_thr))
        print(f"Best: macro F1 = {best[0]:.3f} at talc_threshold={best[1]:.1f}%, thin_share_threshold={best[2]:.1f}%")
        print("(current production rule: talc>10%, thin_share>=50% i.e. thin>=ordinary)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(rows),
            "macro_f1_all": score,
            "per_class_f1_all": per_class,
            "macro_f1_val_holdout": v_score if val_rows else None,
            "n_val_holdout": len(val_rows),
            "calibrated": {"macro_f1": best[0], "talc_thr": best[1], "thin_share_thr": best[2]} if best else None,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
