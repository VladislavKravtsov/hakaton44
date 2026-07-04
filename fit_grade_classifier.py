r"""
Fits the image-level grade classifier (рядовая / труднообогатимая /
оталькованная) on top of the segmentation model's per-image features, using
the raw dataset's folder structure as expert ground truth.

Why this exists: single-threshold rules on pixel shares max out around
macro F1 ~0.75 on this data (measured) -- 'thin share of ore pixels' alone
doesn't capture 'sulfides significantly replaced'. A depth-3 decision tree
over share + grain-morphology features stays fully interpretable (it IS an
expert rule set, just with data-calibrated cutoffs) and buys the missing
accuracy.

Honesty notes, also printed with the results:
  - Features come from the trained U-Net's masks; most of these images were
    in its training set, so aggregate numbers are optimistic. The val-
    holdout rows (never seen by the U-Net OR the tree) and 5-fold CV are
    the honest numbers.
  - The spec's hard rule 'talc > 10% -> оталькованная' is preserved as an
    override: the cutoff on our *estimated* talc share is calibrated so the
    decision agrees with the experts' 10%-of-true-talc labeling.

Runs the U-Net over all ~178 labeled images (~8 min on GPU), caches
features to grade_features.json, fits the tree, prints metrics, and saves
checkpoints/grade_tree.joblib -- which inference.py then picks up
automatically (restart the web server after running this).

Usage:
    python fit_grade_classifier.py
    python fit_grade_classifier.py --reuse-features   # skip the U-Net pass
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(__file__))
import inference
from build_own_dataset import imread_unicode, _safe_ascii, _stem, _list_images
from evaluate_grades import TRUTH_FOLDERS, GRADES, _val_stems, macro_f1

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CACHE = os.path.join(_THIS_DIR, "grade_features.json")
OUT_PATH = os.path.join(_THIS_DIR, "checkpoints", "grade_tree.joblib")

# Must match inference.GRADE_FEATURES
FEATURE_NAMES = inference.GRADE_FEATURES


def extract_features() -> list:
    rows = []
    val_stems = _val_stems()
    for folder, truth in TRUTH_FOLDERS:
        imgs = _list_images(folder)
        names = sorted(imgs)
        print(f"[{truth}] {len(names)} images from {folder}")
        for i, name in enumerate(names, start=1):
            img = imread_unicode(imgs[name])
            if img is None:
                continue
            class_mask = inference.predict_class_mask(img)
            m = inference.compute_metrics(class_mask)
            rows.append({
                "file": name,
                "truth": truth,
                "in_val_holdout": _safe_ascii(_stem(name)) in val_stems,
                **{k: m[k] for k in FEATURE_NAMES},
            })
            if i % 20 == 0 or i == len(names):
                print(f"  {i}/{len(names)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse-features", action="store_true", help="use cached grade_features.json")
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    # IMPORTANT: features must be produced by the raw threshold path, not by a
    # previously fitted tree -- drop any existing tree before extracting.
    if os.path.isfile(OUT_PATH) and not args.reuse_features:
        os.remove(OUT_PATH)
        inference._grade_model_cache.update({"tried": False, "bundle": None})
        print("(removed old grade_tree.joblib so feature extraction is unbiased)")

    if args.reuse_features and os.path.isfile(FEATURES_CACHE):
        rows = json.load(open(FEATURES_CACHE, encoding="utf-8"))
        print(f"Loaded {len(rows)} cached feature rows")
    else:
        t0 = time.time()
        rows = extract_features()
        with open(FEATURES_CACHE, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"Extracted features for {len(rows)} images in {time.time() - t0:.0f}s -> {FEATURES_CACHE}")

    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows])
    y = np.array([r["truth"] for r in rows])
    is_val = np.array([r["in_val_holdout"] for r in rows])

    # 5-fold CV on everything: the stable honest estimate
    print(f"\n5-fold cross-validation (depth={args.max_depth}):")
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        tree = DecisionTreeClassifier(max_depth=args.max_depth, class_weight="balanced", random_state=42)
        tree.fit(X[tr], y[tr])
        s, per = macro_f1(list(y[te]), list(tree.predict(X[te])))
        cv_scores.append(s)
        print(f"  fold {fold}: macro F1 = {s:.3f}")
    print(f"  CV mean macro F1 = {np.mean(cv_scores):.3f} +- {np.std(cv_scores):.3f}")

    # train on non-val, evaluate on the 18 val-holdout images
    tree = DecisionTreeClassifier(max_depth=args.max_depth, class_weight="balanced", random_state=42)
    tree.fit(X[~is_val], y[~is_val])
    if is_val.any():
        s, per = macro_f1(list(y[is_val]), list(tree.predict(X[is_val])))
        print(f"\nVal-holdout ({int(is_val.sum())} images): macro F1 = {s:.3f}")
        for g in GRADES:
            print(f"  {g}: {per[g]:.3f}")

    # final model trained on ALL labeled data (standard once validated)
    final_tree = DecisionTreeClassifier(max_depth=args.max_depth, class_weight="balanced", random_state=42)
    final_tree.fit(X, y)
    s_all, per_all = macro_f1(list(y), list(final_tree.predict(X)))
    print(f"\nFinal tree on all data (fit quality, optimistic): macro F1 = {s_all:.3f}")
    for g in GRADES:
        print(f"  {g}: {per_all[g]:.3f}")

    print("\nLearned rules (human-readable):")
    print(export_text(final_tree, feature_names=FEATURE_NAMES))

    # Talc hard-override: smallest estimated-talc value at which every image
    # at/above it is truly оталькованная (calibrated stand-in for the spec's
    # 'true talc > 10%' rule).
    talc_vals = X[:, FEATURE_NAMES.index("pct_talc")]
    candidates = sorted(set(talc_vals[y == "оталькованная"])) or [1e9]
    override = None
    for c in candidates:
        above = talc_vals >= c
        if above.any() and (y[above] == "оталькованная").all():
            override = float(c)
            break
    override = override if override is not None else 1e9
    print(f"Talc hard-override threshold (estimated share): >{override:.1f}%")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump({
        "model": final_tree,
        "feature_names": FEATURE_NAMES,
        "talc_hard_override_pct": override,
        "cv_macro_f1_mean": float(np.mean(cv_scores)),
        "grades": GRADES,
    }, OUT_PATH)
    print(f"\nSaved grade classifier to {OUT_PATH}")
    print("inference.py will pick it up automatically (restart the web server).")


if __name__ == "__main__":
    main()
