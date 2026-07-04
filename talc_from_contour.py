"""
Extracts a binary talc mask from a (plain, marked) image pair, where the
marked version has the talc-bearing area outlined with a colored line
(as in your own dataset's "Область оталькования" folder -- same filename in
both the plain folder and its "Области оталькования" subfolder).

This is your best source of GOLD-STANDARD talc ground truth: unlike
LumenStone (which has no talc at all), these are real target-domain photos
with a geologist's own outline. Use this to build the actual training
target for the talc class, instead of the dark-patch heuristic in
inference.py (which is only a fallback for images that were never
annotated).

Usage:
    python talc_from_contour.py plain.jpg marked.jpg out_mask.png
"""
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi

# The annotation line is assumed to be a vivid, saturated color that barely
# occurs in the ore/gangue photo itself (geologists typically draw with a
# pure red/green/cyan brush in Photoshop). We detect it as "large per-pixel
# change from the plain image AND high saturation", rather than hard-coding
# one exact color, so it works regardless of which color was used.
DIFF_THRESHOLD = 30          # min per-pixel BGR distance to count as "drawn on"
MIN_SATURATION = 90          # marked-image HSV saturation the line pixels must have
LINE_CLOSE_KERNEL = 7        # morphological closing to bridge small gaps in the line
MIN_TALC_AREA_PX = 200       # drop tiny stray marks/noise


def _align(plain: np.ndarray, marked: np.ndarray) -> np.ndarray:
    if plain.shape[:2] != marked.shape[:2]:
        marked = cv2.resize(marked, (plain.shape[1], plain.shape[0]), interpolation=cv2.INTER_LINEAR)
    return marked


def extract_line_mask(plain_bgr: np.ndarray, marked_bgr: np.ndarray) -> np.ndarray:
    marked_bgr = _align(plain_bgr, marked_bgr)

    diff = cv2.absdiff(plain_bgr, marked_bgr).astype(np.int32).sum(axis=2)
    hsv = cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    line_mask = (diff > DIFF_THRESHOLD) & (saturation > MIN_SATURATION)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_CLOSE_KERNEL, LINE_CLOSE_KERNEL))
    line_mask = cv2.morphologyEx(line_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return line_mask


def fill_enclosed_area(line_mask: np.ndarray) -> np.ndarray:
    """Get the talc region bounded by the drawn line.

    Geologists' contours don't always close up inside the frame -- some
    exit through the image border (the annotation continues past what was
    photographed). `binary_fill_holes` only works for loops fully closed
    within the image, so it silently leaves border-touching contours as a
    bare unfilled line, losing most of the annotated area.

    Instead: the drawn line always splits the image into exactly two sides
    (whether it's a closed loop or crosses the border twice). The talc
    patch is the minority side -- consistent with the task's own framing of
    talc as a localized, non-dominant feature -- so we take every connected
    component of "not line" except the single largest one.
    """
    non_line = ~line_mask
    labeled, n = ndi.label(non_line)
    if n == 0:
        return np.zeros_like(line_mask)

    sizes = ndi.sum(non_line, labeled, range(1, n + 1))
    background_id = int(np.argmax(sizes)) + 1

    talc_mask = (labeled != background_id) & (labeled != 0)
    talc_mask |= line_mask  # include the boundary itself

    labeled2, n2 = ndi.label(talc_mask)
    sizes2 = ndi.sum(talc_mask, labeled2, range(1, n2 + 1))
    for comp_id, size in enumerate(sizes2, start=1):
        if size < MIN_TALC_AREA_PX:
            talc_mask[labeled2 == comp_id] = False

    return talc_mask


def extract_talc_mask(plain_bgr: np.ndarray, marked_bgr: np.ndarray) -> np.ndarray:
    line_mask = extract_line_mask(plain_bgr, marked_bgr)
    return fill_enclosed_area(line_mask)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    plain_path, marked_path, out_path = sys.argv[1:4]
    plain = cv2.imread(plain_path, cv2.IMREAD_COLOR)
    marked = cv2.imread(marked_path, cv2.IMREAD_COLOR)
    if plain is None or marked is None:
        raise SystemExit("Could not read one of the input images.")

    mask = extract_talc_mask(plain, marked)
    cv2.imwrite(out_path, (mask.astype(np.uint8) * 255))
    print(f"Talc coverage: {100 * mask.mean():.1f}% -- saved to {out_path}")
