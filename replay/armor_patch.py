"""Cut the armor plate between two light bars out of a frame, straightened.

The number sticker sits between the bars and is taller than they are, so the
quadrilateral spanned by the four bar endpoints is stretched vertically before
it is warped to a fixed-size patch. The same patch feeds the pair classifier
and the number classifier in ``replay/learned.py``.
"""

import cv2
import numpy as np

PATCH_SIZE = 32          # side of the square patch, px
BAR_EXTEND = 2.0         # patch height in units of the light-bar length
INSET = 0.15             # fraction of the bar spacing cut off left and right, removes the bars themselves


def plate_quad(a, b, extend=BAR_EXTEND, inset=INSET):
    """Corners (tl, tr, br, bl) of the region between bars ``a`` and ``b``.

    Bars are ``LightBar``s; their ``top``/``bottom`` endpoints are used, so a
    tilted plate gives a tilted quadrilateral.
    """
    left, right = (a, b) if a.cx <= b.cx else (b, a)
    lt, lb = np.array(left.top, float), np.array(left.bottom, float)
    rt, rb = np.array(right.top, float), np.array(right.bottom, float)

    def stretch(top, bottom):
        mid, half = (top + bottom) / 2, (bottom - top) / 2
        return mid - half * extend, mid + half * extend

    lt, lb = stretch(lt, lb)
    rt, rb = stretch(rt, rb)
    # pull both sides inwards so the bright bars do not dominate the patch
    lt, rt = lt + (rt - lt) * inset, rt + (lt - rt) * inset
    lb, rb = lb + (rb - lb) * inset, rb + (lb - rb) * inset
    return np.array([lt, rt, rb, lb], dtype=np.float32)


def warp_patch(frame, a, b, size=PATCH_SIZE):
    """Grayscale ``size`` x ``size`` patch of the plate between two bars, values in [0, 1].

    Contrast is stretched per patch (min to max), so the number reads the same
    whether the plate is lit or in shadow.
    """
    quad = plate_quad(a, b)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    patch = cv2.warpPerspective(gray, m, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    lo, hi = float(patch.min()), float(patch.max())
    return (patch - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(patch)
