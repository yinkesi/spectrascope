"""Absorption-feature extraction from a continuum-removed spectrum.

We run peak-prominence analysis (the classic algorithm behind
scipy.signal.find_peaks, implemented with monotonic stacks) on the inverted
CR curve, so each detected feature is a genuine local absorption whose depth
is significant against its surrounding saddles — noise ripples on the flank
of a deep band never become features of their own.

For each feature we report center, depth, FWHM, integrated area, asymmetry,
and whether it falls inside an atmospheric-noise zone (where field data is
unreliable and matching confidence is reduced).
"""
from __future__ import annotations

import numpy as np

from .preprocess import savgol_smooth

PROMINENCE_MIN = 0.035
DEPTH_MIN = 0.05
WIDTH_MIN_NM = 8.0
ATMOSPHERIC_ZONES = [(1330.0, 1480.0), (1780.0, 1980.0)]


def in_atmospheric_zone(nm: float) -> bool:
    return any(lo <= nm <= hi for lo, hi in ATMOSPHERIC_ZONES)


def _peak_prominences(z: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Local maxima of `z` and their topographic prominence.

    scipy find_peaks semantics: extend a horizontal line from the peak until
    it exits the signal or hits a strictly higher value on each side; the
    bases are the minima of the two remaining intervals. Implemented with
    monotonic stacks + a sparse table for O(1) range-min (O(n log n) total).
    """
    n = z.size
    peaks = [i for i in range(1, n - 1) if z[i] >= z[i - 1] and z[i] > z[i + 1]]
    prom = np.zeros(n)

    # nearest index to the left/right with a strictly higher value
    prev_greater = np.full(n, -1)
    stack: list[int] = []
    for i in range(n):
        while stack and z[stack[-1]] <= z[i]:
            stack.pop()
        prev_greater[i] = stack[-1] if stack else -1
        stack.append(i)
    next_greater = np.full(n, n)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and z[stack[-1]] <= z[i]:
            stack.pop()
        next_greater[i] = stack[-1] if stack else n
        stack.append(i)

    # sparse table over z for range-min queries
    levels = [z.astype(float)]
    k = 1
    while (1 << k) <= n:
        half = 1 << (k - 1)
        prev = levels[k - 1]
        m = n - (1 << k) + 1
        levels.append(np.minimum(prev[:m], prev[half : half + m]))
        k += 1

    def range_min(lo: int, hi: int) -> float:  # inclusive bounds
        if lo > hi:
            lo = hi
        kb = (hi - lo + 1).bit_length() - 1
        row = levels[kb]
        return float(min(row[lo], row[hi - (1 << kb) + 1]))

    for i in peaks:
        j = int(prev_greater[i])
        left_base = range_min(0, i) if j == -1 else range_min(j + 1, i)
        g = int(next_greater[i])
        right_base = range_min(i, n - 1) if g == n else range_min(i, g - 1)
        prom[i] = z[i] - max(left_base, right_base)
    return peaks, prom


def _fwhm(z: np.ndarray, i: int, wl: np.ndarray) -> float:
    level = z[i] / 2.0
    left = i
    while left > 0 and z[left] > level:
        left -= 1
    right = i
    while right < z.size - 1 and z[right] > level:
        right += 1
    return float(max(wl[right] - wl[left], 1.0))


def extract_features(
    cr_wl: np.ndarray,
    cr: np.ndarray,
    noise_sigma: float | None = None,
    envelope: np.ndarray | None = None,
) -> list[dict]:
    """Extract absorption features from a local-continuum-removed curve.

    `noise_sigma` + `envelope` enable a per-point SNR floor: a feature must be
    deeper than 3x the local noise expressed in CR units (sigma / envelope).
    Library-side signatures (idealized noiseless curves) omit both.
    """
    z = np.clip(1.0 - savgol_smooth(cr, window=15, order=2), 0.0, 0.99)
    z = savgol_smooth(z, window=7, order=2)
    peaks, prom = _peak_prominences(z)
    out: list[dict] = []
    for i in peaks:
        if prom[i] < PROMINENCE_MIN or z[i] < DEPTH_MIN:
            continue
        if noise_sigma is not None and envelope is not None:
            local_floor = 3.0 * noise_sigma / max(float(envelope[i]), 1e-6)
            if z[i] < min(local_floor, 0.95):
                continue
        edge = 25.0  # steep blue/red ends breed edge artifacts; no diagnostic bands live there
        if cr_wl[i] < cr_wl[0] + edge or cr_wl[i] > cr_wl[-1] - edge:
            continue
        width = _fwhm(z, i, cr_wl)
        if width < WIDTH_MIN_NM:
            continue
        lo = np.searchsorted(cr_wl, cr_wl[i] - width)
        hi = min(z.size, np.searchsorted(cr_wl, cr_wl[i] + width) + 1)
        area = float(np.trapezoid(z[lo:hi], cr_wl[lo:hi]))
        li = min(np.searchsorted(cr_wl, cr_wl[i] - 25), z.size - 1)
        ri = min(np.searchsorted(cr_wl, cr_wl[i] + 25), z.size - 1)
        asym = float(z[li] - z[ri])  # >0: left flank deeper (shortwave-side asymmetry)
        out.append(
            {
                "center_nm": round(float(cr_wl[i]), 1),
                "depth": round(float(z[i]), 3),
                "fwhm_nm": round(width, 1),
                "area": round(area, 2),
                "asymmetry": round(asym, 3),
                "atmospheric": in_atmospheric_zone(float(cr_wl[i])),
            }
        )
    merged: list[dict] = []
    for f in sorted(out, key=lambda d: d["center_nm"]):
        if merged and f["center_nm"] - merged[-1]["center_nm"] < 20.0:
            if f["depth"] > merged[-1]["depth"]:
                merged[-1] = f
        else:
            merged.append(f)
    return merged
