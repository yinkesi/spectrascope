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
    """Local maxima of `z` and their topographic prominence (O(n))."""
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

    # valley-floor min between the peak and its barrier on each side (DP over barrier segments)
    lm = np.zeros(n)
    for i in range(n):
        j = prev_greater[i]
        lm[i] = z[i] if j == i - 1 or j == -1 and i == 0 else min(z[i], lm[i - 1])
    rm = np.zeros(n)
    for i in range(n - 1, -1, -1):
        j = next_greater[i]
        rm[i] = z[i] if j == i + 1 or j == n and i == n - 1 else min(z[i], rm[i + 1])

    global_min = float(z.min())
    for i in peaks:
        left_base = lm[i - 1] if prev_greater[i] != -1 else global_min
        right_base = rm[i + 1] if next_greater[i] != n else global_min
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


def extract_features(cr_wl: np.ndarray, cr: np.ndarray) -> list[dict]:
    z = np.clip(1.0 - savgol_smooth(cr, window=15, order=2), 0.0, 0.99)
    z = savgol_smooth(z, window=7, order=2)
    peaks, prom = _peak_prominences(z)
    out: list[dict] = []
    for i in peaks:
        if prom[i] < PROMINENCE_MIN or z[i] < DEPTH_MIN:
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
        if merged and f["center_nm"] - merged[-1]["center_nm"] < 10.0:
            if f["depth"] > merged[-1]["depth"]:
                merged[-1] = f
        else:
            merged.append(f)
    return merged
