"""Preprocessing: resampling, Savitzky-Golay smoothing, convex-hull continuum removal.

Continuum removal after Clark & Roush (1984): the upper convex hull of the
reflectance curve is flattened to 1.0 so absorption depth/shape become
albedo-independent and comparable across materials.
"""
from __future__ import annotations

import numpy as np

from .spectra import Spectrum

STEP_NM = 1.0


def estimate_noise_sigma(reflectance: np.ndarray) -> float:
    """Robust noise estimate from second differences of the RAW (unsmoothed)
    resampled reflectance. MAD-based so spikes don't dominate; sqrt(6)
    normalizes the second difference of white noise."""
    d2 = np.diff(reflectance, n=2)
    if d2.size < 8:
        return 0.0
    med = np.median(d2)
    return float(1.4826 * np.median(np.abs(d2 - med)) / np.sqrt(6.0))


def resample(spec: Spectrum, step: float = STEP_NM) -> Spectrum:
    grid = np.arange(
        np.ceil(spec.wavelength[0] / step) * step,
        spec.wavelength[-1] + step / 2,
        step,
    )
    grid = grid[(grid >= spec.wavelength[0]) & (grid <= spec.wavelength[-1])]
    return Spectrum(grid, np.interp(grid, spec.wavelength, spec.reflectance), spec.name, spec.source, spec.metadata)


def savgol_smooth(y: np.ndarray, window: int = 11, order: int = 3) -> np.ndarray:
    """Savitzky-Golay via least-squares filter (no scipy dependency)."""
    if window % 2 == 0:
        window += 1
    half = window // 2
    if y.size <= window:
        return y.copy()
    x = np.arange(-half, half + 1, dtype=float)
    coeffs = np.linalg.pinv(np.vander(x, order + 1, increasing=True))[0]  # row for value at center
    padded = np.pad(y, half, mode="edge")
    out = np.convolve(padded, coeffs[::-1], mode="valid")
    return out[: y.size]


def upper_convex_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Upper hull of point set, evaluated at every x (piecewise-linear envelope)."""
    pts = np.column_stack([x, y])
    hull: list[int] = []
    for i in range(pts.shape[0]):
        while len(hull) >= 2:
            o, a, b = pts[hull[-2]], pts[hull[-1]], pts[i]
            if (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    hx = x[np.asarray(hull)]
    hy = y[np.asarray(hull)]
    return np.interp(x, hx, hy)


def continuum_remove(spec: Spectrum) -> tuple[Spectrum, np.ndarray]:
    """Return (continuum-removed spectrum, continuum values at each sample)."""
    wl, rf = spec.wavelength, spec.reflectance
    positive = rf > 1e-6
    cr = np.ones_like(rf)
    continuum = rf.copy()
    if positive.sum() >= 10:
        env = upper_convex_hull(wl[positive], rf[positive])
        continuum = np.interp(wl, wl[positive], env)
        ok = continuum > 1e-6
        cr[ok] = np.clip(rf[ok] / continuum[ok], 0.0, 1.5)
    out = Spectrum(wl, cr, spec.name, spec.source, spec.metadata)
    return out, continuum


def local_envelope(rf: np.ndarray, window: int = 301) -> np.ndarray:
    """Rolling-max upper envelope, smoothed — a local continuum that does not
    flatten broad absorptions the way a global convex hull does."""
    from numpy.lib.stride_tricks import sliding_window_view

    pad = window // 2
    padded = np.pad(rf, pad, mode="edge")
    roll = sliding_window_view(padded, window).max(axis=1)
    return savgol_smooth(roll, window=101, order=2)


def local_continuum_remove(spec: Spectrum, window: int = 301) -> tuple[Spectrum, np.ndarray]:
    """Local-continuum removal: feature depths stay honest for broad bands on slopes."""
    rf = np.clip(spec.reflectance, 1e-6, None)
    env = local_envelope(rf, window)
    env = np.maximum(env, 1e-6)
    cr = np.clip(rf / env, 0.0, 1.5)
    return Spectrum(spec.wavelength, cr, spec.name, spec.source, spec.metadata), env


def preprocess(spec: Spectrum) -> tuple[Spectrum, Spectrum, Spectrum, np.ndarray, np.ndarray, float]:
    """Full chain -> (resampled+smoothed, global-hull CR, local CR, global continuum,
    local envelope, raw-noise sigma)."""
    rs_raw = resample(spec)
    sigma = estimate_noise_sigma(rs_raw.reflectance)
    rs = Spectrum(rs_raw.wavelength, savgol_smooth(rs_raw.reflectance), spec.name, spec.source, spec.metadata)
    cr, continuum = continuum_remove(rs)
    cr_local, env = local_continuum_remove(rs)
    return rs, cr, cr_local, continuum, env, sigma
