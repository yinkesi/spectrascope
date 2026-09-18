"""Reference library: loading, idealized curve synthesis, and two-channel matching.

Both sides of the match — the unknown and every library entry — go through the
same detector: continuum removal + peak-prominence feature extraction, so
depths and widths are directly comparable (Tetracorder-style feature fitting).
Matching score = diagnostic-band coverage (55%) + continuum-removed shape
cosine similarity (30%) - unexplained-feature penalty (15%). Bands inside
atmospheric water-vapor zones are down-weighted on both sides.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .features import extract_features
from .preprocess import continuum_remove, local_continuum_remove
from .spectra import Spectrum

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "spectral_library.json"
ATMOSPHERIC_PENALTY = 0.35  # weight multiplier for features in noise zones

_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_library() -> dict:
    with _lock:
        with open(DATA_PATH, encoding="utf-8") as fh:
            return json.load(fh)


def entry_by_id(entry_id: str) -> dict:
    for e in load_library()["entries"]:
        if e["id"] == entry_id:
            return e
    raise KeyError(entry_id)


def synthesize(entry: dict, rng: np.random.Generator, tilt: float = 0.05, noise_sigma: float = 0.004) -> Spectrum:
    """Rebuild an idealized measured spectrum: continuum + Gaussians + tilt + noise.

    tilt multiplies the continuum by (1 + tilt * (wl-350)/2150) to emulate
    illumination/instrument drift; noise emulates ASD-class SNR.
    """
    cont_pts = np.asarray(entry["continuum"], dtype=float)
    wl = np.arange(350.0, 2500.5, 1.0)
    continuum = np.interp(wl, cont_pts[:, 0], cont_pts[:, 1])
    rf = continuum * (1.0 + tilt * (wl - 350.0) / 1800.0)
    for f in entry["features"]:
        c, w, d = f["c"], f["w"], f["d"]
        rf = rf * (1.0 - d * np.exp(-4.0 * np.log(2.0) * ((wl - c) / w) ** 2))
    rf = rf + rng.normal(0.0, noise_sigma, wl.size)
    return Spectrum(wl, np.clip(rf, 0.001, 1.2), name=entry["name_cn"], source="library", metadata={"entry": entry["id"]})


def _curve_cr(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Idealized entry curve -> (global-hull CR, local-envelope CR)."""
    wl = np.arange(350.0, 2500.5, 1.0)
    cont_pts = np.asarray(entry["continuum"], dtype=float)
    continuum = np.interp(wl, cont_pts[:, 0], cont_pts[:, 1])
    rf = continuum.copy()
    for f in entry["features"]:
        rf = rf * (1.0 - f["d"] * np.exp(-4.0 * np.log(2.0) * ((wl - f["c"]) / f["w"]) ** 2))
    rs = Spectrum(wl, rf)
    cr, _ = continuum_remove(rs)
    cr_local, _ = local_continuum_remove(rs)
    return cr.reflectance, cr_local.reflectance, wl


@lru_cache(maxsize=64)
def _cr_signature(entry_id: str) -> tuple:
    """The entry's own local-CR features, extracted with the same detector as the unknown."""
    entry = entry_by_id(entry_id)
    cr_glob, cr_local, wl = _curve_cr(entry)
    feats = tuple(
        (f["center_nm"], f["depth"], f["fwhm_nm"], f["atmospheric"]) for f in extract_features(wl, cr_local)
    )
    labels = tuple(
        _label_for(entry, c) for c, _, _, _ in feats
    )
    return tuple(wl), tuple(cr_glob), feats, labels


def _label_for(entry: dict, center_nm: float) -> str:
    best, best_dist = "吸收特征", 1e9
    for f in entry["features"]:
        d = abs(f["c"] - center_nm)
        if d < best_dist:
            best_dist = d
            best = (f.get("label_cn") or f.get("label", "吸收特征")) if d <= 30 else best
    return best


TOL_CAP_NM = 60.0  # broad valley features must not inflate match tolerance


def _band_tol(width_nm: float) -> float:
    return max(12.0, min(0.6 * width_nm, TOL_CAP_NM))


def _coverage(unknown: list[dict], sig_feats: tuple, sig_labels: tuple) -> tuple[float, list[dict]]:
    """Fraction of the entry's own CR absorption depth explained by `unknown` features.

    Library bands are visited deepest-first and each observed feature can be
    consumed only once (greedy one-to-one assignment).
    """
    bands = [(lf, label) for lf, label in zip(sig_feats, sig_labels) if lf[1] >= 0.06]
    if not bands:
        return 0.5, []
    order = sorted(range(len(bands)), key=lambda k: -bands[k][0][1])
    used: set[int] = set()
    total_w = 0.0
    got_w = 0.0
    evidence: list[dict] = []
    for k in order:
        lf, label = bands[k]
        c, d, w, atmospheric = lf
        tol = _band_tol(w)
        zone_weight = ATMOSPHERIC_PENALTY if atmospheric else 1.0
        weight = d * zone_weight
        total_w += weight
        best, best_u, best_i = 0.0, None, -1
        for i, uf in enumerate(unknown):
            if i in used:
                continue
            dist = abs(uf["center_nm"] - c)
            if dist > tol * 1.6:
                continue
            ratio = uf["depth"] / max(d, 1e-6)
            s = 0.0
            if dist <= tol and 0.35 <= ratio <= 1.9:
                s = 1.0
            elif dist <= tol * 1.4 and 0.25 <= ratio <= 2.2:
                s = 0.55
            elif dist <= tol * 1.6:
                s = 0.25
            if s > best:
                best, best_u, best_i = s, uf, i
        if best >= 0.55 and best_i >= 0:
            used.add(best_i)
        got_w += best * weight
        evidence.append(
            {
                "library_band_nm": c,
                "library_label": label,
                "library_depth": round(d, 3),
                "matched": bool(best >= 0.55),
                "partial": 0.0 < best < 0.55,
                "observed_nm": best_u["center_nm"] if best_u else None,
                "observed_depth": best_u["depth"] if best_u else None,
                "score": round(best, 2),
                "atmospheric": bool(atmospheric),
            }
        )
    return (got_w / total_w if total_w else 0.5), evidence


def _unexplained_fraction(unknown: list[dict], sig_feats: tuple) -> float:
    """Weighted share of the unknown's deep features that the entry cannot explain."""
    sig = [(c, min(w, 150.0)) for c, d, w, _ in sig_feats if d >= 0.06]
    total = 0.0
    explained = 0.0
    for uf in unknown:
        zone = ATMOSPHERIC_PENALTY if uf.get("atmospheric") else 1.0
        w = uf["depth"] * zone
        total += w
        for c, lw in sig:
            tol = max(14.0, min(0.7 * lw, 70.0))
            if abs(uf["center_nm"] - c) <= tol:
                explained += w
                break
    return 0.0 if total <= 0 else 1.0 - explained / total


def _shape_similarity(entry: dict, cr_wl: np.ndarray, cr: np.ndarray) -> float:
    """Pearson correlation between the unknown's global-hull CR curve and the entry's."""
    _, cr_glob_entry, _ = _curve_cr(entry)
    lw = np.arange(350.0, 2500.5, 1.0)
    lo = max(400.0, max(cr_wl[0], lw[0]))
    hi = min(2450.0, min(cr_wl[-1], lw[-1]))
    if hi <= lo:
        return 0.0
    grid = np.arange(lo, hi, 2.0)
    a = np.interp(grid, cr_wl, cr)
    b = np.interp(grid, lw, np.asarray(cr_glob_entry))
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0:
        return 0.0
    return max(0.0, float(np.dot(a, b)) / den)


def match(unknown_features: list[dict], cr_wl: np.ndarray, cr: np.ndarray, top_k: int = 3) -> list[dict]:
    results = []
    for entry in load_library()["entries"]:
        _, _, sig_feats, sig_labels = _cr_signature(entry["id"])
        cov, evidence = _coverage(unknown_features, sig_feats, sig_labels)
        shape = _shape_similarity(entry, cr_wl, cr)
        extras = _unexplained_fraction(unknown_features, sig_feats)
        score = 0.55 * cov + 0.30 * shape + 0.15 * (1.0 - min(extras, 1.0))
        results.append(
            {
                "entry_id": entry["id"],
                "name": entry["name"],
                "name_cn": entry["name_cn"],
                "category_cn": entry["category_cn"],
                "formula": entry["formula"],
                "note": entry["note"],
                "score": round(score, 4),
                "coverage": round(cov, 4),
                "shape_similarity": round(shape, 4),
                "unexplained": round(extras, 4),
                "evidence": evidence,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:top_k]
    if len(top) >= 2 and top[0]["score"] > 0:
        margin = top[0]["score"] - top[1]["score"]
        confidence = min(0.95, max(0.05, top[0]["score"] * 0.7 + margin * 2.5))
    else:
        confidence = top[0]["score"] if top else 0.0
    top[0]["confidence"] = round(confidence, 3)
    return top
