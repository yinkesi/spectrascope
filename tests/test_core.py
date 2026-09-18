import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.features import (  # noqa: E402
    _peak_prominences,
    extract_features,
    in_atmospheric_zone,
)
from app.core.preprocess import estimate_noise_sigma  # noqa: E402
from app.core.library import _band_tol, entry_by_id, load_library, match, synthesize  # noqa: E402
from app.core.pipeline import analyze  # noqa: E402
from app.core.preprocess import local_continuum_remove, preprocess, upper_convex_hull  # noqa: E402
from app.core.spectra import (  # noqa: E402
    Spectrum,
    demo_samples,
    parse_spectrum_text,
)


# ---------------- parsing ----------------

def test_parse_csv_basic():
    wl = np.arange(400.0, 1600.0, 100.0)
    rf = 0.3 + 0.0001 * (wl - 400)
    text = "wavelength,reflectance\n" + "\n".join(f"{a},{b:.4f}" for a, b in zip(wl, rf))
    spec = parse_spectrum_text(text)
    assert spec.wavelength[0] == 400
    assert abs(spec.reflectance[1] - rf[1]) < 1e-6


def test_parse_um_and_percent():
    wl = np.arange(0.4, 2.4, 0.001)
    rf = 0.5 * np.ones_like(wl) * 100  # percent reflectance, µm wavelengths
    text = "\n".join(f"{a:.3f},{b:.2f}" for a, b in zip(wl, rf))
    spec = parse_spectrum_text(text)
    assert 390 < spec.wavelength[0] < 410 and abs(spec.reflectance.mean() - 0.5) < 0.01


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_spectrum_text("hello\nworld\n")
    with pytest.raises(ValueError):
        parse_spectrum_text("")


# ---------------- continuum / features ----------------

def test_hull_flat_top():
    wl = np.arange(350.0, 2500.0, 5.0)
    rf = 0.8 * np.ones_like(wl)
    hull = upper_convex_hull(wl, rf)
    assert np.allclose(hull, 0.8, atol=1e-6)


def test_local_cr_recovers_gaussian_depth():
    wl = np.arange(350.0, 2500.0, 1.0)
    rf = 0.7 * (1 - 0.4 * np.exp(-4 * np.log(2) * ((wl - 2200) / 25) ** 2))
    spec = Spectrum(wl, rf)
    cr, _env = local_continuum_remove(spec)
    i = int(np.argmin(cr.reflectance))
    assert abs(cr.wavelength[i] - 2200) <= 2
    assert abs(1 - cr.reflectance[i] - 0.4) < 0.06


def test_feature_detection_kaolinite():
    spec = demo_samples()["clay_alteration"]
    result = analyze(spec)
    centers = [f["center_nm"] for f in result.features]
    assert any(abs(c - 2200) <= 8 for c in centers)
    assert any(abs(c - 2160) <= 10 for c in centers)
    assert abs(centers[0] - 1400) <= 8


def test_atmospheric_zone_flag():
    assert in_atmospheric_zone(1400)
    assert in_atmospheric_zone(1900)
    assert not in_atmospheric_zone(2200)


# ---------------- prominence correctness ----------------

def test_peak_prominences_matches_analytic():
    z = np.array([0.0, 0.2, 0.8, 0.3, 0.1, 0.5, 0.2, 0.0])
    peaks, prom = _peak_prominences(z)
    p = {i: prom[i] for i in peaks}
    # peak 2 (0.8): highest -> both bases are interval minima, both reach 0.0
    assert abs(p[2] - 0.8) < 1e-9
    # peak 5 (0.5): left barrier at 2 -> valley floor min(z[3..5]) = 0.1
    assert abs(p[5] - 0.4) < 1e-9


def test_peak_prominences_deep_hidden_valley():
    """Valley floor must see past local ripples inside the barrier segment."""
    z = np.array([9.0, 3.0, 4.0, 3.5, 5.0, 4.5, 3.0, 20.0])
    peaks, prom = _peak_prominences(z)
    p = {i: prom[i] for i in peaks}
    # peak 4 (5.0): left barrier 0 -> floor min(z[1..4]) = 3.0.
    # A truncated running-min chain would stop at the ripple min 3.5 -> prom 1.5.
    assert abs(p[4] - 2.0) < 1e-9


def test_dark_target_noise_floor():
    """A water-like dark spectrum must not explode into noise 'features'."""
    rng = np.random.default_rng(5)
    wl = np.arange(350.0, 2500.0, 1.0)
    rf = 0.05 * np.exp(-(wl - 400) / 900.0) + 0.004
    rf = rf + rng.normal(0, 0.006, wl.size)
    spec = Spectrum(wl, np.clip(rf, 0.0005, None))
    result = analyze(spec)
    assert len(result.features) <= 8, f"noise bred {len(result.features)} fake features"


# ---------------- parser robustness ----------------

def test_parse_index_first_three_columns():
    rng = np.random.default_rng(1)
    wl = np.arange(400.0, 2400.0, 20.0)
    rf = 0.5 + rng.normal(0, 0.01, wl.size)
    text = "idx,wl,refl\n" + "\n".join(f"{i},{a:.1f},{b:.4f}" for i, (a, b) in enumerate(zip(wl, rf)))
    spec = parse_spectrum_text(text)
    assert abs(spec.wavelength[0] - 400) < 1e-6
    assert abs(spec.reflectance.mean() - 0.5) < 0.05


def test_parse_descending_wavelengths():
    wl = np.arange(2400.0, 399.0, -20.0)
    rf = 0.3 * np.ones_like(wl)
    text = "\n".join(f"{a:.1f} {b:.3f}" for a, b in zip(wl, rf))
    spec = parse_spectrum_text(text)
    assert spec.wavelength[0] == 400.0
    assert np.all(np.diff(spec.wavelength) > 0)


def test_parse_dark_percent_reflectance():
    wl = np.arange(400.0, 2400.0, 20.0)
    rf = 1.2 * np.ones_like(wl)  # percent export of a very dark target
    text = "\n".join(f"{a:.1f},{b:.3f}" for a, b in zip(wl, rf))
    spec = parse_spectrum_text(text)
    assert abs(spec.reflectance.mean() - 0.012) < 1e-6


# ---------------- matching ----------------

def test_matching_top1_all_entries():
    """Every library endmember, synthesized with noise/tilt, must be identified."""
    lib = load_library()
    rng = np.random.default_rng(7)
    misses = []
    for e in lib["entries"]:
        spec = synthesize(e, rng, tilt=0.04, noise_sigma=0.006)
        r = analyze(spec)
        if r.candidates[0]["entry_id"] != e["id"]:
            misses.append((e["id"], r.candidates[0]["entry_id"]))
    # spectral near-twins allowed to swap (same chromophore / same substance):
    # Fe3+ oxides, Fe-Mg vs Ca minerals, ice vs water (CR erases albedo),
    # dark urban impervious surfaces
    twins = {("goethite", "hematite"), ("hematite", "goethite"),
             ("epidote", "chlorite"), ("chlorite", "epidote"),
             ("water", "snow"), ("snow", "water"),
             ("asphalt", "concrete"), ("concrete", "asphalt")}
    for true_id, got in misses:
        assert (true_id, got) in twins, f"{true_id} misread as {got}"


def test_tolerance_cap():
    assert _band_tol(2000.0) == 60.0
    assert _band_tol(20.0) == 12.0


def test_evidence_has_one_to_one_mapping():
    spec = demo_samples()["clay_alteration"]
    r = analyze(spec)
    obs = [e["observed_nm"] for e in r.candidates[0]["evidence"] if e["observed_nm"] is not None]
    assert len(obs) == len(set(obs)), "one observed feature used for two library bands"


# ---------------- pipeline / demos ----------------

def test_demo_samples_identify():
    expect = {
        "clay_alteration": "kaolinite",
        "marble": "calcite",
        "canopy": "green_vegetation",
    }
    for key, entry in expect.items():
        r = analyze(demo_samples()[key])
        assert r.candidates[0]["entry_id"] == entry


def test_report_structure():
    r = analyze(demo_samples()["canopy"])
    rep = r.report
    assert rep["mode"] == "deterministic"
    assert "headline" in rep and rep["headline"]
    assert len(rep["caveats"]) >= 2
    assert isinstance(rep["reasoning"], list)


def test_synthetic_mix_gets_both_endmembers_in_top3():
    r = analyze(demo_samples()["mixed_pixel"])
    ids = [c["entry_id"] for c in r.candidates]
    assert "green_vegetation" in ids
    assert "dry_grass" in ids
