import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.features import extract_features, in_atmospheric_zone  # noqa: E402
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
    # near-twins allowed to swap only within the same spectral family;
    # everything else must hit
    twins = {("goethite", "hematite"), ("hematite", "goethite"),
             ("epidote", "chlorite"), ("chlorite", "epidote"),
             ("water", "snow"), ("snow", "water")}
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
