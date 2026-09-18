import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.imaging import (  # noqa: E402
    GRID,
    SCENE_ENTRIES,
    S2_BANDS,
    build_scene,
    fcls,
    pixel_spectrum,
    project_simplex,
    scene_products,
    s2_response,
)


def test_project_simplex():
    v = np.array([[2.0, 0.5], [-1.0, 0.3], [0.5, 0.1]])
    p = project_simplex(v)
    assert np.allclose(p.sum(axis=0), 1.0)
    assert (p >= 0).all()
    # already-on-simplex input is a fixed point
    q = project_simplex(np.array([[0.2], [0.3], [0.5]]))
    assert np.allclose(q, [[0.2], [0.3], [0.5]])


def test_s2_response_shape():
    _wl, A = s2_response()
    assert A.shape == (len(S2_BANDS), len(SCENE_ENTRIES))
    assert (A > 0).all() and A.max() < 1.0  # reflectance-scale, not garbage


def test_fcls_recovers_pure_pixels():
    _wl, A = s2_response()
    X = fcls(A, A)
    assert np.allclose(X.sum(axis=0), 1.0, atol=1e-6)
    assert (X >= 0).all()
    assert (X.argmax(axis=0) == np.arange(len(SCENE_ENTRIES))).all()
    # broad-band conditioning is finite; allow honest split on the worst pair
    assert np.abs(X - np.eye(len(SCENE_ENTRIES))).max() < 0.2


def test_scene_products_consistent():
    prods = scene_products()
    n = GRID * GRID
    assert len(prods["class_map"]) == n == len(prods["truth_class_map"])
    assert len(prods["endmembers"]) == len(SCENE_ENTRIES)
    for em in prods["endmembers"]:
        fracs = np.asarray(prods["abundance"][em["id"]])
        assert fracs.shape == (n,) and fracs.min() >= 0 and fracs.max() <= 100
    # summed abundances == 100 per pixel
    total = np.sum([np.asarray(v) for v in prods["abundance"].values()], axis=0)
    assert np.allclose(total, 100.0, atol=0.5)
    # unmixing quality on the simulated scene
    assert prods["agreement"] >= 0.90


def test_scene_all_classes_present_and_recovered():
    prods = scene_products()
    import collections

    # public truth covers exactly the 7 working classes (hidden vein excluded)
    truth = collections.Counter(prods["truth_class_map"])
    assert set(truth) == set(range(len(SCENE_ENTRIES))), "every working endmember must own some pixels"
    pred = np.asarray(prods["class_map"])
    truth_arr = np.asarray(prods["truth_class_map"])
    for i, em in enumerate(prods["endmembers"]):
        m = truth_arr == i
        hit = float((pred[m] == i).mean())
        # 0.50 not higher: the hidden vein deliberately crosses some zones
        assert hit >= 0.50, f"{em['id']} argmax-hit {hit:.2f} too low"


def test_hidden_anomaly_discoverable():
    """The planted goethite vein must flag as residual hotspot and win its test."""
    from app.core.imaging import ANOMALY_ENTRY, build_scene, hotspots, test_hypothesis

    scene = build_scene()
    vein = scene["truth_working"] == scene["n_hidden"]
    assert vein.sum() >= 25, "vein must exist"

    hs = hotspots()
    assert hs["regions"], "vein must produce a residual hotspot"

    r_goethite = test_hypothesis(ANOMALY_ENTRY)
    assert r_goethite["verdict"] == "accepted", r_goethite
    # decoys (mineral cousins included) must NOT be accepted
    for decoy in ("hematite", "kaolinite", "alunite"):
        assert test_hypothesis(decoy)["verdict"] != "accepted"


def test_route_planning():
    from app.core.imaging import plan_route

    r = plan_route(6, 9)
    assert len(r["stops"]) == 6
    pts = [(s["x"], s["y"]) for s in r["stops"]]
    assert len(set(pts)) == 6
    # min separation (Chebyshev) between all stops
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert max(abs(pts[i][0] - pts[j][0]), abs(pts[i][1] - pts[j][1])) >= 9
    assert r["length_m"] > 500  # a real route, not a degenerate cluster


def test_pixel_spectrum_endpoint_shape():
    p = pixel_spectrum(30, 63)  # inside the lake
    assert p["top_id"] == "water"
    assert abs(sum(a["frac"] for a in p["abundance"]) - 1.0) < 1e-3
    assert p["ndwi"] > p["ndvi"]  # water is water
    q = pixel_spectrum(67, 27)  # forest
    assert q["top_id"] == "green_vegetation"
    assert q["ndvi"] > 0.3
