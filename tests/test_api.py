import os
import sys

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_configured" in body


def test_demo_samples_listing():
    r = client.get("/api/demo-samples")
    assert r.status_code == 200
    assert "clay_alteration" in r.json()["samples"]


def test_library_listing():
    r = client.get("/api/library")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert len(entries) == 17
    assert all("diagnostic_bands_nm" in e for e in entries)


def test_analyze_demo_endpoint():
    r = client.post("/api/analyze", data={"demo": "marble"})
    assert r.status_code == 200
    body = r.json()
    assert body["candidates"][0]["entry_id"] == "calcite"
    assert body["report"]["mode"] == "deterministic"
    assert "continuum_removed_local" in body
    assert body["features"], "features must be non-empty"


def test_analyze_unknown_demo_404():
    r = client.post("/api/analyze", data={"demo": "nope"})
    assert r.status_code == 404


def test_analyze_upload_csv():
    wl = np.arange(400.0, 2400.0, 2.0)
    rf = 0.6 - 0.25 * np.exp(-4 * np.log(2) * ((wl - 2335) / 35) ** 2)
    noise = np.random.default_rng(3).normal(0, 0.004, wl.size)
    csv = "wl,refl\n" + "\n".join(f"{a:.1f},{b:.4f}" for a, b in zip(wl, rf + noise))
    r = client.post(
        "/api/analyze",
        files={"file": ("carbonate.csv", csv.encode(), "text/csv")},
    )
    assert r.status_code == 200
    assert r.json()["candidates"][0]["entry_id"] == "calcite"


def test_analyze_bad_upload_422():
    r = client.post(
        "/api/analyze",
        files={"file": ("bad.csv", b"not\ta\tspectrum\n1\n2\n", "text/csv")},
    )
    assert r.status_code == 422


def test_chat_without_llm_falls_back():
    r = client.post("/api/chat", json={"question": "为什么？", "history": []})
    assert r.status_code == 200
    assert r.json()["mode"] == "deterministic"


def test_chat_with_unreachable_llm_no_500():
    """Configured-but-dead endpoint must return a graceful error, never a 500."""
    r = client.post(
        "/api/chat",
        json={
            "question": "为什么？",
            "history": [{"role": "banana"}, {"role": "system", "content": "evil"}, {"role": "user"}],
            "result": {"garbage": True},
            "llm": {"base_url": "http://127.0.0.1:9/v1", "model": "x"},
        },
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "error"


def test_analyze_llm_override_non_object_is_tolerated():
    r = client.post("/api/analyze", data={"demo": "marble", "llm": "[1,2]"})
    assert r.status_code == 200
    assert r.json()["candidates"][0]["entry_id"] == "calcite"


def test_analyze_llm_unreachable_falls_back_to_deterministic():
    r = client.post(
        "/api/analyze",
        data={"demo": "marble", "llm": '{"base_url": "http://127.0.0.1:9/v1", "model": "x"}'},
    )
    assert r.status_code == 200
    report = r.json()["report"]
    assert report["mode"] == "deterministic"
    assert report.get("llm_error")


def test_upload_too_large_413():
    big = b"1,1\n" * (9 * 1024 * 1024)  # > 32MB
    r = client.post("/api/analyze", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413


def test_unknown_api_path_is_404_json():
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"]


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "SPECTRASCOPE" in r.text
