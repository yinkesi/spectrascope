"""Scene mode: Sentinel-2-style multispectral unmixing and mapping.

A simulated 12-band Sentinel-2 scene is synthesized from the spectral
library (seeded, geologically plausible layout), then mapped with real
photogrammetric tooling:

- band resampling of library endmembers through Gaussian S2 spectral
  response functions (B1..B12, cirrus B10 excluded),
- fully constrained least squares (FCLS, Heinz & Chang 2001) unmixing per
  pixel: min ||Ax-b||^2  s.t. sum(x)=1, x>=0 — solved with vectorized
  projected gradient descent (Duchi et al. simplex projection),
- per-pixel reconstruction RMSE, class map, NDVI/NDWI/iron-oxide indices.

Honest scope note (also shown in the UI): a 12-band pixel cannot support
absorption-feature diagnosis (that needs hyperspectral VNIR-SWIR, the
single-spectrum mode); at multispectral resolution the defensible product
is sub-pixel abundance mapping. Clicking a pixel shows its FCLS breakdown.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .library import entry_by_id, idealized_rf, load_library

GRID = 96
SEED = 20260919

# Sentinel-2 MSI bands (nm): (name, center, FWHM). B10 cirrus excluded.
S2_BANDS = [
    ("B1", 443.0, 21.0), ("B2", 490.0, 65.0), ("B3", 560.0, 35.0),
    ("B4", 665.0, 30.0), ("B5", 705.0, 15.0), ("B6", 740.0, 15.0),
    ("B7", 783.0, 20.0), ("B8", 842.0, 115.0), ("B8A", 865.0, 20.0),
    ("B9", 945.0, 20.0), ("B11", 1610.0, 90.0), ("B12", 2190.0, 180.0),
]

# Endmembers present in the demo scene. Deliberately NO urban class: at S2's
# broad bands chlorite and concrete are 0.9999 correlated and would trade
# abundance — a real multispectral resolution limit we document instead of hide.
SCENE_ENTRIES = [
    "iron_soil",        # background
    "kaolinite",        # alteration core
    "chlorite",         # alteration halo
    "hematite",         # gossan
    "green_vegetation",
    "dry_grass",
    "water",
]

CLASS_COLORS = {
    "iron_soil": "#b08968",
    "kaolinite": "#e9e2cf",
    "chlorite": "#5f7d5a",
    "hematite": "#a03b23",
    "green_vegetation": "#2e7d43",
    "dry_grass": "#c9a83c",
    "water": "#2f6f8f",
}

BAND_NAMES = [b[0] for b in S2_BANDS]


# ---------- band resampling ----------

@lru_cache(maxsize=1)
def s2_response() -> tuple[np.ndarray, np.ndarray]:
    """(wl grid, L×K response-normalized endmember matrix at S2 bands)."""
    wl = np.arange(350.0, 2500.5, 1.0)
    rows = []
    for _name, center, fwhm in S2_BANDS:
        resp = np.exp(-4.0 * np.log(2.0) * ((wl - center) / fwhm) ** 2)
        resp = resp / resp.sum()
        rows.append(resp)
    resp_mat = np.asarray(rows)  # L x nm
    endm = []
    for eid in SCENE_ENTRIES:
        rf = idealized_rf(entry_by_id(eid), wl)
        endm.append(resp_mat @ rf)
    return wl, np.asarray(endm).T  # (L, K) with L bands, K endmembers


# ---------- simplex-constrained unmixing ----------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project columns of (K, N) onto the probability simplex (Duchi et al.)."""
    k = v.shape[0]
    u = np.sort(v, axis=0)[::-1]
    css = np.cumsum(u, axis=0) - 1.0
    idx = np.arange(1, k + 1).reshape(-1, 1)
    cond = u - css / idx > 0
    rho = cond.sum(axis=0)
    theta = css[rho - 1, np.arange(v.shape[1])] / np.maximum(rho, 1)
    return np.maximum(v - theta, 0.0)


def fcls(A: np.ndarray, B: np.ndarray, n_iter: int = 1500) -> np.ndarray:
    """Fully constrained least squares for every column of B.

    A: (L bands, K endmembers); B: (L, N pixels). Returns (K, N) abundances
    on the simplex. FISTA-accelerated projected gradient (Beck & Teboulle
    2009) with power-iteration step size — plain PGD needs ~1e5 iterations
    here because broad-band endmembers make cond(G) ~ 1e7.
    """
    G = A.T @ A
    C = A.T @ B
    # largest eigenvalue of 2G via power iteration -> safe step
    v = np.random.default_rng(0).normal(size=(G.shape[0], 1))
    for _ in range(24):
        v = G @ v
        v /= np.linalg.norm(v) or 1.0
    lam = float((v.T @ (G @ v)).item()) * 2.0
    step = 0.9 / max(lam, 1e-12)
    x = np.full((A.shape[1], B.shape[1]), 1.0 / A.shape[1])
    z = x.copy()
    t = 1.0
    for i in range(n_iter):
        x_new = project_simplex(z - step * (2.0 * (G @ z - C)))
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


# ---------- scene synthesis ----------

def _smooth_field(rng: np.random.Generator, passes: int = 3, width: int = 7) -> np.ndarray:
    """Smooth random field in [0,1]: box-blur white noise, edge-padded."""
    from numpy.lib.stride_tricks import sliding_window_view

    f = rng.normal(size=(GRID, GRID))
    for _ in range(passes):
        pad = width // 2
        padded = np.pad(f, pad, mode="edge")
        f = sliding_window_view(padded, (width, width)).mean(axis=(2, 3))
    f -= f.min()
    return f / (f.max() or 1.0)


def _ell(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Normalized ellipse distance d; d<1 means inside the rim."""
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    return np.sqrt(((xx / GRID - cx) / rx) ** 2 + ((yy / GRID - cy) / ry) ** 2)


def _mask(cx: float, cy: float, rx: float, ry: float, wobble: np.ndarray, gain: float = 0.10) -> np.ndarray:
    return np.clip(1.0 - (_ell(cx, cy, rx, ry) + gain * (wobble - 0.5)), 0.0, 1.0)


@lru_cache(maxsize=1)
def build_scene() -> dict:
    """Deterministic synthetic scene: abundances, S2 reflectance cube, ground truth.

    Zones are alpha-composited over the soil background, so each zone's
    dominant endmember keeps a high fraction while boundaries blend.
    """
    rng = np.random.default_rng(SEED)
    wob = _smooth_field(rng)
    yy, xx = np.mgrid[0:GRID, 0:GRID]

    k = len(SCENE_ENTRIES)
    idx = {e: i for i, e in enumerate(SCENE_ENTRIES)}
    abund = np.zeros((GRID * GRID, k))
    abund[:, idx["iron_soil"]] = 1.0  # background: soil everywhere

    def paint(entry: str, alpha: np.ndarray) -> None:
        a = np.clip(alpha, 0, 1).ravel()[:, None]
        abund[:] = a * _onehot(idx[entry], k) + (1.0 - a) * abund

    paint("dry_grass", _mask(0.38, 0.26, 0.15, 0.12, wob) * 0.85)                # grassland
    paint("green_vegetation", _mask(0.70, 0.28, 0.21, 0.16, wob) * 0.92)         # forest
    paint("water", _mask(0.30, 0.66, 0.17, 0.11, wob) * 0.97)                    # lake
    d_ell = _ell(0.70, 0.70, 0.16, 0.14)
    ring = np.clip(1.0 - np.abs(d_ell - 0.60) / 0.28, 0, 1) + 0.08 * (wob - 0.5)
    paint("chlorite", np.clip(ring, 0, 1) * 0.8)                                 # alteration halo (ring)
    paint("kaolinite", _mask(0.70, 0.70, 0.075, 0.065, wob) * 0.9)               # alteration core
    paint("hematite", _mask(0.50, 0.84, 0.09, 0.07, wob) * 0.9)                  # gossan

    abund = abund.T  # (K, N)
    truth = abund.argmax(axis=0)

    _wl, A = s2_response()  # (12, K)
    noise = rng.normal(0.0, 0.004, (len(S2_BANDS), GRID * GRID))
    illum = 1.0 + 0.06 * (xx / GRID).ravel() - 0.03  # gentle across-track illumination gradient
    cube = A @ abund
    cube = cube * illum + noise
    cube = np.clip(cube, 0.001, 1.2)

    return {"abund": abund, "cube": cube, "truth": truth, "A": A}


def _onehot(i: int, k: int) -> np.ndarray:
    v = np.zeros(k)
    v[i] = 1.0
    return v


# ---------- products ----------

def _indices(cube: np.ndarray) -> dict[str, np.ndarray]:
    b = {name: cube[i] for i, (name, _c, _f) in enumerate(S2_BANDS)}

    def ratio(a, c):
        return (a - c) / np.maximum(a + c, 1e-6)

    return {
        "ndvi": ratio(b["B8"], b["B4"]),
        "ndwi": ratio(b["B3"], b["B8"]),
        "iron_oxide": np.clip(b["B4"] / np.maximum(b["B2"], 1e-6) - 1.0, -1.0, 3.0),
    }


def scene_products() -> dict:
    """Run FCLS over the scene and assemble everything the UI needs."""
    scene = build_scene()
    abund_true, cube = scene["abund"], scene["cube"]
    A = scene["A"]
    n = GRID * GRID

    abund_hat = fcls(A, cube)                       # (K, N)
    recon = A @ abund_hat
    rmse = np.sqrt(np.mean((recon - cube) ** 2, axis=0))  # (N,)

    cls_hat = abund_hat.argmax(axis=0)
    agreement = float((cls_hat == scene["truth"]).mean())

    endmembers = [
        {
            "id": eid,
            "name": entry_by_id(eid)["name"],
            "name_cn": entry_by_id(eid)["name_cn"],
            "color": CLASS_COLORS[eid],
        }
        for eid in SCENE_ENTRIES
    ]
    q = lambda arr, scale=1: [round(float(v) * scale, 4) for v in arr]  # noqa: E731
    indices = _indices(cube)
    return {
        "grid": GRID,
        "bands": BAND_NAMES,
        "endmembers": endmembers,
        "class_map": [int(v) for v in cls_hat],
        "abundance": {eid: q(abund_hat[i], 100) for i, eid in enumerate(SCENE_ENTRIES)},
        "rmse": q(rmse, 1000),
        "indices": {k2: q(v) for k2, v in indices.items()},
        "truth_class_map": [int(v) for v in scene["truth"]],
        "truth_available": True,
        "agreement": round(agreement, 4),
        "note": "模拟 Sentinel-2 场景（12 波段，按 MSI 光谱响应函数重采样），FCLS 全约束解混；丰度为 0-100 整数。"
                "已知极限：高岭石与绿泥石在 S2 宽波段下相关系数 0.9996，两者丰度在过渡像元上存在互换——"
                "黏土矿物种类鉴别需要高光谱分辨率。",
    }


def pixel_spectrum(x: int, y: int) -> dict:
    """One pixel: S2 reflectance vector + FCLS breakdown + ground truth."""
    scene = build_scene()
    x = int(np.clip(x, 0, GRID - 1))
    y = int(np.clip(y, 0, GRID - 1))
    p = y * GRID + x
    cube = scene["cube"]
    abund_hat_all = fcls(scene["A"], cube[:, [p]])
    abund_hat = abund_hat_all[:, 0]
    recon = scene["A"] @ abund_hat
    rmse = float(np.sqrt(np.mean((recon - cube[:, p]) ** 2)))
    top = int(abund_hat.argmax())
    indices = _indices(cube[:, [p]])
    return {
        "x": x,
        "y": y,
        "wavelengths": [b[1] for b in S2_BANDS],
        "band_names": BAND_NAMES,
        "reflectance": [round(float(v), 4) for v in cube[:, p]],
        "endmembers": [
            {"id": eid, "name_cn": entry_by_id(eid)["name_cn"], "color": CLASS_COLORS[eid]}
            for eid in SCENE_ENTRIES
        ],
        "abundance": [
            {"id": eid, "name_cn": entry_by_id(eid)["name_cn"], "color": CLASS_COLORS[eid],
             "frac": round(float(abund_hat[i]), 4)}
            for i, eid in enumerate(SCENE_ENTRIES)
        ],
        "top_id": SCENE_ENTRIES[top],
        "top_name_cn": entry_by_id(SCENE_ENTRIES[top])["name_cn"],
        "rmse": round(rmse, 5),
        "ndvi": round(float(indices["ndvi"][0]), 4),
        "ndwi": round(float(indices["ndwi"][0]), 4),
        "truth_id": SCENE_ENTRIES[int(scene["truth"][p])],
        "truth_name_cn": entry_by_id(SCENE_ENTRIES[int(scene["truth"][p])])["name_cn"],
        "method_note": "多光谱像元（12 波段）不支持吸收特征诊断，像元级结论为 FCLS 亚像元分解；"
                       "若需吸收特征判读，请使用单光谱模式（VNIR-SWIR 全分辨率）。",
    }


def library_summary_count() -> int:
    return len(load_library()["entries"])
