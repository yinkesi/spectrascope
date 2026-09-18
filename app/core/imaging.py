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
# Gypsum is planted as a HIDDEN vein (anomaly) that the working set cannot
# model — the discovery loop must find it from residuals alone.
SCENE_ENTRIES = [
    "iron_soil",        # background
    "kaolinite",        # alteration core
    "chlorite",         # alteration halo
    "hematite",         # gossan
    "green_vegetation",
    "dry_grass",
    "water",
]
# Goethite is planted as a HIDDEN vein (supergene oxidation) that the working
# set cannot model — the discovery loop must find it from residuals alone.
# (Gypsum was tried first and is genuinely invisible to S2: its diagnostic
# 1450/1750/1950 nm bands all fall between MSI passes — itself a nice lesson.)
ANOMALY_ENTRY = "goethite"
ANOMALY_FRACTION = 0.55       # abundance inside the vein
PIXEL_SIZE_M = 20.0           # S2 20m band pixel size, for route lengths

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

    # ---- hidden gypsum vein (the discovery loop's secret) ----
    # a diagonal vein crossing the alteration field; NOT in the working set
    d_line = np.abs((xx / GRID - 0.42) * 0.55 - (yy / GRID - 0.52) * 0.83) / np.sqrt(0.55**2 + 0.83**2)
    away_from_lake = np.clip((_ell(0.30, 0.66, 0.26, 0.20) - 1.0) * 2.0, 0, 1)  # 0 at the lake, 1 outside
    vein = np.clip(1.0 - d_line / 0.025, 0, 1) * away_from_lake
    a = (np.clip(vein, 0, 1) * ANOMALY_FRACTION).ravel()  # (N,) gypsum abundance

    abund_full = np.column_stack([abund * (1.0 - a)[:, None], a])  # (N, K+1)
    truth_full = abund_full.argmax(axis=1)  # includes hidden class index K

    _wl, A_visible = s2_response()  # (12, K)
    wl_grid = np.arange(350.0, 2500.5, 1.0)
    resp = np.asarray([np.exp(-4.0 * np.log(2.0) * ((wl_grid - c) / f) ** 2) for _n, c, f in S2_BANDS])
    resp /= resp.sum(axis=1, keepdims=True)
    A_gyp = (resp @ idealized_rf(entry_by_id(ANOMALY_ENTRY), wl_grid)).reshape(-1, 1)
    A_full = np.hstack([A_visible, A_gyp])  # (12, K+1)

    noise = rng.normal(0.0, 0.004, (len(S2_BANDS), GRID * GRID))
    illum = 1.0 + 0.06 * (xx / GRID).ravel() - 0.03  # gentle across-track illumination gradient
    cube = (A_full @ abund_full.T)  # (12, N)
    cube = cube * illum + noise
    cube = np.clip(cube, 0.001, 1.2)

    return {
        "abund_full": abund_full.T,  # (K+1, N)
        "cube": cube,
        "truth_working": truth_full,  # includes hidden class index K
        "n_hidden": len(SCENE_ENTRIES),
        "A": A_visible,
        "A_full": A_full,
    }


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
    """Run FCLS (working set, gypsum NOT included) and assemble everything the UI needs."""
    scene = build_scene()
    cube = scene["cube"]
    truth_public = np.where(
        scene["truth_working"] == scene["n_hidden"],
        np.argmax(scene["abund_full"][: len(SCENE_ENTRIES)], axis=0),
        scene["truth_working"],
    )
    A = scene["A"]
    n = GRID * GRID

    abund_hat = fcls(A, cube)                       # (K, N)
    recon = A @ abund_hat
    rmse = np.sqrt(np.mean((recon - cube) ** 2, axis=0))  # (N,)

    cls_hat = abund_hat.argmax(axis=0)
    agreement = float((cls_hat == truth_public).mean())

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
        "truth_class_map": [int(v) for v in truth_public],
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
    truth_idx = int(scene["truth_working"][p])
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
        "truth_id": (SCENE_ENTRIES[truth_idx] if truth_idx < len(SCENE_ENTRIES) else ANOMALY_ENTRY),
        "truth_name_cn": entry_by_id(SCENE_ENTRIES[truth_idx] if truth_idx < len(SCENE_ENTRIES) else ANOMALY_ENTRY)["name_cn"],
        "method_note": "多光谱像元（12 波段）不支持吸收特征诊断，像元级结论为 FCLS 亚像元分解；"
                       "若需吸收特征判读，请使用单光谱模式（VNIR-SWIR 全分辨率）。",
    }


def library_summary_count() -> int:
    return len(load_library()["entries"])


# ---------- discovery: residual hotspots + hypothesis testing ----------

def _fcls_for(A: np.ndarray, cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = fcls(A, cube)
    recon = A @ X
    rmse = np.sqrt(np.mean((recon - cube) ** 2, axis=0))
    return X, rmse


def _connect_components(mask: np.ndarray, min_pixels: int = 6) -> list[list[int]]:
    """8-connected components of a boolean grid; returns lists of flat pixel ids."""
    g = GRID
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for start in np.argwhere(mask):
        y0, x0 = int(start[0]), int(start[1])
        if seen[y0, x0]:
            continue
        stack = [(x0, y0)]
        seen[y0, x0] = True
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append(y * g + x)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < g and 0 <= ny_ < g and mask[ny_, nx_] and not seen[ny_, nx_]:
                        seen[ny_, nx_] = True
                        stack.append((nx_, ny_))
        if len(comp) >= min_pixels:
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


@lru_cache(maxsize=1)
def hotspots(top: int = 3) -> dict:
    """RMSE hotspots = where the working endmember set cannot explain the data."""
    scene = build_scene()
    cube = scene["cube"]
    _X, rmse = _fcls_for(scene["A"], cube)
    thr = float(np.percentile(rmse, 97.0))
    mask = (rmse > max(thr, rmse.mean() + 1.5 * rmse.std())).reshape(GRID, GRID)
    comps = _connect_components(mask)[:top]
    regions = []
    for comp in comps:
        px = np.asarray(comp)
        resid = cube[:, px] - (scene["A"] @ fcls(scene["A"], cube[:, px]))
        mean_resid = resid.mean(axis=1)
        xs = px % GRID
        ys = px // GRID
        regions.append({
            "pixels": int(px.size),
            "cx": int(xs.mean()),
            "cy": int(ys.mean()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "mean_rmse": round(float(rmse[px].mean()), 5),
            "background_rmse": round(float(np.percentile(rmse, 50)), 5),
            "mean_residual": [round(float(v), 4) for v in mean_resid],
        })
    regions.sort(key=lambda r: -r["mean_rmse"])
    return {"regions": regions, "band_names": BAND_NAMES, "note": "RMSE 显著高于背景的区域 = 工作端元集无法解释的信号。"}


def test_hypothesis(entry_id: str, region_index: int = 0) -> dict:
    """Add a candidate endmember and refit the hotspot region. Physics decides."""
    entry_id = entry_id.strip().lower()
    try:
        entry_by_id(entry_id)
    except KeyError:
        raise KeyError(entry_id)
    scene = build_scene()
    cube = scene["cube"]
    hs = hotspots()
    if not hs["regions"]:
        return {"entry_id": entry_id, "verdict": "no_hotspots", "improvement": 0.0}
    reg = hs["regions"][min(max(region_index, 0), len(hs["regions"]) - 1)]
    x0, y0, x1, y1 = reg["bbox"]
    pad = 2
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(GRID - 1, x1 + pad), min(GRID - 1, y1 + pad)
    sel = np.asarray([y * GRID + x for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)])
    sub = cube[:, sel]

    _X0, rmse0 = _fcls_for(scene["A"], sub)
    A_aug = np.hstack([scene["A"], _endmember_col(entry_id)])
    X1, rmse1 = _fcls_for(A_aug, sub)

    # score on the pixels that flagged the anomaly: does the hypothesis
    # explain THEIR residual? (bbox-average would dilute with background)
    anomalous = rmse0 > max(1.35 * float(np.median(rmse0)), 1e-6)
    n_anom = int(anomalous.sum())
    if n_anom < 4:
        n_anom = min(len(rmse0), max(4, n_anom))
        anomalous = np.argsort(rmse0)[-n_anom:]
    before = float(rmse0[anomalous].mean())
    after = float(rmse1[anomalous].mean())
    improvement = (before - after) / max(before, 1e-9)
    new_frac = float(X1[-1][anomalous].max())
    verdict = "accepted" if improvement >= 0.35 and new_frac >= 0.2 else ("weak" if improvement >= 0.15 else "rejected")
    return {
        "entry_id": entry_id,
        "name_cn": entry_by_id(entry_id)["name_cn"],
        "region_bbox": [x0, y0, x1, y1],
        "region_pixels": int(sel.size),
        "anomalous_pixels": n_anom,
        "rmse_before": round(before, 5),
        "rmse_after": round(after, 5),
        "improvement": round(improvement, 4),
        "new_frac_max": round(new_frac, 4),
        "verdict": verdict,
    }


def _endmember_col(entry_id: str) -> np.ndarray:
    wl = np.arange(350.0, 2500.5, 1.0)
    resp = np.asarray([np.exp(-4.0 * np.log(2.0) * ((wl - c) / f) ** 2) for _n, c, f in S2_BANDS])
    resp /= resp.sum(axis=1, keepdims=True)
    return (resp @ idealized_rf(entry_by_id(entry_id), wl)).reshape(-1, 1)


def candidate_ids() -> list[str]:
    """Hypothesis pool: library entries not already in the working set."""
    working = set(SCENE_ENTRIES)
    hidden = {ANOMALY_ENTRY}
    others = [e["id"] for e in load_library()["entries"] if e["id"] not in working and e["id"] not in hidden]
    return others + [ANOMALY_ENTRY]


# ---------- active sampling: entropy field + route ----------

def plan_route(k_stops: int = 6, min_sep: int = 9) -> dict:
    """Greedy max-entropy sampling sites, nearest-neighbour ordered, metric length."""
    scene = build_scene()
    X, _rmse = _fcls_for(scene["A"], scene["cube"])
    P = np.clip(X, 1e-9, None)
    H = -(P * np.log(P)).sum(axis=0) / np.log(P.shape[0])  # normalized entropy (N,)

    picked: list[int] = []
    for i in np.argsort(-H):
        x, y = i % GRID, i // GRID
        if all(max(abs(x - p % GRID), abs(y - p // GRID)) >= min_sep for p in picked):
            picked.append(int(i))
        if len(picked) >= k_stops:
            break

    # nearest-neighbour ordering starting from the highest-entropy site
    route = [picked[0]]
    remaining = picked[1:]
    while remaining:
        cur = route[-1]
        remaining.sort(key=lambda j: abs(j % GRID - cur % GRID) + abs(j // GRID - cur // GRID))
        route.append(remaining.pop(0))

    length_px = sum(
        np.hypot(route[i + 1] % GRID - route[i] % GRID, route[i + 1] // GRID - route[i] // GRID)
        for i in range(len(route) - 1)
    )
    stops = [
        {
            "order": n + 1,
            "x": int(p % GRID),
            "y": int(p // GRID),
            "entropy": round(float(H[p]), 4),
            "est_frac": {eid: round(float(X[i, p]), 3) for i, eid in enumerate(SCENE_ENTRIES) if X[i, p] >= 0.05},
        }
        for n, p in enumerate(route)
    ]
    return {
        "stops": stops,
        "length_m": int(round(length_px * PIXEL_SIZE_M)),
        "pixel_size_m": PIXEL_SIZE_M,
        "note": "按丰度熵贪心选点（混合像元信息量最大），间隔≥%d 像元，最近邻排序；像元 20m。" % min_sep,
    }
