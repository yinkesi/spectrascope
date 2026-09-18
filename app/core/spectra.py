"""Spectrum container, text/CSV parsing, and synthetic sample generation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NM_MIN, NM_MAX = 350.0, 2500.0


@dataclass
class Spectrum:
    """A 1-D reflectance spectrum sampled on an arbitrary wavelength grid (nm)."""

    wavelength: np.ndarray  # nm, ascending
    reflectance: np.ndarray  # unitless 0..1+
    name: str = "unknown"
    source: str = "upload"  # upload | demo | library
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.reflectance = np.asarray(self.reflectance, dtype=float)
        if self.wavelength.ndim != 1 or self.wavelength.size < 10:
            raise ValueError("spectrum needs at least 10 samples in a single column")
        if self.wavelength.size != self.reflectance.size:
            raise ValueError("wavelength/reflectance length mismatch")
        order = np.argsort(self.wavelength)
        self.wavelength = self.wavelength[order]
        self.reflectance = self.reflectance[order]


def _parse_numeric_block(lines: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the first column pair in `lines` that parses as increasing wl + reflectance."""
    for width in (2, 3):
        for start in range(0, min(30, len(lines))):
            cols: list[list[float]] = [[] for _ in range(width)]
            ok = True
            used = 0
            for ln in lines[start:]:
                parts = ln.replace(";", ",").replace("\t", ",").split(",")
                vals: list[float] = []
                for p in parts:
                    p = p.strip()
                    try:
                        vals.append(float(p))
                    except ValueError:
                        continue
                if len(vals) < width:
                    continue
                for i in range(width):
                    cols[i].append(vals[i])
                used += 1
                if used >= 4000:
                    break
            if used < 10:
                ok = False
            if ok:
                wl = np.asarray(cols[0])
                rf = np.asarray(cols[1])
                if wl.size >= 10 and np.all(np.diff(wl) > 0) and rf.size == wl.size:
                    # Wavelengths in µm (0.35–2.6) are common in USGS exports.
                    if np.median(wl) < 3.0:
                        wl = wl * 1000.0
                    # Some instruments export percent reflectance.
                    if np.nanmedian(rf) > 1.5:
                        rf = rf / 100.0
                    return wl, rf
    return None


def parse_spectrum_text(text: str, name: str = "upload") -> Spectrum:
    """Parse CSV/TSV/whitespace text of the form `wavelength, reflectance`."""
    if not text or not text.strip():
        raise ValueError("empty file")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    parsed = _parse_numeric_block(lines)
    if parsed is None:
        raise ValueError(
            "could not find (wavelength, reflectance) columns; expected two numeric columns, "
            "wavelength in nm or µm, one point per line"
        )
    wl, rf = parsed
    if np.any(~np.isfinite(rf)):
        good = np.isfinite(rf)
        wl, rf = wl[good], rf[good]
    rf = np.clip(rf, -0.05, 5.0)
    return Spectrum(wl, rf, name=name, source="upload")


def parse_spectrum_bytes(data: bytes, name: str) -> Spectrum:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return parse_spectrum_text(data.decode(enc), name=name)
        except UnicodeDecodeError:
            continue
        except ValueError as exc:
            if "empty file" in str(exc):
                raise
            last_error = exc
    raise ValueError(str(last_error))


def demo_samples() -> dict[str, dict]:
    """Three seeded synthetic field samples + one adversarial mixture."""
    from .library import entry_by_id, load_library, synthesize

    lib = load_library()
    rng = np.random.default_rng(20260918)

    def build(entry_id: str, label: str, note: str, tilt: float, noise: float, mix: dict | None = None):
        entry = entry_by_id(entry_id)
        spec = synthesize(entry, rng, tilt=tilt, noise_sigma=noise)
        if mix:
            other = synthesize(entry_by_id(mix["id"]), rng, tilt=tilt, noise_sigma=noise)
            rf = (1 - mix["fraction"]) * spec.reflectance + mix["fraction"] * other.reflectance
            spec.reflectance = rf
        spec.name = label
        spec.source = "demo"
        spec.metadata = {"note": note, "recipe": entry_id if not mix else f"{entry_id}+{mix['id']}"}
        return spec

    return {
        "clay_alteration": build(
            "kaolinite", "黏土蚀变带样品 · Clay alteration", "浙江某火山岩区手标本粉末，ASD 式 1nm 采样", 0.06, 0.006
        ),
        "marble": build(
            "calcite", "大理岩样品 · Marble", "碳酸盐岩区新鲜面，含微量水汽噪声", 0.04, 0.005
        ),
        "canopy": build(
            "green_vegetation", "冠层光谱 · Canopy", "夏季行道树冠层，叶面积指数较高", 0.03, 0.007
        ),
        "mixed_pixel": build(
            "green_vegetation", "混合像元 · Mixed pixel", "植被-枯草混合，检验可分解性", 0.03, 0.006,
            mix={"id": "dry_grass", "fraction": 0.45},
        ),
    }


def spectrum_to_response(spec: Spectrum, max_points: int = 900) -> dict:
    """Downsample for JSON transport."""
    step = max(1, spec.wavelength.size // max_points)
    wl = spec.wavelength[::step]
    rf = spec.reflectance[::step]
    return {
        "name": spec.name,
        "source": spec.source,
        "n": int(spec.wavelength.size),
        "wavelength": [round(float(v), 2) for v in wl],
        "reflectance": [round(float(v), 5) for v in rf],
        "metadata": spec.metadata,
    }


def spectrum_from_csv_bytes(data: bytes, filename: str) -> Spectrum:
    return parse_spectrum_bytes(data, filename)
