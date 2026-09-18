"""End-to-end deterministic analysis: parse -> preprocess -> features -> match -> report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .features import ATMOSPHERIC_ZONES, extract_features, in_atmospheric_zone
from .library import load_library, match
from .preprocess import preprocess
from .spectra import Spectrum, spectrum_to_response


@dataclass
class AnalysisResult:
    spectrum: dict
    processed: dict
    continuum_removed: dict
    continuum_removed_local: dict
    continuum: dict
    features: list[dict]
    candidates: list[dict]
    report: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "spectrum": self.spectrum,
            "processed": self.processed,
            "continuum_removed": self.continuum_removed,
            "continuum_removed_local": self.continuum_removed_local,
            "continuum": self.continuum,
            "features": self.features,
            "candidates": self.candidates,
            "report": self.report,
            "timings_ms": self.timings_ms,
        }


def _downsample(wl: np.ndarray, vals: np.ndarray, max_points: int = 900) -> dict:
    step = max(1, wl.size // max_points)
    return {
        "wavelength": [round(float(v), 2) for v in wl[::step]],
        "values": [round(float(v), 5) for v in vals[::step]],
    }


def deterministic_report(features: list[dict], candidates: list[dict], spec: Spectrum) -> dict:
    """Template interpretation generated without any LLM — always available."""
    top = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    margin = round(top["score"] - second["score"], 4) if top and second else None

    diagnostic_hits = [
        ev
        for ev in (top["evidence"] if top else [])
        if ev["matched"] and not ev["atmospheric"]
    ]
    head = (
        f"检出 {len(features)} 个吸收特征；与参考库 17 类端元匹配后，"
        f"最优候选为 {top['name_cn']}（{top['name']}，{top['category_cn']}），得分 {top['score']:.3f}。"
        if top
        else "未检出有效特征。"
    )
    if top and margin is not None:
        head += (
            f" 与次优候选 {second['name_cn']}（{second['score']:.3f}）的差距为 {margin:.3f}，"
            + ("判读具有较强的排他性。" if margin > 0.03 else "二者接近，建议结合野外记录甄别。")
        )

    reasoning = []
    if top:
        reasoning.append(
            f"连续统去除后，诊断波段覆盖率 {top['coverage']:.0%}，形状相似度 {top['shape_similarity']:.0%}。"
        )
        for ev in diagnostic_hits[:4]:
            reasoning.append(
                f"观测到 {ev['observed_nm']} nm 吸收（深度 {ev['observed_depth']}），"
                f"对应 {top['name_cn']} 的 {ev['library_label']}（{ev['library_band_nm']} nm）。"
            )
        unexplained_deep = [
            f for f in features if f["depth"] >= 0.12 and not f["atmospheric"] and all(
                abs(f["center_nm"] - e["library_band_nm"]) > 30 for e in top["evidence"]
            )
        ]
        if unexplained_deep:
            reasoning.append(
                "存在参考库未解释的强吸收："
                + "、".join(f"{f['center_nm']} nm" for f in unexplained_deep[:3])
                + "，提示可能为混合端元或库外物质。"
            )

    caveats = [
        "单条光谱判读存在多解性，切勿作为唯一依据；建议结合野外产状、显微镜鉴定或 XRD 验证。",
        f"{ATMOSPHERIC_ZONES[0][0]}–{ATMOSPHERIC_ZONES[0][1]} nm 与 {ATMOSPHERIC_ZONES[1][0]}–{ATMOSPHERIC_ZONES[1][1]} nm 受大气水汽/探测器信噪比影响，区内特征已在评分中降权。",
        "参考库曲线为按诊断波段重建的理想化端元，实测样品的粒度、含水量与照度会改变连续统形态与表观深度。",
    ]
    followups = [
        "若为野外测量：检查白板标定与太阳高度角记录。",
        "尝试对 2100–2400 nm 诊断窗口做二次拟合，确认双峰结构。",
        "如怀疑混合像元，可用「混合像元」演示样例观察评分如何拆分到两个端元。",
    ]
    if any(in_atmospheric_zone(f["center_nm"]) for f in features):
        followups.insert(0, "部分特征落在大气干扰区内，建议改用实验室光谱复核这些波段。")

    return {
        "mode": "deterministic",
        "headline": head,
        "reasoning": reasoning,
        "caveats": caveats,
        "followups": followups,
        "sample_note": spec.metadata.get("note", ""),
    }


def analyze(spec: Spectrum) -> AnalysisResult:
    t0 = time.perf_counter()
    rs, cr, cr_local, continuum = preprocess(spec)
    t1 = time.perf_counter()
    features = extract_features(cr_local.wavelength, cr_local.reflectance)
    t2 = time.perf_counter()
    candidates = match(features, cr.wavelength, cr.reflectance, top_k=3)
    t3 = time.perf_counter()

    result = AnalysisResult(
        spectrum=spectrum_to_response(spec),
        processed=_downsample(rs.wavelength, rs.reflectance),
        continuum_removed=_downsample(cr.wavelength, cr.reflectance),
        continuum_removed_local=_downsample(cr_local.wavelength, cr_local.reflectance),
        continuum=_downsample(rs.wavelength, continuum),
        features=features,
        candidates=candidates,
        timings_ms={
            "preprocess": round((t1 - t0) * 1000, 1),
            "features": round((t2 - t1) * 1000, 1),
            "match": round((t3 - t2) * 1000, 1),
        },
    )
    result.report = deterministic_report(features, candidates, spec)
    result.timings_ms["total_deterministic"] = round((t3 - t0) * 1000, 1)
    return result


def library_summary() -> list[dict]:
    lib = load_library()
    out = []
    for e in lib["entries"]:
        out.append(
            {
                "id": e["id"],
                "name": e["name"],
                "name_cn": e["name_cn"],
                "category_cn": e["category_cn"],
                "formula": e["formula"],
                "n_features": len(e["features"]),
                "diagnostic_bands_nm": [f["c"] for f in e["features"] if f.get("diagnostic")],
            }
        )
    return out
