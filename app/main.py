"""SpectraScope — FastAPI application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core import agent as agent_mod
from .core import pipeline
from .core.agent import LLMConfig
from .core.spectra import demo_samples, spectrum_from_csv_bytes

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

app = FastAPI(title="SpectraScope", version="1.0.0")


class ChatBody(BaseModel):
    question: str
    result: dict | None = None
    history: list[dict] = Field(default_factory=list)
    llm: dict = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict:
    cfg = LLMConfig.from_env()
    return {
        "status": "ok",
        "llm_configured": cfg.configured,
        "llm_base_url": cfg.base_url or None,
        "llm_model": cfg.model or None,
    }


@app.get("/api/demo-samples")
def list_demo_samples() -> dict:
    return {"samples": {k: {"name": v.name, "note": v.metadata.get("note", "")} for k, v in demo_samples().items()}}


@app.get("/api/library")
def get_library() -> dict:
    return {"entries": pipeline.library_summary()}


@app.get("/api/scene")
def get_scene() -> dict:
    """Sentinel-2 风格模拟场景的 FCLS 解混产品（类图/丰度/RMSE/指数，含地面真值）。"""
    from .core import imaging

    return imaging.scene_products()


@app.get("/api/scene/pixel")
def get_scene_pixel(x: int, y: int) -> dict:
    """单像元 FCLS 分解 + 指数 + 地面真值（多光谱像元不做吸收特征诊断）。"""
    from .core import imaging

    return imaging.pixel_spectrum(x, y)


@app.get("/api/scene/hotspots")
def get_scene_hotspots() -> dict:
    """RMSE 残差热点：工作端元集无法解释的区域。"""
    from .core import imaging

    return imaging.hotspots()


class DiscoverBody(BaseModel):
    llm: dict = Field(default_factory=dict)
    max_tests: int = 5


@app.post("/api/scene/discover")
async def discover_endpoint(body: DiscoverBody) -> dict:
    """假设驱动的发现循环：LLM 排序候选（无 LLM 则库扫描）→ 物理管线逐一裁决。"""
    from .core import imaging

    hs = imaging.hotspots()
    if not hs["regions"]:
        return {"discovered": None, "tested": []}
    candidates = imaging.candidate_ids()
    cfg = LLMConfig.merge(body.llm)
    plan = await agent_mod.discover(hs["regions"][0], imaging.SCENE_ENTRIES, candidates, cfg)
    # LLM mode: test its top-K ranked hypotheses; sweep mode: exhaustive until found
    limit = min(body.max_tests, len(plan["ranked"])) if plan["llm_used"] else len(plan["ranked"])
    tested = []
    for cid in plan["ranked"][:limit]:
        r = imaging.test_hypothesis(cid)
        tested.append(r)
        if r["verdict"] == "accepted":
            break
    discovered = next((r for r in tested if r["verdict"] == "accepted"), None)
    return {
        "discovered": discovered,
        "tested": tested,
        "llm_used": plan["llm_used"],
        "llm_reasoning": plan.get("reasoning", []),
        "hidden_truth": {"id": imaging.ANOMALY_ENTRY, "name_cn": "针铁矿", "note": "模拟场景预埋的异常体（发现演示用）"},
    }


@app.get("/api/scene/route")
def get_scene_route(stops: int = 6, min_sep: int = 9) -> dict:
    """丰度熵驱动的野外采样路线规划。"""
    from .core import imaging

    return imaging.plan_route(max(2, min(stops, 12)), max(4, min(min_sep, 24)))


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(default=None),
    demo: str | None = Form(default=None),
    llm: str | None = Form(default=None),
) -> dict:
    """Analyze an uploaded spectrum file (`file=`) or a built-in demo sample (`demo=`)."""
    import json as _json

    override = {}
    if llm:
        try:
            override = _json.loads(llm)
        except _json.JSONDecodeError:
            raise HTTPException(400, "llm 字段不是合法 JSON")
    try:
        if file is not None:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "文件超过 32MB 上限")
            spec = spectrum_from_csv_bytes(data, file.filename or "upload.csv")
        elif demo:
            samples = demo_samples()
            if demo not in samples:
                raise HTTPException(404, f"未知演示样例: {demo}")
            spec = samples[demo]
        else:
            raise HTTPException(400, "需要 file 或 demo 参数")
        result = pipeline.analyze(spec)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    result.report = await agent_mod.interpret(result.to_dict(), LLMConfig.merge(override))
    return result.to_dict()


@app.post("/api/chat")
async def chat(body: ChatBody) -> dict:
    cfg = LLMConfig.merge(body.llm)
    return await agent_mod.chat(body.question, body.result, body.history, cfg)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
