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
