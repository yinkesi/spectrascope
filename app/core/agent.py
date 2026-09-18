"""LLM interpretation layer.

The agent never invents spectral numbers: it receives the measured feature
table and the ranked candidate list produced by the deterministic pipeline and
writes a differential-diagnosis narrative constrained to those numbers.

Endpoint is OpenAI-compatible (llama.cpp / vLLM / Zhipu / OpenAI all work):
  - request override  (UI settings)
  - SPECTRASCOPE_LLM_BASE_URL / SPECTRASCOPE_LLM_API_KEY / SPECTRASCOPE_LLM_MODEL
Falls back to the deterministic template report whenever no endpoint is
configured or the call fails — the demo must never hard-fail.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

SYSTEM_PROMPT = """你是 SpectraScope 的野外光谱判读智能体，服务于测绘遥感与地质工程场景。
你收到的是确定性物理管线（连续统去除、吸收特征提取、光谱角/覆盖率匹配）的测量结果，包括：
1) 吸收特征表（中心波长 nm、深度、FWHM、是否位于大气干扰区）
2) 参考库匹配的前 3 名候选（得分、诊断波段覆盖、形状相似度、证据映射）

你的任务：像一位谨慎的遥感地质学家那样做鉴别诊断。规则：
- 只允许引用输入表中出现过的波长和数值，严禁编造波段或分数；
- 逐条解释证据（哪个观测波段对应候选的哪个诊断吸收），说明支持与反对；
- 明确指出大气干扰区（1330-1480 / 1780-1980 nm）内的特征可信度低；
- 若第一名与第二名分差 < 0.03，必须给出野外甄别方案；
- 给出后续验证建议（如 XRD、显微镜、二次测量、混合像元分解）。
输出严格的 JSON：{"headline": str, "reasoning": [str], "evidence_comments": [{"band_nm": number, "comment": str}],
"caveats": [str], "followups": [str]}。字段全用中文（波长/数字保留阿拉伯数字）。"""


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.environ.get("SPECTRASCOPE_LLM_BASE_URL", "").rstrip("/"),
            api_key=os.environ.get("SPECTRASCOPE_LLM_API_KEY", ""),
            model=os.environ.get("SPECTRASCOPE_LLM_MODEL", ""),
        )

    @classmethod
    def merge(cls, override: dict | None) -> "LLMConfig":
        cfg = cls.from_env()
        if isinstance(override, dict):  # tolerate any JSON the client sends
            if override.get("base_url"):
                cfg.base_url = str(override["base_url"]).strip().rstrip("/")
            if override.get("api_key"):
                cfg.api_key = str(override["api_key"])
            if override.get("model"):
                cfg.model = str(override["model"]).strip()
        return cfg


def _features_block(features: list[dict]) -> str:
    lines = ["center_nm | depth | fwhm_nm | area | atmospheric"]
    for f in features:
        lines.append(
            f"{f['center_nm']} | {f['depth']} | {f['fwhm_nm']} | {f['area']} | {'是' if f['atmospheric'] else '否'}"
        )
    return "\n".join(lines)


def _candidates_block(candidates: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(candidates, 1):
        ev_lines = [
            f"  - 库波段 {e['library_band_nm']} nm（{e['library_label']}，深度 {e['library_depth']}）: "
            + (
                f"匹配到观测 {e['observed_nm']} nm（深度 {e['observed_depth']}）"
                if e["matched"]
                else ("部分匹配" if e["partial"] else "未观测到")
            )
            + ("（大气干扰区）" if e["atmospheric"] else "")
            for e in c["evidence"]
        ]
        blocks.append(
            f"候选{i}: {c['name_cn']}（{c['name']}, {c['category_cn']}, {c['formula']}）\n"
            f"  score={c['score']} coverage={c['coverage']} shape={c['shape_similarity']} unexplained={c['unexplained']}\n"
            + "\n".join(ev_lines)
        )
    return "\n".join(blocks)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


async def interpret(result_dict: dict, cfg: LLMConfig, timeout: float = 45.0) -> dict:
    """Return an agent report; on any failure return the deterministic report untouched."""
    report = result_dict["report"]
    if not cfg.configured:
        report["mode"] = "deterministic"
        report["llm_error"] = None
        return report

    user_prompt = (
        f"样品名：{result_dict['spectrum']['name']}\n\n"
        f"【吸收特征表】\n{_features_block(result_dict['features'])}\n\n"
        f"【候选匹配】\n{_candidates_block(result_dict['candidates'])}\n\n"
        "请按系统规则输出 JSON 判读。"
    )
    payload = {
        "model": cfg.model or "default",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1400,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    try:
        # trust_env=False: never route LLM calls through system/env proxies —
        # on Windows a global proxy would otherwise hijack localhost llama.cpp
        # endpoints and see the Authorization header.
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(f"{cfg.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not parsed or "headline" not in parsed:
            raise ValueError("LLM 输出无法解析为规定 JSON")
        parsed["mode"] = "agent"
        parsed["llm_model"] = cfg.model or "(server default)"
        parsed["deterministic_fallback"] = report
        return parsed
    except Exception as exc:  # noqa: BLE001 — demo must never hard-fail
        report["mode"] = "deterministic"
        report["llm_error"] = f"智能体调用失败，已回退到确定性报告：{exc}"
        return report


def _context_block(result_dict: dict | None) -> str:
    """Build the analysis context defensively — the client may send any JSON."""
    if not isinstance(result_dict, dict):
        return ""
    spectrum = result_dict.get("spectrum") if isinstance(result_dict.get("spectrum"), dict) else {}
    features = result_dict.get("features") if isinstance(result_dict.get("features"), list) else []
    candidates = result_dict.get("candidates") if isinstance(result_dict.get("candidates"), list) else []
    if not features and not candidates:
        return ""
    return (
        f"【当前分析上下文】\n样品：{spectrum.get('name', '未知')}\n"
        f"特征表：{_features_block(features)}\n"
        f"候选：{_candidates_block(candidates)[:1800]}\n\n"
    )


async def chat(question: str, result_dict: dict | None, history: list[dict], cfg: LLMConfig, timeout: float = 45.0) -> dict:
    if not cfg.configured:
        return {
            "answer": "未配置 LLM 端点。可在右上角「智能体设置」填写 OpenAI 兼容地址（本地 llama.cpp / Zhipu / OpenAI 均可）。"
                      "当前仍可使用全部确定性分析功能。",
            "mode": "deterministic",
        }
    context = ""
    context = _context_block(result_dict)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n现在是追问环节：直接用中文回答用户关于本次判读的问题，保持数值忠实。"}]
    # only user/assistant turns survive; anything else the client sends is dropped
    for h in history[-6:] if isinstance(history, list) else []:
        if not isinstance(h, dict) or h.get("role") not in ("user", "assistant"):
            continue
        content = str(h.get("content", ""))[:1500]
        if content:
            messages.append({"role": h["role"], "content": content})
    messages.append({"role": "user", "content": context + question})
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(
                f"{cfg.base_url}/chat/completions",
                json={"model": cfg.model or "default", "messages": messages, "temperature": 0.4, "max_tokens": 900},
                headers=headers,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        return {"answer": answer, "mode": "agent", "llm_model": cfg.model or "(server default)"}
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"智能体调用失败：{exc}", "mode": "error"}
