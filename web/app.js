/* SpectraScope frontend — vanilla JS, SVG chart, no dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = {
  result: null,
  history: [],
  busy: false,
  chatBusy: false,
};

/* ---------------- bootstrap ---------------- */

async function boot() {
  // bind interactions first so the page stays usable even if listing calls fail
  bindInput();
  bindChat();
  bindSettings();
  const results = await Promise.allSettled([refreshAgentStatus(), loadDemoList(), loadLibrary()]);
  for (const r of results) {
    if (r.status === "rejected") {
      console.warn("boot:", r.reason);
      toast("部分初始化失败，请刷新重试");
    }
  }
}

async function api(path, options) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      msg = j.detail || msg;
    } catch (_) { /* keep status */ }
    throw new Error(msg);
  }
  return resp.json();
}

async function refreshAgentStatus() {
  const chip = $("agent-chip");
  try {
    const h = await api("/api/health");
    // effective config = per-request UI override, else server env
    const o = llmOverride();
    const base = (o.base_url || h.llm_base_url || "").trim();
    const model = (o.model || h.llm_model || "").trim();
    if (base) {
      chip.textContent = `智能体 · ${model || "默认模型"} @ ${base}`;
      chip.className = "chip on";
    } else {
      chip.textContent = "确定性模式（未配置 LLM）";
      chip.className = "chip off";
    }
  } catch (_) {
    chip.textContent = "服务未连接";
    chip.className = "chip off";
  }
}

function llmOverride() {
  try {
    const parsed = JSON.parse(localStorage.getItem("spectrascope.llm") || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (_) {
    return {};
  }
}

async function loadDemoList() {
  const data = await api("/api/demo-samples");
  const wrap = $("demo-list");
  wrap.innerHTML = "";
  for (const [key, info] of Object.entries(data.samples)) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "demo-btn";
    b.innerHTML = `${info.name}<span class="note">${info.note || ""}</span>`;
    b.addEventListener("click", () => analyzeDemo(key));
    wrap.appendChild(b);
  }
}

async function loadLibrary() {
  const data = await api("/api/library");
  const wrap = $("library-list");
  wrap.innerHTML = "";
  for (const e of data.entries) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "lib-chip";
    chip.title = `${e.name} · ${e.formula}\n诊断波段：${e.diagnostic_bands_nm.join(", ")} nm`;
    chip.innerHTML = `<b>${e.name_cn}</b>`;
    chip.addEventListener("click", () => {
      toast(`${e.name_cn}（${e.name}）· 诊断波段 ${e.diagnostic_bands_nm.join(" / ")} nm`);
    });
    wrap.appendChild(chip);
  }
}

/* ---------------- input ---------------- */

function bindInput() {
  const dz = $("dropzone");
  const fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fi.click(); });
  fi.addEventListener("change", () => {
    if (fi.files && fi.files[0]) uploadFile(fi.files[0]);
    fi.value = "";
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
}

async function uploadFile(file) {
  if (state.busy) {
    toast("分析进行中，请稍候");
    return;
  }
  const fd = new FormData();
  fd.append("file", file, file.name);
  fd.append("llm", JSON.stringify(llmOverride()));
  await runAnalysis(() => api("/api/analyze", { method: "POST", body: fd }), file.name);
}

async function analyzeDemo(key) {
  if (state.busy) {
    toast("分析进行中，请稍候");
    return;
  }
  const fd = new FormData();
  fd.append("demo", key);
  fd.append("llm", JSON.stringify(llmOverride()));
  await runAnalysis(() => api("/api/analyze", { method: "POST", body: fd }), key);
}

async function runAnalysis(call, label) {
  state.busy = true;
  $("chart-loading").classList.remove("hidden");
  try {
    const result = await call();
    state.result = result;
    state.history = [];
    renderAll(result);
  } catch (err) {
    toast(`分析失败：${err.message}`);
  } finally {
    state.busy = false;
    $("chart-loading").classList.add("hidden");
  }
}

/* ---------------- rendering ---------------- */

function renderAll(result) {
  renderChart(result);
  renderCandidates(result.candidates);
  renderFeatureTable(result.features);
  renderReport(result.report);
  const note = result.spectrum.metadata && result.spectrum.metadata.note;
  const el = $("sample-note");
  if (note) {
    el.textContent = `样品备注：${note}`;
    el.classList.remove("hidden");
  } else {
    el.classList.add("hidden");
  }
  $("chat-log").innerHTML = "";
}

function renderCandidates(cands) {
  const wrap = $("candidates");
  wrap.innerHTML = "";
  cands.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "cand" + (i === 0 ? " first" : "");
    const evItems = (c.evidence || [])
      .map((e) => {
        const cls = e.matched ? "ev-hit" : e.partial ? "ev-partial" : "ev-miss";
        const mark = e.matched ? "✓" : e.partial ? "≈" : "✗";
        const obs = e.observed_nm != null ? `← 观测 ${e.observed_nm} nm（深度 ${e.observed_depth}）` : "← 未观测到";
        return `<li class="${cls}"><span>${mark}</span><span class="ev-band">${e.library_band_nm} nm</span>` +
          `<span>${e.library_label}${e.atm ? ' <span class="ev-atm">大气区</span>' : ""} ${obs}</span></li>`;
      })
      .join("");
    div.innerHTML = `
      <div class="cand-rank"><span>${i === 0 ? "最优候选 TOP-1" : `候选 #${i + 1}`}</span>` +
      `${c.confidence != null ? `<span class="conf">置信度 ${(c.confidence * 100).toFixed(0)}%</span>` : ""}</div>
      <div class="cand-name">${esc(c.name_cn)}</div>
      <div class="cand-meta">${esc(c.name)} · ${esc(c.category_cn)} · ${esc(c.formula)}</div>
      <div class="bar"><i style="width:${(c.score * 100).toFixed(1)}%"></i></div>
      <div class="cand-meta">综合 ${(c.score).toFixed(3)} · 覆盖 ${(c.coverage).toFixed(2)} · 形状 ${(c.shape_similarity).toFixed(2)} · 未解释 ${(c.unexplained).toFixed(2)}</div>
      ${c.note ? `<div class="cand-note">${esc(c.note)}</div>` : ""}
      ${evItems ? `<details><summary>证据映射（诊断波段）</summary><ul class="ev-list">${evItems}</ul></details>` : ""}`;
    wrap.appendChild(div);
  });
}

function renderFeatureTable(feats) {
  const wrap = $("feature-table");
  if (!feats.length) {
    wrap.innerHTML = '<div class="placeholder">未检出超过阈值的吸收特征。</div>';
    return;
  }
  const rows = feats
    .map(
      (f) => `<tr>
        <td>${f.center_nm}</td>
        <td>${f.depth.toFixed(3)}</td>
        <td>${f.fwhm_nm}</td>
        <td>${f.area}</td>
        <td>${f.atmospheric ? '<span class="ft-atm">大气区·降权</span>' : "—"}</td>
      </tr>`
    )
    .join("");
  wrap.innerHTML = `<table>
    <thead><tr><th>中心 nm</th><th>深度</th><th>FWHM nm</th><th>面积</th><th>备注</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function renderReport(rep) {
  const wrap = $("report");
  const modeChip = $("report-mode");
  if (rep.mode === "agent") {
    modeChip.textContent = `智能体判读 · ${rep.llm_model || ""}`;
    modeChip.className = "chip small on";
  } else {
    modeChip.textContent = "确定性报告";
    modeChip.className = "chip small off";
  }
  // report text can originate from an LLM (prompt-injectable via crafted
  // spectra/filenames) — every string goes through esc() before innerHTML
  const sec = (title, items) =>
    items && items.length
      ? `<div class="rp-sec"><h4>${esc(title)}</h4><ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`
      : "";
  let html = `<h3 class="rp-headline">${esc(rep.headline || "")}</h3>`;
  html += sec("推理链", rep.reasoning);
  if (rep.evidence_comments && rep.evidence_comments.length) {
    html += `<div class="rp-sec"><h4>波段点评</h4><ul>` +
      rep.evidence_comments.map((e) => `<li><b>${esc(e.band_nm)} nm</b> — ${esc(e.comment)}</li>`).join("") +
      `</ul></div>`;
  }
  html += sec("注意事项", rep.caveats);
  html += sec("后续建议", rep.followups);
  if (rep.llm_error) html += `<div class="rp-err">${esc(rep.llm_error)}</div>`;
  wrap.innerHTML = html;
}

/* ---------------- SVG chart ---------------- */

const M = { left: 56, right: 18, top: 26, mid: 300, gap: 44, bottom: 40 };
const W = 1000, H = 560;
const PLOT_W = W - M.left - M.right;
const WL_MIN = 350, WL_MAX = 2500;
const ATM_ZONES = [[1330, 1480], [1780, 1980]];

function xScale(wl) { return M.left + ((wl - WL_MIN) / (WL_MAX - WL_MIN)) * PLOT_W; }

function renderChart(result) {
  $("chart-empty").classList.add("hidden");
  const svg = $("chart");
  const pA_top = M.top, pA_bot = M.mid;               // panel A: reflectance
  const pB_top = M.mid + M.gap, pB_bot = H - M.bottom; // panel B: CR

  const wl = result.processed.wavelength.map(Number);
  const rf = result.processed.values.map(Number);
  const cont = result.continuum.values.map(Number);
  const crWl = result.continuum_removed_local.wavelength.map(Number);
  const cr = result.continuum_removed_local.values.map(Number);

  const rfMax = Math.max(0.2, ...rf, ...cont) * 1.08;
  const yA = (v) => pA_bot - (v / rfMax) * (pA_bot - pA_top);
  const yB = (v) => pB_bot - Math.min(v, 1.05) / 1.05 * (pB_bot - pB_top);

  const parts = [];
  // atmospheric zones (both panels)
  for (const [lo, hi] of ATM_ZONES) {
    const x0 = xScale(lo), x1 = xScale(hi);
    parts.push(`<rect x="${x0}" y="${pA_top}" width="${x1 - x0}" height="${pA_bot - pA_top}" fill="url(#atm)"/>`);
    parts.push(`<rect x="${x0}" y="${pB_top}" width="${x1 - x0}" height="${pB_bot - pB_top}" fill="url(#atm)"/>`);
  }
  // grid + axes
  for (let t = 350; t <= 2500; t += 350) {
    parts.push(`<line x1="${xScale(t)}" y1="${pA_top}" x2="${xScale(t)}" y2="${pB_bot}" stroke="rgba(43,36,23,.10)" stroke-width="1"/>`);
    parts.push(`<text class="axis-label" x="${xScale(t)}" y="${H - M.bottom + 18}" text-anchor="middle">${t}</text>`);
  }
  for (let f = 0; f <= 1.001; f += 0.25) {
    const y = yB(f);
    parts.push(`<line x1="${M.left}" y1="${y}" x2="${W - M.right}" y2="${y}" stroke="rgba(43,36,23,.08)"/>`);
    parts.push(`<text class="axis-label" x="${M.left - 8}" y="${y + 3}" text-anchor="end">${f.toFixed(2)}</text>`);
  }
  parts.push(`<text class="axis-label" x="${M.left - 8}" y="${(pA_top + pA_bot) / 2}" text-anchor="end" transform="rotate(-90 ${M.left - 8} ${(pA_top + pA_bot) / 2})">反射率</text>`);
  parts.push(`<text class="axis-label" x="${M.left - 8}" y="${(pB_top + pB_bot) / 2}" text-anchor="end" transform="rotate(-90 ${M.left - 8} ${(pB_top + pB_bot) / 2})">CR</text>`);

  const line = (xs, ys, yfn) => xs.map((v, i) => `${i ? "L" : "M"}${xScale(v).toFixed(1)},${yfn(ys[i]).toFixed(1)}`).join("");
  const area = (xs, ys, yfn) =>
    `${line(xs, ys, yfn)}L${xScale(xs[xs.length - 1]).toFixed(1)},${pB_bot} L${xScale(xs[0]).toFixed(1)},${pB_bot} Z`;

  // panel A: continuum then reflectance
  parts.push(`<path d="${line(wl, cont, yA)}" fill="none" stroke="#7a6548" stroke-width="1.4" stroke-dasharray="5 4" opacity=".9"/>`);
  parts.push(`<path d="${line(wl, rf, yA)}" fill="none" stroke="#2b2417" stroke-width="1.7"/>`);
  parts.push(`<text class="panel-label" x="${M.left}" y="${pA_top - 8}">Ⅰ · 反射率与连续统</text>`);

  // panel B: local CR area
  parts.push(`<path d="${area(crWl, cr, yB)}" fill="rgba(47,111,143,.14)" stroke="none"/>`);
  parts.push(`<path d="${line(crWl, cr, yB)}" fill="none" stroke="#2f6f8f" stroke-width="1.5"/>`);

  // features on both panels
  for (const f of result.features) {
    const x = xScale(f.center_nm);
    const inRange = f.center_nm >= Math.min(...crWl) && f.center_nm <= Math.max(...crWl);
    if (!inRange) continue;
    const y = yB(1 - f.depth);
    parts.push(`<line x1="${x}" y1="${pB_top}" x2="${x}" y2="${pB_bot}" stroke="rgba(194,68,42,.28)" stroke-width="1"/>`);
    parts.push(`<circle cx="${x}" cy="${y}" r="4" fill="#c2442a"/>`);
    const flip = f.depth > 0.5;
    const label = `${f.center_nm}`;
    parts.push(`<text class="feat-label" x="${x}" y="${flip ? y - 8 : y + 15}" text-anchor="middle">${label}</text>`);
    parts.push(`<line x1="${x}" y1="${pA_top}" x2="${x}" y2="${pA_bot}" stroke="rgba(194,68,42,.14)" stroke-width="1"/>`);
  }
  parts.push(`<text class="panel-label" x="${M.left}" y="${pB_top - 8}">Ⅱ · 连续统去除（局部包络）与检出特征</text>`);

  svg.innerHTML = `<defs>
    <pattern id="atm" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="8" height="8" fill="rgba(194,68,42,.045)"/>
      <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(194,68,42,.16)" stroke-width="2"/>
    </pattern>
  </defs>` + parts.join("");
}

/* ---------------- chat ---------------- */

function bindChat() {
  $("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("chat-input");
    const q = input.value.trim();
    if (!q) return;
    if (state.chatBusy) {
      toast("上一条还在回答中…");
      return;
    }
    state.chatBusy = true;
    $("chat-form").querySelector(".btn").disabled = true;
    input.value = "";
    appendMsg("user", q);
    const msgEl = appendMsg("bot", "……");
    try {
      const resp = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          result: state.result,
          history: state.history,
          llm: llmOverride(),
        }),
      });
      msgEl.querySelector("span.txt").textContent = resp.answer;
      state.history.push({ role: "user", content: q });
      state.history.push({ role: "assistant", content: resp.answer });
    } catch (err) {
      msgEl.querySelector("span.txt").textContent = `请求失败：${err.message}`;
    } finally {
      state.chatBusy = false;
      $("chat-form").querySelector(".btn").disabled = false;
    }
  });
}

function appendMsg(role, text) {
  const log = $("chat-log");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `${role === "bot" ? '<span class="who">SPECTRASCOPE</span>' : ""}<span class="txt"></span>`;
  div.querySelector("span.txt").textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

/* ---------------- settings ---------------- */

function bindSettings() {
  const modal = $("settings-modal");
  const open = () => {
    const cfg = llmOverride();
    $("set-base").value = cfg.base_url || "";
    $("set-key").value = cfg.api_key || "";
    $("set-model").value = cfg.model || "";
    modal.classList.remove("hidden");
  };
  $("btn-settings").addEventListener("click", open);
  $("btn-close-settings").addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); });
  $("btn-save-settings").addEventListener("click", () => {
    const cfg = {
      base_url: $("set-base").value.trim(),
      api_key: $("set-key").value.trim(),
      model: $("set-model").value.trim(),
    };
    localStorage.setItem("spectrascope.llm", JSON.stringify(cfg));
    modal.classList.add("hidden");
    refreshAgentStatus();
    toast(cfg.base_url ? "已保存；下次分析将调用智能体。" : "已保存（空地址 = 确定性模式）。");
  });
  $("btn-clear-settings").addEventListener("click", () => {
    localStorage.removeItem("spectrascope.llm");
    $("set-base").value = ""; $("set-key").value = ""; $("set-model").value = "";
    refreshAgentStatus();
    toast("已清除，回到确定性模式。");
  });
}

/* ---------------- toast ---------------- */

let toastTimer = null;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 3400);
}

boot();
