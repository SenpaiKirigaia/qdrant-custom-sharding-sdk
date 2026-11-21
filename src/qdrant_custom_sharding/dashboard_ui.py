from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Qdrant Custom Sharding Dashboard</title>
<style>
  :root {
    --bg: #0c111d;
    --bg-elev: #131b2e;
    --bg-card: #172136;
    --bg-input: #0e1526;
    --border: #253352;
    --border-soft: #1d2942;
    --text: #e6edf7;
    --muted: #8b9bb8;
    --accent: #dc244c;
    --accent-hover: #f03860;
    --blue: #3b82f6;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #f43f5e;
    --purple: #8b5cf6;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.45; }
  a { color: var(--blue); text-decoration: none; }
  button, input, select { font-family: inherit; font-size: 13px; color: var(--text); }
  .app { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }
  @media (max-width: 900px) { .app { grid-template-columns: 1fr; } }

  .sidebar { background: var(--bg-elev); border-right: 1px solid var(--border); padding: 16px; display: flex; flex-direction: column; gap: 16px; }
  .brand { display: flex; align-items: center; gap: 10px; padding-bottom: 12px; border-bottom: 1px solid var(--border-soft); }
  .brand-mark { width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, var(--accent), #9b1433); display: grid; place-items: center; font-weight: 700; font-size: 16px; color: #fff; }
  .brand-title { font-weight: 700; font-size: 14px; }
  .brand-sub { font-size: 11px; color: var(--muted); }
  .section-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
  .coll-list { display: flex; flex-direction: column; gap: 4px; max-height: 45vh; overflow-y: auto; }
  .coll-item { padding: 8px 10px; border-radius: 8px; border: 1px solid transparent; cursor: pointer; display: flex; justify-content: space-between; align-items: center; background: transparent; width: 100%; text-align: left; }
  .coll-item:hover { background: var(--bg-card); }
  .coll-item.active { background: rgba(220, 36, 76, 0.14); border-color: rgba(220, 36, 76, 0.45); }
  .coll-name { font-weight: 600; font-family: var(--mono); font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .coll-meta { font-size: 11px; color: var(--muted); }
  .pill { display: inline-flex; align-items: center; gap: 4px; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .pill-green { background: rgba(16, 185, 129, 0.15); color: #34d399; }
  .pill-amber { background: rgba(245, 158, 11, 0.18); color: #fbbf24; }
  .pill-red { background: rgba(244, 63, 94, 0.18); color: #fb7185; }
  .pill-blue { background: rgba(59, 130, 246, 0.18); color: #60a5fa; }
  .pill-muted { background: rgba(139, 155, 184, 0.15); color: var(--muted); }

  .main { display: flex; flex-direction: column; min-width: 0; }
  .topbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 1px solid var(--border); background: var(--bg-elev); }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .tab { background: transparent; border: 1px solid transparent; padding: 6px 12px; border-radius: 7px; cursor: pointer; color: var(--muted); font-weight: 500; }
  .tab:hover { color: var(--text); background: var(--bg-card); }
  .tab.active { color: #fff; background: var(--accent); border-color: var(--accent); }
  .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

  .btn { padding: 6px 12px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg-card); cursor: pointer; font-weight: 500; }
  .btn:hover { border-color: #3d527f; background: #1e2b45; }
  .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
  .btn-blue { background: var(--blue); border-color: var(--blue); color: #fff; }
  .btn-sm { padding: 4px 8px; font-size: 12px; }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .input, select.input { background: var(--bg-input); border: 1px solid var(--border); border-radius: 7px; padding: 6px 10px; outline: none; }
  .input:focus { border-color: var(--blue); }

  .content { padding: 20px; display: flex; flex-direction: column; gap: 18px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .kpi { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .kpi-value { font-size: 22px; font-weight: 700; margin-top: 4px; font-family: var(--mono); }
  .kpi-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .card-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
  .card-title { font-weight: 600; font-size: 14px; }
  .card-sub { font-size: 12px; color: var(--muted); }

  .chart-wrap { width: 100%; overflow-x: auto; }
  svg.chart { width: 100%; display: block; }
  .heatmap { display: grid; grid-template-columns: repeat(auto-fill, minmax(76px, 1fr)); gap: 6px; }
  .heat-cell { border-radius: 7px; padding: 8px 6px; border: 1px solid var(--border); cursor: pointer; text-align: center; transition: transform 0.1s; }
  .heat-cell:hover { transform: translateY(-1px); border-color: var(--text); }
  .heat-slot { font-family: var(--mono); font-size: 12px; font-weight: 700; }
  .heat-pts { font-family: var(--mono); font-size: 11px; color: #dbeafe; margin-top: 2px; }
  .heat-meta { font-size: 10px; color: rgba(255,255,255,0.75); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-soft); vertical-align: middle; }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
  tr:hover td { background: rgba(255, 255, 255, 0.02); }
  .mono { font-family: var(--mono); }
  .actions-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .form-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 11px; color: var(--muted); }

  .toast { position: fixed; right: 20px; bottom: 20px; background: var(--bg-card); border: 1px solid var(--border); padding: 10px 14px; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.45); max-width: 420px; z-index: 90; display: none; }
  .toast.show { display: block; }
  .toast.error { border-color: var(--red); }
  .toast.ok { border-color: var(--green); }

  .modal-bg { position: fixed; inset: 0; background: rgba(4, 8, 16, 0.72); display: none; align-items: center; justify-content: center; z-index: 80; padding: 16px; }
  .modal-bg.open { display: flex; }
  .modal { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 12px; padding: 20px; width: 100%; max-width: 460px; display: flex; flex-direction: column; gap: 14px; }
  .modal h3 { margin: 0; font-size: 16px; }
  pre.code { background: var(--bg-input); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: var(--mono); font-size: 12px; overflow: auto; max-height: 360px; margin: 0; white-space: pre; }
  .banner { padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border); background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.4); display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
  .empty { text-align: center; padding: 48px 20px; color: var(--muted); }
  .hidden { display: none !important; }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">Q</div>
      <div>
        <div class="brand-title">Qdrant Custom Sharding</div>
        <div class="brand-sub" id="env-status">Connecting...</div>
      </div>
    </div>

    <div>
      <div class="section-label">
        <span>Collections</span>
        <button class="btn btn-sm btn-primary" id="btn-open-create">+ New</button>
      </div>
      <div class="coll-list" id="collection-list"></div>
    </div>

    <div style="margin-top:auto; border-top:1px solid var(--border-soft); padding-top:12px; display:flex; flex-direction:column; gap:8px;">
      <div class="section-label"><span>Endpoints</span></div>
      <div style="display:flex; gap:8px; flex-wrap:wrap; font-size:12px;">
        <a href="/metrics" target="_blank">/metrics</a>
        <a href="/api/overview" target="_blank">/api/overview</a>
        <a href="/healthz" target="_blank">/healthz</a>
      </div>
      <button class="btn btn-sm" id="btn-auth-token">Set API Token</button>
    </div>
  </aside>

  <main class="main">
    <header class="topbar">
      <div class="tabs" id="nav-tabs">
        <button class="tab active" data-tab="overview">Overview & Graphs</button>
        <button class="tab" data-tab="features">Features & Slots</button>
        <button class="tab" data-tab="rebalance">Rebalance & Moves</button>
        <button class="tab" data-tab="inspector">Route Inspector</button>
        <button class="tab" data-tab="metrics">Metrics & Jobs</button>
      </div>
      <div class="toolbar">
        <label style="display:flex; align-items:center; gap:5px; font-size:12px; color:var(--muted);">
          <input type="checkbox" id="toggle-live" /> Live Qdrant counts
        </label>
        <label style="display:flex; align-items:center; gap:5px; font-size:12px; color:var(--muted);">
          <input type="checkbox" id="toggle-auto" /> Auto refresh (5s)
        </label>
        <button class="btn btn-sm" id="btn-refresh-stats">Refresh Stats</button>
        <button class="btn btn-sm" id="btn-reload">Reload</button>
      </div>
    </header>

    <section class="content" id="content-area">
      <div class="empty">Loading collections...</div>
    </section>
  </main>
</div>

<div class="modal-bg" id="modal-bg">
  <div class="modal" id="modal-box"></div>
</div>
<div class="toast" id="toast"></div>

<script>
const state = {
  collections: [],
  selected: null,
  detail: null,
  plan: null,
  overview: null,
  jobs: [],
  promText: "",
  tab: "overview",
  live: false,
  readOnly: false,
  hasVectors: true,
  slotFilter: "",
  featureQuery: "",
  stateFilter: "all",
  autoTimer: null,
};

function authHeaders() {
  const token = sessionStorage.getItem("qcs_api_token");
  return token ? { "Authorization": "Bearer " + token } : {};
}

async function api(path, opts = {}) {
  const headers = Object.assign({ "Accept": "application/json" }, authHeaders(), opts.headers || {});
  if (opts.body && typeof opts.body === "object") {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, Object.assign({}, opts, { headers }));
  const ctype = resp.headers.get("content-type") || "";
  const data = ctype.includes("application/json") ? await resp.json() : await resp.text();
  if (!resp.ok) {
    const msg = (data && data.error) ? data.error : ("HTTP " + resp.status);
    throw new Error(msg);
  }
  return data;
}

function toast(msg, kind = "ok") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show " + kind;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = "toast"; }, 4200);
}

function esc(v) {
  return String(v == null ? "" : v).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function fmtNum(n) {
  return Number(n || 0).toLocaleString();
}

function openModal(html) {
  document.getElementById("modal-box").innerHTML = html;
  document.getElementById("modal-bg").classList.add("open");
}

function closeModal() {
  document.getElementById("modal-bg").classList.remove("open");
}
document.getElementById("modal-bg").addEventListener("click", e => {
  if (e.target.id === "modal-bg") closeModal();
});

async function loadAll() {
  try {
    const health = await api("/healthz");
    state.readOnly = !!health.read_only;
    state.hasVectors = !!health.has_vector_store;
    document.getElementById("env-status").textContent =
      (state.readOnly ? "Read-only" : "Manage") + (state.hasVectors ? " | Live Qdrant" : " | Map-only");
    document.getElementById("btn-open-create").disabled = state.readOnly || !state.hasVectors;
    document.getElementById("btn-refresh-stats").disabled = state.readOnly || !state.hasVectors;

    const q = state.live ? "?live=1" : "";
    state.overview = await api("/api/overview" + q);
    state.collections = state.overview.collections || [];
    if (!state.selected && state.collections.length) {
      state.selected = state.collections[0].collection;
    } else if (state.selected && !state.collections.some(c => c.collection === state.selected)) {
      state.selected = state.collections.length ? state.collections[0].collection : null;
    }
    if (state.selected) {
      state.detail = await api("/api/collections/" + encodeURIComponent(state.selected) + q);
    } else {
      state.detail = null;
    }
    const jobsResp = await api("/api/jobs");
    state.jobs = jobsResp.jobs || [];
    if (state.tab === "metrics") {
      state.promText = await api("/metrics" + q, { headers: { "Accept": "text/plain" } });
    }
    renderSidebar();
    renderMain();
  } catch (err) {
    toast(err.message, "error");
  }
}

function renderSidebar() {
  const list = document.getElementById("collection-list");
  if (!state.collections.length) {
    list.innerHTML = '<div class="coll-meta" style="padding:8px;">No sharded collections yet.</div>';
    return;
  }
  list.innerHTML = state.collections.map(c => {
    const active = c.collection === state.selected ? "active" : "";
    const badge = c.active_moves > 0
      ? `<span class="pill pill-amber">${c.active_moves} moving</span>`
      : (c.needs_rebalance
        ? `<span class="pill pill-red">${c.imbalance_ratio.toFixed(2)}x</span>`
        : `<span class="pill pill-green">${c.imbalance_ratio.toFixed(2)}x</span>`);
    return `<button class="coll-item ${active}" onclick="selectCollection('${esc(c.collection)}')">
      <div style="min-width:0;">
        <div class="coll-name">${esc(c.collection)}</div>
        <div class="coll-meta">${fmtNum(c.total_points)} pts | ${c.slot_count} slots | ${c.feature_count} features</div>
      </div>
      ${badge}
    </button>`;
  }).join("");
}

window.selectCollection = async function(name) {
  state.selected = name;
  state.plan = null;
  state.slotFilter = "";
  await loadAll();
};

function renderMain() {
  const area = document.getElementById("content-area");
  if (state.tab === "metrics") {
    area.innerHTML = renderMetricsTab();
    return;
  }
  if (!state.detail) {
    area.innerHTML = `<div class="empty">
      <div style="font-size:16px; color:var(--text); margin-bottom:8px;">No sharded collection selected</div>
      <div>Create a sharded collection or point the dashboard at a map folder containing collection JSON maps.</div>
    </div>`;
    return;
  }
  if (state.tab === "overview") area.innerHTML = renderOverviewTab(state.detail.stats);
  else if (state.tab === "features") area.innerHTML = renderFeaturesTab(state.detail.stats);
  else if (state.tab === "rebalance") area.innerHTML = renderRebalanceTab(state.detail.stats);
  else if (state.tab === "inspector") area.innerHTML = renderInspectorTab(state.detail.stats);
}

function renderOverviewTab(s) {
  const imbClass = s.needs_rebalance ? "color:var(--red)" : "color:var(--green)";
  const moveBanner = s.active_moves.length ? `<div class="banner">
    <div><strong>${s.active_moves.length} active move(s)</strong> in progress on <span class="mono">${esc(s.collection)}</span>.</div>
    <button class="btn btn-sm" onclick="switchTab('rebalance')">Manage Moves</button>
  </div>` : "";
  return `
    ${moveBanner}
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Total Points</div><div class="kpi-value">${fmtNum(s.total_points)}</div><div class="kpi-sub">source: ${esc(s.source)} | v${s.version}</div></div>
      <div class="kpi"><div class="kpi-label">Slots</div><div class="kpi-value">${s.slot_count}</div><div class="kpi-sub">mean ${fmtNum(Math.round(s.mean_slot_load))} / slot</div></div>
      <div class="kpi"><div class="kpi-label">Features</div><div class="kpi-value">${fmtNum(s.feature_count)}</div><div class="kpi-sub">field: ${esc(s.feature_field)} (${esc(s.feature_type)})</div></div>
      <div class="kpi"><div class="kpi-label">Imbalance Ratio</div><div class="kpi-value" style="${imbClass}">${s.imbalance_ratio.toFixed(2)}x</div><div class="kpi-sub">threshold ${s.balance_threshold}x | p95 ${fmtNum(Math.round(s.p95_slot_load))}</div></div>
      <div class="kpi"><div class="kpi-label">Gini Skew</div><div class="kpi-value">${s.gini_coefficient.toFixed(3)}</div><div class="kpi-sub">0 = even, 1 = skewed</div></div>
      <div class="kpi"><div class="kpi-label">Active Moves</div><div class="kpi-value">${s.active_moves.length}</div><div class="kpi-sub">${s.split_recommendations.length} split rec(s)</div></div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Slot Load Distribution</div>
          <div class="card-sub">Points per slot with mean (blue) and ${s.balance_threshold}x rebalance threshold (red). Click a slot bar to inspect its features.</div>
        </div>
        <div class="actions-row">
          <span class="pill pill-green">balanced</span>
          <span class="pill pill-red">hot</span>
          <span class="pill pill-amber">busy</span>
        </div>
      </div>
      <div class="chart-wrap">${svgSlotBarChart(s)}</div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">Top Features & Cumulative Share (Pareto)</div>
            <div class="card-sub">Largest features by point count and cumulative % of collection.</div>
          </div>
        </div>
        <div class="chart-wrap">${svgParetoChart(s)}</div>
      </div>
      <div class="card">
        <div class="card-head">
          <div>
            <div class="card-title">Feature Size Histogram</div>
            <div class="card-sub">Number of features per size bucket.</div>
          </div>
        </div>
        <div class="chart-wrap">${svgHistogramChart(s)}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Slot Occupancy Map</div>
          <div class="card-sub">Intensity reflects share of max slot load. Click any slot to filter features.</div>
        </div>
      </div>
      <div class="heatmap">${renderSlotHeatmap(s)}</div>
    </div>
  `;
}

function svgSlotBarChart(s) {
  const slots = s.slots || [];
  if (!slots.length) return '<div class="empty">No slots.</div>';
  const w = Math.max(640, slots.length * 24 + 80);
  const h = 230;
  const padL = 56, padR = 20, padT = 18, padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const limit = s.mean_slot_load * s.balance_threshold;
  const maxY = Math.max(1, s.max_slot_load, limit * 1.05);
  const barW = Math.max(6, (plotW / slots.length) - 4);
  const y = v => padT + plotH - (v / maxY) * plotH;

  let out = `<svg class="chart" viewBox="0 0 ${w} ${h}" style="min-width:${Math.min(w, 960)}px">`;
  for (let i = 0; i <= 4; i++) {
    const val = (maxY / 4) * i;
    const py = y(val);
    out += `<line x1="${padL}" y1="${py}" x2="${w - padR}" y2="${py}" stroke="#22304d" stroke-dasharray="2,3"/>`;
    out += `<text x="${padL - 8}" y="${py + 4}" fill="#8b9bb8" font-size="10" text-anchor="end">${fmtNum(Math.round(val))}</text>`;
  }
  slots.forEach((slot, idx) => {
    const x = padL + idx * (plotW / slots.length) + 2;
    const bh = (slot.points / maxY) * plotH;
    const py = padT + plotH - bh;
    const color = slot.busy ? "#f59e0b" : (slot.status === "hot" ? "#f43f5e" : (slot.points === 0 ? "#334155" : "#3b82f6"));
    out += `<g style="cursor:pointer" onclick="filterBySlot('${esc(slot.slot)}')">
      <title>${esc(slot.slot)}: ${fmtNum(slot.points)} pts (${slot.share_pct}%), ${slot.feature_count} features [${slot.status}]</title>
      <rect x="${x}" y="${py}" width="${barW}" height="${Math.max(2, bh)}" rx="3" fill="${color}" />
      ${slots.length <= 28 || idx % 2 === 0 ? `<text x="${x + barW / 2}" y="${h - 12}" fill="#8b9bb8" font-size="10" text-anchor="middle">${esc(slot.slot)}</text>` : ""}
    </g>`;
  });
  if (s.mean_slot_load > 0) {
    const my = y(s.mean_slot_load);
    const ly = y(limit);
    out += `<line x1="${padL}" y1="${my}" x2="${w - padR}" y2="${my}" stroke="#60a5fa" stroke-width="1.5" stroke-dasharray="5,4"/>`;
    out += `<text x="${w - padR}" y="${my - 4}" fill="#60a5fa" font-size="10" text-anchor="end">mean ${fmtNum(Math.round(s.mean_slot_load))}</text>`;
    out += `<line x1="${padL}" y1="${ly}" x2="${w - padR}" y2="${ly}" stroke="#f43f5e" stroke-width="1.5" stroke-dasharray="4,4"/>`;
    out += `<text x="${w - padR}" y="${ly - 4}" fill="#fb7185" font-size="10" text-anchor="end">${s.balance_threshold}x (${fmtNum(Math.round(limit))})</text>`;
  }
  out += "</svg>";
  return out;
}

function svgParetoChart(s) {
  const feats = (s.features || []).slice(0, 14);
  if (!feats.length) return '<div class="empty">No features tracked in map yet.</div>';
  const w = 520, h = 230, padL = 56, padR = 44, padT = 18, padB = 46;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const maxPts = Math.max(1, feats[0].points);
  const step = plotW / feats.length;
  const barW = Math.max(10, step - 6);
  let cum = 0;
  const total = Math.max(1, s.total_points);
  const poly = [];
  let out = `<svg class="chart" viewBox="0 0 ${w} ${h}">`;
  feats.forEach((f, i) => {
    const x = padL + i * step + 3;
    const bh = (f.points / maxPts) * plotH;
    const py = padT + plotH - bh;
    cum += f.points;
    const pct = Math.min(100, (cum / total) * 100);
    const cy = padT + plotH - (pct / 100) * plotH;
    poly.push(`${x + barW / 2},${cy}`);
    const label = f.name.length > 8 ? f.name.slice(0, 7) + "…" : f.name;
    out += `<g>
      <title>${esc(f.name)}: ${fmtNum(f.points)} pts (${f.share_pct}%), width ${f.width}, cumulative ${pct.toFixed(1)}%</title>
      <rect x="${x}" y="${py}" width="${barW}" height="${Math.max(2, bh)}" rx="3" fill="${f.width > 1 ? "#8b5cf6" : "#dc244c"}" />
      <text x="${x + barW / 2}" y="${h - 24}" fill="#8b9bb8" font-size="9.5" text-anchor="middle">${esc(label)}</text>
      <text x="${x + barW / 2}" y="${h - 11}" fill="#64748b" font-size="9" text-anchor="middle">w=${f.width}</text>
    </g>`;
  });
  if (poly.length > 1) {
    out += `<polyline fill="none" stroke="#34d399" stroke-width="2" points="${poly.join(" ")}" />`;
  }
  out += `<text x="${w - 4}" y="${padT + 8}" fill="#34d399" font-size="10" text-anchor="end">100%</text>`;
  out += `<text x="${w - 4}" y="${padT + plotH}" fill="#34d399" font-size="10" text-anchor="end">0%</text>`;
  out += `<text x="${padL - 8}" y="${padT + 8}" fill="#8b9bb8" font-size="10" text-anchor="end">${fmtNum(maxPts)}</text>`;
  out += "</svg>";
  return out;
}

function svgHistogramChart(s) {
  const buckets = s.size_histogram || [];
  if (!buckets.length) return '<div class="empty">No histogram data.</div>';
  const w = 520, h = 230, padL = 44, padR = 16, padT = 22, padB = 34;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const maxC = Math.max(1, ...buckets.map(b => b.feature_count));
  const step = plotW / buckets.length;
  const barW = Math.max(18, step - 12);
  let out = `<svg class="chart" viewBox="0 0 ${w} ${h}">`;
  buckets.forEach((b, i) => {
    const x = padL + i * step + 6;
    const bh = (b.feature_count / maxC) * plotH;
    const py = padT + plotH - bh;
    out += `<g>
      <title>Bucket ${esc(b.label)} pts: ${fmtNum(b.feature_count)} features (${fmtNum(b.total_points)} pts total)</title>
      <rect x="${x}" y="${py}" width="${barW}" height="${Math.max(2, bh)}" rx="4" fill="#10b981" fill-opacity="0.82" />
      <text x="${x + barW / 2}" y="${py - 5}" fill="#e6edf7" font-size="10.5" text-anchor="middle">${b.feature_count}</text>
      <text x="${x + barW / 2}" y="${h - 12}" fill="#8b9bb8" font-size="10" text-anchor="middle">${esc(b.label)}</text>
    </g>`;
  });
  out += "</svg>";
  return out;
}

function renderSlotHeatmap(s) {
  const maxPts = Math.max(1, s.max_slot_load);
  return s.slots.map(slot => {
    const intensity = Math.min(1, slot.points / maxPts);
    let bg = `rgba(59, 130, 246, ${(0.12 + intensity * 0.58).toFixed(2)})`;
    if (slot.busy) bg = "rgba(245, 158, 11, 0.35)";
    else if (slot.status === "hot") bg = `rgba(244, 63, 94, ${(0.28 + intensity * 0.52).toFixed(2)})`;
    else if (slot.points === 0) bg = "rgba(30, 41, 59, 0.55)";
    return `<div class="heat-cell" style="background:${bg}" onclick="filterBySlot('${esc(slot.slot)}')">
      <div class="heat-slot">${esc(slot.slot)}</div>
      <div class="heat-pts">${fmtNum(slot.points)}</div>
      <div class="heat-meta">${slot.feature_count}f | ${slot.share_pct}%</div>
    </div>`;
  }).join("");
}

window.filterBySlot = function(slot) {
  state.slotFilter = slot;
  switchTab("features");
};

function renderFeaturesTab(s) {
  const q = state.featureQuery.toLowerCase();
  const rows = s.features.filter(f => {
    if (state.slotFilter && !f.shard_keys.includes(state.slotFilter) && !(f.write_to || []).includes(state.slotFilter)) return false;
    if (state.stateFilter !== "all" && f.state !== state.stateFilter) return false;
    if (q && !f.name.toLowerCase().includes(q) && !f.aliases.some(a => a.toLowerCase().includes(q))) return false;
    return true;
  });

  const disabled = state.readOnly ? "disabled" : "";
  const vecDisabled = (state.readOnly || !state.hasVectors) ? "disabled" : "";

  return `
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Add Feature</div>
        <div class="form-row">
          <div class="field"><label>Feature Value</label><input class="input" id="inp-add-feature" placeholder="e.g. books or 42" /></div>
          <button class="btn btn-primary" ${disabled} onclick="doAddFeature()">Add Feature</button>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Add Slot</div>
        <div class="form-row">
          <div class="field"><label>Slot Name</label><input class="input" id="inp-add-slot" placeholder="e.g. s${s.slot_count}" /></div>
          <button class="btn btn-primary" ${vecDisabled} onclick="doAddSlot()">Add Slot</button>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Features (${rows.length} / ${s.feature_count})</div>
          <div class="card-sub">${state.slotFilter ? `Filtered by slot <strong class="mono">${esc(state.slotFilter)}</strong> <a href="#" onclick="clearSlotFilter();return false;">(clear)</a>` : "Sorted by point count descending."}</div>
        </div>
        <div class="toolbar">
          <input class="input" placeholder="Search feature or alias..." value="${esc(state.featureQuery)}" oninput="state.featureQuery=this.value;renderMain()" />
          <select class="input" onchange="state.stateFilter=this.value;renderMain()">
            <option value="all" ${state.stateFilter === "all" ? "selected" : ""}>All states</option>
            <option value="stable" ${state.stateFilter === "stable" ? "selected" : ""}>stable</option>
            <option value="copying" ${state.stateFilter === "copying" ? "selected" : ""}>copying</option>
            <option value="dual" ${state.stateFilter === "dual" ? "selected" : ""}>dual</option>
            <option value="cutover" ${state.stateFilter === "cutover" ? "selected" : ""}>cutover</option>
          </select>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr><th>Feature</th><th>Points</th><th>Share</th><th>Slots</th><th>State</th><th>Aliases</th><th>Actions</th></tr>
          </thead>
          <tbody>
            ${rows.slice(0, 200).map(f => {
              const stClass = f.state === "stable" ? "pill-green" : "pill-amber";
              const target = f.write_to ? ` → ${f.write_to.join(", ")}` : "";
              return `<tr>
                <td class="mono"><strong>${esc(f.name)}</strong></td>
                <td class="mono">${fmtNum(f.points)}</td>
                <td class="mono">${f.share_pct}%</td>
                <td class="mono">${esc(f.shard_keys.join(", "))}${esc(target)}</td>
                <td><span class="pill ${stClass}">${esc(f.state)}</span></td>
                <td class="mono">${f.aliases.length ? f.aliases.map(a => `<span class="pill pill-muted">${esc(a)}</span>`).join(" ") : "—"}</td>
                <td>
                  <div class="actions-row">
                    ${f.state === "stable" ? `
                      <button class="btn btn-sm" ${vecDisabled} onclick="openMigrateModal('${esc(f.name)}')">Migrate</button>
                      <button class="btn btn-sm" ${vecDisabled} onclick="openSplitModal('${esc(f.name)}')">Split</button>
                      <button class="btn btn-sm" ${vecDisabled} onclick="openAliasModal('${esc(f.name)}')">Alias</button>
                    ` : `
                      <button class="btn btn-sm btn-primary" ${vecDisabled} onclick="doResumeMove('${esc(f.name)}')">Resume</button>
                    `}
                  </div>
                </td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
        ${!rows.length ? '<div class="empty">No matching features.</div>' : ""}
      </div>
    </div>
  `;
}

window.clearSlotFilter = function() {
  state.slotFilter = "";
  renderMain();
};

function renderRebalanceTab(s) {
  const vecDisabled = (state.readOnly || !state.hasVectors) ? "disabled" : "";
  const planHtml = state.plan ? `
    <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center;">
      <span class="pill ${state.plan.moves.length ? "pill-blue" : "pill-green"}">reason: ${esc(state.plan.reason)}</span>
      <span class="card-sub">Planned moves: <strong>${state.plan.moves.length}</strong></span>
      <span class="card-sub">Max slot load: <strong>${fmtNum(state.plan.max_before)}</strong> → <strong>${fmtNum(state.plan.max_after)}</strong></span>
    </div>
    ${state.plan.moves.length ? `<table>
      <thead><tr><th>Feature</th><th>Target Slots</th><th>Points</th></tr></thead>
      <tbody>${state.plan.moves.map(m => `<tr><td class="mono">${esc(m.feature)}</td><td class="mono">${esc(m.to_slots.join(", "))}</td><td class="mono">${fmtNum(m.points)}</td></tr>`).join("")}</tbody>
    </table>` : '<div class="card-sub">No same-width rebalance moves needed or legal.</div>'}
  ` : '<div class="card-sub">Click "Preview Rebalance Plan" to compute moves from live Qdrant facet counts.</div>';

  const splits = (state.plan && state.plan.split_recommendations) ? state.plan.split_recommendations : s.split_recommendations;

  return `
    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Rebalance Planner</div>
          <div class="card-sub">Same-width moves that reduce max slot load below ${s.balance_threshold}x mean.</div>
        </div>
        <div class="actions-row">
          <button class="btn" ${!state.hasVectors ? "disabled" : ""} onclick="doPreviewPlan()">Preview Rebalance Plan</button>
          <button class="btn btn-primary" ${vecDisabled} onclick="doApplyRebalance()">Apply Rebalance</button>
        </div>
      </div>
      ${planHtml}
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Split Recommendations (${splits.length})</div>
          <div class="card-sub">Single features too large for their current slot width.</div>
        </div>
      </div>
      ${splits.length ? `<table>
        <thead><tr><th>Feature</th><th>Points</th><th>Current Width</th><th>Recommended Partitions</th><th>Reason</th><th>Action</th></tr></thead>
        <tbody>${splits.map(r => `<tr>
          <td class="mono"><strong>${esc(r.feature)}</strong></td>
          <td class="mono">${fmtNum(r.points)}</td>
          <td class="mono">${r.current_width}</td>
          <td class="mono">${r.recommended_partitions}</td>
          <td>${esc(r.reason)}</td>
          <td><button class="btn btn-sm btn-primary" ${vecDisabled} onclick="doQuickSplit('${esc(r.feature)}', ${r.recommended_partitions})">Split to ${r.recommended_partitions}</button></td>
        </tr>`).join("")}</tbody>
      </table>` : '<div class="card-sub">No features currently require splitting.</div>'}
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Active Moves (${s.active_moves.length})</div>
          <div class="card-sub">Step-by-step control (copying → dual → cutover → stable) or automatic resume.</div>
        </div>
      </div>
      ${s.active_moves.length ? `<table>
        <thead><tr><th>Feature</th><th>State</th><th>Old → New</th><th>Copied</th><th>Grace Left</th><th>Steps</th></tr></thead>
        <tbody>${s.active_moves.map(m => `<tr>
          <td class="mono"><strong>${esc(m.name)}</strong></td>
          <td><span class="pill pill-amber">${esc(m.state)}</span></td>
          <td class="mono">${esc(m.shard_keys.join(","))} → ${esc((m.write_to || []).join(","))}</td>
          <td class="mono">${m.copied ? "yes" : "no"}</td>
          <td class="mono">${m.remaining_grace_seconds}s</td>
          <td>
            <div class="actions-row">
              ${m.state === "copying" && !m.copied ? `<button class="btn btn-sm" ${vecDisabled} onclick="doMoveStep('${esc(m.name)}','copy')">Copy</button>` : ""}
              ${m.state === "copying" && m.copied ? `<button class="btn btn-sm" ${vecDisabled} onclick="doMoveStep('${esc(m.name)}','open-dual')">Open Dual</button>` : ""}
              ${m.state === "dual" ? `<button class="btn btn-sm" ${vecDisabled} onclick="doMoveStep('${esc(m.name)}','cutover')">Cutover</button>` : ""}
              ${m.state === "cutover" ? `<button class="btn btn-sm" ${vecDisabled} onclick="doMoveStep('${esc(m.name)}','finish')">Finish</button>` : ""}
              <button class="btn btn-sm btn-primary" ${vecDisabled} onclick="doResumeMove('${esc(m.name)}')">Auto Resume</button>
            </div>
          </td>
        </tr>`).join("")}</tbody>
      </table>` : '<div class="card-sub">All features are currently in state stable.</div>'}
    </div>
  `;
}

function renderInspectorTab(s) {
  return `
    <div class="card">
      <div class="card-title">Route Inspector — <span class="mono">${esc(s.collection)}</span></div>
      <div class="card-sub">Test how the SDK routes a feature value and optional point ID using the current map.</div>
      <div class="form-row">
        <div class="field"><label>Feature (${esc(s.feature_field)})</label><input class="input" id="insp-feature" placeholder="e.g. books or 7" /></div>
        <div class="field"><label>Point ID (optional)</label><input class="input" id="insp-point-id" placeholder="e.g. 12345" /></div>
        <button class="btn btn-primary" onclick="doInspectRoute()">Inspect Route</button>
      </div>
      <pre class="code" id="insp-output">Enter a feature value and click Inspect Route.</pre>
    </div>
    <div class="card">
      <div class="card-head">
        <div class="card-title">Raw Map JSON (v${s.version})</div>
      </div>
      <pre class="code">${esc(JSON.stringify(state.detail.map, null, 2))}</pre>
    </div>
  `;
}

function renderMetricsTab() {
  const r = (state.overview && state.overview.routing) ? state.overview.routing : null;
  const totalReads = r ? (r.total_routed_reads + r.total_fanout_reads) : 0;
  const hitRate = totalReads > 0 ? ((r.total_routed_reads / totalReads) * 100).toFixed(1) + "%" : "—";
  return `
    ${r ? `<div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Routed Reads</div><div class="kpi-value">${fmtNum(r.total_routed_reads)}</div><div class="kpi-sub">hit rate ${hitRate}</div></div>
      <div class="kpi"><div class="kpi-label">Fan-out Reads</div><div class="kpi-value">${fmtNum(r.total_fanout_reads)}</div></div>
      <div class="kpi"><div class="kpi-label">Routed Writes</div><div class="kpi-value">${fmtNum(r.total_routed_writes)}</div></div>
      <div class="kpi"><div class="kpi-label">Fan-out Writes</div><div class="kpi-value">${fmtNum(r.total_fanout_writes)}</div></div>
      <div class="kpi"><div class="kpi-label">Feature Moves</div><div class="kpi-value">${fmtNum(r.total_feature_moves)}</div></div>
      <div class="kpi"><div class="kpi-label">Write Denied</div><div class="kpi-value">${fmtNum(r.total_write_denied)}</div></div>
    </div>` : ""}

    <div class="card">
      <div class="card-head">
        <div class="card-title">Background Jobs (${state.jobs.length})</div>
        <button class="btn btn-sm" onclick="loadAll()">Refresh</button>
      </div>
      ${state.jobs.length ? `<table>
        <thead><tr><th>ID</th><th>Collection</th><th>Action</th><th>Feature</th><th>Status</th><th>Detail</th></tr></thead>
        <tbody>${state.jobs.map(j => {
          const cls = j.status === "completed" ? "pill-green" : (j.status === "failed" ? "pill-red" : "pill-amber");
          return `<tr>
            <td class="mono">${esc(j.id.slice(0, 8))}</td>
            <td class="mono">${esc(j.collection)}</td>
            <td>${esc(j.action)}</td>
            <td class="mono">${esc(j.feature || "—")}</td>
            <td><span class="pill ${cls}">${esc(j.status)}</span></td>
            <td class="mono">${esc(j.error || (j.result ? JSON.stringify(j.result) : ""))}</td>
          </tr>`;
        }).join("")}</tbody>
      </table>` : '<div class="card-sub">No background jobs have been started yet.</div>'}
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Prometheus Metrics (<span class="mono">GET /metrics</span>)</div>
          <div class="card-sub">Scrape directly from Prometheus, VictoriaMetrics, or Grafana Agent.</div>
        </div>
        <button class="btn btn-sm" onclick="navigator.clipboard && navigator.clipboard.writeText(state.promText).then(() => toast('Copied /metrics'))">Copy</button>
      </div>
      <pre class="code">${esc(state.promText || "Click Reload to fetch /metrics.")}</pre>
    </div>
  `;
}

window.switchTab = function(tab) {
  state.tab = tab;
  document.querySelectorAll("#nav-tabs .tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  if (tab === "metrics") loadAll();
  else renderMain();
};

document.querySelectorAll("#nav-tabs .tab").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

document.getElementById("toggle-live").addEventListener("change", e => {
  state.live = e.target.checked;
  loadAll();
});

document.getElementById("toggle-auto").addEventListener("change", e => {
  clearInterval(state.autoTimer);
  if (e.target.checked) state.autoTimer = setInterval(loadAll, 5000);
});

document.getElementById("btn-reload").addEventListener("click", loadAll);

document.getElementById("btn-refresh-stats").addEventListener("click", async () => {
  if (!state.selected) return;
  try {
    await api("/api/collections/" + encodeURIComponent(state.selected) + "/refresh", { method: "POST", body: {} });
    toast("Stats refreshed from Qdrant");
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
});

document.getElementById("btn-auth-token").addEventListener("click", () => {
  const existing = sessionStorage.getItem("qcs_api_token") || "";
  openModal(`
    <h3>API Bearer Token</h3>
    <div class="field"><label>Token (leave blank to clear)</label><input class="input" id="tok-val" type="password" value="${esc(existing)}" /></div>
    <div class="actions-row" style="justify-content:flex-end;">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="saveToken()">Save</button>
    </div>
  `);
});

window.saveToken = function() {
  const v = document.getElementById("tok-val").value.trim();
  if (v) sessionStorage.setItem("qcs_api_token", v);
  else sessionStorage.removeItem("qcs_api_token");
  closeModal();
  loadAll();
};

document.getElementById("btn-open-create").addEventListener("click", () => {
  openModal(`
    <h3>Create Sharded Collection</h3>
    <div class="field"><label>Collection Name</label><input class="input" id="cr-name" placeholder="people" /></div>
    <div class="field"><label>Feature Payload Field</label><input class="input" id="cr-field" placeholder="subject_id" /></div>
    <div class="form-row">
      <div class="field" style="flex:1"><label>Feature Type</label><select class="input" id="cr-type"><option value="keyword">keyword</option><option value="integer">integer</option></select></div>
      <div class="field" style="flex:1"><label>Slots</label><input class="input" id="cr-slots" type="number" min="1" value="12" /></div>
    </div>
    <div class="form-row">
      <div class="field" style="flex:1"><label>Vector Size</label><input class="input" id="cr-vsize" type="number" min="1" value="128" /></div>
      <div class="field" style="flex:1"><label>Distance</label><select class="input" id="cr-dist"><option value="Cosine">Cosine</option><option value="Dot">Dot</option><option value="Euclid">Euclid</option><option value="Manhattan">Manhattan</option></select></div>
    </div>
    <div class="actions-row" style="justify-content:flex-end;">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doCreateCollection()">Create</button>
    </div>
  `);
});

window.doCreateCollection = async function() {
  try {
    const body = {
      collection_name: document.getElementById("cr-name").value.trim(),
      feature_field: document.getElementById("cr-field").value.trim(),
      feature_type: document.getElementById("cr-type").value,
      n_slots: Number(document.getElementById("cr-slots").value),
      vector_size: Number(document.getElementById("cr-vsize").value),
      distance: document.getElementById("cr-dist").value,
    };
    await api("/api/collections", { method: "POST", body });
    closeModal();
    state.selected = body.collection_name;
    toast("Created collection " + body.collection_name);
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doAddFeature = async function() {
  try {
    const raw = document.getElementById("inp-add-feature").value.trim();
    if (!raw) return;
    const val = (state.detail.stats.feature_type === "integer" && /^-?\\d+$/.test(raw)) ? Number(raw) : raw;
    await api("/api/collections/" + encodeURIComponent(state.selected) + "/features", { method: "POST", body: { value: val } });
    toast("Added feature " + raw);
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doAddSlot = async function() {
  try {
    const slot = document.getElementById("inp-add-slot").value.trim();
    if (!slot) return;
    await api("/api/collections/" + encodeURIComponent(state.selected) + "/slots", { method: "POST", body: { slot } });
    toast("Added slot " + slot);
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.openMigrateModal = function(feature) {
  const s = state.detail.stats;
  const f = s.features.find(x => x.name === feature);
  const free = s.slots.filter(sl => !sl.busy && !(f && f.shard_keys.includes(sl.slot)));
  openModal(`
    <h3>Migrate Feature <span class="mono">${esc(feature)}</span></h3>
    <div class="field"><label>Target Slot</label>
      <select class="input" id="mig-slot">${free.map(sl => `<option value="${esc(sl.slot)}">${esc(sl.slot)} (${fmtNum(sl.points)} pts)</option>`).join("")}</select>
    </div>
    <div class="field"><label>Grace Seconds</label><input class="input" id="mig-grace" type="number" min="0" value="60" /></div>
    <div class="actions-row" style="justify-content:flex-end;">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doMigrate('${esc(feature)}')">Start Migrate</button>
    </div>
  `);
};

window.doMigrate = async function(feature) {
  try {
    const to_slot = document.getElementById("mig-slot").value;
    const grace_seconds = Number(document.getElementById("mig-grace").value);
    const res = await api("/api/collections/" + encodeURIComponent(state.selected) + "/migrate", {
      method: "POST", body: { feature, to_slot, grace_seconds, background: true }
    });
    closeModal();
    toast("Queued migrate job " + res.id.slice(0, 8));
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.openSplitModal = function(feature) {
  openModal(`
    <h3>Split Feature <span class="mono">${esc(feature)}</span></h3>
    <div class="field"><label>Partitions (number of free slots)</label><input class="input" id="spl-parts" type="number" min="2" value="2" /></div>
    <div class="field"><label>Grace Seconds</label><input class="input" id="spl-grace" type="number" min="0" value="60" /></div>
    <div class="actions-row" style="justify-content:flex-end;">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doSplit('${esc(feature)}')">Start Split</button>
    </div>
  `);
};

window.doSplit = async function(feature) {
  try {
    const partitions = Number(document.getElementById("spl-parts").value);
    const grace_seconds = Number(document.getElementById("spl-grace").value);
    const res = await api("/api/collections/" + encodeURIComponent(state.selected) + "/split", {
      method: "POST", body: { feature, partitions, grace_seconds, background: true }
    });
    closeModal();
    toast("Queued split job " + res.id.slice(0, 8));
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doQuickSplit = async function(feature, partitions) {
  try {
    const res = await api("/api/collections/" + encodeURIComponent(state.selected) + "/split", {
      method: "POST", body: { feature, partitions, grace_seconds: 60, background: true }
    });
    toast("Queued split job " + res.id.slice(0, 8));
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.openAliasModal = function(feature) {
  openModal(`
    <h3>Alias Feature <span class="mono">${esc(feature)}</span></h3>
    <div class="field"><label>New Canonical Name</label><input class="input" id="al-new" placeholder="new name" /></div>
    <div class="field"><label>Grace Seconds</label><input class="input" id="al-grace" type="number" min="0" value="60" /></div>
    <div class="actions-row" style="justify-content:flex-end;">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doAlias('${esc(feature)}')">Alias</button>
    </div>
  `);
};

window.doAlias = async function(oldName) {
  try {
    const newName = document.getElementById("al-new").value.trim();
    const grace_seconds = Number(document.getElementById("al-grace").value);
    const res = await api("/api/collections/" + encodeURIComponent(state.selected) + "/aliases", {
      method: "POST", body: { old: oldName, new: newName, grace_seconds, background: true }
    });
    closeModal();
    toast("Queued alias job " + res.id.slice(0, 8));
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doMoveStep = async function(feature, step) {
  try {
    await api(`/api/collections/${encodeURIComponent(state.selected)}/moves/${encodeURIComponent(feature)}/${step}`, {
      method: "POST", body: {}
    });
    toast(`Completed ${step} for ${feature}`);
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doResumeMove = async function(feature) {
  try {
    const res = await api(`/api/collections/${encodeURIComponent(state.selected)}/moves/${encodeURIComponent(feature)}/resume`, {
      method: "POST", body: { grace_seconds: 60, background: true }
    });
    toast("Queued resume job " + res.id.slice(0, 8));
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doPreviewPlan = async function() {
  try {
    state.plan = await api("/api/collections/" + encodeURIComponent(state.selected) + "/plan");
    renderMain();
    toast("Rebalance plan updated from live counts");
  } catch (err) { toast(err.message, "error"); }
};

window.doApplyRebalance = async function() {
  try {
    const res = await api("/api/collections/" + encodeURIComponent(state.selected) + "/rebalance", {
      method: "POST", body: { grace_seconds: 60, background: true }
    });
    toast("Queued rebalance job " + res.id.slice(0, 8));
    await loadAll();
  } catch (err) { toast(err.message, "error"); }
};

window.doInspectRoute = async function() {
  try {
    const feature = document.getElementById("insp-feature").value.trim();
    const pointId = document.getElementById("insp-point-id").value.trim();
    if (!feature) return;
    let url = `/api/collections/${encodeURIComponent(state.selected)}/route?feature=${encodeURIComponent(feature)}`;
    if (pointId) url += `&point_id=${encodeURIComponent(pointId)}`;
    const res = await api(url);
    document.getElementById("insp-output").textContent = JSON.stringify(res, null, 2);
  } catch (err) { toast(err.message, "error"); }
};

loadAll();
</script>
</body>
</html>
"""
