DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Twitter Sentiment Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#111827; --muted:#6b7280; --border:#e5e7eb; --accent:#2563eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1115; --card:#181b22; --text:#e5e7eb; --muted:#9ca3af; --border:#2a2f3a; --accent:#60a5fa; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:20px 28px; border-bottom:1px solid var(--border); background:var(--card); }
  header h1 { margin:0; font-size:20px; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:1200px; margin:0 auto; padding:20px; display:grid; gap:16px; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; }
  .kpi .v { font-size:22px; font-weight:650; }
  .kpi .l { color:var(--muted); font-size:12px; margin-top:2px; }
  .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; }
  h2 { margin:0 0 12px; font-size:15px; }
  .chart { position:relative; height:280px; }
  .chart.tall { height:420px; }
  select, textarea, button { font:inherit; color:var(--text); background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:8px 10px; }
  textarea { width:100%; min-height:80px; resize:vertical; }
  button { background:var(--accent); color:#fff; border:none; cursor:pointer; margin-top:8px; }
  .chip { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; color:#fff; }
  .Positive{background:#16a34a}.Negative{background:#dc2626}.Neutral{background:#2563eb}.Irrelevant{background:#6b7280}
  .bar { height:10px; border-radius:5px; background:var(--border); overflow:hidden; margin:2px 0 8px; }
  .bar > div { height:100%; }
  .row { display:flex; justify-content:space-between; font-size:13px; }
  table { border-collapse:collapse; width:100%; font-size:13px; text-align:center; }
  th, td { padding:6px 8px; border:1px solid var(--border); }
  .sample { padding:8px 0; border-bottom:1px solid var(--border); font-size:13px; }
  .sample:last-child { border:none; }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Twitter Platform Sentiment Analysis Dashboard</h1>
  <p>Positive / Negative / Neutral / Irrelevant classification with a Flask REST API</p>
</header>
<main>
  <section class="kpis" id="kpis"></section>

  <section class="grid2">
    <div class="card"><h2>Sentiment distribution</h2><div class="chart"><canvas id="distChart"></canvas></div></div>
    <div class="card"><h2>Model comparison (test set)</h2><div class="chart"><canvas id="modelChart"></canvas></div></div>
  </section>

  <section class="card"><h2>Sentiment mix by brand (% of tweets)</h2><div class="chart tall"><canvas id="brandMix"></canvas></div></section>

  <section class="grid2">
    <div class="card">
      <h2>Live prediction</h2>
      <textarea id="txt" placeholder="Type a tweet, e.g. I absolutely love this new update!"></textarea>
      <button id="go">Analyze sentiment</button>
      <div id="result" style="margin-top:12px"></div>
    </div>
    <div class="card">
      <h2>Confusion matrix <span class="muted" style="font-weight:400">(rows = actual, columns = predicted)</span></h2>
      <div id="cm"></div>
    </div>
  </section>

  <section class="card">
    <h2>Brand explorer</h2>
    <select id="brandSel"></select>
    <div class="grid2" style="margin-top:12px">
      <div><div class="chart"><canvas id="brandChart"></canvas></div><p id="brandInfo" class="muted"></p></div>
      <div id="samples"></div>
    </div>
  </section>
</main>

<script>
const LABELS = ["Positive","Negative","Neutral","Irrelevant"];
const COLORS = {Positive:"#16a34a", Negative:"#dc2626", Neutral:"#2563eb", Irrelevant:"#9ca3af"};
const get = u => fetch(u).then(r => r.json());
const pct = v => (v * 100).toFixed(1) + "%";
const charts = {};

function draw(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function kpi(label, value) {
  const c = el("div", "card kpi");
  c.append(el("div", "v", value), el("div", "l", label));
  document.getElementById("kpis").append(c);
}

async function init() {
  const css = getComputedStyle(document.documentElement);
  Chart.defaults.color = css.getPropertyValue("--muted").trim();
  Chart.defaults.borderColor = css.getPropertyValue("--border").trim();
  Chart.defaults.maintainAspectRatio = false;

  const [stats, metrics] = await Promise.all([get("/api/stats"), get("/api/metrics")]);
  const best = metrics.models[metrics.best_model];
  kpi("Tweets", stats.total.toLocaleString());
  kpi("Brands", stats.brand_count);
  kpi("Best model", metrics.best_model);
  kpi("Accuracy", pct(best.accuracy));
  kpi("Macro F1", pct(best.macro_f1));

  draw("distChart", {
    type: "doughnut",
    data: { labels: LABELS, datasets: [{ data: LABELS.map(l => stats.distribution[l]), backgroundColor: LABELS.map(l => COLORS[l]), borderWidth: 0 }] },
    options: { plugins: { legend: { position: "bottom" } } }
  });

  const names = Object.keys(metrics.models);
  draw("modelChart", {
    type: "bar",
    data: { labels: names, datasets: [
      { label: "Accuracy", data: names.map(n => metrics.models[n].accuracy), backgroundColor: "#2563eb" },
      { label: "Macro F1", data: names.map(n => metrics.models[n].macro_f1), backgroundColor: "#f59e0b" } ] },
    options: { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: "bottom" } } }
  });

  // Brand mix: 100% stacked horizontal bars, sorted by volume
  const brands = Object.keys(stats.by_brand);
  const total = b => LABELS.reduce((s, l) => s + stats.by_brand[b][l], 0);
  brands.sort((a, b) => total(b) - total(a));
  draw("brandMix", {
    type: "bar",
    data: { labels: brands, datasets: LABELS.map(l => ({
      label: l, backgroundColor: COLORS[l],
      data: brands.map(b => +(100 * stats.by_brand[b][l] / total(b)).toFixed(1)) })) },
    options: { indexAxis: "y", scales: { x: { stacked: true, max: 100 }, y: { stacked: true, ticks: { autoSkip: false, font: { size: 10 } } } },
               plugins: { legend: { position: "bottom" } } }
  });

  // Confusion matrix table
  const cm = metrics.confusion_matrix;
  const t = el("table");
  const head = el("tr"); head.append(el("th", "", ""));
  metrics.labels.forEach(l => head.append(el("th", "", l)));
  t.append(head);
  cm.forEach((row, i) => {
    const tr = el("tr"); tr.append(el("th", "", metrics.labels[i]));
    const sum = row.reduce((a, b) => a + b, 0) || 1;
    row.forEach((v, j) => {
      const td = el("td", "", v.toLocaleString());
      td.style.background = "rgba(37,99,235," + (0.08 + 0.6 * v / sum) + ")";
      if (i === j) td.style.fontWeight = "700";
      tr.append(td);
    });
    t.append(tr);
  });
  document.getElementById("cm").append(t);

  // Brand explorer
  const sel = document.getElementById("brandSel");
  stats.brands.forEach(b => sel.append(new Option(b, b)));
  sel.onchange = () => loadBrand(sel.value);
  loadBrand(sel.value);
}

async function loadBrand(name) {
  const q = encodeURIComponent(name);
  const [info, s] = await Promise.all([get("/api/brand?name=" + q), get("/api/samples?limit=6&brand=" + q)]);
  draw("brandChart", {
    type: "doughnut",
    data: { labels: LABELS, datasets: [{ data: LABELS.map(l => info.distribution[l]), backgroundColor: LABELS.map(l => COLORS[l]), borderWidth: 0 }] },
    options: { plugins: { legend: { position: "bottom" } } }
  });
  document.getElementById("brandInfo").textContent =
    info.total.toLocaleString() + " tweets - net sentiment " + (info.net_sentiment * 100).toFixed(1) + " (positive minus negative share)";
  const box = document.getElementById("samples");
  box.replaceChildren(el("h2", "", "Random tweets"));
  s.samples.forEach(x => {
    const d = el("div", "sample");
    d.append(el("span", "chip " + x.Sentiment, x.Sentiment), el("div", "", x.Text));
    box.append(d);
  });
}

document.getElementById("go").onclick = async () => {
  const text = document.getElementById("txt").value.trim();
  const out = document.getElementById("result");
  if (!text) return;
  const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  const data = await r.json();
  out.replaceChildren();
  if (data.error) { out.append(el("div", "muted", data.error)); return; }
  const top = el("div"); top.append(el("span", "chip " + data.sentiment, data.sentiment), el("span", "muted", "  confidence " + pct(data.confidence)));
  out.append(top, el("div", "", ""));
  LABELS.forEach(l => {
    const p = data.probabilities[l] || 0;
    const row = el("div", "row"); row.append(el("span", "", l), el("span", "muted", pct(p)));
    const bar = el("div", "bar"); const fill = el("div"); fill.style.width = (p * 100) + "%"; fill.style.background = COLORS[l];
    bar.append(fill); out.append(row, bar);
  });
};

init();
</script>
</body>
</html>
"""
