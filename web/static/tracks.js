// Tracks: all tracks on a map coloured by class, a sortable per-track table, and one track in detail (trajectory in µm,
// its MSD curve and the two fits behind D and α).
import { $, $$, app, esc, list, route } from './core.js';
import { drawMap, msdPlot, trackSvg } from './charts.js';
import { frameMsd, frameSpan, linregress, num, scaleOf } from './engine.js';
import { COLORS, TRACK_COLUMNS, TRUE_NAMES, cls, f, legend, provenance } from './fmt.js';
import { getResult, sourceLine, tracksOf } from './state.js';

const PAGE = 25;
let sortKey = 'track', sortDir = 1, page = 0, selected = null;

route('tracks', async (args, query) => {
  const r = await getResult(), tracks = await tracksOf(r);
  const rows = r.tracks.map((t, i) => ({ ...t, i }));
  if (query.id && rows.some(t => t.particle === query.id)) selected = query.id;
  if (!rows.some(t => t.particle === selected)) selected = rows[0]?.particle ?? null;
  app().innerHTML = `<p class="kicker">Per track</p><h1>Tracks</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  ${legend()}
  <div class="two">
    <div><div class="mapwrap"><canvas id="map" width="900" height="500" aria-label="Map of all tracks coloured by class; click a track to select it"></canvas></div>
      <p class="small mute">Image orientation (y down), coordinates as in the file. Click near a track to select it.</p></div>
    <div id="detail" class="panel" aria-live="polite"></div>
  </div>
  <h2>All tracks</h2>
  <div class="scroll"><table class="tbl"><thead><tr>${list(TRACK_COLUMNS, ([k, l]) => `<th><button type="button" data-k="${k}">${esc(l)}</button></th>`)}${r.has_labels ? '<th><button type="button" data-k="true_mobility">simulated as</button></th>' : ''}</tr></thead><tbody id="tb"></tbody></table></div>
  <div class="pager"><button type="button" class="alt" id="prev">Previous</button><span id="pg" class="small mute"></span><button type="button" class="alt" id="next">Next</button></div>`;

  const canvas = $('#map');
  let tf = null;
  const redraw = () => { tf = drawMap(canvas, tracks, id => COLORS[r.tracks.find(t => t.particle === id)?.mobility_class] || '#888', selected); };
  const pick = id => { selected = id; redraw(); detail(r, tracks, id); table(); };
  canvas.addEventListener('click', e => {
    const b = canvas.getBoundingClientRect(), s = canvas.width / b.width;
    const X = (e.clientX - b.left) * s, Y = (e.clientY - b.top) * s;
    let best = null, bd = 12 * s;
    for (const t of tracks) for (let i = 0; i < t.xs.length; i++) {
      const d = Math.hypot(tf.pad + (t.xs[i] - tf.x0) * tf.k - X, tf.pad + (t.ys[i] - tf.y0) * tf.k - Y);
      if (d < bd) { bd = d; best = t.id; }
    }
    if (best !== null) pick(best);
  });

  const sorted = () => {
    if (sortKey === 'track') return rows;
    const v = t => (typeof t[sortKey] === 'string' ? t[sortKey] : num(t[sortKey]));
    return [...rows].sort((a, b) => {
      const x = v(a), y = v(b);
      if (typeof x === 'string') return sortDir * x.localeCompare(y);
      if (Number.isNaN(x)) return 1; if (Number.isNaN(y)) return -1;
      return sortDir * (x - y);
    });
  };
  const table = () => {
    const all = sorted(), pages = Math.max(1, Math.ceil(all.length / PAGE));
    page = Math.min(page, pages - 1);
    $('#tb').innerHTML = list(all.slice(page * PAGE, page * PAGE + PAGE), t => `<tr class="click${t.particle === selected ? ' sel' : ''}" tabindex="0" data-id="${esc(t.particle)}">
      <td>${esc(t.particle)}</td><td>${t.track_length}</td><td>${f(t.D)}</td><td>${f(t.alpha, 3)}</td><td>${f(t.directedness, 3)}</td>
      <td>${f(t.confinement_ratio, 3)}</td><td>${cls(t.mobility_class)}</td>${r.has_labels ? `<td>${esc(TRUE_NAMES[t.true_mobility] || t.true_mobility)}</td>` : ''}</tr>`);
    $('#pg').textContent = `page ${page + 1} of ${pages} · ${all.length} tracks`;
    $$('#tb tr').forEach(tr => {
      tr.onclick = () => pick(tr.dataset.id);
      tr.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(tr.dataset.id); } };
    });
  };
  $$('th button').forEach(b => { b.onclick = () => { const k = b.dataset.k === 'particle' ? 'track' : b.dataset.k; sortDir = sortKey === k ? -sortDir : 1; sortKey = k; table(); }; });
  $('#prev').onclick = () => { page = Math.max(0, page - 1); table(); };
  $('#next').onclick = () => { page++; table(); };
  redraw(); table(); if (selected !== null) detail(r, tracks, selected);
});

function detail(r, tracks, id) {
  const t = tracks.find(q => q.id === id), m = r.tracks.find(q => q.particle === id);
  if (!t || !m) { $('#detail').innerHTML = '<p class="mute">No track selected.</p>'; return; }
  const { pixelSize, fps } = scaleOf(r.params);
  const x = t.xs.map(v => v * pixelSize), y = t.ys.map(v => v * pixelSize), n = x.length;
  const am = frameMsd(t.frames, x, y, Math.max(1, Math.min(Math.floor(frameSpan(t.frames) / 2), 50) - 1));
  const lags = am.lags.map(l => l / fps), msd = am.msds;
  const fits = [];
  const dmf = frameMsd(t.frames, x, y, r.d_fit_lags), dl = dmf.lags.map(l => l / fps), dm = dmf.msds;
  if (dl.length > 2) { const L = linregress(dl, dm); fits.push({ name: `linear fit, frame lags 1–${dmf.lags[dmf.lags.length - 1]}: slope ${f(L.slope)} µm²/s = 4D`, color: '#ffd23f', range: [dl[0], dl[dl.length - 1]], fn: s => L.slope * s + L.intercept }); }
  const pos = lags.map((l, i) => [l, msd[i]]).filter(([, v]) => v > 0);
  if (pos.length > 5) { const P = linregress(pos.map(p => Math.log(p[0])), pos.map(p => Math.log(p[1]))); fits.push({ name: `log-log fit: slope α = ${f(P.slope, 3)}`, color: '#ff4fd8', fn: s => Math.exp(P.intercept) * s ** P.slope }); }
  $('#detail').innerHTML = `<h3>Track ${esc(id)} · ${cls(m.mobility_class)}${m.true_mobility ? ` <span class="small mute">simulated as ${esc(TRUE_NAMES[m.true_mobility] || m.true_mobility)}</span>` : ''}</h3>
    <dl class="kv"><dt>points</dt><dd>${n} (${f((t.frames[n - 1] - t.frames[0]) * r.params.frame_interval_s)} s)</dd>
      <dt>D</dt><dd>${f(m.D)} µm²/s</dd><dt>α</dt><dd>${f(m.alpha, 3)}</dd>
      <dt>directedness</dt><dd>${f(m.directedness, 3)} (directed above ${r.params.directedness_threshold})</dd>
      <dt>confinement ratio</dt><dd>${f(m.confinement_ratio, 3)} (confined below ${r.params.confinement_threshold})</dd></dl>
    <div class="charts mini">
      <div>${trackSvg(x, y, { color: COLORS[m.mobility_class] })}<p class="small mute">Trajectory in µm: start amber, end magenta.</p></div>
      <div>${msdPlot([{ name: 'time-averaged MSD', color: COLORS[m.mobility_class], x: lags, y: msd, dots: true, fits }])}
        <p class="small mute">${fits.map(q => `<span style="color:${q.color}">- -</span> ${esc(q.name)}`).join('<br>') || 'Too few points for the fits.'}</p></div>
    </div>`;
}
