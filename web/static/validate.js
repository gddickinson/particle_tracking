// Validation: assigned class against the motion model each simulated track was generated from (true_mobility),
// with live threshold sliders (re-classifying the computed metrics in the browser).
import { $, app, esc, list, route, slider } from './core.js';
import { CLASSES, reclassify } from './engine.js';
import { TRUE_NAMES, cls, pct, provenance } from './fmt.js';
import { getInfo, getResult, sourceLine } from './state.js';

const TRUE = ['brownian', 'confined', 'directed', 'subdiffusive', 'superdiffusive'];

// Smallest n for which a straight, evenly spaced track of n points has a confinement ratio below thr.
function lineLimit(thr) {
  for (let n = 3; n < 100000; n++) if (Math.sqrt(12 * (n - 1) / (n + 1)) / Math.sqrt(n) < thr) return n;
  return null;
}

route('validate', async () => {
  const [info, r] = await Promise.all([getInfo(), getResult()]);
  if (!r.has_labels) {
    app().innerHTML = `<p class="kicker">Ground truth</p><h1>Validation</h1>
      <p>This page needs tracks with a <code>${esc(info.label_column)}</code> column (the motion model each track was simulated
      with). The current data (${esc(sourceLine(r))}) has none. Pick one of the demo datasets on <a href="#/analyze">Analyze</a>.</p>`;
    return;
  }
  const p = { ...r.params };
  const lengths = r.tracks.map(t => t.track_length);
  app().innerHTML = `<p class="kicker">Ground truth</p><h1>Validation</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  <p class="lead">Rows: the model each track was simulated with (<code>TrajectorySimulator.generate_mixed_population</code>:
    Brownian, confined in a reflecting disc, directed with drift, fractional Brownian motion with H &lt; 0.5 as
    subdiffusive and H &gt; 0.5 as superdiffusive). Columns: the class MobilityAnalyzer assigns.</p>
  <div id="cm"></div><p id="acc" class="small"></p>
    <div class="panel"><h3>Thresholds (live)</h3><div id="sl"></div>
      <p class="small mute">Moving a slider re-applies the class rule to the metrics already computed (α, directedness and the
        confinement ratio do not depend on the thresholds). Use <a href="#/analyze">Analyze</a> to recompute everything.</p></div>
  <h2>Why long tracks come out “confined”</h2>
  <div class="callout"><p>The confinement test is <code>end-to-end / (Rg · √n) &lt; ${p.confinement_threshold}</code>
    (<code>MobilityAnalyzer.calculate_confinement_ratio</code>), and it is checked before the other classes. Because of the √n,
    the ratio shrinks as tracks get longer whatever the motion: a perfectly straight track of evenly spaced points has ratio
    √(12(n−1)/(n+1))/√n, which is below the threshold from <b id="nline"></b> points on. The tracks here have
    ${Math.min(...lengths)}–${Math.max(...lengths)} points. Compare the two demo datasets (50 and 200 frames) and the
    “confinement ratio against track length” plot on <a href="#/populations">Populations</a>.</p>
    <p class="small mute">This is how the code behaves today; whether the √n normalisation is intended is an open question for
      the author, so the web app reproduces it unchanged.</p></div>`;
  const draw = () => {
    const rows = reclassify(r.tracks, p).filter(t => TRUE.includes(t.true_mobility));
    const pred = CLASSES.filter(c => rows.some(t => t.mobility_class === c));
    const cell = (tr, pr) => rows.filter(t => t.true_mobility === tr && t.mobility_class === pr).length;
    const hit = rows.filter(t => t.true_mobility === t.mobility_class).length;
    $('#cm').innerHTML = `<div class="scroll"><table class="tbl cm"><thead><tr><th>simulated as ↓ / assigned →</th>${list(pred, c => `<th class="rot">${cls(c)}</th>`)}<th>recall</th></tr></thead><tbody>
      ${list(TRUE.filter(tr => rows.some(t => t.true_mobility === tr)), tr => { const n = rows.filter(t => t.true_mobility === tr).length;
        return `<tr><td>${esc(TRUE_NAMES[tr])} <span class="mute small">(${n})</span></td>${list(pred, pr => { const k = cell(tr, pr), a = n ? k / n : 0;
          return `<td class="n"><span style="background:color-mix(in srgb, ${tr === pr ? 'var(--accent)' : 'var(--accent2)'} ${Math.round(a * 55)}%, transparent)">${k}</span></td>`; })}
          <td class="n">${pct(cell(tr, tr), n)}</td></tr>`; })}</tbody></table></div>`;
    $('#acc').innerHTML = `<b>${pct(hit, rows.length)}</b> of ${rows.length} tracks get the class they were simulated with.`;
    $('#nline').textContent = lineLimit(p.confinement_threshold) ?? 'never';
  };
  const host = $('#sl');
  const sl = (id, label, min, max, step) => slider(host, { id, label, min, max, step, value: p[id], fmt: v => (+v).toFixed(2) }, v => { p[id] = v; draw(); });
  sl('confinement_threshold', 'Confined below (confinement ratio)', 0, 1, 0.01);
  sl('directedness_threshold', 'Directed above (directedness)', 0, 1, 0.01);
  sl('subdiffusion_threshold', 'Subdiffusive below (α)', 0, 2, 0.01);
  sl('superdiffusion_threshold', 'Superdiffusive above (α)', 0, 2, 0.01);
  draw();
});
