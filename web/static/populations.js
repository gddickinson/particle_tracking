// Populations: ensemble MSD per class with power-law fits, D and α distributions, and the per-track metrics against
// the class thresholds (mixed populations at a glance).
import { app, esc, list, route } from './core.js';
import { histogram, msdPlot, scatter } from './charts.js';
import { CLASSES, num } from './engine.js';
import { COLORS, cls, f, legend, provenance } from './fmt.js';
import { getResult, sourceLine } from './state.js';

route('populations', async () => {
  const r = await getResult(), p = r.params, t = r.tracks;
  const series = CLASSES.filter(c => r.ensemble[c]).map(c => {
    const e = r.ensemble[c], fit = e.fit;
    return { name: `${c} (${e.n_tracks} tracks)`, color: COLORS[c], x: e.lag_s, y: e.msd_um2.map(num),
      fits: Number.isFinite(num(fit.D)) ? [{ name: `${c}: 4D t^α, D = ${f(fit.D)} µm²/s, α = ${f(fit.alpha, 3)}`, fn: s => 4 * fit.D * s ** fit.alpha }] : [] };
  });
  const lengths = t.map(q => q.track_length), nmin = Math.min(...lengths), nmax = Math.max(...lengths);
  const line = [];
  for (let n = Math.max(3, nmin); n <= Math.max(nmax, nmin + 1); n += Math.max(1, Math.round((nmax - nmin) / 60))) line.push([n, Math.sqrt(12 * (n - 1) / (n + 1)) / Math.sqrt(n)]);
  app().innerHTML = `<p class="kicker">Mixed populations</p><h1>Populations</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  ${legend(CLASSES.filter(c => t.some(q => q.mobility_class === c)))}
  <div class="charts">
    <div class="panel"><h3>Ensemble MSD per class</h3>${msdPlot(series)}
      <p class="small mute">Solid: mean of the per-track MSD curves of each class (as <code>_calculate_msd_manual</code>);
        dashed: the power-law fit MSD = 4D t<sup>α</sup> over all lags. Long lags average few tracks.</p>
      <div class="scroll"><table class="tbl"><thead><tr><th>class</th><th>D <span class="unit">µm²/s</span></th><th>α</th><th>R²</th></tr></thead><tbody>
      ${list(CLASSES.filter(c => r.ensemble[c]), c => { const e = r.ensemble[c].fit; return `<tr><td>${cls(c)}</td><td>${f(e.D)}</td><td>${f(e.alpha, 3)}</td><td>${f(e.r_squared, 3)}</td></tr>`; })}</tbody></table></div></div>
    <div class="panel"><h3>Diffusion coefficient</h3>${histogram(t.filter(q => num(q.D) > 0).map(q => ({ v: Math.log10(q.D), c: q.mobility_class })), { xlabel: 'log10 D (D in µm²/s)' })}
      <p class="small mute">${t.filter(q => !(num(q.D) > 0)).length} tracks with D ≤ 0 or undefined are not shown (log scale).</p></div>
    <div class="panel"><h3>Anomalous exponent <span class="gk">α</span></h3>${histogram(t.map(q => ({ v: num(q.alpha), c: q.mobility_class })), { xlabel: 'α', marks: [{ x: p.subdiffusion_threshold, label: 'sub' }, { x: p.superdiffusion_threshold, label: 'super' }] })}
      <p class="small mute">Dashed: α thresholds (${p.subdiffusion_threshold}, ${p.superdiffusion_threshold}). Stacked by assigned class.</p></div>
    <div class="panel"><h3><span class="gk">α</span> against D</h3>${scatter(t.map(q => ({ x: num(q.D), y: num(q.alpha), c: q.mobility_class })), { xlabel: 'D (µm²/s)', ylabel: 'α', xlog: true, lines: { y: [p.subdiffusion_threshold, p.superdiffusion_threshold] } })}</div>
    <div class="panel"><h3>Directedness</h3>${histogram(t.map(q => ({ v: num(q.directedness), c: q.mobility_class })), { xlabel: 'directedness (end-to-end / path length)', range: [0, 1], bins: 20, marks: [{ x: p.directedness_threshold, label: 'directed' }] })}</div>
    <div class="panel"><h3>Confinement level against track length</h3>${scatter(t.map(q => ({ x: q.track_length, y: num(q.confinement_level), c: q.mobility_class })), { xlabel: 'points in track', ylabel: 'confinement level −log10 ψ', lines: { y: [p.confinement_level_threshold] } })}
      <p class="small mute">Dashed yellow: confined above ${p.confinement_level_threshold}. For free diffusion the level does not depend on
        track length; for confined motion it grows with duration. See <a href="#/method">Method</a>.</p></div>
    <div class="panel"><h3>Legacy ratio against track length</h3>${scatter(t.map(q => ({ x: q.track_length, y: num(q.confinement_ratio_legacy), c: q.mobility_class })), { xlabel: 'points in track', ylabel: 'end-to-end / (Rg √n)', lines: { y: [0.3] }, curve: line, curveLabel: 'straight line of n points' })}
      <p class="small mute">The former criterion (confined below 0.3), kept for comparison only. Dotted white: a perfectly straight track,
        √(12(n−1)/(n+1))/√n; the ratio falls as 1/√n for any motion, which is why it was replaced.</p></div>
  </div>`;
});
