// Pure JS port of the repo's mobility analysis (mobility_analyzer.py, MobilityAnalyzer) as the web app calls it
// (web/analysis.py): per-track diffusion coefficient, anomalous exponent, directedness, confinement level (and the legacy ratio) and class
// (get_mobility_summary), and the per-class ensemble MSD (_calculate_msd_manual + fit_msd_model 'powerlaw').
// engine.test.mjs checks it value by value against the Python code (tests/fixtures/engine_cases.json).
//
// Units: x / y in pixels (pixel_size_um, µm/pixel) or in µm; frame_interval_s in seconds; fps = 1 / frame_interval_s.
// As in MobilityAnalyzer (mobility_metrics.frame_msd), MSD lags are frame differences, so tracks with gaps get
// correct lag times.

export const CLASSES = ['brownian', 'confined', 'directed', 'subdiffusive', 'superdiffusive', 'unknown'];
export const D_FIT_LAGS = 10, ENSEMBLE_MAX_LAG = 100;
export const COLUMNS = {
  particle: ['particle', 'track_id', 'track_number', 'track', 'TRACK_ID', 'id'],
  frame: ['frame', 'FRAME', 't', 'frame_number'],
  x: ['x', 'POSITION_X', 'x_um', 'x [px]'],
  y: ['y', 'POSITION_Y', 'y_um', 'y [px]'],
};
export const LABEL_COLUMN = 'true_mobility';

// JSON numbers: null = NaN.
export const num = v => (v === null || v === undefined ? NaN : v);
const jsonNum = v => (Number.isFinite(v) ? v : null);
const mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };

// scipy.stats.linregress (slope, intercept, r)
export function linregress(x, y) {
  const n = x.length, xm = mean(x), ym = mean(y);
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) { const dx = x[i] - xm, dy = y[i] - ym; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
  sxx /= n; sxy /= n; syy /= n;
  const slope = sxy / sxx;
  let r = sxx === 0 || syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
  r = Math.max(-1, Math.min(1, r));
  return { slope, intercept: ym - slope * xm, r };
}

// mobility_metrics.frame_msd: time-averaged MSD (µm²) at frame lags 1..maxLag over all pairs exactly `lag` frames
// apart (frames sorted ascending; a repeated frame keeps its first localisation). Returns {lags, msds}.
export function frameMsd(frames, x, y, maxLag) {
  const out = { lags: [], msds: [] };
  if (!frames.length) return out;
  const f = frames.map(Math.round), f0 = f[0], span = f[f.length - 1] - f0;
  const X = new Float64Array(span + 1).fill(NaN), Y = new Float64Array(span + 1).fill(NaN);
  f.forEach((v, i) => { if (Number.isNaN(X[v - f0])) { X[v - f0] = x[i]; Y[v - f0] = y[i]; } });
  for (let lag = 1; lag <= Math.min(maxLag, span); lag++) {
    let s = 0, k = 0;
    for (let i = 0; i + lag <= span; i++) {
      const dx = X[i + lag] - X[i], dy = Y[i + lag] - Y[i], d = dx * dx + dy * dy;
      if (!Number.isNaN(d)) { s += d; k++; }
    }
    if (k) { out.lags.push(lag); out.msds.push(s / k); }
  }
  return out;
}
export const frameSpan = frames => (frames.length ? Math.round(frames[frames.length - 1]) - Math.round(frames[0]) + 1 : 0);

// calculate_diffusion_coefficients: linear fit MSD = 4 D t over frame lags 1..10 (needs > 2 lags).
export function diffusionCoefficient(frames, x, y, fps, maxLag = D_FIT_LAGS) {
  const { lags, msds } = frameMsd(frames, x, y, maxLag);
  if (msds.length <= 2) return NaN;
  return linregress(lags.map(l => l / fps), msds).slope / 4;
}

// calculate_anomalous_exponent: log-log fit over frame lags 1..min(span//2, 50)-1 with MSD > 0 (needs > 5 lags).
export function anomalousExponent(frames, x, y, fps) {
  const maxLag = Math.min(Math.floor(frameSpan(frames) / 2), 50), lt = [], lm = [];
  const { lags, msds } = frameMsd(frames, x, y, maxLag - 1);
  lags.forEach((lag, i) => { if (msds[i] > 0) { lt.push(Math.log(lag / fps)); lm.push(Math.log(msds[i])); } });
  return lm.length > 5 ? linregress(lt, lm).slope : NaN;
}

// calculate_directedness: end-to-end distance / path length (0 if the path length is 0).
export function directedness(x, y) {
  const n = x.length, e = Math.sqrt((x[n - 1] - x[0]) ** 2 + (y[n - 1] - y[0]) ** 2);
  let p = 0;
  for (let i = 1; i < n; i++) p += Math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2);
  return p > 0 ? e / p : 0;
}

// mobility_metrics.confinement_level (Saxton 1993, doi:10.1016/S0006-3495(93)81548-0; Simson et al. 1995,
// doi:10.1016/S0006-3495(95)79972-6): -log10 psi, log10 psi = 0.2048 - 2.5117 D t / R², D from MSD frame lags 1-4,
// t = track duration, R = maximum distance from the first position. NaN if R = 0 or D undefined.
export const CONFINEMENT_FIT_LAGS = 4;
export function confinementLevel(frames, x, y, fps) {
  if (x.length < 2) return NaN;
  const { lags, msds } = frameMsd(frames, x, y, CONFINEMENT_FIT_LAGS);
  if (msds.length < 3) return NaN;
  const t = lags.map(l => l / fps), tm = mean(t), mm = mean(msds);
  let sxy = 0, sxx = 0;
  t.forEach((v, i) => { sxy += (v - tm) * (msds[i] - mm); sxx += (v - tm) ** 2; });
  const D = sxy / sxx / 4, dur = (Math.round(frames[frames.length - 1]) - Math.round(frames[0])) / fps;
  let r2 = 0;
  for (let i = 0; i < x.length; i++) r2 = Math.max(r2, (x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2);
  if (!Number.isFinite(D) || r2 === 0) return NaN;
  return 2.5117 * D * dur / r2 - 0.2048;
}

// LEGACY calculate_confinement_ratio: end-to-end / (Rg * sqrt(n)), dimensionless (no longer used to classify).
export function confinementRatio(x, y) {
  const n = x.length, xm = mean(x), ym = mean(y);
  let s = 0;
  for (let i = 0; i < n; i++) s += (x[i] - xm) ** 2 + (y[i] - ym) ** 2;
  const rg = Math.sqrt(s / n), e = Math.sqrt((x[n - 1] - x[0]) ** 2 + (y[n - 1] - y[0]) ** 2);
  return e / (rg * Math.sqrt(n));
}

// classify_mobility (order matters: unknown, confined, directed, sub-, superdiffusive, Brownian).
export function classify(alpha, level, dir, p) {
  if (Number.isNaN(alpha)) return 'unknown';
  if (level > p.confinement_level_threshold) return 'confined';
  if (dir > p.directedness_threshold) return 'directed';
  if (alpha < p.subdiffusion_threshold) return 'subdiffusive';
  if (alpha > p.superdiffusion_threshold) return 'superdiffusive';
  return 'brownian';
}

export function scaleOf(p) {
  return { pixelSize: p.coord_unit === 'um' ? 1 : p.pixel_size_um, fps: 1 / p.frame_interval_s };
}

// Tracks in the order of first appearance, each sorted by frame: [{id, frames, xs, ys, label}] (table units).
export function toTracks(table) {
  const idx = {};
  for (const [std, names] of Object.entries(COLUMNS)) {
    const hit = names.find(n => table.columns.includes(n));
    if (hit !== undefined) idx[std] = table.columns.indexOf(hit);
  }
  const missing = Object.keys(COLUMNS).filter(k => idx[k] === undefined);
  if (missing.length) throw new Error(`Missing required columns: ${missing.join(', ')} (found: ${table.columns.join(', ')})`);
  const li = table.columns.indexOf(LABEL_COLUMN);
  const numv = s => (s === undefined || String(s).trim() === '' ? NaN : Number(s));
  const map = new Map();
  for (const row of table.rows) {
    const fr = numv(row[idx.frame]), x = numv(row[idx.x]), y = numv(row[idx.y]);
    if (Number.isNaN(fr) || Number.isNaN(x) || Number.isNaN(y)) continue;
    const id = String(row[idx.particle]);
    if (!map.has(id)) map.set(id, { id, pts: [], label: li >= 0 ? String(row[li]) : undefined });
    map.get(id).pts.push([fr, x, y]);
  }
  if (!map.size) throw new Error('No rows with numeric frame, x and y values.');
  return [...map.values()].map(t => {
    t.pts.sort((a, b) => a[0] - b[0]);
    return { id: t.id, label: t.label, frames: t.pts.map(q => q[0]), xs: t.pts.map(q => q[1]), ys: t.pts.map(q => q[2]) };
  });
}

// Coordinates object as returned by the API ({id: [frames, xs, ys]}) -> track list.
export function fromCoords(coords, labels = null) {
  return Object.entries(coords).map(([id, [frames, xs, ys]]) => ({ id, frames, xs, ys, label: labels ? labels[id] : undefined }));
}
export const toCoords = tracks => Object.fromEntries(tracks.map(t => [t.id, [t.frames, t.xs, t.ys]]));

export function frameWarnings(tracks) {
  let gaps = 0, dup = 0;
  for (const t of tracks) {
    let g = false, d = false;
    for (let i = 1; i < t.frames.length; i++) { const s = t.frames[i] - t.frames[i - 1]; if (s === 0) d = true; if (s > 1) g = true; }
    gaps += g; dup += d;
  }
  return { tracks_with_gaps: gaps, tracks_with_repeated_frames: dup };
}

// fit_msd_model(msd, 'powerlaw'): MSD = 4 D t^alpha by a log-log line (needs >= 3 valid points).
export function fitPowerLaw(lags, msds) {
  const t = [], m = [];
  lags.forEach((v, i) => { if (msds[i] > 0 && v > 0) { t.push(Math.log(v)); m.push(Math.log(msds[i])); } });
  if (t.length < 3) return { D: NaN, alpha: NaN, r_squared: NaN };
  const r = linregress(t, m);
  return { D: Math.exp(r.intercept) / 4, alpha: r.slope, r_squared: r.r * r.r };
}

// _calculate_msd_manual: per-track MSD at lags 1..min(maxLag, n)-1, averaged over the tracks reaching each lag.
export function ensembleMsd(tracks, pixelSize, fps, maxLag = ENSEMBLE_MAX_LAG) {
  const dt = 1 / fps, sums = new Map();
  for (const t of tracks) {
    const x = t.xs.map(v => v * pixelSize), y = t.ys.map(v => v * pixelSize);
    const m = frameMsd(t.frames, x, y, maxLag - 1);
    m.lags.forEach((lag, i) => {
      const key = lag * dt, e = sums.get(key) || [0, 0];
      e[0] += m.msds[i]; e[1]++;
      sums.set(key, e);
    });
  }
  const lags = [...sums.keys()].sort((a, b) => a - b);
  return { lags, msds: lags.map(k => sums.get(k)[0] / sums.get(k)[1]) };
}

// Per-track metrics for one track (µm, s).
export function trackMetrics(t, p) {
  const { pixelSize, fps } = scaleOf(p);
  const x = t.xs.map(v => v * pixelSize), y = t.ys.map(v => v * pixelSize);
  const D = diffusionCoefficient(t.frames, x, y, fps), alpha = anomalousExponent(t.frames, x, y, fps);
  const dir = directedness(x, y), level = confinementLevel(t.frames, x, y, fps), ratio = confinementRatio(x, y);
  const out = { particle: t.id, track_length: x.length, D: jsonNum(D), alpha: jsonNum(alpha), directedness: jsonNum(dir),
    confinement_level: jsonNum(level), confinement_ratio_legacy: jsonNum(ratio), mobility_class: classify(alpha, level, dir, p) };
  if (t.label !== undefined) out.true_mobility = t.label;
  return out;
}

// The whole analysis, same JSON shape as web/analysis.py analyse().
export function analyse(tracks, p, name = 'tracks.csv') {
  if (!tracks.length) throw new Error('The table has no rows.');
  const rows = tracks.map(t => trackMetrics(t, p));
  const { pixelSize, fps } = scaleOf(p), ensemble = {};
  for (const c of CLASSES) {
    const sub = tracks.filter((_, i) => rows[i].mobility_class === c);
    if (!sub.length) continue;
    const e = ensembleMsd(sub, pixelSize, fps);
    if (!e.lags.length) continue;
    const fit = fitPowerLaw(e.lags, e.msds);
    ensemble[c] = { n_tracks: sub.length, lag_s: e.lags, msd_um2: e.msds.map(jsonNum),
      fit: { D: jsonNum(fit.D), alpha: jsonNum(fit.alpha), r_squared: jsonNum(fit.r_squared) } };
  }
  return { name, params: { ...p }, computed_by: 'browser', n_rows: tracks.reduce((s, t) => s + t.xs.length, 0),
    n_tracks: rows.length, has_labels: tracks.some(t => t.label !== undefined), d_fit_lags: D_FIT_LAGS,
    ensemble_max_lag: ENSEMBLE_MAX_LAG, warnings: frameWarnings(tracks), tracks: rows, ensemble };
}

// Validate form parameters (same rules as web/analysis.py parse_params).
export function checkParams(p) {
  for (const [k, v] of Object.entries(p)) if (k !== 'coord_unit' && !Number.isFinite(v)) throw new Error(`${k} must be a number`);
  if (!['px', 'um'].includes(p.coord_unit)) throw new Error("coord_unit must be 'px' or 'um'");
  if (p.coord_unit === 'px' && !(p.pixel_size_um > 0)) throw new Error('pixel_size_um must be > 0');
  if (!(p.frame_interval_s > 0)) throw new Error('frame_interval_s must be > 0');
  return p;
}

// Minimal RFC 4180 CSV reader -> {columns, rows}.
export function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else cell += c;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) throw new Error('The file is empty.');
  return { columns: rows[0].map(s => s.trim()), rows: rows.slice(1) };
}

// Track table -> CSV text (particle, frame, x, y[, true_mobility]).
export function tracksCsv(tracks) {
  const lab = tracks.some(t => t.label !== undefined);
  const out = ['particle,frame,x,y' + (lab ? ',true_mobility' : '')];
  for (const t of tracks) t.frames.forEach((fr, i) => out.push(`${t.id},${fr},${t.xs[i]},${t.ys[i]}${lab ? ',' + (t.label ?? '') : ''}`));
  return out.join('\n') + '\n';
}

// Re-apply the class rule with other thresholds to already computed metrics (alpha, confinement level, directedness do not
// depend on the thresholds), as the Validation page's threshold sliders do.
export function reclassify(rows, p) {
  return rows.map(r => ({ ...r, mobility_class: classify(num(r.alpha), num(r.confinement_level), num(r.directedness), p) }));
}
