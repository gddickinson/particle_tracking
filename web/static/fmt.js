// Formatting, class colours and names, units, per-track CSV export.
import { esc } from './core.js';
import { CLASSES, num } from './engine.js';

export const COLORS = {
  brownian: '#59b8ff', confined: '#45f0a0', directed: '#ff5c7a', subdiffusive: '#ffb02e', superdiffusive: '#c58cff',
  unknown: '#7d89a8',
};
export const NAMES = {
  brownian: 'Brownian', confined: 'confined', directed: 'directed', subdiffusive: 'subdiffusive',
  superdiffusive: 'superdiffusive', unknown: 'unknown (too short for α)',
};
export const TRUE_NAMES = { ...NAMES, edge: 'edge case' };
export const nice = s => NAMES[s] || String(s ?? '');
export const cls = s => `<span class="cls" style="--c:${COLORS[s] || '#94a3c6'}">${esc(nice(s))}</span>`;
export const legend = (keys = CLASSES) => `<div class="legend">${keys.map(cls).join('')}</div>`;

// NaN/null -> '–', otherwise 4 significant digits (integers as is).
export function f(v, digits = 4) {
  const x = num(v);
  if (Number.isNaN(x)) return '–';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '-∞';
  if (Number.isInteger(x) && Math.abs(x) < 1e6) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(digits - 1);
  return String(Number(x.toPrecision(digits)));
}
export const pct = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : '–');
export function median(values) {
  const v = values.map(num).filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// Units line for a parameter set.
export function unitsText(p) {
  const xy = p.coord_unit === 'um' ? 'x / y in µm' : `x / y in pixels × ${f(p.pixel_size_um)} µm/pixel`;
  return `${xy}; frame interval ${f(p.frame_interval_s)} s (${f(1 / p.frame_interval_s)} Hz)`;
}
export const umPerUnit = p => (p.coord_unit === 'um' ? 1 : p.pixel_size_um);

export function provenance(r) {
  return r.computed_by === 'python'
    ? '<span class="badge py">computed by the Python MobilityAnalyzer</span>'
    : '<span class="badge js">computed in your browser (engine.js, tested against the Python code)</span>';
}

export const TRACK_COLUMNS = [
  ['particle', 'particle'], ['track_length', 'points'], ['D', 'D (µm²/s)'], ['alpha', 'α'],
  ['directedness', 'directedness'], ['confinement_level', 'confinement level'], ['confinement_ratio_legacy', 'legacy ratio'],
  ['mobility_class', 'class'],
];

export function trackCsv(r) {
  const head = ['particle', 'track_length', 'diffusion_coefficient_um2_per_s', 'anomalous_exponent', 'directedness',
    'confinement_level', 'confinement_ratio_legacy', 'mobility_class'].concat(r.has_labels ? ['true_mobility'] : []);
  const q = s => (/[",\n]/.test(s) ? `"${String(s).replace(/"/g, '""')}"` : s);
  const rows = r.tracks.map(t => [q(t.particle), t.track_length, t.D ?? '', t.alpha ?? '', t.directedness ?? '',
    t.confinement_level ?? '', t.confinement_ratio_legacy ?? '', t.mobility_class].concat(r.has_labels ? [q(t.true_mobility ?? '')] : []).join(','));
  const p = r.params;
  const meta = `# ${r.name}; ${unitsText(p)}; thresholds: alpha < ${p.subdiffusion_threshold} sub, > ${p.superdiffusion_threshold} super, ` +
    `confinement level > ${p.confinement_level_threshold}, directedness > ${p.directedness_threshold}; ${r.computed_by}`;
  return [meta, head.join(','), ...rows].join('\n') + '\n';
}
