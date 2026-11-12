// The current analysis result, shared by the Analyze, Tracks, Populations and Validation pages. Until the user runs
// an analysis, the first demo dataset analysed by the Python code (default thresholds) is shown.
import { api } from './core.js';
import { fromCoords } from './engine.js';

let info = null, current = null;
export async function getInfo() { return info || (info = await api('/info')); }

export async function demoResult(id) {
  return { ...(await api(`/demo/${id}/result`)), coords_ref: `/demo/${id}/tracks` };
}

export function setResult(r) { current = r; return r; }

export async function getResult() {
  if (current) return current;
  const i = await getInfo();
  return setResult(await demoResult(i.demos[0].id));
}

// Track list [{id, frames, xs, ys, label}] of a result (table units), in the result's track order.
export async function tracksOf(r) {
  if (r._tracks) return r._tracks;
  let list;
  if (r.coords) list = fromCoords(r.coords);
  else { const d = await api(r.coords_ref); list = fromCoords(d.coords, d.labels); }
  const byId = new Map(list.map(t => [t.id, t]));
  r._tracks = r.tracks.map(t => byId.get(t.particle)).filter(Boolean);
  return r._tracks;
}

export function sourceLine(r) {
  const s = r.source || {};
  return `${s.title || r.name} · ${r.n_tracks} tracks · ${r.n_rows.toLocaleString()} localisations`;
}
