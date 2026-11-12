// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs the
// analysis itself in the browser (engine.js, tested against the Python MobilityAnalyzer) for uploads and for demos
// analysed with other units or thresholds. Nothing is sent anywhere: an uploaded file stays on your computer.
import { analyse, checkParams, fromCoords, parseCsv, toCoords, toTracks } from './engine.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'analyze') return analyze(body);
  return file(clean);
}

const NUMS = ['pixel_size_um', 'frame_interval_s', 'subdiffusion_threshold', 'superdiffusion_threshold',
  'confinement_threshold', 'directedness_threshold'];

async function analyze(fd) {
  const defaults = (await file('info')).defaults, p = { coord_unit: fd.get('coord_unit') || defaults.coord_unit };
  for (const k of NUMS) { const v = fd.get(k); p[k] = v === null || v === '' ? defaults[k] : Number(v); }
  checkParams(p);
  const upload = fd.get('file');
  if (upload && upload.name) {
    const tracks = toTracks(parseCsv(await upload.text()));
    const r = analyse(tracks, p, upload.name);
    r.source = { id: 'upload', title: upload.name, description: 'Uploaded file (analysed in your browser)' };
    r.coords = toCoords(tracks);
    return r;
  }
  const demo = fd.get('demo'), info = await file('info'), meta = info.demos.find(d => d.id === demo);
  if (!meta) throw new Error('Choose a demo dataset or upload a CSV file.');
  const d = await file(`demo/${demo}/tracks`);
  const r = analyse(fromCoords(d.coords, d.labels), p, `${demo}.csv`);
  r.source = meta;
  r.coords_ref = `/demo/${demo}/tracks`;
  return r;
}
