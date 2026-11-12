// Analyze: data source, units and thresholds -> run the analysis (Python server, or engine.js on the static site)
// -> per-class summary, ensemble fits, warnings and exports.
import { $, api, app, download, esc, isStatic, list, route } from './core.js';
import { CLASSES, tracksCsv } from './engine.js';
import { COLORS, cls, f, median, pct, provenance, trackCsv, unitsText } from './fmt.js';
import { getInfo, getResult, setResult, sourceLine, tracksOf } from './state.js';

const num = (id, label, unit, attrs, hint = '') => `<label for="${id}">${label} <span class="unit">${unit}</span></label>
  <input id="${id}" name="${id}" type="number" ${attrs}>${hint ? `<p class="hint">${hint}</p>` : ''}`;
const KEYS = ['coord_unit', 'pixel_size_um', 'frame_interval_s', 'subdiffusion_threshold', 'superdiffusion_threshold',
  'confinement_threshold', 'directedness_threshold'];

route('analyze', async () => {
  const [info, result] = await Promise.all([getInfo(), getResult()]);
  const d = info.defaults, cols = info.columns;
  app().innerHTML = `<p class="kicker">Run the analysis</p><h1>Analyze tracks</h1>
  <p class="lead">Per track: D from a linear fit of the first ${info.d_fit_lags} MSD points, α from a log-log fit,
    directedness and a confinement ratio, then a class. ${isStatic() ? 'Runs in your browser.' : 'Runs <code>MobilityAnalyzer.get_mobility_summary</code> in Python.'}</p>
  <div class="side">
  <form class="params panel" id="pf" novalidate>
    <fieldset class="src"><legend>Data</legend>
      ${list(info.demos, x => `<label><input type="radio" name="source" value="${x.id}"> <span>${esc(x.title)}<br><small class="mute">${x.n_tracks} tracks · labelled</small></span></label>`)}
      <label><input type="radio" name="source" value="upload"> <span>Upload a CSV file</span></label>
      <div id="upwrap" class="hide"><label for="file">Track CSV</label>
        <input id="file" name="file" type="file" accept=".csv,text/csv">
        <p class="hint">Columns (first match): particle = ${cols.particle.map(esc).join(' / ')}; frame = ${cols.frame.map(esc).join(' / ')};
          x = ${cols.x.map(esc).join(' / ')}; y = ${cols.y.map(esc).join(' / ')}. Optional <code>${esc(info.label_column)}</code> enables the Validation page.</p></div>
    </fieldset>
    <fieldset><legend>Units</legend>
      <label for="coord_unit">x / y columns are in</label>
      <select id="coord_unit" name="coord_unit"><option value="px">pixels</option><option value="um">µm</option></select>
      ${num('pixel_size_um', 'Pixel size', 'µm / pixel', 'min="0" step="any"')}
      ${num('frame_interval_s', 'Frame interval', 's / frame', 'min="0" step="any"', 'fps = 1 / frame interval. D comes out in µm²/s.')}
    </fieldset>
    <fieldset><legend>Class thresholds</legend>
      ${num('subdiffusion_threshold', 'Subdiffusive below', 'α', 'step="any"')}
      ${num('superdiffusion_threshold', 'Superdiffusive above', 'α', 'step="any"')}
      ${num('confinement_threshold', 'Confined below', 'confinement ratio', 'min="0" step="any"')}
      ${num('directedness_threshold', 'Directed above', 'directedness, 0–1', 'min="0" max="1" step="any"')}
    </fieldset>
    <button type="submit" id="run">Run analysis</button><button type="button" class="alt" id="reset">Default thresholds</button>
    <p id="msg" class="small" role="status"></p>
  </form>
  <div id="out"></div></div>`;

  const form = $('#pf');
  const set = (id, v) => { $('#' + id).value = v ?? ''; };
  const toggle = () => {
    $('#upwrap').classList.toggle('hide', form.source.value !== 'upload');
    $('#pixel_size_um').disabled = $('#coord_unit').value !== 'px';
  };
  for (const k of KEYS) set(k, result.params[k]);
  (form.querySelector(`input[name=source][value="${result.source?.id}"]`) || form.querySelector('input[name=source][value=upload]')).checked = true;
  toggle();
  form.addEventListener('change', e => {
    if (e.target.name === 'source' && e.target.value !== 'upload') {
      const m = info.demos.find(x => x.id === e.target.value);
      set('coord_unit', m.coord_unit); set('pixel_size_um', m.pixel_size_um); set('frame_interval_s', m.frame_interval_s);
    }
    toggle();
  });
  $('#reset').onclick = () => { for (const k of KEYS.slice(3)) set(k, d[k]); };

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(), msg = $('#msg');
    if (form.source.value === 'upload') {
      const file = $('#file').files[0];
      if (!file) { msg.innerHTML = '<span class="err">Choose a CSV file first.</span>'; return; }
      fd.append('file', file);
    } else fd.append('demo', form.source.value);
    for (const k of KEYS) {
      const v = $('#' + k).value;
      if (v === '' && !(k === 'pixel_size_um' && $('#coord_unit').value === 'um')) { msg.innerHTML = `<span class="err">${esc(k)} is empty.</span>`; return; }
      fd.append(k, v);
    }
    $('#run').disabled = true; msg.textContent = 'Running...';
    try {
      const r = setResult(await api('/analyze', { body: fd }));
      msg.innerHTML = `<span class="ok">Done: ${r.n_tracks} tracks analysed.</span>`;
      show(r);
    } catch (err) {
      msg.innerHTML = `<span class="err">${esc(err.message)}</span>`;
    } finally { $('#run').disabled = false; }
  });
  show(result);
});

function show(r) {
  const p = r.params, n = r.tracks.length, w = r.warnings || {};
  const by = c => r.tracks.filter(t => t.mobility_class === c);
  const present = CLASSES.filter(c => by(c).length);
  $('#out').innerHTML = `${provenance(r)}
  <p class="small mute">${esc(sourceLine(r))}<br>${esc(unitsText(p))}</p>
  <div class="tiles">${list(present.slice(0, 6), c => `<div class="tile" style="border-left-color:${COLORS[c]}"><b>${by(c).length}</b><span>${cls(c)} ${pct(by(c).length, n)}</span></div>`)}</div>
  ${w.tracks_with_gaps ? `<p class="small mute">${w.tracks_with_gaps} tracks have gaps in their frame numbers; MSD lags are frame differences, so their lag times are correct.</p>` : ''}
  ${w.tracks_with_repeated_frames ? `<p class="callout small">${w.tracks_with_repeated_frames} tracks repeat a frame number: only the first localisation of a repeated frame is used for the MSD.</p>` : ''}
  <div class="scroll"><table class="tbl"><thead><tr><th>class</th><th>tracks</th><th>median D <span class="unit">µm²/s</span></th><th>median α</th>
    <th>ensemble D <span class="unit">µm²/s</span></th><th>ensemble α</th><th>median points</th></tr></thead><tbody>
    ${list(present, c => { const t = by(c), e = r.ensemble[c]?.fit || {};
      return `<tr><td>${cls(c)}</td><td>${t.length}</td><td>${f(median(t.map(x => x.D)))}</td><td>${f(median(t.map(x => x.alpha)), 3)}</td>
        <td>${f(e.D)}</td><td>${f(e.alpha, 3)}</td><td>${f(median(t.map(x => x.track_length)))}</td></tr>`; })}
  </tbody></table></div>
  <p class="small mute">Per track, D is the slope / 4 of MSD against t over lags 1–${r.d_fit_lags} (MSD = 4Dt) and α the slope of
    log MSD against log t over lags up to half the track (at most 49). Ensemble: the mean of the per-track MSD curves of a class
    (lags up to ${r.ensemble_max_lag - 1}) fitted as MSD = 4D t<sup>α</sup>.</p>
  <h3>Export</h3>
  <button type="button" id="x-tracks">Per-track CSV</button>
  <button type="button" class="alt" id="x-json">Result JSON</button>
  <button type="button" class="alt" id="x-input">Track table CSV</button>
  <p><a href="#/tracks">Tracks →</a> · <a href="#/populations">Populations →</a>${r.has_labels ? ' · <a href="#/validate">Validation →</a>' : ''}</p>`;
  const stem = (r.name || 'tracks').replace(/\.csv$/i, '');
  $('#x-tracks').onclick = () => download(`${stem}_mobility.csv`, trackCsv(r), 'text/csv');
  $('#x-json').onclick = () => download(`${stem}_mobility.json`, JSON.stringify({ ...r, coords: undefined, _tracks: undefined }), 'application/json');
  $('#x-input').onclick = async () => download(`${stem}_tracks.csv`, tracksCsv(await tracksOf(r)), 'text/csv');
}
