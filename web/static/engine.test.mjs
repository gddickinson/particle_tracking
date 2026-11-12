// Node test: engine.js reproduces the Python analysis (web/analysis.py -> MobilityAnalyzer) on the same CSV.
// Fixture: tests/fixtures/engine_cases.json (scripts/make_engine_fixture.py). Run: node web/static/engine.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { analyse, checkParams, classify, confinementLevel, confinementRatio, diffusionCoefficient, frameMsd, num, parseCsv, toTracks, tracksCsv } from './engine.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/engine_cases.json', import.meta.url), 'utf8'));
const close = (a, b, what, tol = 1e-9) => {
  const x = num(a), y = num(b);
  if (Number.isNaN(y)) return assert.ok(Number.isNaN(x), `${what}: expected NaN, got ${x}`);
  assert.ok(Math.abs(x - y) <= tol * Math.max(1, Math.abs(y)), `${what}: ${x} vs ${y}`);
};
let checks = 0;
const tracks = toTracks(parseCsv(fx.csv));

for (const py of fx.runs) {
  const js = analyse(tracks, py.params, py.name);
  const tag = `${py.params.coord_unit}/${py.params.frame_interval_s}s`;
  assert.equal(js.n_tracks, py.n_tracks); assert.equal(js.n_rows, py.n_rows);
  assert.deepEqual(js.warnings, py.warnings);
  assert.equal(js.has_labels, py.has_labels);
  py.tracks.forEach((p, i) => {
    const j = js.tracks[i];
    assert.equal(j.particle, p.particle, `${tag} track order`);
    assert.equal(j.track_length, p.track_length);
    assert.equal(j.true_mobility, p.true_mobility);
    for (const k of ['D', 'alpha', 'directedness']) { close(j[k], p[k], `${tag} ${p.particle} ${k}`); checks++; }
    // A stationary track has end-to-end 0 and Rg 0 up to rounding: Python and JS may give 0 or NaN (both unconfined
    // for alpha = NaN, so the class is 'unknown' either way).
    if (p.particle === 'still') assert.ok(!(j.confinement_ratio_legacy > 0)); else { close(j.confinement_ratio_legacy, p.confinement_ratio_legacy, `${tag} ${p.particle} ratio`); checks++; }
    close(j.confinement_level, p.confinement_level, `${tag} ${p.particle} level`); checks++;
    assert.equal(j.mobility_class, p.mobility_class, `${tag} ${p.particle} class`); checks++;
  });
  assert.deepEqual(Object.keys(js.ensemble), Object.keys(py.ensemble), `${tag} ensemble classes`);
  for (const [c, e] of Object.entries(py.ensemble)) {
    const j = js.ensemble[c];
    assert.equal(j.n_tracks, e.n_tracks);
    assert.equal(j.lag_s.length, e.lag_s.length, `${tag} ${c} lags`);
    e.lag_s.forEach((v, i) => { close(j.lag_s[i], v, `${tag} ${c} lag ${i}`); close(j.msd_um2[i], e.msd_um2[i], `${tag} ${c} msd ${i}`); checks += 2; });
    for (const k of ['D', 'alpha', 'r_squared']) { close(j.fit[k], e.fit[k], `${tag} ${c} fit ${k}`); checks++; }
  }
}
assert.ok(new Set(fx.runs[0].tracks.map(t => t.mobility_class)).size >= 5, 'fixture covers most classes');

// Classification order and helpers.
const P = fx.runs[0].params;
assert.equal(classify(NaN, 0, 1, P), 'unknown');
assert.equal(classify(1, 5, 0.9, P), 'confined');
assert.equal(classify(1, 1, 0.9, P), 'directed');
assert.equal(classify(1, NaN, 0.9, P), 'directed');
assert.equal(classify(0.5, 1, 0.1, P), 'subdiffusive');
assert.equal(classify(1.5, 1, 0.1, P), 'superdiffusive');
assert.equal(classify(1.0, 1, 0.1, P), 'brownian');
// Straight lines: never confined, whatever their length.
for (const n of [30, 200, 1000]) {
  const fr = Array.from({ length: n }, (_, i) => i);
  assert.ok(confinementLevel(fr, fr.map(v => v * 0.1), fr.map(() => 0), 10) < P.confinement_level_threshold, `line n=${n}`); checks++;
}
// A straight, evenly spaced line: ratio = sqrt(12 (n-1)/(n+1)) / sqrt(n).
for (const n of [10, 50, 200]) {
  const xs = Array.from({ length: n }, (_, i) => i), ys = xs.map(() => 0);
  close(confinementRatio(xs, ys), Math.sqrt(12 * (n - 1) / (n + 1)) / Math.sqrt(n), `line ratio n=${n}`); checks++;
}
// Frame lags: a constant-speed line with missing frames has MSD = lag² and the same D as the complete line.
const gapF = [0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24];
const fullF = Array.from({ length: 25 }, (_, i) => i);
assert.deepEqual(frameMsd(gapF, gapF, gapF.map(() => 0), 4).msds, [1, 4, 9, 16]);
close(diffusionCoefficient(gapF, gapF, gapF.map(() => 0), 1), diffusionCoefficient(fullF, fullF, fullF.map(() => 0), 1), 'gapped line D'); checks++;
// CSV round trip, quoting and errors.
const again = toTracks(parseCsv(tracksCsv(tracks)));
assert.deepEqual(again.map(t => t.id), tracks.map(t => t.id));
assert.deepEqual(parseCsv('a,b\n"x, ""y""",2\r\n3,\n').rows, [['x, "y"', '2'], ['3', '']]);
assert.throws(() => toTracks(parseCsv('particle,frame,x\n1,0,1\n')), /Missing required columns: y/);
assert.equal(toTracks(parseCsv('TRACK_ID,FRAME,POSITION_X,POSITION_Y\n1,0,1,2\n1,1,,3\n1,2,2,2\n'))[0].xs.length, 2);
assert.throws(() => checkParams({ ...P, frame_interval_s: 0 }), /frame_interval_s/);
assert.throws(() => checkParams({ ...P, pixel_size_um: -1 }), /pixel_size_um/);

console.log(`engine.test.mjs: ok (${checks} checks, ${tracks.length} tracks, ${fx.runs.length} runs)`);
