// Home: what the tool does, live counts from the demo analysis, links to the pages.
import { app, isStatic, list, route } from './core.js';
import { CLASSES } from './engine.js';
import { cls, pct } from './fmt.js';
import { demoResult, getInfo } from './state.js';

route('home', async () => {
  const info = await getInfo();
  const first = await demoResult(info.demos[0].id);
  const nTracks = info.demos.reduce((s, d) => s + d.n_tracks, 0);
  const agree = first.tracks.filter(t => t.mobility_class === t.true_mobility).length;
  app().innerHTML = `<section class="hero">
    <p class="kicker">Single-particle tracking · mixed mobility</p>
    <h1 class="glow">Particle Tracking</h1>
    <p class="lead">Mobility analysis for fluorescently-tagged proteins with mixed behaviours: mean square displacement,
      diffusion coefficients (µm²/s), anomalous exponents α and a class per track (Brownian, confined, directed,
      subdiffusive, superdiffusive). This is the repo's <code>MobilityAnalyzer</code>, run on your own track CSV.</p>
    <a class="btn" href="#/analyze">Analyze tracks</a><a class="btn alt" href="#/validate">See how well it classifies</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>5</b><span>motion classes (+ unknown when a track is too short for α)</span></div>
    <div class="tile"><b>${info.demos.length}</b><span>labelled demo datasets, ${nTracks} simulated tracks</span></div>
    <div class="tile"><b>${pct(agree, first.tracks.length)}</b><span>of “${info.demos[0].title}” tracks get their simulated class (default thresholds)</span></div>
    <div class="tile"><b>µm · s</b><span>explicit units: pixel size in µm/pixel, frame interval in s</span></div>
  </div>
  <div class="grid">
    <a class="card" href="#/analyze"><b>Analyze</b><small>upload or demo</small><span>Load a track table (particle, frame, x, y), set the units and thresholds, get per-class counts and ensemble fits.</span></a>
    <a class="card" href="#/tracks"><b>Tracks</b><small>map and per-track MSD</small><span>Every track coloured by class; pick one to see its trajectory, MSD curve and the two fits behind D and α.</span></a>
    <a class="card" href="#/populations"><b>Populations</b><small>mixed mobility</small><span>Ensemble MSD per class, D and α distributions, and the metrics against the class thresholds.</span></a>
    <a class="card" href="#/validate"><b>Validation</b><small>simulated ground truth</small><span>Confusion matrix against the motion model each simulated track came from, with live threshold sliders.</span></a>
    <a class="card" href="#/method"><b>Method</b><small>formulas and thresholds</small><span>Exactly what is computed, from <code>mobility_analyzer.py</code>.</span></a>
    <a class="card" href="#/about"><b>About</b><small>desktop app, credits</small><span>The PyQt6 desktop app (NanoPyx processing, trackpy detection and linking) and how to run everything.</span></a>
  </div>
  <h2>Classes</h2>
  <div class="legend">${list(CLASSES, cls)}</div>
  <p class="small mute">${isStatic() ? 'On this static site the analysis runs in your browser; uploaded files never leave your computer.' : 'Running locally: the analysis runs in Python.'}
    Image processing (NanoPyx drift correction, eSRRF) and particle detection and linking (trackpy) are in the desktop app only.</p>`;
});
