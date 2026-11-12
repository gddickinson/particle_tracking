// Method (what MobilityAnalyzer computes, with units) and About (desktop app, web app, credits from the README).
import { app, isStatic, route } from './core.js';
import { CLASSES } from './engine.js';
import { cls } from './fmt.js';
import { getInfo } from './state.js';

route('method', async () => {
  const info = await getInfo(), d = info.defaults;
  app().innerHTML = `<p class="kicker">Docs</p><h1>Method</h1>
  <p class="lead">What is computed for each track, as in <code>mobility_analyzer.py</code> (<code>MobilityAnalyzer</code>).
    Positions are converted to µm first (x · pixel size); times are lag / fps with fps = 1 / frame interval.</p>
  <h2>Units</h2>
  <p>Give the pixel size in µm/pixel (default ${d.pixel_size_um}, the 65 nm used in the README) or say the coordinates are
    already in µm, and the frame interval in s (default ${d.frame_interval_s} s = 10 Hz). D is reported in µm²/s, lag times in s,
    MSD in µm². α, directedness and the confinement level are dimensionless. Only displacements are used, so the
    coordinate origin (pixel centre or corner) does not matter. The units travel with the results (CSV header line, JSON).</p>
  <h2>Per-track quantities</h2>
  <p>Each track is sorted by frame. The time-averaged MSD at a lag of k frames is the mean of (Δx² + Δy²) over all pairs of
    localisations exactly k frames apart (<code>mobility_metrics.frame_msd</code>), so tracks with gaps (trackpy memory) get correct lag times.</p>
  <ul>
    <li><b>D</b> (<code>calculate_diffusion_coefficients</code>): linear regression of MSD against t over frame lags 1–${info.d_fit_lags};
      D = slope / 4 (MSD = 4Dt in 2D; the intercept, e.g. localisation error, is fitted and then ignored). Needs at least 4 points.</li>
    <li><b>α</b> (<code>calculate_anomalous_exponent</code>): slope of log MSD against log t over frame lags 1 to min(span/2, 50) − 1 (span = frames covered by the track),
      using lags with MSD &gt; 0. Needs more than 5 such lags, i.e. about 14 points; otherwise the track is “unknown”.</li>
    <li><b>Directedness</b> (<code>calculate_directedness</code>): end-to-end distance / path length, from 0 (returns to start) to 1 (straight).</li>
    <li><b>Confinement level</b> (<code>calculate_confinement_level</code>, used by <code>detect_confinement</code>):
      L = −log₁₀ψ = 2.5117·D·t/R² − 0.2048, Saxton's probability ψ that a free random walker with diffusion coefficient D stays
      within a radius R for a time t (log₁₀ψ = 0.2048 − 2.5117·Dt/R²; 2.5117·ln 10 = 5.783 is the first zero of J₀ squared,
      i.e. the leading term of the survival probability in a disc). Here D is the initial slope of the MSD (frame lags 1–4),
      t the track duration and R the largest distance from the first position, as in Simson et al.'s confinement test (applied here to the whole track rather than to sliding windows).
      For free diffusion L does not depend on track length; a confined particle accumulates evidence (L grows with t).</li>
    <li><b>Legacy ratio</b> (<code>calculate_confinement_ratio</code>, no longer used): end-to-end distance / (Rg · √n). It falls
      as 1/√n for any motion (a straight line drops below the old 0.3 threshold from 132 points), so it labelled every
      200-frame simulated track “confined”; it is kept as a column for comparison.</li>
  </ul>
  <h2>Classes (<code>classify_mobility</code>, first match wins)</h2>
  <ol>
    <li>${cls('unknown')}: α could not be fitted.</li>
    <li>${cls('confined')}: confinement level &gt; ${d.confinement_level_threshold} (ψ below about 0.07 %, the level used by Simson et al.).</li>
    <li>${cls('directed')}: directedness &gt; ${d.directedness_threshold}.</li>
    <li>${cls('subdiffusive')}: α &lt; ${d.subdiffusion_threshold}.</li>
    <li>${cls('superdiffusive')}: α &gt; ${d.superdiffusion_threshold}.</li>
    <li>${cls('brownian')}: otherwise.</li>
  </ol>
  <p class="small mute">All four thresholds can be changed on the Analyze page. The confinement criterion was changed on 2026-11-12
    (classifications differ from earlier versions by design); see <a href="#/validate">Validation</a>.</p>
  <h2>References</h2>
  <ul class="small">
    <li>Saxton MJ (1993) Lateral diffusion in an archipelago. Single-particle diffusion. <i>Biophys J</i> 64:1766–1780.
      <a href="https://doi.org/10.1016/S0006-3495(93)81548-0">doi:10.1016/S0006-3495(93)81548-0</a></li>
    <li>Simson R, Sheets ED, Jacobson K (1995) Detection of temporary lateral confinement of membrane proteins using
      single-particle tracking analysis. <i>Biophys J</i> 69:989–993.
      <a href="https://doi.org/10.1016/S0006-3495(95)79972-6">doi:10.1016/S0006-3495(95)79972-6</a></li>
  </ul>
  <h2>Ensemble MSD</h2>
  <p>Per class, the mean of the per-track MSD curves over the tracks that reach each lag (lags up to ${info.ensemble_max_lag - 1}),
    as <code>_calculate_msd_manual</code>, fitted with <code>fit_msd_model(…, 'powerlaw')</code>: log MSD = log(4D) + α log t.
    The desktop app's MSD plot uses trackpy's <code>emsd</code> when trackpy is installed, which weights lags by the number of
    displacements instead, so its curve can differ slightly.</p>
  <h2>Browser and Python</h2>
  <p>The local web app runs the Python code. The static site runs <code>engine.js</code>, a line-by-line port; a node test
    (<code>web/static/engine.test.mjs</code>) checks every per-track value, class and ensemble point against the Python output
    on simulated tracks plus edge cases (relative tolerance 1e-9).</p>
  <div class="legend">${CLASSES.map(cls).join('')}</div>`;
});

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `<p class="kicker">Docs</p><h1>About</h1>
  <p class="lead">Advanced Particle Tracking with NanoPyx: a Python application for analysing fluorescently-tagged proteins with
    mixed mobility behaviours in microscopy data. This web interface covers the mobility analysis of finished tracks.</p>
  <h2>Desktop app (full pipeline)</h2>
  <p>The PyQt6 app (<code>main.py</code>) loads an image stack (.tif/.tiff/.nd2), corrects drift and optionally applies eSRRF
    super-resolution with NanoPyx, detects and links particles with trackpy, then runs the same mobility analysis and exports
    <code>tracks_with_mobility.csv</code>, <code>diffusion_coefficients.csv</code> and <code>msd_data.csv</code>.</p>
  <pre>pip install -r requirements.txt
python main.py</pre>
  <h2>Web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8130</pre>
  <p>${isStatic() ? 'This static site runs the analysis in your browser: uploaded files are read locally and never sent anywhere.' : 'You are running the local app: uploads are analysed by the Python code on this computer.'}
    Accepted columns: ${Object.entries(info.columns).map(([k, v]) => `<b>${k}</b> (${v.join(', ')})`).join('; ')}.
    These cover trackpy (<code>particle</code>) and TrackMate (<code>TRACK_ID, FRAME, POSITION_X, POSITION_Y</code>, usually
    already in µm) naming; rows without numeric frame, x and y are skipped.</p>
  <h2>Demo data</h2>
  <ul>${info.demos.map(d => `<li><b>${d.title}</b>: ${d.description}<br><code class="small">${d.generator}</code></li>`).join('')}</ul>
  <h2>Credits</h2>
  <p>Image processing: NanoPyx (Saraiva et al. 2023, “NanoPyx: super-fast bioimage analysis powered by adaptive machine
    learning”, bioRxiv). Detection and linking: trackpy (Allan et al., soft-matter.github.io/trackpy). Code: George Dickinson,
    MIT licence. Source: <a href="https://github.com/gddickinson/particle_tracking">github.com/gddickinson/particle_tracking</a>.</p>`;
});
