"""Build the standalone HTML results page, with every figure inlined as a data URI.

The published artifact must be self-contained (no external requests), so the PNGs written
by make_figures.py are base64-embedded here.
"""

import base64
import os

REPO = "/workspace/fluoresceAnything"
FIGURES = f"{REPO}/figures"
OUT = f"{REPO}/results/report.html"


def img(name, alt):
    with open(f"{FIGURES}/{name}.png", "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" loading="lazy">'


def plate(name, alt, caption):
    return f'<figure class="plate">{img(name, alt)}<figcaption>{caption}</figcaption></figure>'


HEAD = """<title>ESM-C Taggability Grid</title>
<style>
:root {
  color-scheme: light;
  --ground:    #fbfcfb;
  --surface:   #f1f4f2;
  --surface-2: #e7ece9;
  --line:      #d5ddd8;
  --line-soft: #e4eae6;
  --ink:       #12171a;
  --ink-2:     #414c47;
  --ink-3:     #6a7671;
  --accent:    #2a78d6;
  --accent-dim:#e4eefb;
  --confirm:   #147c58;
  --confirm-bg:#e0f3ea;
  --null:      #6a7671;
  --null-bg:   #e9edeb;
  --warn:      #a5521f;
  --warn-bg:   #f9ebe1;
  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:    #101513;
    --surface:   #171e1b;
    --surface-2: #1f2723;
    --line:      #2c3733;
    --line-soft: #232c28;
    --ink:       #eef3f0;
    --ink-2:     #c2ccc7;
    --ink-3:     #8d9a94;
    --accent:    #6ba6ec;
    --accent-dim:#17293c;
    --confirm:   #4cc796;
    --confirm-bg:#123227;
    --null:      #8d9a94;
    --null-bg:   #212a26;
    --warn:      #e0955f;
    --warn-bg:   #33231a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:    #101513;
  --surface:   #171e1b;
  --surface-2: #1f2723;
  --line:      #2c3733;
  --line-soft: #232c28;
  --ink:       #eef3f0;
  --ink-2:     #c2ccc7;
  --ink-3:     #8d9a94;
  --accent:    #6ba6ec;
  --accent-dim:#17293c;
  --confirm:   #4cc796;
  --confirm-bg:#123227;
  --null:      #8d9a94;
  --null-bg:   #212a26;
  --warn:      #e0955f;
  --warn-bg:   #33231a;
}

* { box-sizing: border-box; }
body {
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.6;
  margin: 0;
  padding: 0 20px 96px;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1080px; margin: 0 auto; }
.col  { max-width: 68ch; }

/* ---------- masthead ---------- */
header.masthead {
  border-bottom: 2px solid var(--ink);
  padding: 56px 0 20px;
  margin-bottom: 40px;
}
.eyebrow {
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin: 0 0 14px;
}
h1 {
  font-family: var(--serif);
  font-weight: 600;
  font-size: clamp(30px, 4.4vw, 46px);
  line-height: 1.12;
  letter-spacing: -0.012em;
  text-wrap: balance;
  margin: 0 0 14px;
}
.standfirst {
  font-size: 18px;
  line-height: 1.55;
  color: var(--ink-2);
  max-width: 62ch;
  margin: 0 0 26px;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 28px;
  font-family: var(--mono);
  font-size: 12.5px;
  color: var(--ink-3);
  font-variant-numeric: tabular-nums;
}
.meta b { color: var(--ink-2); font-weight: 600; }

/* ---------- verdict strip ---------- */
.verdicts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(184px, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  margin: 0 0 52px;
}
.verdict { background: var(--surface); padding: 16px 18px 18px; display: flex; flex-direction: column; gap: 8px; }
.verdict .q {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.verdict .a { font-family: var(--serif); font-size: 20px; line-height: 1.2; font-weight: 600; }
.verdict .n { font-size: 13.5px; color: var(--ink-2); line-height: 1.45; }
.a.yes  { color: var(--confirm); }
.a.no   { color: var(--null); }
.a.open { color: var(--warn); }

/* ---------- sections ---------- */
section { margin: 0 0 56px; }
h2 {
  font-family: var(--serif);
  font-size: 27px;
  font-weight: 600;
  letter-spacing: -0.008em;
  text-wrap: balance;
  margin: 0 0 6px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
  padding-block-start: 22px;
}
h3 {
  font-family: var(--sans);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.02em;
  margin: 32px 0 8px;
}
p { margin: 0 0 16px; }
.col p, .col ul, .col ol { max-width: 68ch; }
ul, ol { margin: 0 0 16px; padding-left: 22px; }
li { margin-bottom: 8px; }
strong { font-weight: 650; }
code, .mono { font-family: var(--mono); font-size: 0.88em; font-variant-numeric: tabular-nums; }
code {
  background: var(--surface-2);
  padding: 1px 5px;
  border-radius: 2px;
  word-break: break-word;
}
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }

.lede { font-size: 17.5px; color: var(--ink-2); }

/* ---------- callout ---------- */
.callout {
  background: var(--warn-bg);
  border-left: 3px solid var(--warn);
  padding: 20px 24px 6px;
  margin: 0 0 32px;
}
.callout h3 { margin-top: 0; color: var(--warn); }
.callout ol { margin-bottom: 14px; }

.note {
  background: var(--surface);
  border: 1px solid var(--line);
  padding: 16px 20px 4px;
  margin: 0 0 28px;
  font-size: 15px;
  color: var(--ink-2);
}

/* ---------- tables ---------- */
.scroll { overflow-x: auto; margin: 0 0 12px; border: 1px solid var(--line); }
table { border-collapse: collapse; width: 100%; font-size: 14px; background: var(--surface); }
th, td { padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
thead th {
  background: var(--surface-2);
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--ink-3);
  font-weight: 600;
  border-bottom: 1px solid var(--line);
}
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
tr.rule td { background: var(--surface-2); color: var(--ink-3); font-style: italic; }
tr.win td { background: var(--accent-dim); }
tr.win td:first-child { font-weight: 650; }
caption {
  caption-side: bottom;
  text-align: left;
  padding: 10px 14px;
  font-size: 13px;
  color: var(--ink-3);
  background: var(--ground);
  border-top: 1px solid var(--line);
}
.tag {
  display: inline-block;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 2px;
}
.tag.yes  { background: var(--confirm-bg); color: var(--confirm); }
.tag.no   { background: var(--null-bg);    color: var(--null); }
.tag.open { background: var(--warn-bg);    color: var(--warn); }

/* ---------- figures ---------- */
.plate {
  margin: 0 0 28px;
  border: 1px solid var(--line);
  background: #fcfcfb;   /* the PNGs are light-mode plates in both themes, by design */
}
.plate img { display: block; width: 100%; height: auto; }
.plate figcaption {
  background: var(--surface);
  color: var(--ink-2);
  border-top: 1px solid var(--line);
  padding: 11px 16px;
  font-size: 13.5px;
  line-height: 1.5;
}
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
.grid2 .plate { margin: 0; }

details {
  border: 1px solid var(--line);
  background: var(--surface);
  padding: 0 18px;
  margin: 0 0 28px;
}
summary {
  cursor: pointer;
  padding: 14px 0;
  font-weight: 650;
  font-size: 15px;
}
details[open] summary { border-bottom: 1px solid var(--line-soft); margin-bottom: 16px; }
details > *:last-child { margin-bottom: 18px; }

/* ---------- question blocks ---------- */
.qa { border-top: 1px solid var(--line-soft); padding: 22px 0 4px; }
.qa:first-of-type { border-top: none; }
.qa .qnum {
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.1em;
  color: var(--ink-3);
}
.qa h3 {
  font-family: var(--serif);
  font-size: 21px;
  font-weight: 600;
  margin: 4px 0 10px;
  letter-spacing: -0.005em;
}
footer {
  border-top: 1px solid var(--line);
  padding-top: 20px;
  font-size: 13.5px;
  color: var(--ink-3);
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>"""


def build():
    body = f"""
<div class="wrap">

<header class="masthead">
  <p class="eyebrow">OpenCell &middot; Cho et al., Science 2022 &middot; full factorial</p>
  <h1>Can a protein language model predict whether a gene can be tagged?</h1>
  <p class="standfirst">A 2&times;2 factorial over ESM-C embeddings &mdash; model scale (300M vs 6B)
  &times; input sequence (bare protein vs the real mNeonGreen2(11) fusion construct) &mdash; scored
  against a composition-only floor under one shared fold assignment.</p>
  <div class="meta">
    <span><b>1,757</b> knock-in attempts</span>
    <span><b>1,310</b> successful / <b>447</b> failed</span>
    <span>base rate <b>0.746</b></span>
    <span><b>5-fold</b> grouped CV, group = ENSG</span>
    <span><b>80</b> scored configurations</span>
  </div>
</header>

<div class="verdicts">
  <div class="verdict">
    <span class="q">Q1 &middot; beat the floor</span>
    <span class="a yes">Yes</span>
    <span class="n">+0.041 &plusmn; 0.023 PR-AUC, better in 5/5 folds</span>
  </div>
  <div class="verdict">
    <span class="q">Q2 &middot; 300M vs 6B</span>
    <span class="a no">Overlapping</span>
    <span class="n">+0.012 for 19&times; the parameters &mdash; prefer 300M</span>
  </div>
  <div class="verdict">
    <span class="q">Q3 &middot; adding the tag</span>
    <span class="a no">No effect</span>
    <span class="n">&minus;0.005 &plusmn; 0.013, better in 2/5 folds</span>
  </div>
  <div class="verdict">
    <span class="q">Q4 &middot; PaperClip transfer</span>
    <span class="a open">Untested</span>
    <span class="n">No PaperClip data exists in the repo</span>
  </div>
  <div class="verdict">
    <span class="q">Q5 &middot; vs topology baseline</span>
    <span class="a yes">Yes</span>
    <span class="n">Terminus selector beats signal peptide by +0.189</span>
  </div>
</div>

<section class="col">
  <div class="callout">
    <h3>Read these three caveats before the numbers</h3>
    <ol>
      <li><strong>The 447 negatives are confounded.</strong> A failed knock-in is not proof of
      intrinsic un-taggability &mdash; it can be a bad guide RNA, low expression, poor HDR, or a
      failed sort. Every PR-AUC here is capped by that label noise, and a &ldquo;false
      positive&rdquo; is often a perfectly taggable protein that failed for an unrelated reason.
      Read false positives charitably.</li>
      <li><strong>The construct-scoring inference sweep is counterfactual.</strong> Each protein
      carries a real label at only one terminus. We train and score on the real constructs; the
      build-both-termini-and-take-argmax procedure a deployed model would use is never tested,
      because the counterfactual label does not exist.</li>
      <li><strong>No-skill PR-AUC is already 0.746</strong> with positive = successful, which leaves
      little headroom. Every table therefore also carries the minority direction (positive =
      unsuccessful, base rate 0.254), where a real signal has room to show itself.</li>
    </ol>
  </div>

  <div class="note">
    <p><strong>Scope change.</strong> There are no <code>data/paperclip_*.csv</code> files in the
    repo or on <code>origin/main</code>. Grid cells 3 and 6 (PaperClip), the external-generalization
    framing, and figure 7 are <strong>not produced</strong>. PaperClip transfer is
    <strong>untested</strong> &mdash; not &ldquo;no evidence of transfer&rdquo;, simply not
    measured.</p>
  </div>
</section>

<section>
  <h2>The grid as actually run</h2>
  <div class="scroll">
    <table>
      <thead><tr><th>data</th><th>sequence</th><th>model</th><th>framing</th><th>status</th></tr></thead>
      <tbody>
        <tr><td>OpenCell</td><td>protein</td><td>300M</td><td>A &mdash; filter + terminus selector</td><td><span class="tag yes">run</span></td></tr>
        <tr><td>OpenCell</td><td>protein + tag</td><td>300M</td><td>B &mdash; construct scoring</td><td><span class="tag yes">run</span></td></tr>
        <tr><td>OpenCell</td><td>protein</td><td>6B</td><td>A &mdash; filter + terminus selector</td><td><span class="tag yes">run</span></td></tr>
        <tr><td>OpenCell</td><td>protein + tag</td><td>6B</td><td>B &mdash; construct scoring</td><td><span class="tag yes">run</span></td></tr>
        <tr class="rule"><td>PaperClip</td><td>protein</td><td>300M</td><td>C &mdash; external test</td><td><span class="tag open">no data</span></td></tr>
        <tr class="rule"><td>PaperClip</td><td>protein</td><td>6B</td><td>C &mdash; external test</td><td><span class="tag open">no data</span></td></tr>
      </tbody>
    </table>
  </div>

  <div class="col">
    <h3>The tag came from OpenCell's own donors</h3>
    <p>The knock-in cassette was read from <code>data/seq_data/tags_translated.fasta</code> and
    independently confirmed against the HDR donor sequences in table S3 &mdash; the tag DNA appears
    in <strong>1,647 of 1,757 donors</strong> on one strand or the other (the misses are donors whose
    200-nt window clips the cassette).</p>
    <ul>
      <li>mNG2(11) = <code>TELNFKEWQKAFTDMM</code> (16 aa)</li>
      <li>linker = <code>GGGLEVLFQGPGSG</code> (14 aa, carrying the HRV-3C site <code>LEVLFQ&darr;GP</code>)</li>
      <li>N-terminal construct = <code>M</code> + tag + linker + protein[1:] &mdash; the cassette supplies the initiator Met</li>
      <li>C-terminal construct = protein + linker + tag</li>
    </ul>

    <h3>Embedding, held identical across cells</h3>
    <p>300M via the native <code>esm</code> SDK (333M params, d = 960); 6B via
    <code>multimolecule/esmc-6b</code> (6.35B params, 80 layers, d = 2560). Both bf16, <code>no_grad</code>,
    one H100. Both tokenizers were verified by printing decoded token ids rather than assumed: the
    layout is <code>&lt;cls&gt;</code> + residues + <code>&lt;eos&gt;</code> for both models and both
    sequence types, so residues are <code>emb[1:1+L]</code>. Hidden dim is read off the tensor.
    Sequences are middle-truncated at 2046 &mdash; the models' 2048-position limit minus the two
    special tokens &mdash; which touches 48 of 1,756 proteins and preserves both termini and the
    junction.</p>
  </div>
</section>

<section>
  <h2>Filter task &mdash; can it tell taggable from untaggable?</h2>
  <p class="col lede">Best configuration per cell. Positive = successful, base rate 0.746.</p>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>cell</th><th>features</th><th>classifier</th>
          <th class="num">PR-AUC</th><th class="num">PR-AUC (minority)</th>
          <th class="num">recall @ P&ge;0.90</th><th class="num">Brier</th>
        </tr>
      </thead>
      <tbody>
        <tr class="rule"><td>no skill</td><td>&mdash;</td><td>&mdash;</td><td class="num">0.746</td><td class="num">0.254</td><td class="num">&mdash;</td><td class="num">&mdash;</td></tr>
        <tr><td>baseline</td><td>signal peptide only</td><td>logreg</td><td class="num">0.752 &plusmn; 0.010</td><td class="num">0.268 &plusmn; 0.019</td><td class="num">0.000</td><td class="num">0.190</td></tr>
        <tr><td>baseline</td><td>terminus only</td><td>logreg</td><td class="num">0.778 &plusmn; 0.019</td><td class="num">0.292 &plusmn; 0.022</td><td class="num">0.000</td><td class="num">0.187</td></tr>
        <tr><td>baseline</td><td>topology (SP+TM+KD)</td><td>xgboost</td><td class="num">0.818 &plusmn; 0.015</td><td class="num">0.381 &plusmn; 0.076</td><td class="num">0.101 &plusmn; 0.071</td><td class="num">0.225</td></tr>
        <tr><td><strong>baseline</strong></td><td><strong>FLOOR</strong> (length + aa comp)</td><td>logreg</td><td class="num"><strong>0.861 &plusmn; 0.022</strong></td><td class="num">0.459 &plusmn; 0.057</td><td class="num">0.360 &plusmn; 0.242</td><td class="num">0.171</td></tr>
        <tr><td>baseline</td><td>FLOOR + terminus</td><td>logreg</td><td class="num">0.864 &plusmn; 0.020</td><td class="num">0.477 &plusmn; 0.071</td><td class="num">0.309 &plusmn; 0.187</td><td class="num">0.168</td></tr>
        <tr><td>protein / 300M</td><td>TERMINAL</td><td>logreg</td><td class="num">0.890 &plusmn; 0.011</td><td class="num">0.562 &plusmn; 0.074</td><td class="num">0.507 &plusmn; 0.153</td><td class="num">0.155</td></tr>
        <tr><td>protein+tag / 300M</td><td>TERMINAL</td><td>xgboost</td><td class="num">0.888 &plusmn; 0.019</td><td class="num">0.569 &plusmn; 0.080</td><td class="num">0.588 &plusmn; 0.092</td><td class="num">0.162</td></tr>
        <tr class="win"><td>protein / 6B</td><td>TERMINAL</td><td>xgboost</td><td class="num">0.902 &plusmn; 0.016</td><td class="num">0.588 &plusmn; 0.065</td><td class="num">0.629 &plusmn; 0.090</td><td class="num">0.151</td></tr>
        <tr><td>protein+tag / 6B</td><td>TERMINAL</td><td>logreg</td><td class="num">0.897 &plusmn; 0.015</td><td class="num"><strong>0.616 &plusmn; 0.071</strong></td><td class="num">0.598 &plusmn; 0.090</td><td class="num">0.147</td></tr>
      </tbody>
      <caption>Full 80-row table &mdash; every cell &times; feature set &times; classifier &times;
      weighted, both metric directions, precision@recall=0.50, log-loss, n_train/n_pos/n_neg &mdash;
      is in <code>results/master_results.csv</code>.</caption>
    </table>
  </div>

  {plate("fig4_all_cells", "Grouped bar chart of PR-AUC for all four cells against the floor",
         "All four cells against the composition floor, the terminus-only control, and the no-skill line. Every embedding cell clears the floor; the two axes of the factorial barely separate from each other.")}

  <h3 class="col">Paired per-fold deltas</h3>
  <p class="col">Because every cell shares one fold assignment, the honest comparison is paired
  rather than &ldquo;do the error bars overlap&rdquo;. &Delta; is the mean per-fold difference
  &plusmn; its own standard deviation.</p>
  <div class="scroll">
    <table>
      <thead><tr><th>comparison</th><th class="num">&Delta; PR-AUC</th><th class="num">paired t</th><th class="num">folds improved</th><th>verdict</th></tr></thead>
      <tbody>
        <tr><td>300M protein &minus; FLOOR</td><td class="num">+0.029 &plusmn; 0.014</td><td class="num">+4.57</td><td class="num">5/5</td><td><span class="tag yes">real</span></td></tr>
        <tr><td>6B protein &minus; FLOOR</td><td class="num">+0.041 &plusmn; 0.023</td><td class="num">+3.92</td><td class="num">5/5</td><td><span class="tag yes">real</span></td></tr>
        <tr><td>6B &minus; 300M &nbsp;<span class="mono">[SCALE]</span></td><td class="num">+0.012 &plusmn; 0.012</td><td class="num">+2.25</td><td class="num">5/5</td><td><span class="tag no">marginal</span></td></tr>
        <tr><td>6B protein+tag &minus; 6B protein &nbsp;<span class="mono">[TAG]</span></td><td class="num">&minus;0.005 &plusmn; 0.013</td><td class="num">&minus;0.92</td><td class="num">2/5</td><td><span class="tag no">null</span></td></tr>
        <tr><td>300M protein+tag &minus; 300M protein &nbsp;<span class="mono">[TAG]</span></td><td class="num">&minus;0.002 &plusmn; 0.008</td><td class="num">&minus;0.61</td><td class="num">1/5</td><td><span class="tag no">null</span></td></tr>
      </tbody>
      <caption>In the minority direction: 300M &minus; FLOOR = +0.103 &plusmn; 0.057 (t = 4.01),
      6B &minus; FLOOR = +0.129 &plusmn; 0.062 (t = 4.63), scale = +0.026 &plusmn; 0.031 (t = 1.91),
      tag = +0.028 &plusmn; 0.047 (t = 1.32).</caption>
    </table>
  </div>

  <div class="grid2">
    {plate("fig5_scale_axis", "Paired comparison of 300M against 6B",
           "SCALE isolated. The direction is consistent &mdash; 6B wins in 5/5 folds &mdash; but the gain is about one standard deviation of its own delta.")}
    {plate("fig6_tag_axis", "Paired comparison of protein against protein plus tag",
           "TAG isolated. Both models move slightly downward when the 30-aa cassette is added. The construct does not help.")}
  </div>

  <h3 class="col">What that buys you at the bench</h3>
  <p class="col">Operating point = the highest-recall threshold on the pooled out-of-fold PR curve
  with precision &ge; 0.90. Never 0.5.</p>
  <div class="scroll">
    <table>
      <thead>
        <tr><th>configuration</th><th class="num">threshold</th><th class="num">precision</th>
        <th class="num">recall</th><th class="num">candidates kept</th><th class="num">duds among them</th>
        <th class="num">failures rejected</th></tr>
      </thead>
      <tbody>
        <tr><td>FLOOR</td><td class="num">0.848</td><td class="num">0.901</td><td class="num">0.256</td><td class="num">373</td><td class="num">37</td><td class="num">410/447 (92%)</td></tr>
        <tr><td>300M protein</td><td class="num">0.859</td><td class="num">0.900</td><td class="num">0.461</td><td class="num">671</td><td class="num">67</td><td class="num">380/447 (85%)</td></tr>
        <tr class="win"><td>6B protein</td><td class="num">0.844</td><td class="num">0.900</td><td class="num">0.615</td><td class="num">894</td><td class="num">89</td><td class="num">358/447 (80%)</td></tr>
        <tr><td>6B protein+tag</td><td class="num">0.852</td><td class="num">0.900</td><td class="num">0.598</td><td class="num">870</td><td class="num">87</td><td class="num">360/447 (81%)</td></tr>
      </tbody>
      <caption>At a fixed 90% precision the 6B embedding passes <strong>2.4&times; as many true
      targets</strong> as the composition floor (894 vs 373). That, not the 0.04 of PR-AUC, is the
      practically meaningful number.</caption>
    </table>
  </div>

  <div class="grid2">
    {plate("pr_protein_6b", "Precision-recall curve for the 6B protein cell",
           "Best cell: mean PR curve across the five folds with the &plusmn;1 std band, the 0.746 no-skill line, and the 0.90 precision target.")}
    {plate("confusion_protein_6b", "Confusion matrix for the 6B protein cell at the chosen threshold",
           "The same cell at the precision &ge; 0.90 operating point: 894 candidates kept, 89 of them duds, 80% of failures screened out.")}
  </div>
</section>

<section>
  <h2>Terminus selector &mdash; which end should the tag go on?</h2>
  <p class="col lede">1,310 successful targets, positive = N-terminus, base rate 0.559. Terminus is
  never an input. Only the protein cells run this framing &mdash; in the +tag cells, tag placement
  <em>is</em> the label.</p>
  <div class="scroll">
    <table>
      <thead><tr><th>cell</th><th>features</th><th>classifier</th><th class="num">PR-AUC</th><th class="num">balanced acc @ 0.5</th><th class="num">Brier</th></tr></thead>
      <tbody>
        <tr class="rule"><td>no skill</td><td>&mdash;</td><td>&mdash;</td><td class="num">0.559</td><td class="num">0.500</td><td class="num">&mdash;</td></tr>
        <tr><td>baseline</td><td>signal peptide only</td><td>logreg</td><td class="num">0.572 &plusmn; 0.005</td><td class="num">0.526 &plusmn; 0.008</td><td class="num">0.243</td></tr>
        <tr><td>baseline</td><td>topology (SP + TM + hydropathy)</td><td>xgboost</td><td class="num">0.661 &plusmn; 0.023</td><td class="num">0.588 &plusmn; 0.015</td><td class="num">0.240</td></tr>
        <tr><td>baseline</td><td>FLOOR</td><td>xgboost</td><td class="num">0.667 &plusmn; 0.010</td><td class="num">0.590 &plusmn; 0.034</td><td class="num">0.249</td></tr>
        <tr class="win"><td>protein / 300M</td><td>TERMINAL</td><td>xgboost</td><td class="num">0.761 &plusmn; 0.020</td><td class="num">0.652 &plusmn; 0.028</td><td class="num">0.212</td></tr>
        <tr><td>protein / 6B</td><td>TERMINAL</td><td>logreg</td><td class="num">0.731 &plusmn; 0.029</td><td class="num">0.647 &plusmn; 0.010</td><td class="num">0.217</td></tr>
      </tbody>
    </table>
  </div>

  {plate("fig8_terminus_selector", "Terminus selector compared against signal peptide and topology baselines",
         "The selector clears every baseline by a wide margin: +0.189 over signal-peptide-only (t = 25.3), +0.100 over full topology (t = 6.70), +0.094 over composition (t = 10.2) &mdash; all 5/5 folds. Note that 6B is <em>worse</em> than 300M here (&minus;0.030, better in only 2/5 folds).")}

  <p class="col">The signal-peptide rule is real but tiny in reach: of the 30 successful targets with
  an annotated signal peptide, <strong>all 30</strong> were tagged at the C-terminus &mdash; a
  perfect rule covering 2.3% of the set. That is why signal-peptide-only barely clears no-skill
  (0.572 vs 0.559), and why the embedding's +0.189 is not it rediscovering signal peptides.</p>
</section>

<section>
  <h2>The five questions</h2>

  <div class="qa col">
    <span class="qnum">Q1</span>
    <h3>Does any embedding clear the FLOOR, and by how much relative to std?</h3>
    <p>Yes, unambiguously. FLOOR &mdash; sequence length plus 20 amino-acid composition fractions
    &mdash; is 0.861 &plusmn; 0.022, already far above the 0.746 no-skill line, so it is a genuinely
    hard bar. The best 6B cell reaches 0.902 &plusmn; 0.016, a paired gain of
    <strong>+0.041 &plusmn; 0.023 (t = 3.92, better in 5/5 folds)</strong>: 1.8&times; the std of the
    delta itself and 2.6&times; the cell's own fold spread. 300M gains +0.029 &plusmn; 0.014
    (t = 4.57, 5/5). The minority direction is where it is stark &mdash; FLOOR 0.459 &rarr; 6B 0.588,
    a paired +0.129 &plusmn; 0.062, a 28% relative improvement at finding the failures.</p>
  </div>

  <div class="qa col">
    <span class="qnum">Q2</span>
    <h3>300M vs 6B: separated or overlapping error bars?</h3>
    <p><strong>Overlapping &mdash; prefer 300M.</strong> 0.890 &plusmn; 0.011 vs 0.902 &plusmn; 0.016
    overlap heavily. The paired test is kinder (+0.012 &plusmn; 0.012, 5/5 folds, t = 2.25), so the
    direction is consistent, but the effect is about one std of its own delta and costs 19&times; the
    parameters, a 25 GB checkpoint, and roughly 8&times; the embedding wall-clock. On the terminus
    task 6B is actually <em>worse</em> (&minus;0.030 &plusmn; 0.043, better in only 2/5 folds). Scale
    buys about one point of PR-AUC on one of two tasks and loses on the other.</p>
  </div>

  <div class="qa col">
    <span class="qnum">Q3</span>
    <h3>protein vs protein+tag: did the construct move PR-AUC beyond the std?</h3>
    <p><strong>No.</strong> 6B: &minus;0.005 &plusmn; 0.013 (t = &minus;0.92, better in 2/5 folds).
    300M: &minus;0.002 &plusmn; 0.008 (1/5). The construct is if anything marginally worse in the
    headline direction. The minority direction hints at a gain (+0.028 &plusmn; 0.047, t = 1.32, and
    the +tag/6B cell holds the single best minority PR-AUC at 0.616 &plusmn; 0.071) but that sits
    well inside the fold spread and should not be called a result on 447 negatives.</p>
    <p>The terminus-only control is what makes that statement safe. Terminus alone reaches 0.778, so
    a +tag cell could have &ldquo;won&rdquo; purely by leaking terminus through tag placement. It
    didn't win at all, so the question is moot &mdash; but without the control we could not have said
    so. <strong>A 30-aa cassette appended to a 600-aa protein does not visibly change what ESM-C
    encodes about it.</strong></p>
  </div>

  <div class="qa col">
    <span class="qnum">Q4</span>
    <h3>Does the OpenCell selector generalize to PaperClip?</h3>
    <p><strong>Not tested.</strong> No PaperClip data exists in the repo or on
    <code>origin/main</code>. This is an open question, not a negative result, and the cells are
    ready to run the moment the file lands.</p>
  </div>

  <div class="qa col">
    <span class="qnum">Q5</span>
    <h3>Does Model 2 beat the signal-peptide / topology baseline?</h3>
    <p><strong>Decisively.</strong> 0.761 &plusmn; 0.020 against 0.572 &plusmn; 0.005 for
    signal-peptide-only (+0.189, t = 25.3) and 0.661 &plusmn; 0.023 for full topology including
    transmembrane counts and terminal hydropathy (+0.100, t = 6.70). It also beats composition alone
    by +0.094 (t = 10.2). Of the three framings, the terminus selector is where the protein language
    model most clearly earns its keep.</p>
  </div>
</section>

<section>
  <h2>Two things worth knowing about the floor</h2>
  <div class="col">
    <h3>The floor is a hydrophobicity detector &mdash; and so, partly, is the model</h3>
    <p>The strongest FLOOR features are composition, not length: Leu (r = &minus;0.19), Trp
    (&minus;0.17), His (&minus;0.15) and Cys (&minus;0.15) predict failure; Lys (+0.17), Asp (+0.15)
    and Glu (+0.13) predict success. Length barely matters (median 472 aa successful vs 491
    unsuccessful, p = 0.005). That is a membrane/soluble axis, and the UniProt annotations agree:
    transmembrane-annotated proteins succeed 58.6% of the time versus 78.6% for the rest.
    Out-of-fold prediction ranks correlate &rho; = 0.64 between FLOOR and the 6B cell &mdash;
    substantial overlap, but far from identical, so the embedding is adding signal rather than
    re-deriving composition.</p>
    <p>This feeds straight back into caveat 1. Membrane proteins are exactly the class most likely to
    fail for reasons that are <em>not</em> intrinsic untaggability &mdash; expression, trafficking,
    sortability. Part of what every model here has learned is &ldquo;is this a membrane
    protein&rdquo;, which is a real predictor of the recorded label and a poor predictor of true
    taggability.</p>

    <h3>The imbalance knob does nothing</h3>
    <p>Across 40 paired configurations, weighted &minus; unweighted PR-AUC averages
    <strong>&minus;0.0006</strong> (median &minus;0.0004, range &minus;0.005 to +0.005) and helps in
    14/40. Per the locked strategy &mdash; keep only if CV PR-AUC improves &mdash; the recommendation
    is <strong>unweighted</strong>. The single best-scoring filter row happens to be a weighted
    XGBoost, but the paired evidence says that is noise, not the knob working. No SMOTE, no
    subsampling, and no resampling of any kind was used anywhere.</p>
  </div>
</section>

<section>
  <h2>Per-cell diagnostics</h2>
  <details>
    <summary>All twelve per-cell plates &mdash; PR curves, calibration, confusion matrices</summary>
    <div class="grid2">
      {plate("pr_protein_300m", "PR curve, protein 300M", "protein / 300M &mdash; PR curve")}
      {plate("calibration_protein_300m", "Calibration, protein 300M", "protein / 300M &mdash; calibration")}
      {plate("confusion_protein_300m", "Confusion matrix, protein 300M", "protein / 300M &mdash; confusion at P &ge; 0.90")}
      {plate("pr_proteinplustag_300m", "PR curve, protein+tag 300M", "protein+tag / 300M &mdash; PR curve")}
      {plate("calibration_proteinplustag_300m", "Calibration, protein+tag 300M", "protein+tag / 300M &mdash; calibration")}
      {plate("confusion_proteinplustag_300m", "Confusion matrix, protein+tag 300M", "protein+tag / 300M &mdash; confusion at P &ge; 0.90")}
      {plate("pr_protein_6b", "PR curve, protein 6B", "protein / 6B &mdash; PR curve")}
      {plate("calibration_protein_6b", "Calibration, protein 6B", "protein / 6B &mdash; calibration")}
      {plate("confusion_protein_6b", "Confusion matrix, protein 6B", "protein / 6B &mdash; confusion at P &ge; 0.90")}
      {plate("pr_proteinplustag_6b", "PR curve, protein+tag 6B", "protein+tag / 6B &mdash; PR curve")}
      {plate("calibration_proteinplustag_6b", "Calibration, protein+tag 6B", "protein+tag / 6B &mdash; calibration")}
      {plate("confusion_proteinplustag_6b", "Confusion matrix, protein+tag 6B", "protein+tag / 6B &mdash; confusion at P &ge; 0.90")}
    </div>
  </details>
  {plate("fig4b_all_cells_minority", "Grouped bar chart in the minority direction",
         "The same four cells scored with positive = unsuccessful (base rate 0.254), where there is real headroom. The ordering is the same; the separations are wider.")}
</section>

<section class="col">
  <h2>Reproducing</h2>
  <div class="scroll">
    <table>
      <tbody>
        <tr><td><code>src/tags/build_constructs.py</code></td><td>verify the tag against the HDR donors, build <code>data/modeling_table.csv</code></td></tr>
        <tr><td><code>src/embed/esmc_embed.py</code></td><td><code>--model {{300m,6b}} --seqtype {{protein,construct}}</code> &mdash; four passes, ~4 min total on an H100</td></tr>
        <tr><td><code>src/modeling/run_grid.py</code></td><td>the 64-row sweep &rarr; <code>master_results.csv</code>, <code>oof_predictions.csv</code></td></tr>
        <tr><td><code>src/modeling/topology_baseline.py</code></td><td>fetch UniProt topology, append 16 baseline rows</td></tr>
        <tr><td><code>src/figures/make_figures.py</code></td><td>all seventeen figures</td></tr>
        <tr><td><code>src/modeling/summarize.py</code></td><td>markdown views of the master table</td></tr>
      </tbody>
    </table>
  </div>
  <p>The embedding cache is gitignored and holds pooled vectors only &mdash; full residue tensors for
  the 6B model would run to about 11 GB per pass. The modelling grid takes roughly 40 minutes,
  dominated by XGBoost on the 10,240-dimensional 6B TERMINAL features.</p>
</section>

<footer class="col">
  <p>Data: Cho et al., <em>Science</em> 375, eabi6983 (2022), supplementary table S3. Evaluation:
  5-fold <code>StratifiedGroupKFold</code>, group = ENSG, seed 42, one fold assignment shared by every
  cell. No holdout &mdash; this is a comparison study, not a deployment.</p>
</footer>

</div>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(HEAD + body)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
