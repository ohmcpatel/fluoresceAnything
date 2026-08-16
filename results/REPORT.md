# OpenCell taggability + terminus — ESM-C factorial

**Data:** OpenCell (Cho et al., *Science* 2022) supplementary table S3. 1,757 knock-in attempts,
1,310 successful / 447 unsuccessful (base rate 0.746), tagged N 913 : C 844.
**Evaluation:** 5-fold `StratifiedGroupKFold`, group = ENSG, seed 42, one fold assignment reused by
every cell so all deltas are attributable to the varied axis. No holdout — this is a comparison
study, not a deployment.
**No resampling of any kind.** The only imbalance knob is `class_weight` / `scale_pos_weight`, run
both ways everywhere.

---

## Read this first — the three caveats

1. **The 447 negatives are confounded.** A failed knock-in is not proof of intrinsic
   un-taggability: it can be a bad guide RNA, low expression, poor HDR, or a failed sort. Every
   PR-AUC below is therefore capped by label noise, and a "false positive" here is often a protein
   that is perfectly taggable and simply failed for an unrelated reason. Read false positives
   charitably.
2. **Framing B's inference sweep is counterfactual.** Each protein carries a real label at only one
   terminus. We train and score on the real constructs; the build-both-and-take-argmax procedure a
   deployed model would use is never actually tested here, because the counterfactual label does not
   exist.
3. **With positive = successful, no-skill PR-AUC is already 0.746.** That leaves little headroom, so
   every table also carries the minority direction (positive = unsuccessful, base rate 0.254), where
   a real signal has room to show itself.

**Scope change:** there are no `data/paperclip_*.csv` files in the repo or on `origin/main`. Grid
cells 3 and 6 (PaperClip), framing C, and figure 7 are **not produced**. PaperClip generalization is
therefore **untested** — not "no evidence of generalization", simply not measured.

---

## The grid as actually run

| data | sequence | model | status |
|---|---|---|---|
| OpenCell | protein | 300M | run — framing A (filter + terminus selector) |
| OpenCell | protein+tag | 300M | run — framing B (construct scoring) |
| OpenCell | protein | 6B | run — framing A |
| OpenCell | protein+tag | 6B | run — framing B |
| PaperClip | protein | 300M | **not run — no data** |
| PaperClip | protein | 6B | **not run — no data** |

**Embedding.** 300M via the native `esm` SDK (`esmc_300m`, 333M params, d=960); 6B via
`multimolecule/esmc-6b` (6.35B params, 80 layers, d=2560), bf16, `no_grad`, H100. Both tokenizers
verified by printing decoded token ids: layout is `<cls>` + residues + `<eos>` for both models and
both sequence types, so residues are `emb[1:1+L]`. Hidden dim read off the tensor, never hardcoded.
Sequences middle-truncated at 2046 (= the model's 2048 position limit minus the two special tokens),
which affects 48/1,756 proteins and 49/1,756 constructs and preserves both termini and the junction.

**The tag.** Recovered from `data/seq_data/tags_translated.fasta` and independently confirmed against
OpenCell's own HDR donor sequences — the tag DNA appears in 1,647/1,757 donors on one strand or the
other (the misses are donors whose 200-nt window clips the cassette).
mNG2(11) = `TELNFKEWQKAFTDMM` (16 aa), linker = `GGGLEVLFQGPGSG` (14 aa, carrying the HRV-3C site
`LEVLFQ↓GP`). N-terminal construct = `M` + tag + linker + protein[1:] (the cassette supplies the
initiator Met); C-terminal construct = protein + linker + tag.

---

## Filter task (Model 1 / framing B) — best configuration per cell

Positive = successful, base rate **0.746**.

| cell | features | classifier | PR-AUC | PR-AUC (minority) | recall @ P≥0.90 | Brier |
|---|---|---|---|---|---|---|
| no skill | — | — | 0.746 | 0.254 | — | — |
| baseline | TOPOLOGY_SP | logreg | 0.752 ± 0.010 | 0.268 ± 0.019 | 0.000 | 0.190 |
| baseline | TERMINUS_ONLY | logreg | 0.778 ± 0.019 | 0.292 ± 0.022 | 0.000 | 0.187 |
| baseline | TOPOLOGY_FULL | xgboost | 0.818 ± 0.015 | 0.381 ± 0.076 | 0.101 ± 0.071 | 0.225 |
| **baseline** | **FLOOR** | logreg | **0.861 ± 0.022** | 0.459 ± 0.057 | 0.360 ± 0.242 | 0.171 |
| baseline | FLOOR+terminus | logreg | 0.864 ± 0.020 | 0.477 ± 0.071 | 0.309 ± 0.187 | 0.168 |
| OpenCell/protein/300M | TERMINAL | logreg | 0.890 ± 0.011 | 0.562 ± 0.074 | 0.507 ± 0.153 | 0.155 |
| OpenCell/protein+tag/300M | TERMINAL | xgboost | 0.888 ± 0.019 | 0.569 ± 0.080 | 0.588 ± 0.092 | 0.162 |
| **OpenCell/protein/6B** | **TERMINAL** | xgboost | **0.902 ± 0.016** | 0.588 ± 0.065 | 0.629 ± 0.090 | 0.151 |
| OpenCell/protein+tag/6B | TERMINAL | logreg | 0.897 ± 0.015 | **0.616 ± 0.071** | 0.598 ± 0.090 | 0.147 |

Full 80-row table (every cell × feature set × classifier × weighted, both metric directions,
precision@recall=0.50, log-loss, n_train/n_pos/n_neg): `results/master_results.csv`.

### Paired per-fold deltas

Because every cell shares one fold assignment, the honest comparison is paired, not
"do the error bars overlap". Δ is the mean per-fold difference ± its own std.

| comparison | Δ PR-AUC | paired t | folds improved |
|---|---|---|---|
| 300M protein − FLOOR | **+0.029 ± 0.014** | +4.57 | 5/5 |
| 6B protein − FLOOR | **+0.041 ± 0.023** | +3.92 | 5/5 |
| 6B − 300M (**SCALE**) | +0.012 ± 0.012 | +2.25 | 5/5 |
| 6B protein+tag − 6B protein (**TAG**) | −0.005 ± 0.013 | −0.92 | 2/5 |
| 300M protein+tag − 300M protein (**TAG**) | −0.002 ± 0.008 | −0.61 | 1/5 |

In the minority direction the same comparisons read +0.103 ± 0.057 (300M − FLOOR, t = 4.01),
+0.129 ± 0.062 (6B − FLOOR, t = 4.63), +0.026 ± 0.031 (scale, t = 1.91), +0.028 ± 0.047 (tag,
t = 1.32).

### What that buys you at the bench

Operating point = highest-recall threshold on the pooled out-of-fold PR curve with precision ≥ 0.90.
Never 0.5.

| configuration | threshold | precision | recall | candidates kept | duds among them | failures rejected |
|---|---|---|---|---|---|---|
| FLOOR | 0.848 | 0.901 | 0.256 | 373 | 37 | 410/447 (92%) |
| 300M protein | 0.859 | 0.900 | 0.461 | 671 | 67 | 380/447 (85%) |
| **6B protein** | 0.844 | 0.900 | 0.615 | **894** | 89 | 358/447 (80%) |
| 6B protein+tag | 0.852 | 0.900 | 0.598 | 870 | 87 | 360/447 (81%) |

At a fixed 90% precision the 6B embedding passes **2.4× as many true targets** as the
composition floor (894 vs 373). That is the practically meaningful number, not the 0.04 of PR-AUC.

---

## Terminus selector (Model 2) — 1,310 successful targets, positive = N

Base rate 0.559. Terminus is never an input; only protein cells run this framing (in the +tag cells
tag placement *is* the label).

| cell | features | classifier | PR-AUC | balanced acc @ 0.5 | Brier |
|---|---|---|---|---|---|
| no skill | — | — | 0.559 | 0.500 | — |
| baseline | TOPOLOGY_SP (signal peptide only) | logreg | 0.572 ± 0.005 | 0.526 ± 0.008 | 0.243 |
| baseline | TOPOLOGY_FULL (SP + TM + hydropathy) | xgboost | 0.661 ± 0.023 | 0.588 ± 0.015 | 0.240 |
| baseline | FLOOR | xgboost | 0.667 ± 0.010 | 0.590 ± 0.034 | 0.249 |
| **OpenCell/protein/300M** | **TERMINAL** | xgboost | **0.761 ± 0.020** | 0.652 ± 0.028 | 0.212 |
| OpenCell/protein/6B | TERMINAL | logreg | 0.731 ± 0.029 | 0.647 ± 0.010 | 0.217 |

| comparison | Δ PR-AUC | paired t | folds improved |
|---|---|---|---|
| 300M − signal-peptide only | **+0.189 ± 0.017** | +25.3 | 5/5 |
| 300M − full topology | **+0.100 ± 0.034** | +6.70 | 5/5 |
| 300M − FLOOR | **+0.094 ± 0.021** | +10.2 | 5/5 |
| 6B − 300M (**SCALE**) | −0.030 ± 0.043 | −1.59 | 2/5 |

The signal-peptide rule is real but tiny in reach: of the 30 successful targets with an annotated
signal peptide, **all 30** were tagged at the C-terminus — a perfect rule covering 2.3% of the set,
which is why signal-peptide-only barely clears no-skill (0.572 vs 0.559).

---

## The five questions, answered

**1. Does any embedding clear the FLOOR, and by how much relative to std?**
Yes, and unambiguously. FLOOR (length + 20 aa composition fractions) is 0.861 ± 0.022 — already far
above the 0.746 no-skill line, so it is a genuinely hard bar. The best 6B cell reaches 0.902 ± 0.016,
a paired gain of **+0.041 ± 0.023 (t = 3.92, better in 5/5 folds)** — 1.8× the std of the delta
itself, and 2.6× the cell's own fold spread of ±0.016. 300M gains +0.029 ± 0.014 (t = 4.57, 5/5). The
minority direction is where it is stark: FLOOR 0.459 → 6B 0.588, a paired **+0.129 ± 0.062**, i.e. a
28% relative improvement at finding the failures. Embeddings clear the floor.

**2. 300M vs 6B: separated or overlapping error bars?**
**Overlapping — prefer 300M.** 0.890 ± 0.011 vs 0.902 ± 0.016: the error bars overlap heavily. The
paired test is more favourable (+0.012 ± 0.012, better in 5/5 folds, t = 2.25), so the direction is
consistent, but the effect is about one std of its own delta and costs 19× the parameters, a 25 GB
checkpoint, and ~8× the embedding wall-clock. On the terminus task 6B is actually **worse**
(−0.030 ± 0.043, better in only 2/5 folds). Scale buys ~1 point of PR-AUC on one of two tasks and
loses on the other. Use 300M.

**3. protein vs protein+tag: did the construct move PR-AUC beyond the std?**
**No.** 6B: −0.005 ± 0.013 (t = −0.92, better in 2/5 folds). 300M: −0.002 ± 0.008 (1/5 folds). The
construct is, if anything, marginally worse in the headline direction. The minority direction hints
at a gain (+0.028 ± 0.047, t = 1.32, and the +tag/6B cell holds the single best minority PR-AUC at
0.616 ± 0.071) but that is well inside the fold spread and should not be called a result on n = 447
negatives. The terminus-only control explains why this had to be checked: terminus alone reaches
0.778, so a +tag cell could have "won" purely by leaking terminus through tag placement. It didn't
win at all, so the question is moot — but the control is what makes that statement safe. **A 30-aa
cassette appended to a 600-aa protein does not visibly change what ESM-C encodes about it.**

**4. Does the OpenCell selector generalize to PaperClip?**
**Not tested.** No PaperClip data exists in the repo or on `origin/main`. This is an open question,
not a negative result.

**5. Does Model 2 beat the signal-peptide/topology baseline?**
**Decisively yes.** 0.761 ± 0.020 vs 0.572 ± 0.005 for signal-peptide-only (+0.189, t = 25.3) and vs
0.661 ± 0.023 for full topology including transmembrane counts and terminal hydropathy (+0.100,
t = 6.70). It is not a signal-peptide detector — that rule covers 30 proteins. It also beats
composition alone by +0.094 (t = 10.2). Of the three framings, the terminus selector is where the
protein language model most clearly earns its keep.

---

## Two things worth knowing about the floor

**The floor is a hydrophobicity detector, and so, partly, is the model.** The strongest FLOOR
features are amino-acid composition, not length: Leu (r = −0.19), Trp (−0.17), His (−0.15), Cys
(−0.15) predict failure; Lys (+0.17), Asp (+0.15), Glu (+0.13) predict success. Length barely
matters (median 472 successful vs 491 unsuccessful, p = 0.005). That is a membrane/soluble axis, and
the UniProt annotations agree: transmembrane-annotated proteins succeed 58.6% of the time versus
78.6% for the rest. Out-of-fold prediction ranks correlate ρ = 0.64 between FLOOR and the 6B cell —
substantial overlap, but far from identical, so the embedding is adding signal rather than
re-deriving composition.

This matters for caveat 1: membrane proteins are exactly the class most likely to fail for reasons
that are *not* intrinsic untaggability (expression, trafficking, sortability). Some of what every
model here has learned is "is this a membrane protein", which is a real predictor of the recorded
label and a poor predictor of true taggability.

**The imbalance knob does nothing.** Across 40 paired configurations, weighted − unweighted PR-AUC
averages **−0.0006** (median −0.0004, range −0.005 to +0.005) and helps in 14/40. Per the locked
strategy — keep only if CV PR-AUC improves — the recommendation is **unweighted**. The single
best-scoring filter row happens to be a weighted XGBoost, but the paired evidence says that is noise,
not the knob working.

---

## Figures — `figures/`

Per cell: `pr_<cell>.png` (mean ± std PR band across folds, no-skill line, precision target),
`calibration_<cell>.png`, `confusion_<cell>.png` (at the precision ≥ 0.90 threshold).

Master: `fig4_all_cells.png` (all four cells vs FLOOR / terminus-only / no-skill),
`fig4b_all_cells_minority.png` (same, minority direction), `fig5_scale_axis.png` (SCALE isolated),
`fig6_tag_axis.png` (TAG isolated), `fig8_terminus_selector.png` (Model 2 vs topology baselines).
`fig7` is not produced — no PaperClip data.

## Reproducing

```
python src/tags/build_constructs.py                       # verify tag, build data/modeling_table.csv
python src/embed/esmc_embed.py --model {300m,6b} --seqtype {protein,construct}
python src/modeling/run_grid.py                           # -> results/master_results.csv, oof_predictions.csv
python src/modeling/topology_baseline.py                  # appends UniProt topology baselines
python src/figures/make_figures.py
python src/modeling/summarize.py                          # markdown views of the master table
```

Embedding cache lives in `cache/` (gitignored, pooled vectors only — full residue tensors for 6B
would be ~11 GB per pass). All four passes take ~4 minutes total on an H100; the full modelling grid
takes ~40 minutes, dominated by XGBoost on the 10,240-dimensional 6B TERMINAL features.
