# Taggability scoring endpoint

Protein sequence in, one continuous value in `[0, 1]` out. No threshold is applied and no
success/failure label is returned — the score *is* the output.

```
0.0  ── the model expects an mNG2(11) knock-in attempt on this protein to fail
1.0  ── the model expects it to succeed
```

## What the number is

`P(library_success)` from the classifier locked in
[`notebooks/decision_to_classifier.ipynb`](../../notebooks/decision_to_classifier.ipynb),
decision for decision:

| # | decision | shipped |
|---|---|---|
| 1 | embeddings vs. hand-built FLOOR | ESM-C |
| 2 | pooling | `TERMINAL` = `mean` + `first_25` + `last_25` |
| 3 | protein vs. tag-fusion construct | bare protein |
| 4 | backbone scale | **6B** (2560-d blocks → 7,680 features) |
| 5 | estimator | `StandardScaler → LogisticRegression(C=0.001, max_iter=5000, seed=42)` |
| 6 | `class_weight` | **`None`** — see below |
| 7 | remaining upgrades | none (C already optimal to 1e-6; 50-aa windows solidly worse) |

**On decision 6, the one open parameter.** The notebook resolves it by what the output is
for: `balanced` for ranking a candidate list, `None` when the probability has to mean
something on its own scale. This endpoint returns a bare number to a caller who cannot see
the rest of the batch, so it ships **`None`** — the better-calibrated branch (Brier 0.147
vs. 0.164). Ranking is unaffected either way; that is decision 6's whole finding.

Fitted on all 1,757 rows with no holdout, as the notebook specifies — cross-validation was
already spent on the grid.

**Held-out performance** (frozen 5-fold grouped CV, `filter|OpenCell/protein/6b|TERMINAL|logreg|u`):

| metric | value | no-skill |
|---|---|---|
| PR-AUC | 0.895 | 0.746 |
| PR-AUC, minority direction (positive = failure) | 0.605 ± 0.073 | 0.254 |
| Brier | 0.147 | — |

## Running it

```bash
python src/serve/train_final.py                  # writes models/taggability_6b_terminal_logreg.joblib
python src/serve/verify.py --n 8                 # serving path == training path

FLUORESCE_API_KEYS=<key1>,<key2> \
  uvicorn app:app --app-dir src/serve --host 0.0.0.0 --port 8000
```

| variable | default | |
|---|---|---|
| `FLUORESCE_API_KEYS` | *required* | comma-separated bearer tokens; the service refuses to start without one |
| `FLUORESCE_MODEL` | `models/taggability_6b_terminal_logreg.joblib` | bundle to serve |
| `FLUORESCE_DEVICE` | `cuda` | |
| `FLUORESCE_EAGER_LOAD` | unset | `1` loads the backbone at startup instead of on the first request |

The 6B backbone is ~13 GB on the GPU and takes ~9 s to load. Set `FLUORESCE_EAGER_LOAD=1`
behind a readiness probe so the first caller does not pay for it.

## API

### `POST /v1/score`

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"sequence": "MSKGEELFTG..."}'
```

```json
{
  "taggability": 0.9483,
  "percentile": 0.808,
  "model_version": "taggability_6b_terminal_logreg",
  "sequence": {"length": 452, "scored_length": 452, "truncated": false, "ambiguous_residues": []},
  "elapsed_ms": 49
}
```

`percentile` is where the score falls among the 1,757 held-out OpenCell predictions —
context a bare probability lacks, since the base rate is 0.746 and not 0.5.

Accepts a bare sequence or a single FASTA record; whitespace, digits, and a trailing `*`
are stripped. Sequences over 2,046 aa are middle-truncated exactly as in training (both
termini and therefore both tagging sites are preserved) and `truncated` is set.

`422` on: fewer than 20 aa, letters that are not amino acids, more than one FASTA record,
or a sequence whose alphabet is entirely `ACGTUN` — that last one is DNA someone forgot to
translate, and it would otherwise score as a confident, meaningless protein.

### `GET /v1/model`

Model card: full spec, held-out metrics, and the caveats below.

### `GET /healthz`

Unauthenticated liveness probe.

## Auth

Any valid bearer token passes; there are no per-key scopes. Tokens are compared with
`hmac.compare_digest`. Rotate by restarting with a new `FLUORESCE_API_KEYS`. Terminate TLS
in front of this — the tokens are bearer credentials in a header.

## What the score cannot tell you

Carried over from the notebook and `results/REPORT.md`, because they bound every number
this endpoint returns:

1. **The negatives are confounded.** A failed knock-in can be a bad guide RNA, low
   expression, poor HDR, or a failed sort — not intrinsic un-taggability. A low score is
   the model's estimate that *the OpenCell pipeline would have failed*, which is not the
   same claim as "this protein cannot be tagged".
2. **Generalization beyond OpenCell is unmeasured, not disproven.** 1,757 attempts,
   HEK293T, mNG2(11) split-fluorophore knock-in. There is no external evaluation set in
   the repo.
3. **The score is terminus-agnostic.** It does not say which end to tag. That is the
   separate terminus selector (300M, `TERMINAL`, logreg, PR-AUC 0.761), which is not
   served here — and which ranks but cannot gate (recall at 90% precision is 0.098).

## Throughput note

Decision 4 flagged that 6B is free on the 1,757 cached genes and costs 20× a 300M forward
pass on anything new. **Every request to this endpoint is a cache miss**, so that is the
cost being paid here: ~50 ms per sequence warm on an H100, versus ~5 ms for 300M, for
+0.05 minority PR-AUC. If this is ever used to screen large libraries rather than score
candidates one at a time, `train_final.py --model 300m` writes the cheaper head and
`FLUORESCE_MODEL` points at it — the featurizer reads the backbone from the bundle, so
nothing else changes. Requests are scored one sequence at a time and GPU forwards are
serialised behind a lock.
