"""Taggability scoring endpoint.

POST a protein sequence with a bearer token, get back one continuous value in [0, 1]:
the model's probability that an mNG2(11) knock-in tagging attempt on that protein
succeeds. 0 = the model expects failure, 1 = the model expects success. No threshold is
applied and no success/failure label is returned -- the score is the output.

    FLUORESCE_API_KEYS=key1,key2 uvicorn app:app --app-dir src/serve --host 0.0.0.0 --port 8000

Environment:
    FLUORESCE_API_KEYS   required, comma-separated bearer tokens. No default: the service
                         refuses to start rather than come up unauthenticated.
    FLUORESCE_MODEL      path to the joblib bundle from train_final.py
                         (default models/taggability_6b_terminal_logreg.joblib)
    FLUORESCE_DEVICE     cuda (default) or cpu
    FLUORESCE_EAGER_LOAD 1 to pull the backbone onto the GPU at startup instead of on the
                         first request -- the 6B backbone takes ~1 min and 13 GB
"""

import hmac
import os
import sys
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from featurize import Featurizer, SequenceError, clean_sequence

REPO = "/workspace/fluoresceAnything"
DEFAULT_MODEL = f"{REPO}/models/taggability_6b_terminal_logreg.joblib"


# ------------------------------------------------------------------ auth
def _load_keys():
    raw = os.environ.get("FLUORESCE_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError(
            "FLUORESCE_API_KEYS is empty -- set at least one bearer token. The endpoint is "
            "public, so it will not start without one."
        )
    return keys


API_KEYS = _load_keys()
bearer = HTTPBearer(auto_error=False)


def require_key(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    """Any valid bearer token is sufficient -- that is the whole authorization model.
    Compared with compare_digest so a wrong key cannot be recovered by timing."""
    token = creds.credentials if creds else ""
    ok = any(hmac.compare_digest(token, k) for k in API_KEYS)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# ------------------------------------------------------------------ model
class Scorer:
    def __init__(self, path):
        bundle = joblib.load(path)
        self.path = path
        self.estimator = bundle["estimator"]
        self.spec = bundle["spec"]
        self.held_out = bundle["held_out"]
        self.oof_scores = np.asarray(self.held_out["oof_scores"])  # sorted
        self.featurizer = Featurizer(
            self.spec["backbone"],
            self.spec["blocks"],
            device=os.environ.get("FLUORESCE_DEVICE", "cuda"),
        )

    @property
    def version(self):
        return os.path.basename(self.path).replace(".joblib", "")

    def score(self, seq):
        x, truncated = self.featurizer.embed(seq)
        if x.shape[1] != self.spec["n_features"]:
            raise RuntimeError(
                f"featurizer produced {x.shape[1]} features, head expects "
                f"{self.spec['n_features']} -- backbone and bundle disagree"
            )
        p = float(self.estimator.predict_proba(x)[0, 1])
        return p, truncated

    def percentile(self, p):
        """Where this score sits among the 1,757 held-out training predictions -- context
        for a bare probability, since the base rate is 0.746 and not 0.5."""
        return float(np.searchsorted(self.oof_scores, p) / len(self.oof_scores))


scorer = Scorer(os.environ.get("FLUORESCE_MODEL", DEFAULT_MODEL))


# ------------------------------------------------------------------ schema
class ScoreRequest(BaseModel):
    sequence: str = Field(
        ...,
        description="Amino-acid sequence, bare or as a single FASTA record.",
        examples=["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTL"],
    )


class SequenceInfo(BaseModel):
    length: int
    scored_length: int
    truncated: bool
    ambiguous_residues: list[str]


class ScoreResponse(BaseModel):
    taggability: float = Field(
        ..., description="Continuous score in [0, 1]. 0 = untaggable, 1 = perfectly taggable."
    )
    percentile: float = Field(
        ..., description="Fraction of held-out OpenCell proteins scoring at or below this."
    )
    model_version: str
    sequence: SequenceInfo
    elapsed_ms: int


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("FLUORESCE_EAGER_LOAD") == "1":
        t0 = time.time()
        await run_in_threadpool(scorer.featurizer.load)
        print(f"[startup] backbone {scorer.spec['backbone']} resident in {time.time() - t0:.0f}s")
    yield


app = FastAPI(
    title="fluoresceAnything taggability",
    version=scorer.version,
    lifespan=lifespan,
    description=(
        "Continuous taggability score for a protein sequence. ESM-C "
        f"{scorer.spec['backbone'].upper()} TERMINAL embedding -> logistic regression, the "
        "configuration locked in notebooks/decision_to_classifier.ipynb."
    ),
)


@app.exception_handler(SequenceError)
def _bad_sequence(request: Request, exc: SequenceError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/healthz")
def healthz():
    """Unauthenticated liveness probe. Deliberately says nothing about the model."""
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_key)])
def score(req: ScoreRequest):
    t0 = time.time()
    seq, ambiguous = clean_sequence(req.sequence)
    p, truncated = scorer.score(seq)
    return ScoreResponse(
        taggability=round(p, 4),
        percentile=round(scorer.percentile(p), 3),
        model_version=scorer.version,
        sequence=SequenceInfo(
            length=len(seq),
            scored_length=min(len(seq), scorer.spec["max_len"]),
            truncated=truncated,
            ambiguous_residues=ambiguous,
        ),
        elapsed_ms=int((time.time() - t0) * 1000),
    )


@app.get("/v1/model", dependencies=[Depends(require_key)])
def model_card():
    """What the score is, and what it is worth -- held-out numbers from the frozen 5-fold
    sweep, plus the caveats that bound them."""
    h = scorer.held_out
    return {
        "model_version": scorer.version,
        "spec": scorer.spec,
        "backbone_loaded": scorer.featurizer.loaded,
        "held_out": {k: v for k, v in h.items() if k != "oof_scores"},
        "interpretation": {
            "score": "P(knock-in tagging attempt succeeds), continuous in [0, 1], no threshold applied",
            "base_rate": h["base_rate"],
            "note": (
                "A score below the 0.746 base rate is the model arguing against this protein; "
                "the percentile field places it among held-out OpenCell proteins."
            ),
        },
        "caveats": [
            "Negatives are confounded: a failed knock-in can be a bad guide RNA, low expression, "
            "poor HDR, or a failed sort -- not intrinsic un-taggability. Scores are capped by that "
            "label noise and a low score is not proof a protein cannot be tagged.",
            "Trained on 1,757 OpenCell attempts in HEK293T with an mNG2(11) knock-in. "
            "Generalization beyond that assay is unmeasured, not disproven.",
            "The score is terminus-agnostic: it does not say which terminus to tag.",
        ],
    }
