"""Turn one protein sequence into the 7,680-d feature vector the deployment head expects.

This deliberately imports `middle_truncate` and `pool` from the embedding script that built
the training cache rather than reimplementing them. A serving featurizer that drifts from
the training featurizer is the classic silent failure here, and `verify.py` checks the two
still agree by re-embedding cached genes and comparing to `cache/emb/*.npz`.
"""

import os
import re
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embed"))

import numpy as np

from esmc_embed import MAX_LEN, Esmc6B, Esmc300M, middle_truncate, pool

# The 20 canonical residues -- everything the training table contains.
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
# ESM-C tokenizes these too, but no training sequence had one, so they are reported back.
AMBIGUOUS = set("XBUZO")
# A, C, G, T are all valid amino-acid letters, so a submitted nucleotide sequence passes
# every character check and returns a confident, meaningless score. Nothing but the
# composition distinguishes it: a real 20+ aa protein drawn from only these letters is a
# ~1e-14 event, so an exclusively-nucleotide alphabet is a translation the caller forgot.
NUCLEOTIDES = set("ACGTUN")

MIN_LEN = 20  # shortest training protein is 37 aa; below ~20 the terminal windows overlap fully


class SequenceError(ValueError):
    """Input that cannot be scored, with a message meant for the caller."""


def clean_sequence(raw):
    """Accept a bare sequence or a single FASTA record. Returns (sequence, notes)."""
    if not isinstance(raw, str):
        raise SequenceError("sequence must be a string")

    lines = [ln for ln in raw.strip().splitlines()]
    if sum(ln.startswith(">") for ln in lines) > 1:
        raise SequenceError("multiple FASTA records -- submit one sequence per request")
    seq = "".join(ln for ln in lines if not ln.startswith(">"))
    seq = re.sub(r"[\s\d]", "", seq).upper()
    seq = seq.rstrip("*")  # trailing stop codon from a translation

    if not seq:
        raise SequenceError("no residues found in the submitted sequence")

    bad = sorted(set(seq) - CANONICAL - AMBIGUOUS)
    if bad:
        raise SequenceError(
            f"unrecognised residue letters: {''.join(bad)} -- expected amino acids, not nucleotides"
        )
    if len(seq) < MIN_LEN:
        raise SequenceError(f"sequence is {len(seq)} aa; the minimum is {MIN_LEN} aa")
    if set(seq) <= NUCLEOTIDES:
        raise SequenceError(
            "sequence looks like DNA/RNA, not protein (only the letters "
            f"{''.join(sorted(set(seq)))} appear) -- submit the translated amino-acid sequence"
        )

    return seq, sorted(set(seq) & AMBIGUOUS)


class Featurizer:
    """Holds the ESM-C backbone. Loading is lazy so the process can start before the
    weights are resident, and forward passes are serialised -- one GPU, many workers."""

    def __init__(self, backbone, blocks, device="cuda"):
        self.backbone_name = backbone
        self.blocks = list(blocks)
        self.device = device
        self._backend = None
        self._lock = threading.Lock()

    @property
    def loaded(self):
        return self._backend is not None

    def load(self):
        with self._lock:
            if self._backend is None:
                cls = Esmc300M if self.backbone_name == "300m" else Esmc6B
                self._backend = cls(device=self.device)
        return self

    def embed(self, seq):
        """Returns (feature_vector, truncated). Mirrors the cache build exactly:
        middle-truncate at 2046, run the backbone, slice off <cls>/<eos>, pool."""
        if self._backend is None:
            self.load()

        truncated = len(seq) > MAX_LEN
        trimmed, _ = middle_truncate(seq, None)

        with self._lock:
            ids, mask = self._backend.tokenize([trimmed])
            emb = self._backend.forward(ids, mask).float().cpu()

        residues = emb[0, 1 : 1 + len(trimmed)]
        assert residues.shape[0] == len(trimmed), "special-token layout is not <cls>+residues+<eos>"

        feats = pool(residues, None)
        x = np.concatenate([feats[b] for b in self.blocks]).astype(np.float32)
        return x.reshape(1, -1), truncated
