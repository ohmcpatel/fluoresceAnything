"""Build the canonical modeling table: protein sequences + mNG2(11) fusion constructs.

Verifies OpenCell's tag/linker fasta against the HDR donor sequences in table S3 before
using it, then writes one row per library target with both the bare protein and the
construct with the tag placed at the real tagged terminus.
"""

import re

import pandas as pd

REPO = "/workspace/fluoresceAnything"
TABLE_S3 = f"{REPO}/data/science.abi6983_table_s3.xlsx"
SEQS = f"{REPO}/data/opencell_sequences.csv"
TAGS_DNA = f"{REPO}/data/seq_data/tags.fasta"
TAGS_PROT = f"{REPO}/data/seq_data/tags_translated.fasta"
OUT = f"{REPO}/data/modeling_table.csv"

BASES = "TCAG"
AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS = {
    a + b + c: AAS[i]
    for i, (a, b, c) in enumerate((a, b, c) for a in BASES for b in BASES for c in BASES)
}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def translate(dna):
    dna = dna.upper()
    return "".join(CODONS.get(dna[i : i + 3], "X") for i in range(0, len(dna) - 2, 3))


def revcomp(dna):
    return dna.upper().translate(COMPLEMENT)[::-1]


def read_fasta(path):
    records, name = {}, None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].strip()
            records[name] = ""
        else:
            records[name] += line
    return records


# ---------------------------------------------------------------- tag verification
dna = read_fasta(TAGS_DNA)
prot = read_fasta(TAGS_PROT)
print("tag records (dna): ", {k: len(v) for k, v in dna.items()})
print("tag records (prot):", {k: len(v) for k, v in prot.items()})

for dna_key, prot_key in [
    ("N_term_tag", "N_term_tag_prot"),
    ("C_term_tag", "C_term_tag_prot"),
    ("linker", "linker_prot"),
    ("splitGFP", "splitGFP_prot"),
]:
    got = translate(dna[dna_key])
    assert got == prot[prot_key], f"{dna_key}: translate -> {got!r} != {prot[prot_key]!r}"
print("check 1 OK: all four tag records translate to their stated peptides")

TAG = prot["splitGFP_prot"]
LINKER = prot["linker_prot"]
N_TAG = prot["N_term_tag_prot"]
C_TAG = prot["C_term_tag_prot"]
assert TAG == "TELNFKEWQKAFTDMM", TAG
assert LINKER == "GGGLEVLFQGPGSG", LINKER
assert N_TAG == TAG + LINKER and C_TAG == LINKER + TAG
assert "LEVLFQGP" in LINKER, "linker should carry the HRV-3C protease site"
print(f"check 2 OK: mNG2(11)={TAG} ({len(TAG)} aa), linker={LINKER} ({len(LINKER)} aa)")

# ---------------------------------------------------------------- library metadata
raw = pd.read_excel(TABLE_S3, sheet_name="library_success")
lib = raw[raw["library_success"].isin(["successful", "unsuccessful"])].copy()
lib["terminus"] = lib["terminus_tagged"].str.upper()
assert set(lib["terminus"]) == {"N", "C"}, set(lib["terminus"])
print(
    f"library: {len(lib)} rows  "
    f"{lib['library_success'].value_counts().to_dict()}  "
    f"{lib['terminus'].value_counts().to_dict()}"
)

# check 3: the tag fasta really is the cassette in OpenCell's own HDR donors
donors = lib["HDR_donor_sequence"].str.upper()
tag_dna = dna["splitGFP"].upper()
hits = donors.str.contains(tag_dna, regex=False) | donors.str.contains(
    revcomp(tag_dna), regex=False
)
print(f"check 3: tag DNA found in {hits.sum()}/{len(lib)} HDR donors (either strand)")
assert hits.sum() >= 1600, "tag fasta does not match the HDR donors -- wrong tag"

# ---------------------------------------------------------------- join sequences
seqs = pd.read_csv(SEQS)
print(f"sequences: {len(seqs)} rows, {seqs['ENSG'].nunique()} unique ENSG")
df = lib.merge(
    seqs[["ENSG", "uniprot_accession", "sequence", "sequence_length"]], on="ENSG", how="left"
)
assert df["sequence"].notna().all(), f"{df['sequence'].isna().sum()} rows without a sequence"
assert len(df) == len(lib)

bad = df[~df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXBZUO]+")]
assert bad.empty, f"non-standard residues in {len(bad)} sequences"

dupes = df[df.duplicated("ENSG", keep=False)]
print(
    f"duplicated ENSG: {dupes['gene_name'].tolist()} "
    f"terminus={dupes['terminus'].tolist()} label={dupes['library_success'].tolist()}"
)

# ---------------------------------------------------------------- constructs
def build(row):
    seq = row["sequence"]
    if row["terminus"] == "N":
        # the cassette supplies the initiator Met; the native Met is replaced
        return "M" + N_TAG + seq[1:], 1 + len(N_TAG)
    return seq + C_TAG, len(seq)


built = df.apply(build, axis=1, result_type="expand")
df["construct"] = built[0]
df["junction_idx"] = built[1]
df["construct_length"] = df["construct"].str.len()
df["label"] = (df["library_success"] == "successful").astype(int)

# every construct = protein with exactly one 30 aa cassette added
assert (df["construct_length"] - df["sequence_length"] == 30).all()
n_mask = df["terminus"] == "N"
assert df.loc[n_mask, "construct"].str.startswith("M" + N_TAG).all()
assert df.loc[~n_mask, "construct"].str.endswith(C_TAG).all()
# junction index points at the first residue after the boundary
for _, r in df.iterrows():
    assert r["construct"][r["junction_idx"] - 1] != "" and 0 < r["junction_idx"] < r["construct_length"]
print("check 4 OK: constructs well-formed, +30 aa each")

out = df[
    [
        "ENSG",
        "gene_name",
        "uniprot_accession",
        "terminus",
        "library_success",
        "label",
        "sequence_length",
        "sequence",
        "construct_length",
        "construct",
        "junction_idx",
        "log_hek_RNA_tpm",
        "log_hek_conc_nM",
        "is_essential",
    ]
].rename(columns={"sequence": "protein", "sequence_length": "protein_length"})
out.to_csv(OUT, index=False)
print(f"\nwrote {len(out)} rows -> {OUT}")

# ---------------------------------------------------------------- eyeball
for term in ["N", "C"]:
    r = out[out["terminus"] == term].iloc[0]
    c = r["construct"]
    print(f"\n--- {term}-terminal example: {r['gene_name']} ({r['library_success']})")
    print(f"    protein  [{r['protein_length']:5d}] {r['protein'][:40]} ... {r['protein'][-40:]}")
    print(f"    construct[{r['construct_length']:5d}] {c[:50]} ... {c[-50:]}")
    print(f"    junction at {r['junction_idx']}: ...{c[r['junction_idx'] - 12:r['junction_idx']]}|{c[r['junction_idx']:r['junction_idx'] + 12]}...")
