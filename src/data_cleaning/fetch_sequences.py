import io
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

DATA = "/workspace/fluoresceAnything/data/science.abi6983_table_s3.xlsx"
OUT = "/workspace/fluoresceAnything/data/opencell_sequences.csv"
SHEET = "library_success"
WORKERS = 16

FIELDS = "accession,id,gene_names,organism_id,xref_ensembl,length,sequence"
URL = "https://rest.uniprot.org/uniprotkb/search"

df = pd.read_excel(DATA, sheet_name=SHEET)
lib = df[df["library_success"].isin(["successful", "unsuccessful"])].copy()
genes = (
    lib[["ENSG", "gene_name"]]
    .dropna(subset=["ENSG"])
    .drop_duplicates("ENSG")
    .sort_values("ENSG")
)
print(f"{len(genes)} unique ENSGs to fetch")

results = []
errors = []
existing = pd.DataFrame()
if pd.io.common.file_exists(OUT):
    existing = pd.read_csv(OUT)
    done = set(existing["ENSG"].astype(str))
    print(f"resuming: {len(done)} already fetched")
    genes = genes[~genes["ENSG"].astype(str).isin(done)]
results = existing.to_dict("records")

session = requests.Session()


def fetch(ensg, reviewed):
    query = f"(xref:{ensg}) AND (organism_id:9606)"
    if reviewed:
        query += " AND (reviewed:true)"
    params = {
        "format": "tsv",
        "fields": FIELDS,
        "query": query,
        "size": "10",
    }
    url = f"{URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(0.5 + attempt)
                continue
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(0.3)
    return None


def work(gene):
    ensg, gene_name = gene
    r = fetch(ensg, reviewed=True)
    if r is None:
        return {"ENSG": ensg, "gene_name": gene_name, "error": "request_error"}
    hit = pd.read_csv(io.StringIO(r.text), sep="\t")
    if hit.empty:
        r = fetch(ensg, reviewed=False)
        if r is None:
            return {"ENSG": ensg, "gene_name": gene_name, "error": "request_error"}
        hit = pd.read_csv(io.StringIO(r.text), sep="\t")
        if hit.empty:
            return {"ENSG": ensg, "gene_name": gene_name, "error": "no_entry"}
    hit.columns = [c.lower().replace(" ", "_") for c in hit.columns]
    top = hit.iloc[0]
    return {
        "ENSG": ensg,
        "gene_name": gene_name,
        "uniprot_accession": top["entry"],
        "uniprot_id": top["entry_name"],
        "uniprot_gene_names": top["gene_names"],
        "sequence_length": top["length"],
        "sequence": top["sequence"],
    }


results = existing.to_dict("records")
errors = []
todo = list(genes.itertuples(index=False))
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(work, (g.ENSG, g.gene_name)): g for g in todo}
    for i, fut in enumerate(as_completed(futs)):
        res = fut.result()
        if "error" in res:
            errors.append(res)
        else:
            results.append(res)
        if (i + 1) % 400 == 0:
            elapsed = time.time() - t0
            print(
                f"...{i + 1}/{len(todo)}  ({elapsed / 60:.1f} min, {len(results)} ok)",
                flush=True,
            )
            combined = pd.DataFrame(results).drop_duplicates("ENSG", keep="last")
            combined.to_csv(OUT, index=False)

seqs = pd.DataFrame(results)
seqs.to_csv(OUT, index=False)
print(f"\nsaved {len(seqs)} sequences -> {OUT}")
print(f"errors: {len(errors)}")
for e in errors:
    print("   ", e["ENSG"], e["gene_name"], e["error"])
