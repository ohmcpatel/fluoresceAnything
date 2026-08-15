import pandas as pd

DATA = "/workspace/fluoresceAnything/data/science.abi6983_table_s3.xlsx"
SHEET = "library_success"

df = pd.read_excel(DATA, sheet_name=SHEET)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print()
print(df.head(10).to_string())
print()

LIBRARY = df["library_success"].isin(["successful", "unsuccessful"])
lib = df[LIBRARY].copy()
print("=== library rows (successful + unsuccessful):", len(lib), "===")
print(lib["library_success"].value_counts(dropna=False).to_string())
print()

print("=== Q1: clean terminus labels ===")
print("terminus_tagged value counts (ALL rows, dropna=False):")
print(df["terminus_tagged"].value_counts(dropna=False).to_string())
print()
print("terminus_tagged value counts (library rows only):")
print(lib["terminus_tagged"].value_counts(dropna=False).to_string())
print()
clean = lib["terminus_tagged"].isin(["N", "C"])
print("library rows with clean C/N terminus:", clean.sum(), "/", len(lib))
print("library rows with non-C/N or missing terminus:", (~clean).sum())
print()
print("successful subset:")
succ = lib[lib["library_success"] == "successful"]
print("  successful total:", len(succ))
print("  successful with clean C/N terminus:", succ["terminus_tagged"].isin(["N", "C"]).sum())
print(succ["terminus_tagged"].value_counts(dropna=False).to_string())
print()

print("=== Q2: C:N split ===")
def cn_split(sub):
    c = (sub["terminus_tagged"].astype(str).str.upper() == "C").sum()
    n = (sub["terminus_tagged"].astype(str).str.upper() == "N").sum()
    return c, n

for name, sub in [("successful", succ), ("unsuccessful", lib[lib["library_success"] == "unsuccessful"]), ("all library", lib)]:
    c, n = cn_split(sub)
    tot = c + n
    print(f"  {name}: C={c} N={n}  (C {c/tot:.1%} : N {n/tot:.1%})")
print()

print("=== Q3: join key integrity ===")
print("ENSG missing (all rows):", df["ENSG"].isna().sum())
print("ENSG missing (library rows):", lib["ENSG"].isna().sum())
print("ENSG duplicated (all rows):", df["ENSG"].duplicated().sum())
print("ENSG duplicated (library rows):", lib["ENSG"].duplicated().sum())
print("gene_name missing (library rows):", lib["gene_name"].isna().sum())
print("protein_name missing (library rows):", lib["protein_name"].isna().sum())
print("dup (ENSG, terminus) pairs in library:", lib.duplicated(subset=["ENSG", "terminus_tagged"]).sum())
print()

print("=== corner cases to eyeball ===")
print(lib[lib["terminus_tagged"].isin(["c"])].to_string())
