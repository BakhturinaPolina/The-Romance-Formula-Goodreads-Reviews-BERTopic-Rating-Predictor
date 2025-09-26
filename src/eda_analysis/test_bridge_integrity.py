import json
from pathlib import Path
import pandas as pd
import numpy as np

# Set these to your real paths or pass via env in CI
BRIDGE_DIR = Path("bridge_outputs")
NORM_DIR = Path("shelf_norm_outputs")

def _read_parquet(p: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")

def test_files_exist_and_nonempty():
    files = [
        BRIDGE_DIR/"books_with_shelf_norm.parquet",
        BRIDGE_DIR/"shelves_raw_long.parquet",
        BRIDGE_DIR/"shelves_canon_long.parquet",
        BRIDGE_DIR/"segments_long.parquet",
        NORM_DIR/"shelf_canonical.csv",
        NORM_DIR/"shelf_segments.csv",
        NORM_DIR/"noncontent_shelves.csv",
    ]
    for f in files:
        print(f"[exists] {f}")
        assert f.exists(), f"Missing required file: {f}"
        assert f.stat().st_size > 0, f"Empty file: {f}"

def test_book_table_basic_integrity():
    books = _read_parquet(BRIDGE_DIR/"books_with_shelf_norm.parquet")
    print(f"[books] rows={len(books)} cols={books.columns.tolist()}")

    for c in ["work_id","shelves","shelves_canon","shelves_canon_content","shelves_noncontent_flags"]:
        assert c in books.columns, f"Missing column {c}"

    # Flags aligned with canon
    len_diff = (books["shelves_canon"].map(len) - books["shelves_noncontent_flags"].map(len)).abs()
    mism = (len_diff > 0).sum()
    print(f"[books] flag length mismatches: {mism}")
    assert mism == 0, "Noncontent flags misaligned with canon shelves"

def test_long_tables_look_ok():
    raw_long = _read_parquet(BRIDGE_DIR/"shelves_raw_long.parquet")
    canon_long = _read_parquet(BRIDGE_DIR/"shelves_canon_long.parquet")
    seg_long = _read_parquet(BRIDGE_DIR/"segments_long.parquet")

    print(f"[raw_long] {len(raw_long)} rows; cols={raw_long.columns.tolist()[:6]}")
    print(f"[canon_long] {len(canon_long)} rows")
    print(f"[segments_long] {len(seg_long)} rows")

    for df, val_col in [(raw_long,"shelf_raw"), (canon_long,"shelf_canon")]:
        assert df[val_col].notna().all(), f"{val_col} has nulls"
        empties = df[val_col].astype(str).str.strip().eq("").sum()
        print(f"[{val_col}] empties={empties}")
        assert empties == 0, f"{val_col} has empty strings"

    # Segments look sane
    assert seg_long["segment"].notna().all(), "segments have nulls"
    empties = seg_long["segment"].astype(str).str.strip().eq("").sum()
    print(f"[segment] empties={empties}")
    assert empties == 0, "segments have empty strings"

def test_canonical_coverage_print():
    canon_map = pd.read_csv(NORM_DIR/"shelf_canonical.csv", dtype=str).fillna("")
    canon_long = _read_parquet(BRIDGE_DIR/"shelves_canon_long.parquet")
    canon_set = set(canon_map["shelf_canon"].astype(str).str.strip().str.casefold())
    output_set = set(canon_long["shelf_canon"].astype(str).str.strip().str.casefold())
    missing = sorted([x for x in output_set if x not in canon_set])
    print(f"[coverage] canon_in_output={len(output_set)}; in_map={len(canon_set)}; missing={len(missing)}")
    # Not a hard fail—print top 20 to help debugging
    print("[coverage] sample_missing:", missing[:20])

def test_noncontent_alignment_print():
    books = _read_parquet(BRIDGE_DIR/"books_with_shelf_norm.parquet")
    nc = pd.read_csv(NORM_DIR/"noncontent_shelves.csv", dtype=str).fillna("")
    col = "shelf" if "shelf" in nc.columns else ("shelf_canon" if "shelf_canon" in nc.columns else "shelf_raw")
    nc_keys = set(nc[col].astype(str).str.strip().str.casefold())

    pairs = []
    for canon, flags in zip(books["shelves_canon"], books["shelves_noncontent_flags"]):
        for c, f in zip(canon, flags):
            pairs.append((" ".join(str(c).strip().casefold().split()), bool(f)))
    df = pd.DataFrame(pairs, columns=["canon_key","flag"])

    fp = df[(df["flag"]) & (~df["canon_key"].isin(nc_keys))]
    fn = df[(~df["flag"]) & (df["canon_key"].isin(nc_keys))]
    print(f"[noncontent] pairs={len(df)} fp={len(fp)} fn={len(fn)} fp_rate={100*len(fp)/max(len(df),1):.2f}% fn_rate={100*len(fn)/max(len(df),1):.2f}%")

    # Gentle guardrail: they shouldn't explode
    assert len(fp) / max(len(df),1) < 0.05, "Too many FP in noncontent flags"
    assert len(fn) / max(len(df),1) < 0.10, "Too many FN in noncontent flags"
