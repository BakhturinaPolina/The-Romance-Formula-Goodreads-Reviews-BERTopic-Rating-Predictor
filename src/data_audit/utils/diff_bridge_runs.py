#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path, help="old diagnostics_summary.json")
    ap.add_argument("--new", required=True, type=Path, help="new diagnostics_summary.json")
    args = ap.parse_args()

    a = json.loads(Path(args.old).read_text())
    b = json.loads(Path(args.new).read_text())

    keys = [
        ("book_table","n_rows"),
        ("book_table","issues","flag_len_mismatch_rows"),
        ("raw_long","n_rows"),
        ("canon_long","n_rows"),
        ("segments_long","n_rows"),
        ("coverage_vs_canon","canon_values_missing_in_map"),
        ("noncontent_alignment","false_positives"),
        ("noncontent_alignment","false_negatives"),
    ]

    print("=== DIFF ===")
    for k in keys:
        va, vb = a, b
        for part in k:
            va = None if va is None else va.get(part)
            vb = None if vb is None else vb.get(part)
        print(f"{'.'.join(k)}: {va}  ->  {vb}")

if __name__ == "__main__":
    main()
