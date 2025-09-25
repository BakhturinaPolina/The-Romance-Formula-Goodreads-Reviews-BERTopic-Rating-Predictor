# File: src/deduplication/post_cluster_diagnostics.py
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
import time

def load_outputs(out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all output files from dedupe pipeline."""
    out = Path(out_dir)
    pairs = pd.read_parquet(out / "pairs_filtered.parquet")
    samples = pd.read_parquet(out / "clusters_edges_samples.parquet")
    cmap = pd.read_parquet(out / "clusters_token_map.parquet")      # token -> cluster_id
    csum = pd.read_parquet(out / "clusters_summary.parquet")        # cluster stats
    return pairs, samples, cmap, csum

def validate_samples_against_filtered(pairs: pd.DataFrame, samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Validate that all edge samples exist in filtered pairs and meet threshold."""
    # Normalize types
    pairs = pairs.astype({"token_a":"string","token_b":"string"})
    samples = samples.astype({"token_a":"string","token_b":"string"})

    # Join both (both directions)
    key_cols = ["token_a","token_b","cosine_sim"]
    merged = samples.merge(pairs[key_cols], on=["token_a","token_b","cosine_sim"], how="left", indicator=True)
    
    # Try swapped direction for any that didn't match
    if (merged["_merge"] != "both").any():
        swapped = samples.rename(columns={"token_a":"token_b","token_b":"token_a"})
        merged2 = merged.loc[merged["_merge"]!="both"].drop(columns=["_merge"]).merge(
            pairs[key_cols], on=["token_a","token_b","cosine_sim"], how="left", indicator=True
        )
        merged.loc[merged["_merge"]!="both","_merge"] = merged2["_merge"].values

    # Find invalid samples (not in filtered pairs OR below threshold)
    invalid = merged[(merged["_merge"]!="both") | (merged["cosine_sim"] < threshold)].copy()
    return invalid[["cluster_id","token_a","token_b","cosine_sim","_merge"]].sort_values("cosine_sim")

def rebuild_edge_samples_from_filtered(pairs: pd.DataFrame, cmap: pd.DataFrame, sample_per_cluster: int = 5) -> pd.DataFrame:
    """Rebuild edge samples directly from filtered pairs, ensuring threshold compliance."""
    # Map any edge to cluster via token_a (both tokens must fall in same cluster for connected components)
    cmap = cmap.astype({"token":"string"})
    pairs = pairs.astype({"token_a":"string","token_b":"string"})
    
    # Join edges with cluster mappings
    m = pairs.merge(cmap.rename(columns={"token":"token_a","cluster_id":"cid_a"}), on="token_a", how="left")
    m = m.merge(cmap.rename(columns={"token":"token_b","cluster_id":"cid_b"}), on="token_b", how="left")
    
    # Keep only edges within same cluster
    m = m[m["cid_a"].notna() & (m["cid_a"] == m["cid_b"])]
    m["cluster_id"] = m["cid_a"].astype("int64")

    # Take top-k strongest edges per cluster for auditing
    m = m.sort_values(["cluster_id","cosine_sim"], ascending=[True, False])
    sampled = m.groupby("cluster_id").head(sample_per_cluster).reset_index(drop=True)
    return sampled[["cluster_id","token_a","token_b","cosine_sim"]]

def cluster_cohesion_metrics(pairs: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """Compute cluster cohesion metrics: min/mean/max sim, triangle rate."""
    cmap = cmap.astype({"token":"string"})
    pairs = pairs.astype({"token_a":"string","token_b":"string"})

    # Map edges to cluster id (intra-cluster only)
    m = pairs.merge(cmap.rename(columns={"token":"token_a","cluster_id":"cid_a"}), on="token_a", how="left")
    m = m.merge(cmap.rename(columns={"token":"token_b","cluster_id":"cid_b"}), on="token_b", how="left")
    m = m[m["cid_a"].notna() & (m["cid_a"] == m["cid_b"])].copy()
    m["cluster_id"] = m["cid_a"].astype("int64")

    # Basic cohesion stats
    g = m.groupby("cluster_id")["cosine_sim"]
    coh = g.agg(min_sim="min", mean_sim="mean", max_sim="max", edges="count").reset_index()

    # Triangle check: for each cluster, approximate triangle rate via shared-neighbor count
    def triangle_rate(df):
        """Calculate triangle rate for a cluster (local consistency measure)."""
        # Build adjacency for the cluster
        pairs_set = set(zip(df["token_a"], df["token_b"]))
        tokens = pd.unique(df[["token_a","token_b"]].values.ravel("K"))
        neigh = {t:set() for t in tokens}
        for a,b in pairs_set:
            neigh[a].add(b); neigh[b].add(a)
        
        tri = 0; tot = 0
        for a,b in pairs_set:
            tot += 1
            if len(neigh[a].intersection(neigh[b])) > 0:
                tri += 1
        return tri / tot if tot else 0.0

    tri = m.groupby("cluster_id").apply(triangle_rate).rename("triangle_rate").reset_index()
    return coh.merge(tri, on="cluster_id")

def prune_weak_bridges(pairs: pd.DataFrame, cmap: pd.DataFrame, min_sim: float = 0.65, require_triangle: bool = True) -> pd.DataFrame:
    """
    Return a filtered edge list: drop edges below min_sim; if require_triangle=True,
    drop edges that don't share a neighbor ('triangle constraint'). Use this to re-cluster if needed.
    """
    cmap = cmap.astype({"token":"string"})
    pairs = pairs.astype({"token_a":"string","token_b":"string"})
    
    # Basic sim filter
    keep = pairs[pairs["cosine_sim"] >= min_sim].copy()

    # Triangle filter (local consistency)
    if require_triangle:
        # Build neighbor map per connected bucket to avoid O(N^2) across all
        df = keep.copy()
        tokens = pd.unique(df[["token_a","token_b"]].values.ravel("K"))
        neigh = {t:set() for t in tokens}
        for a,b in zip(df["token_a"], df["token_b"]):
            neigh[a].add(b); neigh[b].add(a)
        
        mask = []
        for a,b in zip(df["token_a"], df["token_b"]):
            mask.append(len(neigh[a].intersection(neigh[b])) > 0)
        keep = df[pd.Series(mask, index=df.index)]
    
    return keep

def mutual_nearest_neighbors(pairs: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Keep only edges where both tokens are in each other's top-k nearest neighbors.
    This prevents hub chaining where one token connects to many others.
    """
    # Get top-k neighbors for each token
    top_k_neighbors = {}
    
    # Group by token_a to get top-k for each token
    for token, group in pairs.groupby("token_a"):
        top_k = group.nlargest(k, "cosine_sim")["token_b"].tolist()
        top_k_neighbors[token] = set(top_k)
    
    # Also group by token_b to get top-k for each token
    for token, group in pairs.groupby("token_b"):
        top_k = group.nlargest(k, "cosine_sim")["token_a"].tolist()
        if token in top_k_neighbors:
            top_k_neighbors[token].update(top_k)
        else:
            top_k_neighbors[token] = set(top_k)
    
    # Keep only mutual nearest neighbors
    mutual_edges = []
    for _, row in pairs.iterrows():
        a, b = row["token_a"], row["token_b"]
        if (a in top_k_neighbors.get(b, set()) and 
            b in top_k_neighbors.get(a, set())):
            mutual_edges.append(row)
    
    return pd.DataFrame(mutual_edges)

def flag_low_quality_clusters(coh: pd.DataFrame, min_sim_threshold: float = 0.62, min_triangle_rate: float = 0.10) -> pd.DataFrame:
    """Flag clusters with low cohesion metrics."""
    bad_cids = coh[
        (coh["min_sim"] < min_sim_threshold) | 
        (coh["triangle_rate"] < min_triangle_rate)
    ].sort_values(["min_sim","triangle_rate"])
    return bad_cids

def generate_diagnostic_report(out_dir: Path, threshold: float, 
                             min_sim_threshold: float = 0.62, 
                             min_triangle_rate: float = 0.10) -> Dict:
    """Generate comprehensive diagnostic report."""
    print(f"\n=== DIAGNOSTIC REPORT for {out_dir} ===")
    print(f"Threshold used: {threshold}")
    
    # Load data
    pairs, samples, cmap, csum = load_outputs(out_dir)
    
    print(f"\nLoaded data:")
    print(f"  - Filtered pairs: {len(pairs):,} edges")
    print(f"  - Edge samples: {len(samples):,} edges")
    print(f"  - Token clusters: {len(cmap):,} tokens")
    print(f"  - Cluster summary: {len(csum):,} clusters")
    
    # 1. Validate samples
    print(f"\n=== 1. VALIDATE EDGE SAMPLES ===")
    invalid = validate_samples_against_filtered(pairs, samples, threshold)
    if len(invalid):
        print(f"❌ Found {len(invalid)} invalid samples (below threshold or not from filtered set):")
        print(invalid.head(10))
        print(f"   Min sim in invalid samples: {invalid['cosine_sim'].min():.3f}")
        print(f"   Max sim in invalid samples: {invalid['cosine_sim'].max():.3f}")
    else:
        print("✅ All sample edges are valid and meet threshold.")
    
    # 2. Rebuild samples
    print(f"\n=== 2. REBUILD EDGE SAMPLES ===")
    new_samples = rebuild_edge_samples_from_filtered(pairs, cmap, sample_per_cluster=5)
    print(f"Rebuilt {len(new_samples):,} edge samples from filtered pairs")
    print(f"Sample sim range: {new_samples['cosine_sim'].min():.3f} - {new_samples['cosine_sim'].max():.3f}")
    
    # 3. Cluster cohesion
    print(f"\n=== 3. CLUSTER COHESION METRICS ===")
    coh = cluster_cohesion_metrics(pairs, cmap)
    print(f"Cluster cohesion summary:")
    print(f"  - Min sim range: {coh['min_sim'].min():.3f} - {coh['min_sim'].max():.3f}")
    print(f"  - Mean sim range: {coh['mean_sim'].min():.3f} - {coh['mean_sim'].max():.3f}")
    print(f"  - Triangle rate range: {coh['triangle_rate'].min():.3f} - {coh['triangle_rate'].max():.3f}")
    
    # 4. Flag low quality clusters
    print(f"\n=== 4. LOW QUALITY CLUSTERS ===")
    bad_clusters = flag_low_quality_clusters(coh, min_sim_threshold, min_triangle_rate)
    print(f"Found {len(bad_clusters)} low-quality clusters:")
    if len(bad_clusters):
        print(bad_clusters.head(10))
    
    # 5. Generate report
    report = {
        "timestamp": time.time(),
        "threshold_used": threshold,
        "data_summary": {
            "filtered_pairs": len(pairs),
            "edge_samples": len(samples),
            "tokens": len(cmap),
            "clusters": len(csum)
        },
        "validation": {
            "invalid_samples": len(invalid),
            "min_invalid_sim": float(invalid["cosine_sim"].min()) if len(invalid) else None,
            "max_invalid_sim": float(invalid["cosine_sim"].max()) if len(invalid) else None
        },
        "cohesion_summary": {
            "min_sim_range": [float(coh["min_sim"].min()), float(coh["min_sim"].max())],
            "mean_sim_range": [float(coh["mean_sim"].min()), float(coh["mean_sim"].max())],
            "triangle_rate_range": [float(coh["triangle_rate"].min()), float(coh["triangle_rate"].max())]
        },
        "low_quality_clusters": {
            "count": len(bad_clusters),
            "min_sim_threshold": min_sim_threshold,
            "min_triangle_rate": min_triangle_rate
        }
    }
    
    return report, new_samples, coh, bad_clusters

if __name__ == "__main__":
    # Example usage - update these paths
    OUT = "output_demo"  # change to your output dir
    TH = 0.6            # use the threshold you ran with
    
    out_path = Path(OUT)
    if not out_path.exists():
        print(f"❌ Output directory {OUT} does not exist")
        exit(1)
    
    # Generate report
    report, new_samples, coh, bad_clusters = generate_diagnostic_report(out_path, TH)
    
    # Save outputs
    new_samples.to_parquet(out_path / "clusters_edges_samples_fixed.parquet", index=False)
    coh.to_parquet(out_path / "cluster_cohesion_metrics.parquet", index=False)
    bad_clusters.to_parquet(out_path / "clusters_flagged_low_quality.parquet", index=False)
    
    # Save report
    with open(out_path / "diagnostic_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Saved outputs:")
    print(f"  - clusters_edges_samples_fixed.parquet")
    print(f"  - cluster_cohesion_metrics.parquet") 
    print(f"  - clusters_flagged_low_quality.parquet")
    print(f"  - diagnostic_report.json")
