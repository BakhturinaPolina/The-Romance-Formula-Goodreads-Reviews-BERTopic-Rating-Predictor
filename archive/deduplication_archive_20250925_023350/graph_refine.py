# File: src/deduplication/graph_refine.py
# Adds: graph build (top-k + mutual-NN + hubs + adaptive thresholds), size-aware quality flags, and CLI hooks.
from __future__ import annotations
import typing as t
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import typer

# ---------------- Config for graph building ----------------
@dataclass(frozen=True)
class GraphBuildCfg:
    base_threshold: float = 0.60      # low floor for candidate graph
    k: int = 5                        # top-k per token
    mutual_nn: bool = True
    max_degree: int = 50              # hub cap
    adaptive_short_len: int = 5
    adaptive_short_thr: float = 0.80  # len<adaptive_short_len uses this thr
    require_triangle: bool = False    # optional local consistency
    min_jaccard: float = 0.05         # neighbor overlap threshold if used

# ---------------- Helpers ----------------
def _adaptive_thr(tok: str, cfg: GraphBuildCfg) -> float:
    return cfg.adaptive_short_thr if len(tok) < cfg.adaptive_short_len else cfg.base_threshold

def _topk_per_token(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # Expect columns: token_a, token_b, cosine_sim, rank (optional)
    # Keep top-k by cosine per token_a; if 'rank' exists, break ties.
    order_cols = ["token_a", "cosine_sim"]
    asc = [True, False]
    if "rank" in df.columns:
        order_cols.append("rank"); asc.append(True)
    g = df.sort_values(order_cols, ascending=asc).groupby("token_a").head(k)
    return g

def _mutual_nn(edges: pd.DataFrame) -> pd.DataFrame:
    # Keep edges where (a in top-k of b) and (b in top-k of a)
    # Compute both directions membership
    ab = edges[["token_a","token_b"]].astype("string")
    ba = ab.rename(columns={"token_a":"token_b","token_b":"token_a"})
    key = pd.Series(np.arange(len(ab)), name="eid")
    abk = pd.concat([ab, key], axis=1)
    joined = abk.merge(ba, on=["token_a","token_b"], how="inner")
    return edges.loc[joined["eid"].unique()]

def _cap_hubs(edges: pd.DataFrame, max_degree: int) -> pd.DataFrame:
    deg = pd.concat([edges["token_a"], edges["token_b"]]).astype("string").value_counts()
    bad = set(deg[deg > max_degree].index)
    mask = ~edges["token_a"].isin(bad) & ~edges["token_b"].isin(bad)
    return edges[mask].copy()

def _triangle_filter(edges: pd.DataFrame, min_jaccard: float) -> pd.DataFrame:
    # Keep edges whose endpoints share neighbors (Jaccard > min_jaccard)
    a = edges["token_a"].astype("string").to_numpy()
    b = edges["token_b"].astype("string").to_numpy()
    tokens = pd.unique(np.concatenate([a,b]))
    idx = {t:i for i,t in enumerate(tokens)}
    neigh = [set() for _ in range(len(tokens))]
    for x,y in zip(a,b):
        ix,iy = idx[x], idx[y]
        neigh[ix].add(iy); neigh[iy].add(ix)
    keep_mask = []
    for x,y in zip(a,b):
        nx, ny = neigh[idx[x]], neigh[idx[y]]
        inter = len(nx & ny); union = len(nx | ny) or 1
        keep_mask.append((inter/union) >= min_jaccard)
    return edges[pd.Series(keep_mask, index=edges.index)]

def build_clean_graph(candidates: pd.DataFrame, cfg: GraphBuildCfg) -> pd.DataFrame:
    """Build a clean, dense graph from raw candidates using top-k + mutual-NN + hub caps."""
    df = candidates.copy()
    df[["token_a","token_b"]] = df[["token_a","token_b"]].astype("string")
    
    # Adaptive threshold (per-endpoint max)
    thr_a = df["token_a"].map(lambda x: _adaptive_thr(x, cfg))
    thr_b = df["token_b"].map(lambda x: _adaptive_thr(x, cfg))
    thr = np.maximum(thr_a.to_numpy(dtype=float), thr_b.to_numpy(dtype=float))
    df = df[df["cosine_sim"].astype(float).to_numpy() >= thr].copy()

    # Densify with top-k per endpoint, then mutual-NN (optional)
    topk_a = _topk_per_token(df, cfg.k)
    topk_b = _topk_per_token(df.rename(columns={"token_a":"token_b","token_b":"token_a"}), cfg.k)
    topk_b = topk_b.rename(columns={"token_a":"token_b","token_b":"token_a"})
    edges = pd.concat([topk_a, topk_b], ignore_index=True).drop_duplicates(subset=["token_a","token_b","cosine_sim"])

    if cfg.mutual_nn:
        edges = _mutual_nn(edges)

    # Hub cap
    edges = _cap_hubs(edges, cfg.max_degree)

    # Optional triangle/Jaccard
    if cfg.require_triangle:
        edges = _triangle_filter(edges, cfg.min_jaccard)

    return edges.sort_values(["token_a","cosine_sim"], ascending=[True, False]).reset_index(drop=True)

# ---------------- Size-aware quality flags ----------------
@dataclass(frozen=True)
class QualityCfg:
    pair_min_sim: float = 0.85
    small_min_sim: float = 0.75
    small_min_tri: float = 0.33
    big_min_sim: float = 0.70
    big_min_mean: float = 0.78
    big_min_tri: float = 0.20

def size_aware_flags(coh: pd.DataFrame, qc: QualityCfg) -> pd.DataFrame:
    """Apply size-aware quality criteria to cluster cohesion metrics."""
    # coh must contain: cluster_id, size, min_sim, mean_sim, triangle_rate
    s = coh.copy()
    s["flag_reason"] = ""
    
    # pairs (size == 2): accept if min_sim >= threshold
    is_pair = s["size"] == 2
    s.loc[is_pair & (s["min_sim"] < qc.pair_min_sim), "flag_reason"] = "pair_min_sim"
    
    # small clusters (3..5): require min_sim AND triangle_rate
    small = (s["size"] >= 3) & (s["size"] <= 5)
    s.loc[small & (s["min_sim"] < qc.small_min_sim), "flag_reason"] += ("|min_sim" if s.loc[small, "flag_reason"].ne("").any() else "min_sim")
    s.loc[small & (s["triangle_rate"] < qc.small_min_tri), "flag_reason"] += ("|triangle" if s.loc[small, "flag_reason"].ne("").any() else "triangle")
    
    # big clusters (>5): require min_sim AND mean_sim AND triangle_rate
    big = s["size"] > 5
    s.loc[big & (s["min_sim"] < qc.big_min_sim), "flag_reason"] += ("|min_sim" if s.loc[big, "flag_reason"].ne("").any() else "min_sim")
    s.loc[big & (s["mean_sim"] < qc.big_min_mean), "flag_reason"] += ("|mean_sim" if s.loc[big, "flag_reason"].ne("").any() else "mean_sim")
    s.loc[big & (s["triangle_rate"] < qc.big_min_tri), "flag_reason"] += ("|triangle" if s.loc[big, "flag_reason"].ne("").any() else "triangle")
    
    s["flag_reason"] = s["flag_reason"].str.strip("|")
    s["is_flagged"] = s["flag_reason"].ne("")
    return s

def split_clusters_by_size(coh: pd.DataFrame, token_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split clusters into pairs (size=2) and multi-token clusters (size>=3)."""
    pairs = coh[coh["size"] == 2].copy()
    multi = coh[coh["size"] >= 3].copy()
    
    # Get token mappings for each type
    pair_tokens = token_map[token_map["cluster_id"].isin(pairs["cluster_id"])]
    multi_tokens = token_map[token_map["cluster_id"].isin(multi["cluster_id"])]
    
    return pairs, multi, pair_tokens, multi_tokens

# ---------------- CLI demo (optional integration) ----------------
app = typer.Typer(add_completion=False)

@app.command("graph-build")
def cli_graph_build(
    in_candidates: Path = typer.Argument(..., help="Unfiltered candidate pairs parquet"),
    out_pairs: Path = typer.Argument(..., help="Output cleaned pairs parquet"),
    base_threshold: float = typer.Option(0.60),
    k: int = typer.Option(5),
    mutual_nn: bool = typer.Option(True),
    max_degree: int = typer.Option(50),
    adaptive_short_len: int = typer.Option(5),
    adaptive_short_thr: float = typer.Option(0.80),
    require_triangle: bool = typer.Option(False),
    min_jaccard: float = typer.Option(0.05),
):
    """Build a clean, dense graph from raw candidates."""
    cfg = GraphBuildCfg(
        base_threshold=base_threshold, k=k, mutual_nn=mutual_nn, max_degree=max_degree,
        adaptive_short_len=adaptive_short_len, adaptive_short_thr=adaptive_short_thr,
        require_triangle=require_triangle, min_jaccard=min_jaccard,
    )
    df = pd.read_parquet(in_candidates)
    clean = build_clean_graph(df, cfg)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_pairs, index=False)
    typer.echo(f"✅ Graph built: {len(clean):,} edges → {out_pairs}")

@app.command("quality-flag")
def cli_quality_flag(
    cohesion_path: Path = typer.Argument(..., help="cluster_cohesion_metrics.parquet"),
    out_flagged: Path = typer.Argument(..., help="clusters_flagged_size_aware.parquet"),
    pair_min_sim: float = typer.Option(0.85),
    small_min_sim: float = typer.Option(0.75),
    small_min_tri: float = typer.Option(0.33),
    big_min_sim: float = typer.Option(0.70),
    big_min_mean: float = typer.Option(0.78),
    big_min_tri: float = typer.Option(0.20),
):
    """Apply size-aware quality flags to cluster cohesion metrics."""
    coh = pd.read_parquet(cohesion_path)
    qc = QualityCfg(pair_min_sim, small_min_sim, small_min_tri, big_min_sim, big_min_mean, big_min_tri)
    flagged = size_aware_flags(coh, qc)
    flagged.to_parquet(out_flagged, index=False)
    ok = (~flagged["is_flagged"]).sum()
    typer.echo(f"✅ Size-aware flagged saved. Good clusters: {ok:,} / {len(flagged):,}")

@app.command("split-clusters")
def cli_split_clusters(
    cohesion_path: Path = typer.Argument(..., help="cluster_cohesion_metrics.parquet"),
    token_map_path: Path = typer.Argument(..., help="clusters_token_map.parquet"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
):
    """Split clusters into pairs and multi-token clusters."""
    coh = pd.read_parquet(cohesion_path)
    token_map = pd.read_parquet(token_map_path)
    
    pairs, multi, pair_tokens, multi_tokens = split_clusters_by_size(coh, token_map)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save pairs
    pairs.to_parquet(out_dir / "clusters_pairs.parquet", index=False)
    pair_tokens.to_parquet(out_dir / "clusters_pairs_tokens.parquet", index=False)
    
    # Save multi-token clusters
    multi.to_parquet(out_dir / "clusters_multi.parquet", index=False)
    multi_tokens.to_parquet(out_dir / "clusters_multi_tokens.parquet", index=False)
    
    typer.echo(f"✅ Split clusters: {len(pairs)} pairs, {len(multi)} multi-token clusters")

if __name__ == "__main__":
    app()
