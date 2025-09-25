# File: src/deduplication/grow_runner.py
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import typer

# local imports without package install
THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))

from graph_grow import GraphGrowCfg, grow_by_triangles

# Reuse helpers from your pipeline (why: consistent artifacts + metrics)
sys.path.append(str(THIS_DIR))  # ensure we can import sibling
from dedupe_pipeline import (
    Config, head, save_artifacts, cluster_cohesion_metrics, build_clusters,
    cmd_graph_build, cmd_size_aware_quality
)

app = typer.Typer(add_completion=False)

@app.command("refined-grow")
def refined_grow(
    in_candidates: Path = typer.Argument(..., help="Full candidate pairs parquet"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
    # seed graph params
    base_threshold: float = typer.Option(0.60),
    k: int = typer.Option(8),
    mutual_nn: bool = typer.Option(True),
    max_degree: int = typer.Option(100),
    adaptive_short_thr: float = typer.Option(0.80),
    # growth params
    grow_min_sim: float = typer.Option(0.75),
    grow_min_support: int = typer.Option(2),
    grow_require_triangle: bool = typer.Option(True),
    grow_attach_k: int = typer.Option(2),
    grow_max_new_per_cluster: int = typer.Option(10),
    grow_iterations: int = typer.Option(1),
    # quality params (same defaults as your size-aware)
    pair_min_sim: float = typer.Option(0.85),
    small_min_sim: float = typer.Option(0.75),
    small_min_tri: float = typer.Option(0.33),
    big_min_sim: float = typer.Option(0.70),
    big_min_mean: float = typer.Option(0.78),
    big_min_tri: float = typer.Option(0.20),
):
    """
    Build a clean seed graph, then grow clusters via triangle-based attachment and assess quality.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(in_pairs=in_candidates, out_dir=out_dir, base_threshold=base_threshold).ensure()

    # 1) Seed graph
    cmd_graph_build(in_candidates, out_dir, base_threshold, k, mutual_nn, max_degree, 5, adaptive_short_thr)
    clean_path = out_dir / "pairs_clean_graph.parquet"
    clean = pd.read_parquet(clean_path)

    # 2) Triangle growth from full candidates
    head("TRIANGLE-BASED GROWTH")
    candidates = pd.read_parquet(in_candidates)
    gcfg = GraphGrowCfg(
        min_sim=grow_min_sim,
        min_support=grow_min_support,
        require_triangle=grow_require_triangle,
        attach_k_per_node=grow_attach_k,
        max_new_nodes_per_cluster=grow_max_new_per_cluster,
        iterations=grow_iterations,
    )
    grown = grow_by_triangles(clean, candidates, gcfg)
    grown_path = out_dir / "pairs_grown_graph.parquet"
    grown.to_parquet(grown_path, index=False)
    print(f"✅ Saved grown graph to {grown_path}")

    # 3) Cluster + metrics on grown graph
    head("CLUSTER ON GROWN GRAPH")
    token_cluster, sample_edges = build_clusters(grown_path, sample_per_cluster=5)

    head("CLUSTER QUALITY METRICS")
    cohesion = cluster_cohesion_metrics(grown_path, token_cluster)
    save_artifacts(cfg, token_cluster, sample_edges, cohesion, None)

    # 4) Size-aware flags
    cmd_size_aware_quality(out_dir / "cluster_cohesion_metrics.parquet", out_dir,
                           pair_min_sim, small_min_sim, small_min_tri, big_min_sim, big_min_mean, big_min_tri)

    head("REFINED-GROW PIPELINE COMPLETE")

if __name__ == "__main__":
    app()
