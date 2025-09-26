# File: src/deduplication/param_tuner.py
"""
Hyperparameter tuner for the deduplication pipeline.

Why this exists:
- Systematically searches parameters with Bayesian & multi-objective optimization.
- Optimizes on a small fixed sample for speed; re-validates best Pareto trials.

Requirements: optuna, pandas, pyarrow, typer
Optional (parallel storage): sqlite (via sqlite3)

Usage:
  python -m src.deduplication.param_tuner tune \
    --in-pairs data/intermediate/token_pairs_sample.parquet \
    --out-base organized_outputs/tuning_runs \
    --trials 60 --n-jobs 1 \
    --study dedupe_moo --storage sqlite:///tuning.db

  # After tuning, export top Pareto and optionally rerun at scale:
  python -m src.deduplication.param_tuner best \
    --study dedupe_moo --storage sqlite:///tuning.db \
    --top 5 --export best_trials.csv
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import optuna
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# --------- Configs ---------

PIPELINE_ENTRY = [sys.executable, "src/deduplication/dedupe_pipeline.py", "refined-all"]

REQUIRED_FILES = {
    "flagged": "clusters_flagged_size_aware.parquet",
    "cohesion": "cluster_cohesion_metrics.parquet",
    "token_map": "clusters_token_map.parquet",
    "clean_graph": "pairs_clean_graph.parquet",
}

# --------- Metric extraction ---------

@dataclass
class TrialMetrics:
    good_clusters: int
    coverage: int
    edges: int
    min_sim_mean: float
    total_clusters: int
    pairs_count: int
    multi_count: int

def read_metrics(out_dir: Path) -> TrialMetrics:
    """Reads pipeline outputs and computes metrics. Fail-safe returns zeros if missing.
    Why: objective must be total; missing files indicate a bad config or failure."""
    try:
        flagged = pd.read_parquet(out_dir / REQUIRED_FILES["flagged"])
        token_map = pd.read_parquet(out_dir / REQUIRED_FILES["token_map"])
        clean = pd.read_parquet(out_dir / REQUIRED_FILES["clean_graph"])
        cohesion = pd.read_parquet(out_dir / REQUIRED_FILES["cohesion"])

        good = int((~flagged["is_flagged"]).sum())
        coverage = int(len(token_map))
        edges = int(len(clean))
        min_sim_mean = float(cohesion["min_sim"].mean()) if len(cohesion) else 0.0
        total_clusters = int(len(flagged))
        pairs_count = int((flagged["size"] == 2).sum())
        multi_count = int((flagged["size"] >= 3).sum())
        
        return TrialMetrics(good, coverage, edges, min_sim_mean, total_clusters, pairs_count, multi_count)
    except Exception as e:
        # Return dominated metrics for failed runs
        return TrialMetrics(0, 0, 10**9, 0.0, 0, 0, 0)

# --------- Objective ---------

@dataclass
class ObjectiveCfg:
    in_pairs: Path
    out_base: Path
    timeout_sec: int = 600  # protect from runaway configs
    keep_artifacts: bool = False  # only keep winners by default

class DedupeObjective:
    def __init__(self, cfg: ObjectiveCfg):
        self.cfg = cfg

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """Multi-objective: maximize (good_clusters, coverage, -edges).
        Why -edges: smaller graph is cleaner/faster."""
        # Search space (bounded to realistic ranges from your logs)
        base_threshold = trial.suggest_float("base_threshold", 0.55, 0.85)
        k = trial.suggest_int("k", 3, 20)
        mutual_nn = trial.suggest_categorical("mutual_nn", [True, False])
        max_degree = trial.suggest_int("max_degree", 30, 200)
        adaptive_short_thr = trial.suggest_float("adaptive_short_thr", 0.75, 0.90)

        pair_min_sim = trial.suggest_float("pair_min_sim", 0.82, 0.95)
        small_min_sim = trial.suggest_float("small_min_sim", 0.70, 0.85)
        small_min_tri = trial.suggest_float("small_min_tri", 0.20, 0.60)
        big_min_sim = trial.suggest_float("big_min_sim", 0.65, 0.80)
        big_min_mean = trial.suggest_float("big_min_mean", 0.75, 0.85)
        big_min_tri = trial.suggest_float("big_min_tri", 0.10, 0.30)

        # Derive minor constraints
        if small_min_sim > big_min_mean:
            # Avoid over-strict small clusters vs big clusters; nudge down small_min_sim.
            small_min_sim = big_min_mean - 0.01

        run_id = f"trial_{trial.number:04d}_{uuid.uuid4().hex[:8]}"
        out_dir = self.cfg.out_base / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            *PIPELINE_ENTRY,
            str(self.cfg.in_pairs),
            str(out_dir),
            "--base-threshold",
            f"{base_threshold:.4f}",
            "--k",
            str(k),
            "--max-degree",
            str(max_degree),
            "--adaptive-short-thr",
            f"{adaptive_short_thr:.4f}",
            "--pair-min-sim",
            f"{pair_min_sim:.4f}",
            "--small-min-sim",
            f"{small_min_sim:.4f}",
            "--small-min-tri",
            f"{small_min_tri:.4f}",
            "--big-min-sim",
            f"{big_min_sim:.4f}",
            "--big-min-mean",
            f"{big_min_mean:.4f}",
            "--big-min-tri",
            f"{big_min_tri:.4f}",
        ]
        if mutual_nn:
            cmd.append("--mutual-nn")

        # Run pipeline with timeout
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.cfg.timeout_sec,
            )
            # Attach logs as intermediate notes
            trial.set_user_attr("stdout_tail", proc.stdout[-4000:])
            trial.set_user_attr("return_code", proc.returncode)
        except subprocess.TimeoutExpired:
            # Penalize timeout
            if not self.cfg.keep_artifacts:
                shutil.rmtree(out_dir, ignore_errors=True)
            return 0.0, 0.0, -1e6

        # Compute metrics
        metrics = read_metrics(out_dir)

        # Optionally clean up non-dominant artifacts later; keep all for now
        trial.set_user_attr("out_dir", str(out_dir))
        trial.set_user_attr("metrics", metrics.__dict__)

        # Multi-objective directions = maximize, maximize, maximize(-edges)
        return float(metrics.good_clusters), float(metrics.coverage), float(-metrics.edges)

# --------- CLI ---------

@app.command()
def tune(
    in_pairs: Path = typer.Option(..., exists=True, dir_okay=False, help="Candidate pairs parquet"),
    out_base: Path = typer.Option(..., help="Base directory to store trial outputs"),
    trials: int = typer.Option(40, min=1, help="Number of trials"),
    n_jobs: int = typer.Option(1, help="Parallel workers (>=2 requires RDB storage)"),
    study: str = typer.Option("dedupe_moo", help="Optuna study name"),
    storage: Optional[str] = typer.Option(
        None, help="e.g., sqlite:///tuning.db to enable parallelism & persistence"
    ),
    timeout_sec: int = typer.Option(600, help="Per-trial timeout seconds"),
    keep_artifacts: bool = typer.Option(False, help="Keep all trial artifacts"),
):
    """
    Run multi-objective tuning. Returns a Pareto front of parameter sets.
    Why multi-objective: balances quality (good clusters, coverage) vs graph size.
    """
    out_base.mkdir(parents=True, exist_ok=True)
    objective = DedupeObjective(ObjectiveCfg(in_pairs, out_base, timeout_sec, keep_artifacts))

    directions = ["maximize", "maximize", "maximize"]
    study_obj = optuna.create_study(
        study_name=study,
        directions=directions,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )
    
    typer.echo(f"🔍 Starting tuning with {trials} trials...")
    typer.echo(f"📊 Study: {study}")
    typer.echo(f"💾 Storage: {storage or 'memory'}")
    typer.echo(f"⚡ Parallel jobs: {n_jobs}")
    
    study_obj.optimize(objective, n_trials=trials, n_jobs=n_jobs)

    # Summarize Pareto front
    pf = [t.number for t in study_obj.best_trials]
    summary = []
    for t in study_obj.best_trials:
        m = t.user_attrs.get("metrics", {})
        summary.append(
            {
                "trial": t.number,
                "good_clusters": m.get("good_clusters", None),
                "coverage": m.get("coverage", None),
                "edges": m.get("edges", None),
                "min_sim_mean": m.get("min_sim_mean", None),
                "total_clusters": m.get("total_clusters", None),
                "pairs_count": m.get("pairs_count", None),
                "multi_count": m.get("multi_count", None),
                **t.params,
                "out_dir": t.user_attrs.get("out_dir", ""),
            }
        )
    df = pd.DataFrame(summary).sort_values(["good_clusters", "coverage", "edges"], ascending=[False, False, True])
    out_csv = out_base / "pareto_trials.csv"
    df.to_csv(out_csv, index=False)
    
    typer.echo(f"✅ Tuning complete!")
    typer.echo(f"🎯 Pareto trials: {pf}")
    typer.echo(f"📁 Saved: {out_csv}")
    typer.echo(f"📈 Best trial: {df.iloc[0]['trial']} (good_clusters={df.iloc[0]['good_clusters']}, coverage={df.iloc[0]['coverage']}, edges={df.iloc[0]['edges']})")

@app.command()
def best(
    study: str = typer.Option("dedupe_moo", help="Optuna study name"),
    storage: str = typer.Option(..., help="sqlite:///tuning.db"),
    top: int = typer.Option(5, min=1, help="Export top N Pareto trials"),
    export: Path = typer.Option(Path("best_trials.csv"), help="Output CSV with best trials"),
):
    """
    Export the top-N Pareto trials (sorted by good_clusters, coverage, edges).
    Why: re-run these on larger data/full corpus for confirmation.
    """
    study_obj = optuna.load_study(study_name=study, storage=storage)
    rows = []
    for t in study_obj.best_trials:
        m = t.user_attrs.get("metrics", {})
        rows.append(
            {
                "trial": t.number,
                "good_clusters": m.get("good_clusters", None),
                "coverage": m.get("coverage", None),
                "edges": m.get("edges", None),
                "min_sim_mean": m.get("min_sim_mean", None),
                "total_clusters": m.get("total_clusters", None),
                "pairs_count": m.get("pairs_count", None),
                "multi_count": m.get("multi_count", None),
                **t.params,
                "out_dir": t.user_attrs.get("out_dir", ""),
            }
        )
    df = pd.DataFrame(rows).sort_values(["good_clusters", "coverage", "edges"], ascending=[False, False, True]).head(top)
    export.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(export, index=False)
    typer.echo(f"✅ Exported top {len(df)} Pareto trials → {export}")

@app.command()
def validate(
    trial_dir: Path = typer.Option(..., exists=True, dir_okay=True, help="Trial output directory"),
    in_pairs: Path = typer.Option(..., exists=True, dir_okay=False, help="Original input pairs for comparison"),
):
    """
    Validate a single trial by examining its outputs and comparing to input.
    """
    typer.echo(f"🔍 Validating trial: {trial_dir}")
    
    # Read metrics
    metrics = read_metrics(trial_dir)
    typer.echo(f"📊 Metrics:")
    typer.echo(f"   Good clusters: {metrics.good_clusters}")
    typer.echo(f"   Total clusters: {metrics.total_clusters}")
    typer.echo(f"   Coverage: {metrics.coverage}")
    typer.echo(f"   Edges: {metrics.edges}")
    typer.echo(f"   Mean min_sim: {metrics.min_sim_mean:.4f}")
    typer.echo(f"   Pairs: {metrics.pairs_count}")
    typer.echo(f"   Multi-token: {metrics.multi_count}")
    
    # Check input size
    input_pairs = pd.read_parquet(in_pairs)
    typer.echo(f"📥 Input pairs: {len(input_pairs):,}")
    
    # Check if files exist
    for name, filename in REQUIRED_FILES.items():
        path = trial_dir / filename
        if path.exists():
            typer.echo(f"✅ {name}: {path}")
        else:
            typer.echo(f"❌ {name}: {path} (missing)")

@app.command()
def compare(
    trial_dirs: List[Path] = typer.Option(..., help="List of trial directories to compare"),
    export: Optional[Path] = typer.Option(None, help="Export comparison CSV"),
):
    """
    Compare multiple trial results side by side.
    """
    typer.echo(f"🔍 Comparing {len(trial_dirs)} trials...")
    
    results = []
    for trial_dir in trial_dirs:
        metrics = read_metrics(trial_dir)
        results.append({
            "trial_dir": str(trial_dir),
            "good_clusters": metrics.good_clusters,
            "total_clusters": metrics.total_clusters,
            "coverage": metrics.coverage,
            "edges": metrics.edges,
            "min_sim_mean": metrics.min_sim_mean,
            "pairs_count": metrics.pairs_count,
            "multi_count": metrics.multi_count,
        })
    
    df = pd.DataFrame(results).sort_values(["good_clusters", "coverage", "edges"], ascending=[False, False, True])
    
    typer.echo("📊 Comparison Results:")
    typer.echo(df.to_string(index=False))
    
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export, index=False)
        typer.echo(f"💾 Saved comparison to: {export}")

if __name__ == "__main__":
    app()
