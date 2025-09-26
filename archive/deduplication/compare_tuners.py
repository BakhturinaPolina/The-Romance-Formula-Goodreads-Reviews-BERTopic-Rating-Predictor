#!/usr/bin/env python3
"""
Comparison script between basic and advanced hyperparameter tuners.

This script demonstrates the differences between:
1. Basic TPE tuner (single-objective, full data)
2. Advanced NSGA-II tuner (multi-objective, multi-fidelity)
3. Advanced MOTPE tuner (multi-objective, multi-fidelity)

Usage:
    python src/deduplication/compare_tuners.py
"""

import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

def run_tuner(tuner_type, sample_pairs, output_dir, trials=20):
    """Run a specific tuner and return results."""
    print(f"\n🔧 Running {tuner_type} tuner...")
    
    if tuner_type == "basic_tpe":
        cmd = [
            sys.executable, "-m", "src.deduplication.param_tuner", "tune",
            "--in-pairs", str(sample_pairs),
            "--out-base", str(output_dir / "basic_tpe"),
            "--trials", str(trials),
            "--n-jobs", "1",
            "--study", "basic_tpe_comparison",
            "--timeout-sec", "300",
            "--keep-artifacts"
        ]
    elif tuner_type == "advanced_nsga2":
        cmd = [
            sys.executable, "-m", "src.deduplication.param_tuner_advanced", "tune",
            "--in-pairs", str(sample_pairs),
            "--out-base", str(output_dir / "advanced_nsga2"),
            "--trials", str(trials),
            "--n-jobs", "1",
            "--study-name", "advanced_nsga2_comparison",
            "--storage", f"sqlite:///{output_dir}/nsga2.db",
            "--sampler", "nsga2",
            "--pruner", "hyperband",
            "--budgets", "0.1,1.0",
            "--folds", "1",
            "--min-coverage", "3",
            "--max-edges", "3000",
            "--timeout-sec", "300",
            "--top-k", "3"
        ]
    elif tuner_type == "advanced_motpe":
        cmd = [
            sys.executable, "-m", "src.deduplication.param_tuner_advanced", "tune",
            "--in-pairs", str(sample_pairs),
            "--out-base", str(output_dir / "advanced_motpe"),
            "--trials", str(trials),
            "--n-jobs", "1",
            "--study-name", "advanced_motpe_comparison",
            "--storage", f"sqlite:///{output_dir}/motpe.db",
            "--sampler", "motpe",
            "--pruner", "hyperband",
            "--budgets", "0.1,1.0",
            "--folds", "1",
            "--min-coverage", "3",
            "--max-edges", "3000",
            "--timeout-sec", "300",
            "--top-k", "3"
        ]
    else:
        raise ValueError(f"Unknown tuner type: {tuner_type}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    end_time = time.time()
    
    success = result.returncode == 0
    duration = end_time - start_time
    
    print(f"   {'✅' if success else '❌'} {tuner_type}: {duration:.1f}s")
    
    if not success:
        print(f"   Error: {result.stderr[-500:]}")
    
    return success, duration

def load_results(output_dir, tuner_type):
    """Load results from a tuner run."""
    if tuner_type == "basic_tpe":
        results_file = output_dir / "basic_tpe" / "pareto_trials.csv"
    elif tuner_type == "advanced_nsga2":
        results_file = output_dir / "advanced_nsga2" / "pareto_full_results.csv"
    elif tuner_type == "advanced_motpe":
        results_file = output_dir / "advanced_motpe" / "pareto_full_results.csv"
    else:
        return None
    
    if results_file.exists():
        return pd.read_csv(results_file)
    return None

def compare_results(results_dict):
    """Compare results from different tuners."""
    print(f"\n📊 Results Comparison:")
    print("=" * 80)
    
    for tuner_type, results in results_dict.items():
        if results is not None and len(results) > 0:
            best = results.iloc[0]
            print(f"\n{tuner_type.upper()}:")
            print(f"   Best Trial: {best.get('trial', 'N/A')}")
            print(f"   Good Clusters: {best.get('good_clusters', 'N/A')}")
            print(f"   Coverage: {best.get('coverage', 'N/A')}")
            print(f"   Edges: {best.get('edges', 'N/A')}")
            if 'mean_min_sim' in best:
                print(f"   Mean Min Sim: {best['mean_min_sim']:.4f}")
        else:
            print(f"\n{tuner_type.upper()}: No results found")

def main():
    """Run the tuner comparison."""
    print("🔬 Hyperparameter Tuner Comparison")
    print("=" * 50)
    print()
    print("Comparing:")
    print("• Basic TPE (single-objective, full data)")
    print("• Advanced NSGA-II (multi-objective, multi-fidelity)")
    print("• Advanced MOTPE (multi-objective, multi-fidelity)")
    print()
    
    # Configuration
    base_dir = Path("organized_outputs/tuner_comparison")
    sample_pairs = base_dir / "token_pairs_sample.parquet"
    
    # Check if sample data exists
    if not sample_pairs.exists():
        print("❌ Sample data not found. Please create it first:")
        print(f"   {sample_pairs}")
        print()
        print("Example command:")
        print("python -c \"")
        print("import pandas as pd")
        print("df = pd.read_parquet('data/intermediate/token_pairs.parquet')")
        print("sample = df.sample(n=50000, random_state=42)")
        print(f"sample.to_parquet('{sample_pairs}', index=False)")
        print("print(f'Sampled {len(sample):,} pairs')")
        print("\"")
        return
    
    # Create output directory
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tuners
    tuners = ["basic_tpe", "advanced_nsga2", "advanced_motpe"]
    results = {}
    durations = {}
    
    for tuner in tuners:
        success, duration = run_tuner(tuner, sample_pairs, base_dir, trials=15)
        durations[tuner] = duration
        if success:
            results[tuner] = load_results(base_dir, tuner)
        else:
            results[tuner] = None
    
    # Compare results
    compare_results(results)
    
    # Performance comparison
    print(f"\n⏱️  Performance Comparison:")
    print("=" * 40)
    for tuner, duration in durations.items():
        print(f"   {tuner}: {duration:.1f}s")
    
    # Key differences summary
    print(f"\n💡 Key Differences:")
    print("=" * 30)
    print("Basic TPE:")
    print("   • Single objective (good_clusters)")
    print("   • Full data evaluation")
    print("   • No early pruning")
    print("   • No constraints")
    print()
    print("Advanced NSGA-II/MOTPE:")
    print("   • Multi-objective (good_clusters, coverage, edges)")
    print("   • Multi-fidelity budgets (10% → 100%)")
    print("   • Hyperband pruning")
    print("   • Constraint handling")
    print("   • Pareto front analysis")
    print("   • Full data re-evaluation")
    
    print(f"\n✅ Comparison complete!")
    print(f"📁 Results saved to: {base_dir}")

if __name__ == "__main__":
    main()
