#!/usr/bin/env python3
"""
Example script demonstrating advanced multi-objective hyperparameter tuning.

This script shows how to use the advanced tuner with:
1. Multi-fidelity budgets (5% → 20% → 100% data)
2. NSGA-II/MOTPE samplers for multi-objective optimization
3. Hyperband pruning for early stopping
4. Constraint handling (min coverage, max edges)
5. Pareto front re-evaluation on full data
6. Visualization and analysis

Usage:
    python src/deduplication/example_advanced_tuning.py
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def create_sample_data(input_pairs: Path, output_pairs: Path, sample_size: int = 100000):
    """Create a sample dataset for tuning."""
    print(f"📊 Creating sample dataset ({sample_size:,} pairs)...")
    
    # Read full dataset
    df = pd.read_parquet(input_pairs)
    print(f"   Original dataset: {len(df):,} pairs")
    
    # Sample
    if len(df) > sample_size:
        sample = df.sample(n=sample_size, random_state=42)
        print(f"   Sampled: {len(sample):,} pairs")
    else:
        sample = df
        print(f"   Using full dataset: {len(sample):,} pairs")
    
    # Save sample
    output_pairs.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_pairs, index=False)
    print(f"   Saved to: {output_pairs}")
    
    return sample

def run_command(cmd, description, timeout=3600):
    """Run a command and print its output."""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"   {line}")
        else:
            print("❌ Failed")
            print("Error output:")
            print(result.stderr[-1000:])  # Last 1000 chars
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the complete advanced tuning example."""
    
    # Configuration
    base_dir = Path("organized_outputs/advanced_tuning_example")
    sample_pairs = base_dir / "token_pairs_sample.parquet"
    tuning_dir = base_dir / "tuning_runs"
    storage_db = base_dir / "advanced_tuning.db"
    
    print("🚀 Advanced Multi-Objective Hyperparameter Tuning Example")
    print("=" * 70)
    print()
    print("This example demonstrates:")
    print("• Multi-fidelity budgets (5% → 20% → 100% data)")
    print("• NSGA-II multi-objective optimization")
    print("• Hyperband pruning for early stopping")
    print("• Constraint handling (min coverage, max edges)")
    print("• Pareto front re-evaluation on full data")
    print("• Visualization and analysis")
    print()
    
    # Step 1: Create sample data (if not exists)
    if not sample_pairs.exists():
        print(f"\n📊 Creating sample dataset...")
        # For this example, we'll assume you have a larger dataset
        # In practice, you'd sample from your full token_pairs.parquet
        print("⚠️  Please create a sample dataset first:")
        print(f"   - Take a random sample from your full token_pairs.parquet")
        print(f"   - Save it as: {sample_pairs}")
        print("   - Recommended size: 50k-100k pairs for advanced tuning")
        print()
        print("Example command:")
        print(f"python -c \"")
        print(f"import pandas as pd")
        print(f"df = pd.read_parquet('data/intermediate/token_pairs.parquet')")
        print(f"sample = df.sample(n=100000, random_state=42)")
        print(f"sample.to_parquet('{sample_pairs}', index=False)")
        print(f"print(f'Sampled {{len(sample):,}} pairs')\"")
        return
    
    # Step 2: Run advanced hyperparameter tuning with NSGA-II
    print(f"\n🎯 Starting advanced tuning with NSGA-II...")
    tune_cmd = [
        sys.executable, "-m", "src.deduplication.param_tuner_advanced", "tune",
        "--in-pairs", str(sample_pairs),
        "--out-base", str(tuning_dir),
        "--trials", "30",  # Start with fewer trials for example
        "--n-jobs", "2",
        "--study-name", "advanced_nsga2",
        "--storage", f"sqlite:///{storage_db}",
        "--timeout-sec", "600",
        "--sampler", "nsga2",
        "--pruner", "hyperband",
        "--folds", "2",
        "--budgets", "0.05,0.20,1.00",
        "--min-coverage", "5",
        "--max-edges", "5000",
        "--top-k", "3",
        "--keep-artifacts"
    ]
    
    if not run_command(tune_cmd, "Running advanced NSGA-II tuning", timeout=1800):
        print("❌ NSGA-II tuning failed. Trying with MOTPE...")
        
        # Fallback to MOTPE
        tune_cmd_motpe = tune_cmd.copy()
        tune_cmd_motpe[tune_cmd_motpe.index("nsga2")] = "motpe"
        tune_cmd_motpe[tune_cmd_motpe.index("advanced_nsga2")] = "advanced_motpe"
        
        if not run_command(tune_cmd_motpe, "Running advanced MOTPE tuning", timeout=1800):
            return
    
    # Step 3: Analyze results
    print(f"\n📈 Analyzing tuning results...")
    analyze_cmd = [
        sys.executable, "-m", "src.deduplication.param_tuner_advanced", "analyze",
        "--study-name", "advanced_nsga2",
        "--storage", f"sqlite:///{storage_db}",
        "--out-dir", str(base_dir / "analysis")
    ]
    
    run_command(analyze_cmd, "Analyzing results")
    
    # Step 4: Show results summary
    print(f"\n📊 Advanced Tuning Results Summary:")
    print(f"   📁 Tuning directory: {tuning_dir}")
    print(f"   💾 Storage database: {storage_db}")
    print(f"   📈 Analysis: {base_dir / 'analysis'}")
    
    # Check for results files
    pareto_file = tuning_dir / "pareto_trials.csv"
    full_results_file = tuning_dir / "pareto_full_results.csv"
    
    if pareto_file.exists():
        pareto_results = pd.read_csv(pareto_file)
        print(f"\n🏆 Pareto Front ({len(pareto_results)} trials):")
        for i, row in pareto_results.head(5).iterrows():
            print(f"   {i+1}. Trial {row['trial']}: "
                  f"good_clusters={row['good_clusters']}, "
                  f"coverage={row['coverage']}, "
                  f"edges={row['edges']}")
    
    if full_results_file.exists():
        full_results = pd.read_csv(full_results_file)
        print(f"\n🎯 Full Data Results (Top {len(full_results)}):")
        for i, row in full_results.iterrows():
            print(f"   {i+1}. Trial {row['trial']}: "
                  f"good_clusters={row['good_clusters']}, "
                  f"coverage={row['coverage']}, "
                  f"edges={row['edges']}")
    
    # Step 5: Show visualization files
    viz_files = [
        "pareto_front.html",
        "param_importance.html", 
        "optimization_history.html"
    ]
    
    print(f"\n📊 Visualizations created:")
    for viz_file in viz_files:
        viz_path = tuning_dir / viz_file
        if viz_path.exists():
            print(f"   ✅ {viz_file}")
        else:
            print(f"   ❌ {viz_file} (not found)")
    
    print(f"\n✅ Advanced tuning example complete!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   🎯 Multi-objective optimization (NSGA-II/MOTPE)")
    print(f"   📊 Multi-fidelity budgets (5% → 20% → 100%)")
    print(f"   ✂️  Hyperband pruning for early stopping")
    print(f"   🎯 Constraint handling (min coverage, max edges)")
    print(f"   🔄 Pareto front re-evaluation on full data")
    print(f"   📈 Visualization and analysis tools")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review the Pareto front results")
    print(f"   2. Examine visualizations in {tuning_dir}")
    print(f"   3. Use the best parameters in your production pipeline")
    print(f"   4. Scale up to larger datasets with more trials")

def compare_samplers():
    """Compare NSGA-II vs MOTPE samplers."""
    print("\n🔬 Sampler Comparison Example")
    print("=" * 40)
    
    base_dir = Path("organized_outputs/sampler_comparison")
    sample_pairs = base_dir / "token_pairs_sample.parquet"
    
    if not sample_pairs.exists():
        print("❌ Sample data not found. Run main example first.")
        return
    
    samplers = ["nsga2", "motpe"]
    results = {}
    
    for sampler in samplers:
        print(f"\n🎯 Testing {sampler.upper()} sampler...")
        
        tuning_dir = base_dir / f"tuning_{sampler}"
        storage_db = base_dir / f"tuning_{sampler}.db"
        
        cmd = [
            sys.executable, "-m", "src.deduplication.param_tuner_advanced", "tune",
            "--in-pairs", str(sample_pairs),
            "--out-base", str(tuning_dir),
            "--trials", "20",
            "--n-jobs", "1",
            "--study-name", f"comparison_{sampler}",
            "--storage", f"sqlite:///{storage_db}",
            "--timeout-sec", "300",
            "--sampler", sampler,
            "--pruner", "hyperband",
            "--folds", "1",
            "--budgets", "0.1,1.0",
            "--min-coverage", "3",
            "--max-edges", "3000",
            "--top-k", "3"
        ]
        
        if run_command(cmd, f"Running {sampler} comparison", timeout=900):
            # Load results
            pareto_file = tuning_dir / "pareto_trials.csv"
            if pareto_file.exists():
                results[sampler] = pd.read_csv(pareto_file)
                print(f"✅ {sampler} completed: {len(results[sampler])} Pareto trials")
    
    # Compare results
    if len(results) > 1:
        print(f"\n📊 Sampler Comparison Results:")
        for sampler, df in results.items():
            if len(df) > 0:
                best = df.iloc[0]
                print(f"   {sampler.upper()}: "
                      f"best good_clusters={best['good_clusters']}, "
                      f"coverage={best['coverage']}, "
                      f"edges={best['edges']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced tuning example")
    parser.add_argument("--compare", action="store_true", help="Compare samplers")
    args = parser.parse_args()
    
    if args.compare:
        compare_samplers()
    else:
        main()
