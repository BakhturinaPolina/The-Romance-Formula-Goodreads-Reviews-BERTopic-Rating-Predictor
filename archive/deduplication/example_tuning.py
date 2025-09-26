#!/usr/bin/env python3
"""
Example script demonstrating hyperparameter tuning for the deduplication pipeline.

This script shows how to:
1. Create a sample dataset for fast tuning
2. Run multi-objective optimization
3. Analyze results and select best configurations
4. Validate top configurations on larger data

Usage:
    python src/deduplication/example_tuning.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Success")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Failed")
        print(result.stderr)
    return result.returncode == 0

def main():
    """Run the complete tuning example."""
    
    # Configuration
    base_dir = Path("organized_outputs/tuning_example")
    sample_pairs = base_dir / "token_pairs_sample.parquet"
    tuning_dir = base_dir / "tuning_runs"
    storage_db = base_dir / "tuning.db"
    
    print("🚀 Deduplication Pipeline Hyperparameter Tuning Example")
    print("=" * 60)
    
    # Step 1: Create sample data (if not exists)
    if not sample_pairs.exists():
        print(f"\n📊 Creating sample dataset...")
        # For this example, we'll assume you have a larger dataset
        # In practice, you'd sample from your full token_pairs.parquet
        print("⚠️  Please create a sample dataset first:")
        print(f"   - Take a random sample from your full token_pairs.parquet")
        print(f"   - Save it as: {sample_pairs}")
        print("   - Recommended size: 20k-50k pairs for fast tuning")
        return
    
    # Step 2: Run hyperparameter tuning
    print(f"\n🎯 Starting hyperparameter tuning...")
    tune_cmd = [
        sys.executable, "-m", "src.deduplication.param_tuner", "tune",
        "--in-pairs", str(sample_pairs),
        "--out-base", str(tuning_dir),
        "--trials", "20",  # Start with fewer trials for example
        "--n-jobs", "1",
        "--study", "dedupe_example",
        "--storage", f"sqlite:///{storage_db}",
        "--timeout-sec", "300",
        "--keep-artifacts"
    ]
    
    if not run_command(tune_cmd, "Running hyperparameter tuning"):
        return
    
    # Step 3: Export best trials
    print(f"\n📈 Exporting best trials...")
    best_cmd = [
        sys.executable, "-m", "src.deduplication.param_tuner", "best",
        "--study", "dedupe_example",
        "--storage", f"sqlite:///{storage_db}",
        "--top", "3",
        "--export", str(base_dir / "best_trials.csv")
    ]
    
    if not run_command(best_cmd, "Exporting best trials"):
        return
    
    # Step 4: Validate top trial
    best_trials_file = base_dir / "best_trials.csv"
    if best_trials_file.exists():
        import pandas as pd
        best_trials = pd.read_csv(best_trials_file)
        if len(best_trials) > 0:
            top_trial_dir = best_trials.iloc[0]["out_dir"]
            if top_trial_dir and Path(top_trial_dir).exists():
                print(f"\n🔍 Validating top trial...")
                validate_cmd = [
                    sys.executable, "-m", "src.deduplication.param_tuner", "validate",
                    "--trial-dir", top_trial_dir,
                    "--in-pairs", str(sample_pairs)
                ]
                run_command(validate_cmd, "Validating top trial")
    
    # Step 5: Show results summary
    print(f"\n📊 Tuning Results Summary:")
    print(f"   📁 Tuning directory: {tuning_dir}")
    print(f"   💾 Storage database: {storage_db}")
    print(f"   📈 Best trials: {best_trials_file}")
    
    if best_trials_file.exists():
        import pandas as pd
        best_trials = pd.read_csv(best_trials_file)
        print(f"\n🏆 Top 3 Trials:")
        for i, row in best_trials.head(3).iterrows():
            print(f"   {i+1}. Trial {row['trial']}: "
                  f"good_clusters={row['good_clusters']}, "
                  f"coverage={row['coverage']}, "
                  f"edges={row['edges']}")
    
    print(f"\n✅ Tuning example complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the best_trials.csv file")
    print(f"   2. Re-run top configurations on larger data")
    print(f"   3. Use the best parameters in your production pipeline")

if __name__ == "__main__":
    main()
