#!/usr/bin/env python3
"""
Quick start script for hyperparameter tuning.

This script provides a simple interface to run hyperparameter tuning
with sensible defaults for common use cases.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

def create_sample_data(input_pairs: Path, output_pairs: Path, sample_size: int = 50000):
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

def run_quick_tuning(args):
    """Run quick hyperparameter tuning with sensible defaults."""
    
    # Create sample data if needed
    if args.sample_size > 0:
        sample_pairs = args.output_dir / "token_pairs_sample.parquet"
        if not sample_pairs.exists() or args.force_sample:
            create_sample_data(args.input_pairs, sample_pairs, args.sample_size)
        input_data = sample_pairs
    else:
        input_data = args.input_pairs
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.deduplication.param_tuner", "tune",
        "--in-pairs", str(input_data),
        "--out-base", str(args.output_dir / "tuning_runs"),
        "--trials", str(args.trials),
        "--n-jobs", str(args.n_jobs),
        "--study", args.study_name,
        "--timeout-sec", str(args.timeout),
    ]
    
    # Add storage if specified
    if args.storage:
        cmd.extend(["--storage", args.storage])
    
    # Add keep artifacts if requested
    if args.keep_artifacts:
        cmd.append("--keep-artifacts")
    
    print(f"\n🎯 Starting hyperparameter tuning...")
    print(f"   Input: {input_data}")
    print(f"   Trials: {args.trials}")
    print(f"   Parallel jobs: {args.n_jobs}")
    print(f"   Study: {args.study_name}")
    print(f"   Timeout: {args.timeout}s")
    
    # Run tuning
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Tuning completed successfully!")
        
        # Export best trials
        if args.storage:
            export_cmd = [
                sys.executable, "-m", "src.deduplication.param_tuner", "best",
                "--study", args.study_name,
                "--storage", args.storage,
                "--top", str(args.top_trials),
                "--export", str(args.output_dir / "best_trials.csv")
            ]
            
            print(f"\n📈 Exporting best {args.top_trials} trials...")
            subprocess.run(export_cmd, check=True)
            
            # Show results
            best_file = args.output_dir / "best_trials.csv"
            if best_file.exists():
                results = pd.read_csv(best_file)
                print(f"\n🏆 Top {min(args.top_trials, len(results))} Results:")
                for i, row in results.head(args.top_trials).iterrows():
                    print(f"   {i+1}. Trial {row['trial']}: "
                          f"good_clusters={row['good_clusters']}, "
                          f"coverage={row['coverage']}, "
                          f"edges={row['edges']}")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tuning failed with return code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Tuning interrupted by user")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Quick hyperparameter tuning for deduplication pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_pairs",
        type=Path,
        help="Path to input token pairs parquet file"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for tuning results"
    )
    
    # Tuning parameters
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of tuning trials"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--study-name",
        default="quick_tuning",
        help="Optuna study name"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per trial in seconds"
    )
    
    # Data sampling
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Sample size for tuning (0 = use full dataset)"
    )
    parser.add_argument(
        "--force-sample",
        action="store_true",
        help="Force recreation of sample dataset"
    )
    
    # Storage and persistence
    parser.add_argument(
        "--storage",
        help="Storage URL (e.g., sqlite:///tuning.db) for persistence and parallelism"
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep all trial artifacts (default: only Pareto winners)"
    )
    
    # Results
    parser.add_argument(
        "--top-trials",
        type=int,
        default=5,
        help="Number of top trials to export"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_pairs.exists():
        print(f"❌ Input file not found: {args.input_pairs}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tuning
    run_quick_tuning(args)

if __name__ == "__main__":
    main()
