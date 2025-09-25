#!/usr/bin/env python3
"""
Example usage of the enhanced dedupe pipeline with quality diagnostics.

This script demonstrates how to:
1. Run the full pipeline with quality metrics
2. Diagnose existing results
3. Apply mutual nearest neighbor pruning
4. Re-cluster with improved quality
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str], description: str) -> None:
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        sys.exit(1)

def main():
    """Example usage of the enhanced dedupe pipeline."""
    
    # Example paths - update these to match your data
    input_pairs = "data/raw/token_pairs.parquet"  # Your input file
    output_dir = "outputs/dedupe_analysis"
    threshold = 0.6  # Higher threshold for better quality
    
    print("🚀 Enhanced Dedupe Pipeline Example")
    print(f"Input: {input_pairs}")
    print(f"Output: {output_dir}")
    print(f"Threshold: {threshold}")
    
    # Check if input exists
    if not Path(input_pairs).exists():
        print(f"❌ Input file {input_pairs} not found!")
        print("Please update the input_pairs path in this script.")
        return
    
    # 1. Run full pipeline with quality metrics
    run_command([
        "python", "dedupe_pipeline.py", "all",
        input_pairs, output_dir,
        "--threshold", str(threshold),
        "--min-len", "3"
    ], "Running full pipeline with quality metrics")
    
    # 2. Diagnose the results
    run_command([
        "python", "dedupe_pipeline.py", "diagnose",
        output_dir,
        "--threshold", str(threshold),
        "--min-sim-threshold", "0.62",
        "--min-triangle-rate", "0.10"
    ], "Diagnosing cluster quality")
    
    # 3. Apply mutual nearest neighbor pruning
    run_command([
        "python", "dedupe_pipeline.py", "prune",
        f"{output_dir}/pairs_filtered.parquet",
        f"{output_dir}/pruned",
        "--min-sim", "0.65",
        "--require-triangle",
        "--k", "5"
    ], "Applying mutual nearest neighbor pruning")
    
    # 4. Compare results
    print(f"\n{'='*60}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Check output files
    output_path = Path(output_dir)
    pruned_path = Path(f"{output_dir}/pruned")
    
    files_to_check = [
        ("Original clusters", output_path / "clusters_summary.parquet"),
        ("Cohesion metrics", output_path / "cluster_cohesion_metrics.parquet"),
        ("Low quality clusters", output_path / "clusters_flagged_low_quality.parquet"),
        ("Pruned clusters", pruned_path / "clusters_summary.parquet"),
        ("Pruned cohesion", pruned_path / "cluster_cohesion_metrics.parquet"),
    ]
    
    for name, path in files_to_check:
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (not found)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Review cluster_cohesion_metrics.parquet for quality assessment")
    print(f"2. Check clusters_flagged_low_quality.parquet for problematic clusters")
    print(f"3. Compare original vs pruned results")
    print(f"4. Adjust thresholds based on quality metrics")
    print(f"5. Use mutual NN filtering for cleaner clusters")

if __name__ == "__main__":
    main()
