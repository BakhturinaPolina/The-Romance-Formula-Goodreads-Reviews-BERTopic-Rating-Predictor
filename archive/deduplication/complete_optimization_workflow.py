#!/usr/bin/env python3
"""
Complete Optimization Workflow

This script provides a complete workflow for:
1. Calculating optimization runs
2. Running advanced multi-objective tuning
3. Deploying best parameters to full corpus
4. Analyzing and reporting results

Usage:
    python src/deduplication/complete_optimization_workflow.py \
        --sample-pairs data/intermediate/token_pairs_sample.parquet \
        --full-corpus data/intermediate/token_pairs.parquet \
        --output-dir organized_outputs/complete_workflow \
        --trials 60 --top-k 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

def run_command(cmd: List[str], description: str, timeout: int = 3600) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:
                    if line.strip():
                        print(f"   {line}")
            return True
        else:
            print("❌ Failed")
            print("Error output:")
            print(result.stderr[-1000:])  # Last 1000 chars
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def calculate_optimization_runs(
    trials: int,
    budgets: List[float],
    folds: int,
    n_jobs: int,
    pruner: str
) -> Dict:
    """Calculate optimization run statistics."""
    print("🔢 Calculating Optimization Runs")
    print("=" * 40)
    
    # Basic calculation
    total_runs = trials * len(budgets) * folds
    
    # Estimate pruning effects
    if pruner == "hyperband":
        pruned_ratio = 0.6  # 60% pruned
    elif pruner == "median":
        pruned_ratio = 0.4  # 40% pruned
    else:
        pruned_ratio = 0.0  # No pruning
    
    pruned_runs = int(total_runs * pruned_ratio)
    successful_runs = total_runs - pruned_runs
    
    # Estimate time per run (varies by budget)
    time_per_run = {
        0.05: 30,   # 30 seconds for 5% budget
        0.10: 60,   # 1 minute for 10% budget
        0.20: 120,  # 2 minutes for 20% budget
        0.50: 300,  # 5 minutes for 50% budget
        1.00: 600,  # 10 minutes for 100% budget
    }
    
    total_time_seconds = 0
    for budget in budgets:
        runs_at_budget = trials * folds
        time_per_run_budget = time_per_run.get(budget, 300)
        total_time_seconds += runs_at_budget * time_per_run_budget
    
    # Account for parallelization
    if n_jobs > 1:
        total_time_seconds = total_time_seconds / n_jobs
    
    estimated_time_hours = total_time_seconds / 3600
    
    calculation = {
        'total_trials': trials,
        'total_runs': total_runs,
        'pruned_runs': pruned_runs,
        'successful_runs': successful_runs,
        'estimated_time_hours': estimated_time_hours,
        'budgets': budgets,
        'folds': folds,
        'n_jobs': n_jobs,
        'pruner': pruner
    }
    
    print(f"📊 Run Statistics:")
    print(f"   Total trials: {trials}")
    print(f"   Total runs: {total_runs:,}")
    print(f"   Pruned runs: {pruned_runs:,} ({pruned_runs/total_runs*100:.1f}%)")
    print(f"   Successful runs: {successful_runs:,}")
    print(f"   Estimated time: {estimated_time_hours:.1f} hours")
    print()
    
    return calculation

def run_advanced_tuning(
    sample_pairs: Path,
    output_dir: Path,
    trials: int,
    n_jobs: int,
    study_name: str,
    storage_db: Path,
    sampler: str = "nsga2",
    pruner: str = "hyperband",
    budgets: List[float] = [0.05, 0.20, 1.00],
    folds: int = 2,
    min_coverage: int = 10,
    max_edges: int = 20000,
    top_k: int = 5,
    timeout_sec: int = 1800
) -> bool:
    """Run advanced multi-objective tuning."""
    print("🎯 Running Advanced Multi-Objective Tuning")
    print("=" * 50)
    
    # Create tuning command
    cmd = [
        sys.executable, "src/deduplication/param_tuner_advanced.py", "tune",
        "--in-pairs", str(sample_pairs),
        "--out-base", str(output_dir / "tuning_runs"),
        "--trials", str(trials),
        "--n-jobs", str(n_jobs),
        "--study-name", study_name,
        "--storage", f"sqlite:///{storage_db}",
        "--sampler", sampler,
        "--pruner", pruner,
        "--budgets", ",".join(map(str, budgets)),
        "--folds", str(folds),
        "--min-coverage", str(min_coverage),
        "--max-edges", str(max_edges),
        "--timeout-sec", str(timeout_sec),
        "--top-k", str(top_k),
        "--keep-artifacts"
    ]
    
    return run_command(cmd, "Running advanced tuning", timeout=timeout_sec * 2)

def analyze_tuning_results(
    study_name: str,
    storage_db: Path,
    output_dir: Path
) -> bool:
    """Analyze tuning results and create visualizations."""
    print("📊 Analyzing Tuning Results")
    print("=" * 30)
    
    cmd = [
        sys.executable, "-m", "src.deduplication.param_tuner_advanced", "analyze",
        "--study-name", study_name,
        "--storage", f"sqlite:///{storage_db}",
        "--out-dir", str(output_dir / "analysis")
    ]
    
    return run_command(cmd, "Analyzing results")

def deploy_to_full_corpus(
    best_params_file: Path,
    full_corpus: Path,
    output_dir: Path,
    top_k: int
) -> bool:
    """Deploy best parameters to full corpus."""
    print("🚀 Deploying to Full Corpus")
    print("=" * 30)
    
    cmd = [
        sys.executable, "src/deduplication/deploy_to_full_corpus.py",
        "--best-params", str(best_params_file),
        "--full-corpus", str(full_corpus),
        "--output-dir", str(output_dir / "production"),
        "--top-k", str(top_k)
    ]
    
    return run_command(cmd, "Deploying to full corpus", timeout=7200)

def create_workflow_summary(
    calculation: Dict,
    output_dir: Path,
    sample_pairs: Path,
    full_corpus: Path
) -> None:
    """Create a comprehensive workflow summary."""
    print("📋 Creating Workflow Summary")
    print("=" * 30)
    
    # Check if files exist
    tuning_results = output_dir / "tuning_runs" / "pareto_full_results.csv"
    production_results = output_dir / "production" / "production_results.csv"
    analysis_dir = output_dir / "analysis"
    
    summary = {
        'workflow_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_files': {
            'sample_pairs': str(sample_pairs),
            'full_corpus': str(full_corpus)
        },
        'optimization_calculation': calculation,
        'output_files': {
            'tuning_results': str(tuning_results) if tuning_results.exists() else None,
            'production_results': str(production_results) if production_results.exists() else None,
            'analysis_dir': str(analysis_dir) if analysis_dir.exists() else None
        },
        'status': {
            'tuning_completed': tuning_results.exists(),
            'analysis_completed': analysis_dir.exists(),
            'production_completed': production_results.exists()
        }
    }
    
    # Add tuning results if available
    if tuning_results.exists():
        try:
            tuning_df = pd.read_csv(tuning_results)
            summary['tuning_summary'] = {
                'total_trials': len(tuning_df),
                'best_trial': tuning_df.iloc[0]['trial'] if len(tuning_df) > 0 else None,
                'best_good_clusters': tuning_df.iloc[0]['good_clusters'] if len(tuning_df) > 0 else None,
                'best_coverage': tuning_df.iloc[0]['coverage'] if len(tuning_df) > 0 else None,
                'best_edges': tuning_df.iloc[0]['edges'] if len(tuning_df) > 0 else None
            }
        except Exception as e:
            summary['tuning_summary'] = {'error': str(e)}
    
    # Add production results if available
    if production_results.exists():
        try:
            production_df = pd.read_csv(production_results)
            successful = production_df[production_df['success'] == True]
            if len(successful) > 0:
                best_production = successful.iloc[0]
                summary['production_summary'] = {
                    'total_trials': len(production_df),
                    'successful_trials': len(successful),
                    'best_trial': best_production['trial'],
                    'best_good_clusters': best_production['good_clusters'],
                    'best_coverage': best_production['coverage'],
                    'best_edges': best_production['edges']
                }
            else:
                summary['production_summary'] = {'error': 'No successful trials'}
        except Exception as e:
            summary['production_summary'] = {'error': str(e)}
    
    # Save summary
    summary_file = output_dir / "workflow_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Workflow summary saved to: {summary_file}")
    
    # Print summary
    print(f"\n📊 Workflow Summary:")
    print(f"   Date: {summary['workflow_date']}")
    print(f"   Sample pairs: {sample_pairs}")
    print(f"   Full corpus: {full_corpus}")
    print(f"   Total optimization runs: {calculation['total_runs']:,}")
    print(f"   Estimated time: {calculation['estimated_time_hours']:.1f} hours")
    
    if 'tuning_summary' in summary and 'error' not in summary['tuning_summary']:
        ts = summary['tuning_summary']
        print(f"   Best tuning trial: {ts['best_trial']}")
        print(f"   Best good clusters: {ts['best_good_clusters']}")
        print(f"   Best coverage: {ts['best_coverage']}")
        print(f"   Best edges: {ts['best_edges']}")
    
    if 'production_summary' in summary and 'error' not in summary['production_summary']:
        ps = summary['production_summary']
        print(f"   Production trials: {ps['successful_trials']}/{ps['total_trials']}")
        print(f"   Best production trial: {ps['best_trial']}")
        print(f"   Best good clusters: {ps['best_good_clusters']}")
        print(f"   Best coverage: {ps['best_coverage']}")
        print(f"   Best edges: {ps['best_edges']}")

def main():
    parser = argparse.ArgumentParser(
        description="Complete optimization workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        "--sample-pairs",
        type=Path,
        required=True,
        help="Path to sample token pairs parquet"
    )
    
    parser.add_argument(
        "--full-corpus",
        type=Path,
        required=True,
        help="Path to full corpus token pairs parquet"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all results"
    )
    
    # Optimization parameters
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Number of optimization trials"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs"
    )
    
    parser.add_argument(
        "--sampler",
        type=str,
        default="nsga2",
        choices=["nsga2", "motpe"],
        help="Optimization sampler"
    )
    
    parser.add_argument(
        "--pruner",
        type=str,
        default="hyperband",
        choices=["hyperband", "median"],
        help="Pruning strategy"
    )
    
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.05,0.20,1.00",
        help="Comma-separated budgets"
    )
    
    parser.add_argument(
        "--folds",
        type=int,
        default=2,
        help="Number of folds for robustness"
    )
    
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=10,
        help="Minimum coverage constraint"
    )
    
    parser.add_argument(
        "--max-edges",
        type=int,
        default=20000,
        help="Maximum edges constraint"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top trials to deploy"
    )
    
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Timeout per trial in seconds"
    )
    
    # Workflow control
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip tuning step (use existing results)"
    )
    
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step"
    )
    
    parser.add_argument(
        "--skip-production",
        action="store_true",
        help="Skip production deployment"
    )
    
    args = parser.parse_args()
    
    print("🚀 Complete Optimization Workflow")
    print("=" * 50)
    print(f"📊 Sample pairs: {args.sample_pairs}")
    print(f"📁 Full corpus: {args.full_corpus}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Trials: {args.trials}")
    print(f"⚡ Parallel jobs: {args.n_jobs}")
    print(f"🎯 Sampler: {args.sampler}")
    print(f"✂️  Pruner: {args.pruner}")
    print(f"🔄 Budgets: {args.budgets}")
    print(f"📁 Folds: {args.folds}")
    print(f"🎯 Top K: {args.top_k}")
    print()
    
    # Parse budgets
    budgets = [float(x.strip()) for x in args.budgets.split(",")]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Calculate optimization runs
    calculation = calculate_optimization_runs(
        args.trials, budgets, args.folds, args.n_jobs, args.pruner
    )
    
    # Step 2: Run advanced tuning
    if not args.skip_tuning:
        study_name = f"workflow_{int(time.time())}"
        storage_db = args.output_dir / "optimization.db"
        
        success = run_advanced_tuning(
            args.sample_pairs,
            args.output_dir,
            args.trials,
            args.n_jobs,
            study_name,
            storage_db,
            args.sampler,
            args.pruner,
            budgets,
            args.folds,
            args.min_coverage,
            args.max_edges,
            args.top_k,
            args.timeout_sec
        )
        
        if not success:
            print("❌ Tuning failed. Stopping workflow.")
            sys.exit(1)
    else:
        print("⏭️  Skipping tuning step")
        study_name = "existing_study"
        storage_db = args.output_dir / "optimization.db"
    
    # Step 3: Analyze results
    if not args.skip_analysis:
        success = analyze_tuning_results(study_name, storage_db, args.output_dir)
        if not success:
            print("⚠️  Analysis failed, but continuing...")
    else:
        print("⏭️  Skipping analysis step")
    
    # Step 4: Deploy to full corpus
    if not args.skip_production:
        best_params_file = args.output_dir / "tuning_runs" / "pareto_full_results.csv"
        
        if not best_params_file.exists():
            print(f"❌ Best parameters file not found: {best_params_file}")
            print("   Make sure tuning completed successfully.")
            sys.exit(1)
        
        success = deploy_to_full_corpus(
            best_params_file,
            args.full_corpus,
            args.output_dir,
            args.top_k
        )
        
        if not success:
            print("❌ Production deployment failed.")
            sys.exit(1)
    else:
        print("⏭️  Skipping production deployment")
    
    # Step 5: Create workflow summary
    create_workflow_summary(
        calculation,
        args.output_dir,
        args.sample_pairs,
        args.full_corpus
    )
    
    print("\n🎉 Complete optimization workflow finished!")
    print(f"📁 All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
