#!/usr/bin/env python3
"""
Full Corpus Deployment Script

This script takes the best parameters from optimization and applies them
to the full corpus with proper scaling and resource management.

Usage:
    python src/deduplication/deploy_to_full_corpus.py \
        --best-params best_trials.csv \
        --full-corpus data/intermediate/token_pairs.parquet \
        --output-dir outputs/production \
        --top-k 3
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import psutil

def load_best_parameters(params_file: Path, top_k: int = 3) -> List[Dict]:
    """Load the best parameters from optimization results."""
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")
    
    df = pd.read_csv(params_file)
    
    # Sort by objectives (good_clusters desc, coverage desc, edges asc)
    if 'good_clusters' in df.columns:
        df = df.sort_values(['good_clusters', 'coverage', 'edges'], 
                          ascending=[False, False, True])
    
    # Take top K
    top_params = df.head(top_k)
    
    # Convert to list of dictionaries
    params_list = []
    for _, row in top_params.iterrows():
        params = {
            'trial': row.get('trial', 'unknown'),
            'good_clusters': row.get('good_clusters', 0),
            'coverage': row.get('coverage', 0),
            'edges': row.get('edges', 0),
            'base_threshold': row.get('base_threshold', 0.65),
            'k': int(row.get('k', 8)),
            'max_degree': int(row.get('max_degree', 100)),
            'adaptive_short_thr': row.get('adaptive_short_thr', 0.80),
            'pair_min_sim': row.get('pair_min_sim', 0.85),
            'small_min_sim': row.get('small_min_sim', 0.75),
            'small_min_tri': row.get('small_min_tri', 0.33),
            'big_min_sim': row.get('big_min_sim', 0.70),
            'big_min_mean': row.get('big_min_mean', 0.78),
            'big_min_tri': row.get('big_min_tri', 0.20),
            'mutual_nn': row.get('mutual_nn', True)
        }
        params_list.append(params)
    
    return params_list

def estimate_resource_requirements(corpus_size: int) -> Dict:
    """Estimate resource requirements for full corpus processing."""
    # Rough estimates based on corpus size
    memory_gb = max(8, int(corpus_size * 0.00001))  # ~10MB per 1000 pairs
    cpu_cores = max(4, int(corpus_size * 0.000005))  # ~1 core per 200k pairs
    storage_gb = max(10, int(corpus_size * 0.00002))  # ~20MB per 1000 pairs
    
    return {
        'memory_gb': memory_gb,
        'cpu_cores': cpu_cores,
        'storage_gb': storage_gb,
        'estimated_time_hours': max(1, corpus_size * 0.000001)  # ~1 hour per 1M pairs
    }

def check_system_resources(requirements: Dict) -> bool:
    """Check if system has sufficient resources."""
    # Check memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    
    # Check disk space
    disk = psutil.disk_usage('/')
    available_disk_gb = disk.free / (1024**3)
    
    print(f"🔍 System Resource Check:")
    print(f"   Required memory: {requirements['memory_gb']} GB")
    print(f"   Available memory: {available_gb:.1f} GB")
    print(f"   Required CPU cores: {requirements['cpu_cores']}")
    print(f"   Available CPU cores: {cpu_count}")
    print(f"   Required storage: {requirements['storage_gb']} GB")
    print(f"   Available storage: {available_disk_gb:.1f} GB")
    print()
    
    # Check if resources are sufficient
    if available_gb < requirements['memory_gb']:
        print(f"⚠️  Warning: Insufficient memory ({available_gb:.1f} < {requirements['memory_gb']} GB)")
        return False
    
    if cpu_count < requirements['cpu_cores']:
        print(f"⚠️  Warning: Insufficient CPU cores ({cpu_count} < {requirements['cpu_cores']})")
        return False
    
    if available_disk_gb < requirements['storage_gb']:
        print(f"⚠️  Warning: Insufficient storage ({available_disk_gb:.1f} < {requirements['storage_gb']} GB)")
        return False
    
    print("✅ System resources are sufficient")
    return True

def run_pipeline_on_full_corpus(
    params: Dict,
    full_corpus: Path,
    output_dir: Path,
    trial_name: str
) -> bool:
    """Run the deduplication pipeline on full corpus with given parameters."""
    
    trial_output_dir = output_dir / f"trial_{trial_name}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "src/deduplication/dedupe_pipeline.py", "refined-all",
        str(full_corpus),
        str(trial_output_dir),
        "--base-threshold", f"{params['base_threshold']:.4f}",
        "--k", str(params['k']),
        "--max-degree", str(params['max_degree']),
        "--adaptive-short-thr", f"{params['adaptive_short_thr']:.4f}",
        "--pair-min-sim", f"{params['pair_min_sim']:.4f}",
        "--small-min-sim", f"{params['small_min_sim']:.4f}",
        "--small-min-tri", f"{params['small_min_tri']:.4f}",
        "--big-min-sim", f"{params['big_min_sim']:.4f}",
        "--big-min-mean", f"{params['big_min_mean']:.4f}",
        "--big-min-tri", f"{params['big_min_tri']:.4f}",
    ]
    
    if params['mutual_nn']:
        cmd.append("--mutual-nn")
    
    print(f"🚀 Running trial {trial_name} on full corpus...")
    print(f"   Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Trial {trial_name} completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"❌ Trial {trial_name} failed with return code {result.returncode}")
            print(f"Error: {result.stderr[-1000:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Trial {trial_name} timed out after 2 hours")
        return False
    except Exception as e:
        print(f"❌ Trial {trial_name} failed with error: {e}")
        return False

def analyze_results(output_dir: Path, params_list: List[Dict]) -> pd.DataFrame:
    """Analyze the results from all trials."""
    results = []
    
    for i, params in enumerate(params_list):
        trial_name = f"trial_{params['trial']}"
        trial_dir = output_dir / trial_name
        
        # Check if trial completed successfully
        required_files = [
            "clusters_flagged_size_aware.parquet",
            "cluster_cohesion_metrics.parquet",
            "clusters_token_map.parquet",
            "pairs_clean_graph.parquet"
        ]
        
        success = all((trial_dir / file).exists() for file in required_files)
        
        if success:
            # Load metrics
            try:
                flagged = pd.read_parquet(trial_dir / "clusters_flagged_size_aware.parquet")
                token_map = pd.read_parquet(trial_dir / "clusters_token_map.parquet")
                clean_graph = pd.read_parquet(trial_dir / "pairs_clean_graph.parquet")
                cohesion = pd.read_parquet(trial_dir / "cluster_cohesion_metrics.parquet")
                
                good_clusters = len(flagged[~flagged["is_flagged"]])
                total_clusters = len(flagged)
                coverage = len(token_map)
                edges = len(clean_graph)
                mean_min_sim = cohesion["min_sim"].mean() if len(cohesion) > 0 else 0.0
                
                results.append({
                    'trial': params['trial'],
                    'rank': i + 1,
                    'success': True,
                    'good_clusters': good_clusters,
                    'total_clusters': total_clusters,
                    'coverage': coverage,
                    'edges': edges,
                    'mean_min_sim': mean_min_sim,
                    **params
                })
                
            except Exception as e:
                print(f"⚠️  Error analyzing trial {trial_name}: {e}")
                results.append({
                    'trial': params['trial'],
                    'rank': i + 1,
                    'success': False,
                    'error': str(e),
                    **params
                })
        else:
            results.append({
                'trial': params['trial'],
                'rank': i + 1,
                'success': False,
                'error': 'Missing output files',
                **params
            })
    
    return pd.DataFrame(results)

def create_production_summary(results_df: pd.DataFrame, output_dir: Path):
    """Create a production summary with recommendations."""
    
    successful_results = results_df[results_df['success'] == True]
    
    if len(successful_results) == 0:
        print("❌ No successful trials found!")
        return
    
    # Sort by objectives
    successful_results = successful_results.sort_values(
        ['good_clusters', 'coverage', 'edges'], 
        ascending=[False, False, True]
    )
    
    best_result = successful_results.iloc[0]
    
    summary = {
        'deployment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_trials': len(results_df),
        'successful_trials': len(successful_results),
        'best_trial': {
            'trial_id': best_result['trial'],
            'rank': best_result['rank'],
            'good_clusters': int(best_result['good_clusters']),
            'total_clusters': int(best_result['total_clusters']),
            'coverage': int(best_result['coverage']),
            'edges': int(best_result['edges']),
            'mean_min_sim': float(best_result['mean_min_sim']),
            'parameters': {
                'base_threshold': float(best_result['base_threshold']),
                'k': int(best_result['k']),
                'max_degree': int(best_result['max_degree']),
                'adaptive_short_thr': float(best_result['adaptive_short_thr']),
                'pair_min_sim': float(best_result['pair_min_sim']),
                'small_min_sim': float(best_result['small_min_sim']),
                'small_min_tri': float(best_result['small_min_tri']),
                'big_min_sim': float(best_result['big_min_sim']),
                'big_min_mean': float(best_result['big_min_mean']),
                'big_min_tri': float(best_result['big_min_tri']),
                'mutual_nn': bool(best_result['mutual_nn'])
            }
        },
        'all_results': results_df.to_dict('records')
    }
    
    # Save summary
    summary_file = output_dir / "production_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save results CSV
    results_file = output_dir / "production_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"📊 Production Summary:")
    print(f"   Total trials: {summary['total_trials']}")
    print(f"   Successful trials: {summary['successful_trials']}")
    print(f"   Best trial: {best_result['trial']}")
    print(f"   Good clusters: {best_result['good_clusters']}")
    print(f"   Coverage: {best_result['coverage']}")
    print(f"   Edges: {best_result['edges']}")
    print(f"   Mean min sim: {best_result['mean_min_sim']:.4f}")
    print()
    
    print(f"🎯 Best Parameters:")
    for param, value in summary['best_trial']['parameters'].items():
        print(f"   {param}: {value}")
    print()
    
    print(f"📁 Results saved to:")
    print(f"   Summary: {summary_file}")
    print(f"   Results: {results_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Deploy optimized parameters to full corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--best-params",
        type=Path,
        required=True,
        help="Path to best parameters CSV file"
    )
    
    parser.add_argument(
        "--full-corpus",
        type=Path,
        required=True,
        help="Path to full corpus token pairs parquet"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for production results"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top trials to deploy"
    )
    
    parser.add_argument(
        "--skip-resource-check",
        action="store_true",
        help="Skip system resource check"
    )
    
    args = parser.parse_args()
    
    print("🚀 Full Corpus Deployment")
    print("=" * 40)
    print(f"📊 Best parameters: {args.best_params}")
    print(f"📁 Full corpus: {args.full_corpus}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Top K trials: {args.top_k}")
    print()
    
    # Load best parameters
    try:
        params_list = load_best_parameters(args.best_params, args.top_k)
        print(f"✅ Loaded {len(params_list)} parameter sets")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        sys.exit(1)
    
    # Estimate resource requirements
    corpus_size = 1000000  # Rough estimate, could be improved
    requirements = estimate_resource_requirements(corpus_size)
    
    # Check system resources
    if not args.skip_resource_check:
        if not check_system_resources(requirements):
            print("❌ Insufficient system resources. Use --skip-resource-check to proceed anyway.")
            sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run trials on full corpus
    print(f"🔄 Running {len(params_list)} trials on full corpus...")
    print()
    
    successful_trials = 0
    for i, params in enumerate(params_list, 1):
        trial_name = f"{params['trial']}"
        success = run_pipeline_on_full_corpus(
            params, args.full_corpus, args.output_dir, trial_name
        )
        if success:
            successful_trials += 1
        print()
    
    print(f"✅ Completed {successful_trials}/{len(params_list)} trials successfully")
    
    # Analyze results
    print("📊 Analyzing results...")
    results_df = analyze_results(args.output_dir, params_list)
    
    # Create production summary
    create_production_summary(results_df, args.output_dir)
    
    print("🎉 Full corpus deployment complete!")

if __name__ == "__main__":
    main()
