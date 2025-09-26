#!/usr/bin/env python3
"""
Test script for the hyperparameter tuner.

This script creates a minimal test dataset and runs a few tuning trials
to verify the tuner works correctly.
"""

import tempfile
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def create_test_data(output_path: Path, n_pairs: int = 1000):
    """Create a minimal test dataset for tuning."""
    print(f"Creating test dataset with {n_pairs:,} pairs...")
    
    # Create synthetic token pairs with varying similarities
    np.random.seed(42)
    
    # Generate token pairs
    tokens_a = [f"token_{i:03d}" for i in range(n_pairs)]
    tokens_b = [f"token_{i+1:03d}" for i in range(n_pairs)]
    
    # Create similarities with some structure
    similarities = np.random.beta(2, 5, n_pairs)  # Skewed toward lower similarities
    similarities = np.clip(similarities, 0.1, 0.99)
    
    # Add some high-similarity pairs
    high_sim_indices = np.random.choice(n_pairs, size=n_pairs//10, replace=False)
    similarities[high_sim_indices] = np.random.uniform(0.8, 0.95, len(high_sim_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'token_a': tokens_a,
        'token_b': tokens_b,
        'cosine_sim': similarities,
        'rank': np.random.randint(1, 100, n_pairs)
    })
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Created test dataset: {output_path}")
    return df

def test_tuner():
    """Test the hyperparameter tuner with a small dataset."""
    print("🧪 Testing Hyperparameter Tuner")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        test_pairs = temp_path / "test_pairs.parquet"
        create_test_data(test_pairs, n_pairs=500)  # Small dataset for fast testing
        
        # Create output directory
        output_dir = temp_path / "tuning_test"
        
        # Run tuning with minimal trials (direct script execution)
        cmd = [
            sys.executable, "src/deduplication/param_tuner.py", "tune",
            "--in-pairs", str(test_pairs),
            "--out-base", str(output_dir),
            "--trials", "3",  # Just 3 trials for testing
            "--n-jobs", "1",
            "--study", "test_study",
            "--timeout-sec", "120",
            "--keep-artifacts"
        ]
        
        print(f"\n🔧 Running test tuning...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Tuning completed successfully!")
                print("\nOutput:")
                print(result.stdout)
                
                # Check if output files were created
                if (output_dir / "pareto_trials.csv").exists():
                    print("✅ Pareto trials CSV created")
                    
                    # Load and display results
                    results = pd.read_csv(output_dir / "pareto_trials.csv")
                    print(f"\n📊 Results Summary:")
                    print(f"   Trials completed: {len(results)}")
                    if len(results) > 0:
                        best = results.iloc[0]
                        print(f"   Best trial: {best['trial']}")
                        print(f"   Good clusters: {best['good_clusters']}")
                        print(f"   Coverage: {best['coverage']}")
                        print(f"   Edges: {best['edges']}")
                else:
                    print("❌ Pareto trials CSV not found")
                    
            else:
                print("❌ Tuning failed!")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Tuning timed out!")
            return False
        except Exception as e:
            print(f"❌ Error running tuning: {e}")
            return False
    
    print("\n✅ Test completed successfully!")
    return True

def test_validation():
    """Test the validation functionality."""
    print("\n🔍 Testing Validation Functionality")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        test_pairs = temp_path / "test_pairs.parquet"
        create_test_data(test_pairs, n_pairs=100)
        
        # Create a mock trial directory with some files
        trial_dir = temp_path / "mock_trial"
        trial_dir.mkdir()
        
        # Create mock output files
        mock_flagged = pd.DataFrame({
            'cluster_id': [1, 2, 3],
            'size': [2, 3, 4],
            'is_flagged': [False, True, False]
        })
        mock_flagged.to_parquet(trial_dir / "clusters_flagged_size_aware.parquet", index=False)
        
        mock_tokens = pd.DataFrame({
            'token': ['token_001', 'token_002', 'token_003'],
            'cluster_id': [1, 1, 2]
        })
        mock_tokens.to_parquet(trial_dir / "clusters_token_map.parquet", index=False)
        
        mock_graph = pd.DataFrame({
            'token_a': ['token_001', 'token_002'],
            'token_b': ['token_002', 'token_003'],
            'cosine_sim': [0.85, 0.75]
        })
        mock_graph.to_parquet(trial_dir / "pairs_clean_graph.parquet", index=False)
        
        mock_cohesion = pd.DataFrame({
            'cluster_id': [1, 2],
            'min_sim': [0.85, 0.75],
            'mean_sim': [0.90, 0.80]
        })
        mock_cohesion.to_parquet(trial_dir / "cluster_cohesion_metrics.parquet", index=False)
        
        # Test validation
        cmd = [
            sys.executable, "src/deduplication/param_tuner.py", "validate",
            "--trial-dir", str(trial_dir),
            "--in-pairs", str(test_pairs)
        ]
        
        print(f"Running validation test...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Validation test passed!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Validation test failed!")
            print("Error:")
            print(result.stderr)
            return False
    
    return True

if __name__ == "__main__":
    print("🚀 Hyperparameter Tuner Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic tuning functionality
    if not test_tuner():
        success = False
    
    # Test 2: Validation functionality
    if not test_validation():
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
