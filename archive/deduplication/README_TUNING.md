# Hyperparameter Tuning for Deduplication Pipeline

This document explains how to systematically optimize the deduplication pipeline parameters using Bayesian optimization and multi-objective optimization.

## 🎯 Overview

The tuning system uses **Optuna** with **TPE (Tree-structured Parzen Estimator)** sampling and **multi-objective optimization** to find Pareto-optimal parameter configurations that balance:

**Why TPE over Random/Grid Search:**
- **Early trials**: Explore parameter space broadly to build models
- **Later trials**: Focus on promising regions based on learned distributions  
- **Efficiency**: Get good results in ~50 trials instead of hundreds
- **Multivariate**: Models parameter interactions and correlations

- **Quality**: Number of good clusters (high cohesion, low noise)
- **Coverage**: Number of tokens normalized 
- **Efficiency**: Graph size (fewer edges = cleaner/faster)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install optuna>=3.0.0
```

### 2. Create Sample Dataset

For fast tuning, create a representative sample of your token pairs:

```bash
# Sample 50k pairs from your full dataset
python -c "
import pandas as pd
df = pd.read_parquet('data/intermediate/token_pairs.parquet')
sample = df.sample(n=50000, random_state=42)
sample.to_parquet('data/intermediate/token_pairs_sample.parquet', index=False)
print(f'Sampled {len(sample):,} pairs')
"
```

### 3. Run Tuning

```bash
# Basic tuning (40 trials, single worker)
python -m src.deduplication.param_tuner tune \
  --in-pairs data/intermediate/token_pairs_sample.parquet \
  --out-base organized_outputs/tuning_runs \
  --trials 40 \
  --study dedupe_optimization

# Parallel tuning with persistence (recommended)
python -m src.deduplication.param_tuner tune \
  --in-pairs data/intermediate/token_pairs_sample.parquet \
  --out-base organized_outputs/tuning_runs \
  --trials 100 \
  --n-jobs 4 \
  --study dedupe_optimization \
  --storage sqlite:///tuning.db \
  --keep-artifacts
```

### 4. Analyze Results

```bash
# Export best Pareto trials
python -m src.deduplication.param_tuner best \
  --study dedupe_optimization \
  --storage sqlite:///tuning.db \
  --top 5 \
  --export best_trials.csv

# Validate a specific trial
python -m src.deduplication.param_tuner validate \
  --trial-dir organized_outputs/tuning_runs/trial_0042_abc123 \
  --in-pairs data/intermediate/token_pairs_sample.parquet
```

## 📊 Metrics and Objectives

### Primary Metrics

The tuner optimizes three objectives simultaneously:

1. **Good Clusters** (maximize): Number of clusters passing size-aware quality criteria
2. **Coverage** (maximize): Total number of tokens in clusters (normalization coverage)
3. **Graph Efficiency** (maximize): Negative edge count (fewer edges = cleaner graph)

### Quality Criteria

The system uses size-aware quality assessment:

- **Pairs (size=2)**: `min_sim ≥ 0.85` (no triangle requirement)
- **Small clusters (3-5)**: `min_sim ≥ 0.75` AND `triangle_rate ≥ 0.33`
- **Big clusters (>5)**: `min_sim ≥ 0.70`, `mean_sim ≥ 0.78`, `triangle_rate ≥ 0.20`

### Additional Metrics

- `min_sim_mean`: Average minimum similarity across clusters
- `total_clusters`: Total number of clusters found
- `pairs_count`: Number of size-2 clusters
- `multi_count`: Number of multi-token clusters

## 🔧 Parameter Search Space

### Graph Building Parameters

- `base_threshold`: [0.55, 0.85] - Minimum similarity floor
- `k`: [3, 20] - Top-k neighbors per token
- `mutual_nn`: {True, False} - Mutual nearest neighbor filtering
- `max_degree`: [30, 200] - Hub capping limit
- `adaptive_short_thr`: [0.75, 0.90] - Threshold for short tokens

### Quality Gate Parameters

- `pair_min_sim`: [0.82, 0.95] - Minimum similarity for pairs
- `small_min_sim`: [0.70, 0.85] - Minimum similarity for small clusters
- `small_min_tri`: [0.20, 0.60] - Minimum triangle rate for small clusters
- `big_min_sim`: [0.65, 0.80] - Minimum similarity for big clusters
- `big_min_mean`: [0.75, 0.85] - Minimum mean similarity for big clusters
- `big_min_tri`: [0.10, 0.30] - Minimum triangle rate for big clusters

### Parameter Constraints

The tuner automatically applies domain constraints:

- `small_min_sim ≤ big_min_mean - 0.01` (prevents over-strict small clusters)
- All similarity thresholds are bounded to realistic ranges
- Integer parameters use appropriate ranges

## 🧠 Bayesian Optimization with TPE

### How TPE Works

**Tree-structured Parzen Estimator** is a sophisticated Bayesian optimization method:

1. **Exploration Phase** (first ~20% of trials):
   - Samples parameters uniformly across the search space
   - Builds initial models of the objective landscape
   - No assumptions about parameter relationships

2. **Exploitation Phase** (remaining trials):
   - Models the distribution of "good" vs "bad" parameters
   - Samples more frequently from promising regions
   - Learns parameter interactions and correlations

### Efficiency Comparison

| Method | Trials Needed | Efficiency | Parameter Interactions |
|--------|---------------|------------|----------------------|
| **Grid Search** | 1000s | Very Low | None |
| **Random Search** | 500+ | Low | None |
| **TPE (Bayesian)** | 50-100 | **High** | **Modeled** |

### Why TPE is Superior

- **Adaptive**: Learns from each trial to improve future sampling
- **Multivariate**: Captures parameter correlations (e.g., `k` vs `max_degree`)
- **Efficient**: Focuses computational budget on promising regions
- **Robust**: Handles noisy objectives and parameter interactions

## 🎯 Multi-Objective Optimization

### Pareto Front

The tuner finds a **Pareto front** of non-dominated solutions. A solution is Pareto-optimal if no other solution is better in all objectives simultaneously.

Example Pareto solutions:
- **Solution A**: 50 good clusters, 1000 coverage, 200 edges
- **Solution B**: 45 good clusters, 1200 coverage, 150 edges  
- **Solution C**: 55 good clusters, 800 coverage, 300 edges

All three are Pareto-optimal because:
- A has more good clusters than B, but B has better coverage and efficiency
- C has the most good clusters, but worst efficiency
- No single solution dominates all others

### Selection Strategy

After tuning, select from Pareto solutions using:

1. **Primary**: Maximize `good_clusters` (quality first)
2. **Secondary**: Maximize `coverage` (normalization breadth)
3. **Tertiary**: Minimize `edges` (efficiency)

## 📈 Tuning Strategy

### 1. Subsample First

- Start with 20k-50k pairs for fast iteration
- Use fixed random seed for reproducibility
- Aim for 1-2 minute per trial

### 2. Escalate Winners

- Re-run top 3-5 Pareto solutions on larger data
- Use 200k-500k pairs for validation
- Final selection on full dataset

### 3. Parallel Execution

```bash
# Enable parallel tuning with SQLite storage
python -m src.deduplication.param_tuner tune \
  --n-jobs 4 \
  --storage sqlite:///tuning.db \
  --trials 200
```

### 4. Early Stopping & Bayesian Efficiency

The tuner includes built-in pruning and intelligent sampling:
- **TPE sampler**: Models parameter distributions, focuses on promising regions
- **Timeout protection**: Default 10 minutes per trial prevents runaway configs
- **Failed runs**: Return dominated metrics to guide future trials
- **Adaptive sampling**: Later trials automatically focus on better parameter combinations

## 🔍 Analysis and Validation

### Trial Validation

```bash
# Check a specific trial
python -m src.deduplication.param_tuner validate \
  --trial-dir organized_outputs/tuning_runs/trial_0042_abc123 \
  --in-pairs data/intermediate/token_pairs_sample.parquet
```

### Comparison

```bash
# Compare multiple trials
python -m src.deduplication.param_tuner compare \
  --trial-dirs organized_outputs/tuning_runs/trial_0042_abc123 \
  --trial-dirs organized_outputs/tuning_runs/trial_0043_def456 \
  --export comparison.csv
```

### Results Analysis

```python
import pandas as pd

# Load results
results = pd.read_csv('best_trials.csv')

# Analyze Pareto front
print("Pareto Solutions:")
print(results[['trial', 'good_clusters', 'coverage', 'edges']].head())

# Parameter analysis
print("\nParameter Ranges:")
print(results[['base_threshold', 'k', 'max_degree']].describe())
```

## 🚀 Production Deployment

### 1. Select Best Configuration

```python
# Load best trials
best = pd.read_csv('best_trials.csv')

# Select top configuration
top_config = best.iloc[0]
params = {
    'base_threshold': top_config['base_threshold'],
    'k': int(top_config['k']),
    'max_degree': int(top_config['max_degree']),
    'adaptive_short_thr': top_config['adaptive_short_thr'],
    'pair_min_sim': top_config['pair_min_sim'],
    'small_min_sim': top_config['small_min_sim'],
    'small_min_tri': top_config['small_min_tri'],
    'big_min_sim': top_config['big_min_sim'],
    'big_min_mean': top_config['big_min_mean'],
    'big_min_tri': top_config['big_min_tri'],
    'mutual_nn': top_config['mutual_nn']
}
```

### 2. Run Production Pipeline

```bash
# Use optimized parameters
python src/deduplication/dedupe_pipeline.py refined-all \
  data/intermediate/token_pairs.parquet \
  organized_outputs/production \
  --base-threshold 0.65 \
  --k 8 \
  --max-degree 100 \
  --adaptive-short-thr 0.80 \
  --pair-min-sim 0.88 \
  --small-min-sim 0.75 \
  --small-min-tri 0.35 \
  --big-min-sim 0.70 \
  --big-min-mean 0.78 \
  --big-min-tri 0.20 \
  --mutual-nn
```

## 🔧 Advanced Usage

### Custom Objectives

To add custom objectives, modify the `DedupeObjective.__call__` method:

```python
def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float, float]:
    # ... existing code ...
    
    # Add custom metric
    custom_metric = compute_custom_metric(out_dir)
    
    # Return 4 objectives
    return (good_clusters, coverage, -edges, custom_metric)
```

### Constraint Handling

Add parameter constraints:

```python
# In the objective function
if k > max_degree / 2:
    k = max_degree // 2

if small_min_sim > big_min_mean:
    small_min_sim = big_min_mean - 0.01
```

### Distributed Tuning

For cluster-scale tuning, use Ray Tune:

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Configure Ray
ray.init()

# Define search space
search_space = {
    "base_threshold": tune.uniform(0.55, 0.85),
    "k": tune.randint(3, 20),
    "max_degree": tune.randint(30, 200),
    # ... other parameters
}

# Run distributed tuning
analysis = tune.run(
    objective_function,
    search_alg=OptunaSearch(),
    scheduler=ASHAScheduler(metric="good_clusters", mode="max"),
    config=search_space,
    num_samples=1000,
    resources_per_trial={"cpu": 2, "gpu": 0}
)
```

## 📚 References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Multi-Objective Optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#multi-objective-optimization)
- [TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

## 🐛 Troubleshooting

### Common Issues

1. **Timeout errors**: Increase `--timeout-sec` or reduce sample size
2. **Memory issues**: Use smaller chunks or reduce `max_degree`
3. **No good clusters**: Lower quality thresholds or increase `base_threshold`
4. **Too many edges**: Increase `k` or enable `mutual_nn`

### Debug Mode

```bash
# Run with verbose output
python -m src.deduplication.param_tuner tune \
  --keep-artifacts \
  --timeout-sec 1200 \
  --trials 5
```

### Performance Tips

- Use SSD storage for SQLite database
- Enable parallel execution with `--n-jobs`
- Start with smaller sample sizes
- Use `--keep-artifacts` for debugging
- Monitor memory usage with `psutil`
