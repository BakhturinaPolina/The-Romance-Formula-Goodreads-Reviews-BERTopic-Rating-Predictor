# Complete Optimization and Deployment Guide

This guide provides a comprehensive strategy for calculating optimization runs and deploying results to the full corpus.

## 🎯 **Overview**

The complete workflow consists of:

1. **Optimization Run Calculation** - Calculate total runs and resource requirements
2. **Advanced Multi-Objective Tuning** - Find optimal parameters using NSGA-II/MOTPE
3. **Full Corpus Deployment** - Apply best parameters to production data
4. **Results Analysis** - Analyze and validate production results

## 🔢 **Optimization Run Calculation**

### Formula for Total Runs

```
Total Runs = Trials × Budgets × Folds
```

**Example Configuration:**
- Trials: 60
- Budgets: [0.05, 0.20, 1.00] (3 budgets)
- Folds: 2
- **Total Runs: 60 × 3 × 2 = 360 runs**

### Pruning Effects

**Hyperband Pruner:**
- Prunes ~60% of runs early
- **Successful Runs: 360 × 0.4 = 144 runs**

**Median Pruner:**
- Prunes ~40% of runs early
- **Successful Runs: 360 × 0.6 = 216 runs**

### Time Estimation

| Budget | Time per Run | Runs | Total Time |
|--------|-------------|------|------------|
| 5% | 30s | 120 | 1 hour |
| 20% | 2min | 120 | 4 hours |
| 100% | 10min | 120 | 20 hours |
| **Total** | | **360** | **25 hours** |

**With 4 parallel jobs: 25 ÷ 4 = 6.25 hours**

### Resource Requirements

**Memory:**
- 5% budget: ~2 GB
- 20% budget: ~4 GB
- 100% budget: ~8 GB

**CPU:**
- Recommended: 4-8 cores
- Parallel jobs: 2-4 (depending on cores)

**Storage:**
- Trial outputs: ~100 MB per trial
- Total: ~36 GB for 360 runs

## 🚀 **Complete Workflow**

### Step 1: Calculate Optimization Runs

```bash
# Run the calculator to understand resource requirements
python src/deduplication/optimization_calculator.py
```

**Output:**
```
🔢 Optimization Run Calculation
==================================================
📊 Configuration:
   Trials: 60
   Budgets: [0.05, 0.2, 1.0]
   Folds: 2
   Parallel jobs: 4
   Pruner: hyperband
   Sampler: nsga2

📈 Run Breakdown:
   Total trials: 60
   Total runs: 360
   Pruned runs: 216 (60.0%)
   Successful runs: 144

⏱️  Time Estimation:
   Total time: 6.3 hours
   Estimated cost: $12.60

💰 Budget Breakdown:
    5% budget: 120 runs
   20% budget: 120 runs
  100% budget: 120 runs
```

### Step 2: Run Complete Workflow

```bash
# Run the complete optimization workflow
python src/deduplication/complete_optimization_workflow.py \
  --sample-pairs data/intermediate/token_pairs_sample.parquet \
  --full-corpus data/intermediate/token_pairs.parquet \
  --output-dir organized_outputs/complete_workflow \
  --trials 60 \
  --n-jobs 4 \
  --sampler nsga2 \
  --pruner hyperband \
  --budgets 0.05,0.20,1.00 \
  --folds 2 \
  --min-coverage 10 \
  --max-edges 20000 \
  --top-k 5
```

**What this does:**
1. ✅ Calculates optimization runs
2. ✅ Runs advanced multi-objective tuning
3. ✅ Analyzes results with visualizations
4. ✅ Deploys top 5 trials to full corpus
5. ✅ Creates comprehensive summary

### Step 3: Manual Deployment (Alternative)

If you prefer manual control:

```bash
# 1. Run advanced tuning
python -m src.deduplication.param_tuner_advanced tune \
  --in-pairs data/intermediate/token_pairs_sample.parquet \
  --out-base organized_outputs/tuning \
  --trials 60 --n-jobs 4 \
  --study-name production_tuning \
  --storage sqlite:///tuning.db \
  --sampler nsga2 \
  --pruner hyperband \
  --budgets 0.05,0.20,1.00 \
  --min-coverage 10 --max-edges 20000 \
  --top-k 5

# 2. Analyze results
python -m src.deduplication.param_tuner_advanced analyze \
  --study-name production_tuning \
  --storage sqlite:///tuning.db \
  --out-dir organized_outputs/analysis

# 3. Deploy to full corpus
python src/deduplication/deploy_to_full_corpus.py \
  --best-params organized_outputs/tuning/pareto_full_results.csv \
  --full-corpus data/intermediate/token_pairs.parquet \
  --output-dir organized_outputs/production \
  --top-k 3
```

## 📊 **Results Analysis**

### Tuning Results

**Pareto Front Analysis:**
```python
import pandas as pd

# Load Pareto results
pareto = pd.read_csv('organized_outputs/tuning/pareto_trials.csv')
print("Pareto Front Solutions:")
print(pareto[['trial', 'good_clusters', 'coverage', 'edges']].head())

# Full data re-evaluation results
full_results = pd.read_csv('organized_outputs/tuning/pareto_full_results.csv')
print("\nFull Data Results:")
print(full_results[['rank', 'trial', 'good_clusters', 'coverage', 'edges']])
```

### Production Results

**Production Summary:**
```python
import json

# Load production summary
with open('organized_outputs/production/production_summary.json', 'r') as f:
    summary = json.load(f)

print("Production Results:")
print(f"Best trial: {summary['best_trial']['trial_id']}")
print(f"Good clusters: {summary['best_trial']['good_clusters']}")
print(f"Coverage: {summary['best_trial']['coverage']}")
print(f"Edges: {summary['best_trial']['edges']}")

# Best parameters
params = summary['best_trial']['parameters']
print("\nBest Parameters:")
for param, value in params.items():
    print(f"  {param}: {value}")
```

## 🎯 **Deployment Strategies**

### Strategy 1: Conservative (Recommended)

**Configuration:**
- Trials: 30-60
- Budgets: [0.1, 0.5, 1.0]
- Folds: 2
- Top K: 3-5

**Timeline:**
- Tuning: 3-6 hours
- Production: 1-2 hours
- **Total: 4-8 hours**

**Use Case:** Production systems, limited compute resources

### Strategy 2: Aggressive

**Configuration:**
- Trials: 100-200
- Budgets: [0.05, 0.20, 1.00]
- Folds: 3
- Top K: 10

**Timeline:**
- Tuning: 10-20 hours
- Production: 2-4 hours
- **Total: 12-24 hours**

**Use Case:** Research, maximum optimization, abundant compute resources

### Strategy 3: Quick Validation

**Configuration:**
- Trials: 20
- Budgets: [0.2, 1.0]
- Folds: 1
- Top K: 3

**Timeline:**
- Tuning: 1-2 hours
- Production: 1 hour
- **Total: 2-3 hours**

**Use Case:** Quick validation, parameter sensitivity analysis

## 📈 **Scaling to Full Corpus**

### Scaling Factors

**Time Scaling:**
```
Full Corpus Time = Sample Time × (Full Size / Sample Size)^1.2
```

**Example:**
- Sample size: 100,000 pairs
- Full corpus: 2,000,000 pairs
- Scale factor: 20x
- Time scaling: 20^1.2 = 31.6x

**Resource Scaling:**
- Memory: Scale factor × 2 GB
- CPU: Scale factor × 0.5 cores
- Storage: Scale factor × 0.02 GB

### Production Deployment Phases

**Phase 1: Validation (10% of time)**
- Re-run top Pareto solutions on full data
- Validate parameter performance
- Select final configuration

**Phase 2: Production (5% of time)**
- Apply best parameters to full corpus
- Generate final production results
- Create canonical mappings

## 🔧 **Configuration Examples**

### Example 1: Standard Production

```bash
python src/deduplication/complete_optimization_workflow.py \
  --sample-pairs data/intermediate/token_pairs_sample.parquet \
  --full-corpus data/intermediate/token_pairs.parquet \
  --output-dir organized_outputs/production_standard \
  --trials 60 \
  --n-jobs 4 \
  --sampler nsga2 \
  --pruner hyperband \
  --budgets 0.05,0.20,1.00 \
  --folds 2 \
  --min-coverage 20 \
  --max-edges 15000 \
  --top-k 5
```

**Expected Results:**
- Total runs: 360
- Successful runs: 144
- Time: 6-8 hours
- Best good clusters: 50-100
- Best coverage: 1000-5000
- Best edges: 5000-15000

### Example 2: High-Performance Research

```bash
python src/deduplication/complete_optimization_workflow.py \
  --sample-pairs data/intermediate/token_pairs_sample.parquet \
  --full-corpus data/intermediate/token_pairs.parquet \
  --output-dir organized_outputs/research_high_performance \
  --trials 100 \
  --n-jobs 8 \
  --sampler nsga2 \
  --pruner hyperband \
  --budgets 0.05,0.20,1.00 \
  --folds 3 \
  --min-coverage 50 \
  --max-edges 50000 \
  --top-k 10
```

**Expected Results:**
- Total runs: 900
- Successful runs: 360
- Time: 15-20 hours
- Best good clusters: 100-200
- Best coverage: 5000-10000
- Best edges: 10000-50000

### Example 3: Quick Validation

```bash
python src/deduplication/complete_optimization_workflow.py \
  --sample-pairs data/intermediate/token_pairs_sample.parquet \
  --full-corpus data/intermediate/token_pairs.parquet \
  --output-dir organized_outputs/quick_validation \
  --trials 20 \
  --n-jobs 2 \
  --sampler nsga2 \
  --pruner hyperband \
  --budgets 0.2,1.0 \
  --folds 1 \
  --min-coverage 5 \
  --max-edges 10000 \
  --top-k 3
```

**Expected Results:**
- Total runs: 40
- Successful runs: 16
- Time: 2-3 hours
- Best good clusters: 20-50
- Best coverage: 500-2000
- Best edges: 2000-10000

## 📋 **Checklist for Production Deployment**

### Pre-Deployment

- [ ] Calculate optimization runs and resource requirements
- [ ] Verify sample dataset is representative
- [ ] Check system resources (memory, CPU, storage)
- [ ] Set appropriate constraints (min coverage, max edges)
- [ ] Choose optimization strategy (conservative/aggressive/quick)

### During Optimization

- [ ] Monitor optimization progress
- [ ] Check for early convergence
- [ ] Validate intermediate results
- [ ] Ensure no system resource exhaustion
- [ ] Save intermediate results

### Post-Optimization

- [ ] Analyze Pareto front results
- [ ] Validate top solutions on full data
- [ ] Compare with baseline parameters
- [ ] Generate production summary
- [ ] Create canonical mappings
- [ ] Document best parameters

### Production Validation

- [ ] Verify production results quality
- [ ] Check coverage and efficiency metrics
- [ ] Validate canonical mappings
- [ ] Test downstream applications
- [ ] Monitor system performance
- [ ] Create rollback plan if needed

## 🚨 **Troubleshooting**

### Common Issues

**1. Insufficient Memory**
```
Error: MemoryError during optimization
Solution: Reduce sample size or increase system memory
```

**2. Timeout Errors**
```
Error: Trial timed out
Solution: Increase timeout or reduce budget complexity
```

**3. No Good Clusters**
```
Error: All trials produce 0 good clusters
Solution: Lower quality thresholds or increase base_threshold
```

**4. Too Many Edges**
```
Error: All trials exceed max_edges constraint
Solution: Increase k or enable mutual_nn filtering
```

### Performance Optimization

**1. Faster Tuning**
- Use smaller sample sizes (50k-100k pairs)
- Reduce number of folds (1-2)
- Use fewer budgets (2 instead of 3)
- Increase parallel jobs

**2. Better Results**
- Use more trials (100-200)
- Use more folds (3-4)
- Use more budgets (4-5)
- Use longer timeouts

**3. Resource Management**
- Monitor memory usage
- Use SSD storage for SQLite
- Enable parallel processing
- Clean up intermediate files

## 📚 **References**

- [Optuna Multi-Objective Optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#multi-objective-optimization)
- [NSGA-II Algorithm](https://ieeexplore.ieee.org/document/996017)
- [Hyperband Pruning](https://arxiv.org/abs/1603.06560)
- [Multi-Fidelity Optimization](https://arxiv.org/abs/1807.01838)

## 🎯 **Key Takeaways**

1. **Calculate first** - Understand resource requirements before starting
2. **Start conservative** - Use standard configuration for production
3. **Scale gradually** - Increase complexity as needed
4. **Validate thoroughly** - Always test on full data
5. **Monitor resources** - Watch memory, CPU, and storage usage
6. **Document results** - Keep detailed records of optimization runs
7. **Plan for rollback** - Have fallback parameters ready

This comprehensive guide ensures successful optimization and deployment of your deduplication pipeline to the full corpus.
