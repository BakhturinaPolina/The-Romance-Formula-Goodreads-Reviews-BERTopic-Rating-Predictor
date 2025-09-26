#!/usr/bin/env python3
"""
Optimization Run Calculator and Full Corpus Deployment Strategy

This script calculates the total number of optimization runs and provides
a comprehensive strategy for applying results to the full corpus.

Usage:
    python src/deduplication/optimization_calculator.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    trials: int
    budgets: List[float]
    folds: int
    n_jobs: int
    pruner: str
    sampler: str
    timeout_sec: int
    min_coverage: int
    max_edges: int
    top_k: int

@dataclass
class RunCalculation:
    """Results of optimization run calculations."""
    total_trials: int
    total_runs: int
    pruned_runs: int
    successful_runs: int
    estimated_time_hours: float
    estimated_cost: float
    budget_breakdown: Dict[float, int]
    fold_breakdown: Dict[int, int]

class OptimizationCalculator:
    """Calculate optimization runs and deployment strategy."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def calculate_runs(self) -> RunCalculation:
        """Calculate total number of optimization runs."""
        
        # Basic calculation: trials × budgets × folds
        total_runs = self.config.trials * len(self.config.budgets) * self.config.folds
        
        # Estimate pruning effects based on pruner type
        if self.config.pruner == "hyperband":
            # Hyperband typically prunes 50-70% of runs
            pruned_ratio = 0.6
        elif self.config.pruner == "median":
            # Median pruner typically prunes 30-50% of runs
            pruned_ratio = 0.4
        else:
            # No pruning
            pruned_ratio = 0.0
        
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
        
        # Calculate time breakdown by budget
        budget_breakdown = {}
        fold_breakdown = {}
        total_time_seconds = 0
        
        for budget in self.config.budgets:
            runs_at_budget = self.config.trials * self.config.folds
            budget_breakdown[budget] = runs_at_budget
            
            # Estimate time for this budget
            time_per_run_budget = time_per_run.get(budget, 300)  # Default 5 minutes
            time_at_budget = runs_at_budget * time_per_run_budget
            total_time_seconds += time_at_budget
        
        for fold in range(self.config.folds):
            runs_at_fold = self.config.trials * len(self.config.budgets)
            fold_breakdown[fold] = runs_at_fold
        
        # Account for parallelization
        if self.config.n_jobs > 1:
            total_time_seconds = total_time_seconds / self.config.n_jobs
        
        estimated_time_hours = total_time_seconds / 3600
        
        # Estimate cost (assuming cloud compute)
        cost_per_hour = 2.0  # $2/hour for compute
        estimated_cost = estimated_time_hours * cost_per_hour
        
        return RunCalculation(
            total_trials=self.config.trials,
            total_runs=total_runs,
            pruned_runs=pruned_runs,
            successful_runs=successful_runs,
            estimated_time_hours=estimated_time_hours,
            estimated_cost=estimated_cost,
            budget_breakdown=budget_breakdown,
            fold_breakdown=fold_breakdown
        )
    
    def print_calculation(self, calculation: RunCalculation):
        """Print detailed calculation results."""
        print("🔢 Optimization Run Calculation")
        print("=" * 50)
        print(f"📊 Configuration:")
        print(f"   Trials: {self.config.trials}")
        print(f"   Budgets: {self.config.budgets}")
        print(f"   Folds: {self.config.folds}")
        print(f"   Parallel jobs: {self.config.n_jobs}")
        print(f"   Pruner: {self.config.pruner}")
        print(f"   Sampler: {self.config.sampler}")
        print()
        
        print(f"📈 Run Breakdown:")
        print(f"   Total trials: {calculation.total_trials}")
        print(f"   Total runs: {calculation.total_runs:,}")
        print(f"   Pruned runs: {calculation.pruned_runs:,} ({calculation.pruned_runs/calculation.total_runs*100:.1f}%)")
        print(f"   Successful runs: {calculation.successful_runs:,}")
        print()
        
        print(f"⏱️  Time Estimation:")
        print(f"   Total time: {calculation.estimated_time_hours:.1f} hours")
        print(f"   Estimated cost: ${calculation.estimated_cost:.2f}")
        print()
        
        print(f"💰 Budget Breakdown:")
        for budget, runs in calculation.budget_breakdown.items():
            percentage = budget * 100
            print(f"   {percentage:4.0f}% budget: {runs:,} runs")
        print()
        
        print(f"🔄 Fold Breakdown:")
        for fold, runs in calculation.fold_breakdown.items():
            print(f"   Fold {fold}: {runs:,} runs")
        print()

def create_deployment_strategy(
    sample_size: int,
    full_corpus_size: int,
    best_params: Dict,
    calculation: RunCalculation
) -> Dict:
    """Create deployment strategy for full corpus."""
    
    # Calculate scaling factors
    scale_factor = full_corpus_size / sample_size
    time_scale_factor = scale_factor ** 1.2  # Slightly more than linear due to complexity
    
    # Estimate full corpus processing time
    sample_time_hours = calculation.estimated_time_hours
    full_corpus_time_hours = sample_time_hours * time_scale_factor
    
    # Estimate resource requirements
    memory_gb = max(8, int(scale_factor * 2))  # Scale memory requirements
    cpu_cores = max(4, int(scale_factor * 0.5))  # Scale CPU requirements
    
    # Create deployment phases
    phases = [
        {
            "phase": "Validation",
            "description": "Re-run top Pareto solutions on full data",
            "trials": calculation.total_trials // 10,  # Top 10%
            "time_hours": full_corpus_time_hours * 0.1,
            "purpose": "Validate best parameters on full corpus"
        },
        {
            "phase": "Production",
            "description": "Apply best parameters to full corpus",
            "trials": 1,
            "time_hours": full_corpus_time_hours * 0.05,
            "purpose": "Generate final production results"
        }
    ]
    
    return {
        "sample_size": sample_size,
        "full_corpus_size": full_corpus_size,
        "scale_factor": scale_factor,
        "time_scale_factor": time_scale_factor,
        "full_corpus_time_hours": full_corpus_time_hours,
        "resource_requirements": {
            "memory_gb": memory_gb,
            "cpu_cores": cpu_cores,
            "storage_gb": int(full_corpus_size * 0.001)  # Rough estimate
        },
        "phases": phases,
        "best_parameters": best_params,
        "total_deployment_time": sum(phase["time_hours"] for phase in phases)
    }

def print_deployment_strategy(strategy: Dict):
    """Print deployment strategy."""
    print("🚀 Full Corpus Deployment Strategy")
    print("=" * 50)
    print(f"📊 Corpus Information:")
    print(f"   Sample size: {strategy['sample_size']:,} pairs")
    print(f"   Full corpus size: {strategy['full_corpus_size']:,} pairs")
    print(f"   Scale factor: {strategy['scale_factor']:.1f}x")
    print()
    
    print(f"⏱️  Time Estimation:")
    print(f"   Full corpus time: {strategy['full_corpus_time_hours']:.1f} hours")
    print(f"   Total deployment time: {strategy['total_deployment_time']:.1f} hours")
    print()
    
    print(f"💻 Resource Requirements:")
    reqs = strategy['resource_requirements']
    print(f"   Memory: {reqs['memory_gb']} GB")
    print(f"   CPU cores: {reqs['cpu_cores']}")
    print(f"   Storage: {reqs['storage_gb']} GB")
    print()
    
    print(f"📋 Deployment Phases:")
    for i, phase in enumerate(strategy['phases'], 1):
        print(f"   {i}. {phase['phase']}: {phase['description']}")
        print(f"      Trials: {phase['trials']}")
        print(f"      Time: {phase['time_hours']:.1f} hours")
        print(f"      Purpose: {phase['purpose']}")
        print()
    
    print(f"🎯 Best Parameters:")
    for param, value in strategy['best_parameters'].items():
        print(f"   {param}: {value}")
    print()

def create_optimization_commands(
    config: OptimizationConfig,
    sample_pairs: Path,
    output_dir: Path,
    storage_db: Path
) -> List[str]:
    """Create optimization commands for different scenarios."""
    
    commands = []
    
    # Basic optimization
    basic_cmd = [
        "python -m src.deduplication.param_tuner_advanced tune",
        f"--in-pairs {sample_pairs}",
        f"--out-base {output_dir}",
        f"--trials {config.trials}",
        f"--n-jobs {config.n_jobs}",
        f"--study-name basic_optimization",
        f"--storage sqlite:///{storage_db}",
        f"--sampler {config.sampler}",
        f"--pruner {config.pruner}",
        f"--budgets {','.join(map(str, config.budgets))}",
        f"--folds {config.folds}",
        f"--min-coverage {config.min_coverage}",
        f"--max-edges {config.max_edges}",
        f"--timeout-sec {config.timeout_sec}",
        f"--top-k {config.top_k}"
    ]
    commands.append(" ".join(basic_cmd))
    
    # High-performance optimization
    hp_config = OptimizationConfig(
        trials=config.trials * 2,
        budgets=config.budgets,
        folds=config.folds,
        n_jobs=config.n_jobs * 2,
        pruner=config.pruner,
        sampler=config.sampler,
        timeout_sec=config.timeout_sec * 2,
        min_coverage=config.min_coverage,
        max_edges=config.max_edges,
        top_k=config.top_k
    )
    
    hp_cmd = [
        "python -m src.deduplication.param_tuner_advanced tune",
        f"--in-pairs {sample_pairs}",
        f"--out-base {output_dir}/high_performance",
        f"--trials {hp_config.trials}",
        f"--n-jobs {hp_config.n_jobs}",
        f"--study-name high_performance_optimization",
        f"--storage sqlite:///{storage_db}",
        f"--sampler {hp_config.sampler}",
        f"--pruner {hp_config.pruner}",
        f"--budgets {','.join(map(str, hp_config.budgets))}",
        f"--folds {hp_config.folds}",
        f"--min-coverage {hp_config.min_coverage}",
        f"--max-edges {hp_config.max_edges}",
        f"--timeout-sec {hp_config.timeout_sec}",
        f"--top-k {hp_config.top_k}"
    ]
    commands.append(" ".join(hp_cmd))
    
    # Production deployment
    prod_cmd = [
        "python -m src.deduplication.param_tuner_advanced tune",
        f"--in-pairs data/intermediate/token_pairs.parquet",  # Full corpus
        f"--out-base {output_dir}/production",
        f"--trials {config.top_k}",  # Only top K trials
        f"--n-jobs {config.n_jobs}",
        f"--study-name production_deployment",
        f"--storage sqlite:///{storage_db}",
        f"--sampler {config.sampler}",
        f"--pruner {config.pruner}",
        f"--budgets 1.0",  # Full data only
        f"--folds 1",  # Single fold for production
        f"--min-coverage {config.min_coverage}",
        f"--max-edges {config.max_edges}",
        f"--timeout-sec {config.timeout_sec * 3}",  # Longer timeout
        f"--top-k {config.top_k}"
    ]
    commands.append(" ".join(prod_cmd))
    
    return commands

def main():
    """Main function to demonstrate calculations."""
    print("🔢 Optimization Run Calculator and Deployment Strategy")
    print("=" * 60)
    print()
    
    # Example configurations
    configs = {
        "Basic": OptimizationConfig(
            trials=30,
            budgets=[0.1, 0.5, 1.0],
            folds=2,
            n_jobs=2,
            pruner="hyperband",
            sampler="nsga2",
            timeout_sec=600,
            min_coverage=10,
            max_edges=20000,
            top_k=5
        ),
        "Standard": OptimizationConfig(
            trials=60,
            budgets=[0.05, 0.20, 1.00],
            folds=2,
            n_jobs=4,
            pruner="hyperband",
            sampler="nsga2",
            timeout_sec=1800,
            min_coverage=20,
            max_edges=15000,
            top_k=5
        ),
        "High-Performance": OptimizationConfig(
            trials=100,
            budgets=[0.05, 0.20, 1.00],
            folds=3,
            n_jobs=8,
            pruner="hyperband",
            sampler="nsga2",
            timeout_sec=3600,
            min_coverage=50,
            max_edges=50000,
            top_k=10
        )
    }
    
    # Calculate for each configuration
    for name, config in configs.items():
        print(f"📊 {name} Configuration")
        print("-" * 30)
        
        calculator = OptimizationCalculator(config)
        calculation = calculator.calculate_runs()
        calculator.print_calculation(calculation)
        
        # Create deployment strategy
        sample_size = 100000  # Example sample size
        full_corpus_size = 2000000  # Example full corpus size
        best_params = {
            "base_threshold": 0.68,
            "k": 8,
            "max_degree": 120,
            "adaptive_short_thr": 0.80,
            "pair_min_sim": 0.90,
            "small_min_sim": 0.78,
            "big_min_sim": 0.72,
            "mutual_nn": True
        }
        
        strategy = create_deployment_strategy(
            sample_size, full_corpus_size, best_params, calculation
        )
        print_deployment_strategy(strategy)
        
        print("=" * 60)
        print()
    
    # Create example commands
    print("💻 Example Commands")
    print("=" * 30)
    
    sample_pairs = Path("data/intermediate/token_pairs_sample.parquet")
    output_dir = Path("organized_outputs/optimization")
    storage_db = Path("optimization.db")
    
    commands = create_optimization_commands(
        configs["Standard"], sample_pairs, output_dir, storage_db
    )
    
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")
        print()
    
    print("✅ Calculation complete!")

if __name__ == "__main__":
    main()
