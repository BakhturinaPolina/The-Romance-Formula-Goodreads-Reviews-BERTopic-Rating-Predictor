#!/usr/bin/env python3
"""
Demonstration of Bayesian Optimization (TPE) vs Random Search efficiency.

This script shows how TPE learns from previous trials and focuses on promising
regions, while random search explores uniformly without learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import optuna
from typing import Tuple
import pandas as pd

def objective_function(trial: optuna.Trial) -> float:
    """
    Synthetic objective function with multiple optima.
    Simulates the complexity of deduplication pipeline optimization.
    """
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    
    # Multiple peaks with different heights (simulating parameter interactions)
    peak1 = np.exp(-((x - 2)**2 + (y - 1)**2) / 2) * 0.8
    peak2 = np.exp(-((x + 1)**2 + (y + 2)**2) / 1.5) * 1.0  # Global optimum
    peak3 = np.exp(-((x - 1)**2 + (y - 3)**2) / 3) * 0.6
    
    # Add some noise to simulate real-world variability
    noise = np.random.normal(0, 0.05)
    
    return peak1 + peak2 + peak3 + noise

def run_random_search(n_trials: int = 100) -> Tuple[list, list]:
    """Run random search optimization."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler()
    )
    
    study.optimize(objective_function, n_trials=n_trials)
    
    # Extract trial results
    trials = []
    values = []
    for trial in study.trials:
        trials.append(trial.number)
        values.append(trial.value)
    
    return trials, values

def run_tpe_optimization(n_trials: int = 100) -> Tuple[list, list]:
    """Run TPE (Bayesian) optimization."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )
    
    study.optimize(objective_function, n_trials=n_trials)
    
    # Extract trial results
    trials = []
    values = []
    for trial in study.trials:
        trials.append(trial.number)
        values.append(trial.value)
    
    return trials, values

def plot_convergence_comparison():
    """Plot convergence comparison between Random Search and TPE."""
    print("🧪 Running optimization comparison...")
    print("This may take a moment...")
    
    # Run both optimizers
    random_trials, random_values = run_random_search(100)
    tpe_trials, tpe_values = run_tpe_optimization(100)
    
    # Calculate cumulative best values
    random_cummax = np.maximum.accumulate(random_values)
    tpe_cummax = np.maximum.accumulate(tpe_values)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Convergence comparison
    plt.subplot(2, 2, 1)
    plt.plot(random_trials, random_cummax, 'b-', label='Random Search', linewidth=2)
    plt.plot(tpe_trials, tpe_cummax, 'r-', label='TPE (Bayesian)', linewidth=2)
    plt.xlabel('Trial Number')
    plt.ylabel('Best Value Found')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Parameter space exploration (Random)
    plt.subplot(2, 2, 2)
    random_study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    random_study.optimize(objective_function, n_trials=50)
    
    x_vals = [trial.params['x'] for trial in random_study.trials]
    y_vals = [trial.params['y'] for trial in random_study.trials]
    colors = [trial.value for trial in random_study.trials]
    
    scatter = plt.scatter(x_vals, y_vals, c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Objective Value')
    plt.xlabel('Parameter X')
    plt.ylabel('Parameter Y')
    plt.title('Random Search: Uniform Exploration')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameter space exploration (TPE)
    plt.subplot(2, 2, 3)
    tpe_study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True))
    tpe_study.optimize(objective_function, n_trials=50)
    
    x_vals = [trial.params['x'] for trial in tpe_study.trials]
    y_vals = [trial.params['y'] for trial in tpe_study.trials]
    colors = [trial.value for trial in tpe_study.trials]
    
    scatter = plt.scatter(x_vals, y_vals, c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Objective Value')
    plt.xlabel('Parameter X')
    plt.ylabel('Parameter Y')
    plt.title('TPE: Focused on Promising Regions')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency comparison
    plt.subplot(2, 2, 4)
    trials_to_best = []
    for i in range(10, 101, 10):
        random_best = np.max(random_values[:i])
        tpe_best = np.max(tpe_values[:i])
        trials_to_best.append([i, random_best, tpe_best])
    
    df = pd.DataFrame(trials_to_best, columns=['Trials', 'Random', 'TPE'])
    plt.plot(df['Trials'], df['Random'], 'b-o', label='Random Search')
    plt.plot(df['Trials'], df['TPE'], 'r-s', label='TPE (Bayesian)')
    plt.xlabel('Number of Trials')
    plt.ylabel('Best Value Found')
    plt.title('Efficiency Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_vs_random_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📊 Optimization Results Summary:")
    print(f"Random Search - Best value: {max(random_values):.4f}")
    print(f"TPE (Bayesian) - Best value: {max(tpe_values):.4f}")
    print(f"Improvement: {((max(tpe_values) - max(random_values)) / max(random_values) * 100):.1f}%")
    
    # Convergence analysis
    tpe_converged_trials = np.where(tpe_cummax >= 0.95 * max(tpe_values))[0]
    random_converged_trials = np.where(random_cummax >= 0.95 * max(random_values))[0]
    
    if len(tpe_converged_trials) > 0 and len(random_converged_trials) > 0:
        print(f"\n⚡ Convergence to 95% of best value:")
        print(f"TPE: {tpe_converged_trials[0] + 1} trials")
        print(f"Random: {random_converged_trials[0] + 1} trials")
        print(f"Speedup: {random_converged_trials[0] / tpe_converged_trials[0]:.1f}x faster")

def main():
    """Run the Bayesian vs Random optimization demonstration."""
    print("🚀 Bayesian Optimization (TPE) vs Random Search Demonstration")
    print("=" * 70)
    print()
    print("This demo shows why TPE is superior to random search:")
    print("• TPE learns from previous trials and focuses on promising regions")
    print("• Random search explores uniformly without learning")
    print("• TPE typically finds better solutions in fewer trials")
    print()
    
    try:
        plot_convergence_comparison()
        print("\n✅ Demonstration complete!")
        print("📁 Saved plot: bayesian_vs_random_comparison.png")
        
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        print("Running text-only comparison...")
        
        # Text-only comparison
        print("\n🧪 Running text-only comparison...")
        random_trials, random_values = run_random_search(50)
        tpe_trials, tpe_values = run_tpe_optimization(50)
        
        print(f"\n📊 Results after 50 trials:")
        print(f"Random Search - Best: {max(random_values):.4f}")
        print(f"TPE (Bayesian) - Best: {max(tpe_values):.4f}")
        print(f"Improvement: {((max(tpe_values) - max(random_values)) / max(random_values) * 100):.1f}%")

if __name__ == "__main__":
    main()
