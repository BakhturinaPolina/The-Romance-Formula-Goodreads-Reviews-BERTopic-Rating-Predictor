#!/usr/bin/env python3
"""
Milestone A — Ingestion & Audit: Load and Audit Script
=====================================================

This script performs comprehensive data audit including:
1. Schema validation and missingness analysis
2. Heavy-tail diagnostics using Clauset-Shalizi-Newman (2009) workflow
3. Overdispersion tests for count variables
4. Edge case analysis and data quality checks

Outputs: audit_report.html with comprehensive analysis results

References:
- Clauset et al. (2009, SIAM Review)
- Alstott et al. (2014, powerlaw package)
- Dean & Lawless (1989), Cameron & Trivedi (1990)
"""

import os
import sys
import json
import math
import logging
import pathlib
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import statsmodels.api as sm

# Add project root to path for imports
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class DataAuditor:
    """Comprehensive data audit class implementing CSN workflow and statistical tests."""
    
    def __init__(self, data_path: str, output_dir: str = "./audit_outputs"):
        """
        Initialize the data auditor.
        
        Args:
            data_path: Path to the CSV file
            output_dir: Directory for output files
        """
        self.data_path = data_path
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected schema
        self.expected_cols = [
            'work_id', 'book_id_list_en', 'title', 'publication_year', 'num_pages_median',
            'description', 'language_codes_en', 'author_id', 'author_name', 
            'author_average_rating', 'author_ratings_count', 'series_id', 'series_title',
            'ratings_count_sum', 'text_reviews_count_sum', 'average_rating_weighted_mean',
            'genres_str', 'shelves_str', 'series_works_count_numeric'
        ]
        
        # ID columns to exclude from numerical analysis
        self.id_columns = ['work_id', 'book_id_list_en', 'author_id', 'series_id']
        
        # Count variables for heavy-tail analysis
        self.count_variables = ['ratings_count_sum', 'text_reviews_count_sum', 'author_ratings_count']
        
        self.df = None
        self.audit_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data validation."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def schema_audit(self) -> Dict[str, Any]:
        """Perform comprehensive schema validation."""
        logger.info("Performing schema audit...")
        
        present_cols = list(self.df.columns)
        missing_cols = [c for c in self.expected_cols if c not in present_cols]
        extra_cols = [c for c in present_cols if c not in self.expected_cols]
        schema_ok = len(missing_cols) == 0
        
        # Data type analysis
        dtype_analysis = {}
        for col in self.df.columns:
            dtype_analysis[col] = {
                'dtype': str(self.df[col].dtype),
                'non_null_count': int(self.df[col].count()),
                'null_count': int(self.df[col].isnull().sum()),
                'null_percentage': float(self.df[col].isnull().sum() / len(self.df) * 100),
                'unique_count': int(self.df[col].nunique()),
                'is_id_column': col in self.id_columns
            }
        
        schema_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'n_rows': int(len(self.df)),
            'n_cols': int(len(self.df.columns)),
            'present_cols': present_cols,
            'expected_cols': self.expected_cols,
            'missing_cols': missing_cols,
            'extra_cols': extra_cols,
            'schema_ok': schema_ok,
            'dtype_analysis': dtype_analysis
        }
        
        self.audit_results['schema'] = schema_results
        logger.info(f"Schema audit complete. Schema OK: {schema_ok}")
        return schema_results
    
    def missingness_analysis(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        logger.info("Performing missingness analysis...")
        
        missing_summary = self.df.isnull().sum().sort_values(ascending=False)
        missing_percentage = (missing_summary / len(self.df) * 100).round(2)
        
        # Identify columns with significant missingness
        significant_missing = missing_percentage[missing_percentage > 5].to_dict()
        
        # Pattern analysis for list-like columns
        list_columns = ['book_id_list_en', 'genres_str', 'shelves_str']
        list_analysis = {}
        
        for col in list_columns:
            if col in self.df.columns:
                non_null = self.df[col].dropna()
                if len(non_null) > 0:
                    # Check for empty strings
                    empty_strings = (non_null.astype(str).str.strip() == '').sum()
                    list_analysis[col] = {
                        'total_non_null': len(non_null),
                        'empty_strings': int(empty_strings),
                        'empty_percentage': float(empty_strings / len(non_null) * 100)
                    }
        
        missingness_results = {
            'missing_summary': missing_summary.to_dict(),
            'missing_percentage': missing_percentage.to_dict(),
            'significant_missing': significant_missing,
            'list_column_analysis': list_analysis
        }
        
        self.audit_results['missingness'] = missingness_results
        logger.info("Missingness analysis complete")
        return missingness_results
    
    def heavy_tail_analysis(self) -> Dict[str, Any]:
        """Perform heavy-tail analysis using CSN methodology."""
        logger.info("Performing heavy-tail analysis...")
        
        heavy_tail_results = {}
        
        for var in self.count_variables:
            if var not in self.df.columns:
                continue
                
            logger.info(f"Analyzing {var}...")
            x = self.df[var].values
            x = x[~np.isnan(x)]
            x = x[x >= 0]  # Remove negative values
            
            if len(x) == 0:
                continue
            
            # Basic statistics
            stats = {
                'count': len(x),
                'mean': float(np.mean(x)),
                'median': float(np.median(x)),
                'std': float(np.std(x)),
                'min': float(np.min(x)),
                'max': float(np.max(x)),
                'zeros': int((x == 0).sum()),
                'zero_percentage': float((x == 0).sum() / len(x) * 100)
            }
            
            # Power-law analysis (if powerlaw package available)
            powerlaw_results = None
            try:
                import powerlaw
                
                # Use only positive values for power-law fitting
                x_positive = x[x >= 1]
                if len(x_positive) >= 50:  # Minimum sample size
                    fit = powerlaw.Fit(x_positive, discrete=True, estimate_discrete=True, verbose=False)
                    pl = fit.power_law
                    
                    # Model comparisons
                    comparisons = {}
                    for alt in ['lognormal', 'exponential', 'truncated_power_law', 'stretched_exponential']:
                        try:
                            R, p = fit.distribution_compare('power_law', alt)
                            comparisons[alt] = {'LR': float(R), 'p': float(p)}
                        except:
                            comparisons[alt] = {'LR': None, 'p': None}
                    
                    powerlaw_results = {
                        'xmin': float(pl.xmin),
                        'alpha': float(pl.alpha),
                        'ks_statistic': float(fit.D),
                        'tail_fraction': float((x_positive >= pl.xmin).mean()),
                        'comparisons': comparisons
                    }
                    
            except ImportError:
                logger.warning("powerlaw package not available. Install with: pip install powerlaw")
            
            # Overdispersion test (variance vs mean)
            if stats['mean'] > 0:
                variance_mean_ratio = stats['std']**2 / stats['mean']
                is_overdispersed = variance_mean_ratio > 1.5
            else:
                variance_mean_ratio = np.inf
                is_overdispersed = True
            
            heavy_tail_results[var] = {
                'basic_stats': stats,
                'powerlaw_analysis': powerlaw_results,
                'overdispersion': {
                    'variance_mean_ratio': float(variance_mean_ratio),
                    'is_overdispersed': bool(is_overdispersed)
                }
            }
        
        self.audit_results['heavy_tails'] = heavy_tail_results
        logger.info("Heavy-tail analysis complete")
        return heavy_tail_results
    
    def overdispersion_tests(self, y: np.ndarray, X: np.ndarray = None) -> Dict[str, Any]:
        """
        Formal overdispersion diagnostics for Poisson:
          - Dean–Lawless Pearson chi-square z-test
          - Cameron–Trivedi auxiliary OLS test:
                Y* = ((y - mu)**2 - y) / mu  ~  mu   (no intercept)
        References:
          Dean & Lawless (1989, JASA); Cameron & Trivedi (1990, Journal of Econometrics).
        """
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y)
        y = y[mask]

        # Intercept-only Poisson unless X is provided
        if X is None:
            X = np.ones((len(y), 1))
        else:
            X = np.asarray(X, dtype=float)
            X = X[mask, :]

        # Fit Poisson GLM
        glm = sm.GLM(y, X, family=sm.families.Poisson())
        res = glm.fit()

        mu = np.asarray(res.fittedvalues, dtype=float)

        # --- Dean–Lawless z (Pearson chi-square) ---
        # Pearson chi2 = sum((y - mu)^2 / mu); under H0: E[chi2] ≈ df, Var ≈ 2*df
        pearson_chi2 = float(((y - mu) ** 2 / np.clip(mu, 1e-12, None)).sum())
        df = int(res.df_resid) if hasattr(res, "df_resid") else max(len(y) - X.shape[1], 1)
        # Normal approximation for z
        z_dl = (pearson_chi2 - df) / math.sqrt(2.0 * df) if df > 0 else float("nan")
        # two-sided p using error function (avoid hard SciPy dep)
        p_dl = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_dl) / math.sqrt(2.0))))) if np.isfinite(z_dl) else float("nan")

        dean_lawless = {
            "pearson_chi2": pearson_chi2,
            "df": df,
            "z": float(z_dl),
            "p_two_sided": p_dl,
        }

        # --- Cameron–Trivedi auxiliary OLS ---
        # Variance function test: Var(Y|X) = mu + alpha * mu^2
        # Regress Y* = ((y - mu)^2 - y) / mu  on mu  (no intercept)
        y_star = ((y - mu) ** 2 - y) / np.clip(mu, 1e-12, None)
        X_aux = mu.reshape(-1, 1)  # no intercept
        aux = sm.OLS(y_star, X_aux).fit()
        alpha_hat = float(aux.params[0])
        t_stat = float(aux.tvalues[0])
        p_val = float(aux.pvalues[0])

        cameron_trivedi = {
            "alpha_hat": alpha_hat,
            "t": t_stat,
            "p": p_val,
            "n": int(len(y)),
        }

        return {
            "dean_lawless": dean_lawless,
            "cameron_trivedi": cameron_trivedi,
            "mean": float(y.mean()),
            "variance": float(y.var(ddof=1)),
            "variance_mean_ratio": float(y.var(ddof=1) / y.mean() if y.mean() > 0 else float("inf")),
        }
    
    def formal_overdispersion_analysis(self) -> Dict[str, Any]:
        """Run formal overdispersion tests on count variables."""
        logger.info("Performing formal overdispersion analysis...")
        
        overdisp_summary = {}
        
        for var in self.count_variables:
            if var not in self.df.columns:
                continue
                
            logger.info(f"Running overdispersion tests for {var}...")
            y = self.df[var].to_numpy()
            
            # Run formal tests (intercept-only model)
            tests = self.overdispersion_tests(y, X=None)
            overdisp_summary[var] = tests
        
        # Save detailed results
        overdisp_path = self.output_dir / "overdispersion_tests.json"
        with open(overdisp_path, 'w', encoding='utf-8') as f:
            json.dump(overdisp_summary, f, indent=2)
        
        # Add compact summary to audit results
        self.audit_results["overdispersion"] = {
            var: {
                "dl_z": overdisp_summary[var]["dean_lawless"]["z"],
                "dl_p": overdisp_summary[var]["dean_lawless"]["p_two_sided"],
                "ct_t": overdisp_summary[var]["cameron_trivedi"]["t"],
                "ct_p": overdisp_summary[var]["cameron_trivedi"]["p"],
                "vmr": overdisp_summary[var]["variance_mean_ratio"],
                "is_overdispersed": (
                    overdisp_summary[var]["dean_lawless"]["p_two_sided"] < 0.05 or
                    overdisp_summary[var]["cameron_trivedi"]["p"] < 0.05
                )
            }
            for var in overdisp_summary
        }
        
        logger.info(f"Formal overdispersion analysis complete. Results saved to {overdisp_path}")
        return overdisp_summary
    
    def generate_visualizations(self) -> List[str]:
        """Generate visualization plots for the audit."""
        logger.info("Generating visualizations...")
        
        plot_files = []
        
        # 1. Missingness heatmap
        plt.figure(figsize=(12, 8))
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df) * 100).round(2)
        
        # Create a DataFrame for the heatmap
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        
        # Plot missingness
        plt.subplot(2, 1, 1)
        missing_df['Missing %'].plot(kind='bar', color='coral')
        plt.title('Missing Data by Column')
        plt.ylabel('Missing Percentage')
        plt.xticks(rotation=45, ha='right')
        
        # Plot data types
        plt.subplot(2, 1, 2)
        dtype_counts = self.df.dtypes.value_counts()
        dtype_counts.plot(kind='bar', color='skyblue')
        plt.title('Data Types Distribution')
        plt.ylabel('Number of Columns')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        missing_plot = self.output_dir / 'missingness_analysis.png'
        plt.savefig(missing_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(missing_plot))
        
        # 2. Heavy-tail visualizations
        for var in self.count_variables:
            if var not in self.df.columns:
                continue
                
            x = self.df[var].values
            x = x[~np.isnan(x)]
            x = x[x > 0]  # For log plots
            
            if len(x) == 0:
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # CCDF plot
            x_sorted = np.sort(x)
            y = 1.0 - np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax1.loglog(x_sorted, y, 'o', markersize=2, alpha=0.6)
            ax1.set_xlabel('Value')
            ax1.set_ylabel('CCDF P(X≥x)')
            ax1.set_title(f'{var} - Empirical CCDF (log-log)')
            ax1.grid(True, alpha=0.3)
            
            # Histogram with log bins
            bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
            ax2.hist(x, bins=bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel('Value (log)')
            ax2.set_ylabel('Density (log)')
            ax2.set_title(f'{var} - Log-binned Histogram')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            tail_plot = self.output_dir / f'heavy_tail_{var}.png'
            plt.savefig(tail_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(tail_plot))
        
        logger.info(f"Generated {len(plot_files)} visualization files")
        return plot_files
    
    def generate_html_report(self, plot_files: List[str]) -> str:
        """Generate comprehensive HTML audit report."""
        logger.info("Generating HTML audit report...")
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Audit Report - Romance Novel Dataset</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: #27ae60; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .plot { text-align: center; margin: 20px 0; }
        .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .summary-box { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Data Audit Report</h1>
        <h2>Romance Novel Dataset - Milestone A Analysis</h2>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Dataset:</strong> {{ data_path }}</p>
    </div>

    <div class="summary-box">
        <h3>📋 Executive Summary</h3>
        <div class="metric">
            <div class="metric-value">{{ format_number(schema.n_rows) }}</div>
            <div>Total Rows</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ schema.n_cols }}</div>
            <div>Total Columns</div>
        </div>
        <div class="metric">
            <div class="metric-value {% if schema.schema_ok %}success{% else %}error{% endif %}">
                {% if schema.schema_ok %}✅ PASS{% else %}❌ FAIL{% endif %}
            </div>
            <div>Schema Validation</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ missing_cols_count }}</div>
            <div>Missing Columns</div>
        </div>
    </div>

    <div class="section">
        <h3>🔍 Schema Validation</h3>
        {% if schema.schema_ok %}
            <p class="success">✅ All expected columns are present</p>
        {% else %}
            <p class="error">❌ Schema validation failed</p>
            {% if schema.missing_cols %}
                <h4>Missing Columns:</h4>
                <ul>
                {% for col in schema.missing_cols %}
                    <li class="error">{{ col }}</li>
                {% endfor %}
                </ul>
            {% endif %}
        {% endif %}
        
        {% if schema.extra_cols %}
            <h4>Extra Columns ({{ schema.extra_cols | length }}):</h4>
            <ul>
            {% for col in schema.extra_cols %}
                <li>{{ col }}</li>
            {% endfor %}
            </ul>
        {% endif %}
    </div>

    <div class="section">
        <h3>📈 Data Quality Analysis</h3>
        <h4>Missing Data Summary</h4>
        <table>
            <tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Data Type</th></tr>
            {% for col, info in schema.dtype_analysis.items() %}
            <tr>
                <td>{{ col }}</td>
                <td>{{ format_number(info.null_count) }}</td>
                <td>{{ "%.2f" | format(info.null_percentage) }}%</td>
                <td>{{ info.dtype }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h3>📊 Heavy-Tail Analysis</h3>
        <p><em>Using Clauset-Shalizi-Newman (2009) methodology</em></p>
        <p><em>Note: Log-binned histograms are visualization only; parameters estimated via MLE/KS/LLR.</em></p>
        
        {% for var, analysis in heavy_tails.items() %}
        <h4>{{ var }}</h4>
        <div class="summary-box">
            <div class="metric">
                <div class="metric-value">{{ "%.1f" | format(analysis.basic_stats.mean) }}</div>
                <div>Mean</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f" | format(analysis.basic_stats.median) }}</div>
                <div>Median</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f" | format(analysis.overdispersion.variance_mean_ratio) }}</div>
                <div>Var/Mean Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value {% if analysis.overdispersion.is_overdispersed %}warning{% else %}success{% endif %}">
                    {% if analysis.overdispersion.is_overdispersed %}⚠️ OVERDISPERSED{% else %}✅ NORMAL{% endif %}
                </div>
                <div>Dispersion</div>
            </div>
        </div>
        
        {% if analysis.powerlaw_analysis %}
        <h5>Power-Law Analysis</h5>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>xmin</td><td>{{ "%.0f" | format(analysis.powerlaw_analysis.xmin) }}</td></tr>
            <tr><td>α (alpha)</td><td>{{ "%.3f" | format(analysis.powerlaw_analysis.alpha) }}</td></tr>
            <tr><td>KS Statistic</td><td>{{ "%.4f" | format(analysis.powerlaw_analysis.ks_statistic) }}</td></tr>
            <tr><td>Tail Fraction</td><td>{{ "%.1f" | format(analysis.powerlaw_analysis.tail_fraction * 100) }}%</td></tr>
        </table>
        {% endif %}
        {% endfor %}
    </div>

    <div class="section">
        <h3>🔬 Formal Overdispersion Tests</h3>
        <p><em>Dean–Lawless (1989) and Cameron–Trivedi (1990) methodology</em></p>
        
        {% for var, results in overdispersion.items() %}
        <h4>{{ var }}</h4>
        <div class="summary-box">
            <div class="metric">
                <div class="metric-value">{{ "%.1f" | format(results.vmr) }}</div>
                <div>Var/Mean Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value {% if results.is_overdispersed %}error{% else %}success{% endif %}">
                    {% if results.is_overdispersed %}⚠️ OVERDISPERSED{% else %}✅ EQUIDISPERSED{% endif %}
                </div>
                <div>Poisson Test</div>
            </div>
        </div>
        
        <table>
            <tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Interpretation</th></tr>
            <tr>
                <td>Dean–Lawless (Pearson χ²)</td>
                <td>z = {{ "%.3f" | format(results.dl_z) }}</td>
                <td>{{ "%.4f" | format(results.dl_p) }}</td>
                <td>{% if results.dl_p < 0.05 %}<span class="error">Reject H₀ (overdispersed)</span>{% else %}<span class="success">Fail to reject H₀</span>{% endif %}</td>
            </tr>
            <tr>
                <td>Cameron–Trivedi (Auxiliary OLS)</td>
                <td>t = {{ "%.3f" | format(results.ct_t) }}</td>
                <td>{{ "%.4f" | format(results.ct_p) }}</td>
                <td>{% if results.ct_p < 0.05 %}<span class="error">Reject H₀ (overdispersed)</span>{% else %}<span class="success">Fail to reject H₀</span>{% endif %}</td>
            </tr>
        </table>
        {% endfor %}
    </div>

    <div class="section">
        <h3>📊 Visualizations</h3>
        {% for plot_file in plot_files %}
        <div class="plot">
            <img src="{{ basename(plot_file) }}" alt="Analysis Plot">
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h3>🎯 Recommendations</h3>
        <ul>
            <li><strong>Schema:</strong> {% if schema.schema_ok %}All expected columns present.{% else %}Address missing columns before proceeding.{% endif %}</li>
            <li><strong>Missing Data:</strong> Review columns with >5% missingness for imputation strategies.</li>
            <li><strong>Heavy Tails:</strong> Consider power-law aware modeling approaches for count variables.</li>
            <li><strong>Overdispersion:</strong> 
                {% if overdispersion %}
                    {% for var, results in overdispersion.items() %}
                        {% if results.is_overdispersed %}
                            <strong>{{ var }}:</strong> Reject Poisson assumption (p < 0.05). Use Negative Binomial or Zero-Inflated models.
                        {% else %}
                            <strong>{{ var }}:</strong> Poisson assumption acceptable.
                        {% endif %}
                    {% endfor %}
                {% else %}
                    Use Negative Binomial models instead of Poisson for count regressions if overdispersion detected.
                {% endif %}
            </li>
        </ul>
    </div>

    <div class="section">
        <h3>📚 References</h3>
        <ul>
            <li>Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law distributions in empirical data. SIAM review, 51(4), 661-703.</li>
            <li>Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: a Python package for analysis of heavy-tailed distributions. PloS one, 9(1), e85777.</li>
            <li>Dean, C., & Lawless, J. F. (1989). Tests for detecting overdispersion in Poisson regression models. Journal of the American Statistical Association, 84(406), 467-472.</li>
        </ul>
    </div>
</body>
</html>
        """
        
        # Prepare template data
        template_data = {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'data_path': self.data_path,
            'schema': self.audit_results['schema'],
            'missingness': self.audit_results['missingness'],
            'heavy_tails': self.audit_results['heavy_tails'],
            'overdispersion': self.audit_results.get('overdispersion', {}),
            'plot_files': plot_files,
            'missing_cols_count': len(self.audit_results['schema']['missing_cols']),
            'format_number': lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x),
            'basename': lambda x: os.path.basename(x)
        }
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save HTML report
        report_path = self.output_dir / 'audit_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {report_path}")
        return str(report_path)
    
    def save_audit_data(self) -> str:
        """Save audit results as JSON for programmatic access."""
        audit_data_path = self.output_dir / 'audit_results.json'
        with open(audit_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        
        logger.info(f"Audit data saved to {audit_data_path}")
        return str(audit_data_path)
    
    def run_full_audit(self) -> Dict[str, str]:
        """Run the complete audit pipeline."""
        logger.info("Starting comprehensive data audit...")
        
        # Load data
        self.load_data()
        
        # Run audit components
        self.schema_audit()
        self.missingness_analysis()
        self.heavy_tail_analysis()
        self.formal_overdispersion_analysis()
        
        # Generate visualizations
        plot_files = self.generate_visualizations()
        
        # Generate reports
        html_report = self.generate_html_report(plot_files)
        json_data = self.save_audit_data()
        
        logger.info("Audit complete!")
        
        return {
            'html_report': html_report,
            'json_data': json_data,
            'plot_files': plot_files
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Data Audit Script for Romance Novel Dataset')
    parser.add_argument('--data-path', type=str, 
                       default='../../data/processed/romance_books_main_final.csv',
                       help='Path to the CSV data file')
    parser.add_argument('--output-dir', type=str, default='./audit_outputs',
                       help='Output directory for audit results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--verbose-logs', action='store_true',
                       help='Enable verbose DEBUG logs from matplotlib et al.')
    
    args = parser.parse_args()
    
    # Silence very chatty matplotlib DEBUG logs unless explicitly requested
    if not args.verbose_logs:
        for name in ("matplotlib", "matplotlib.ticker"):
            logging.getLogger(name).setLevel(logging.WARNING)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run audit
    auditor = DataAuditor(args.data_path, args.output_dir)
    results = auditor.run_full_audit()
    
    print("\n" + "="*60)
    print("🎉 AUDIT COMPLETE!")
    print("="*60)
    print(f"📊 HTML Report: {results['html_report']}")
    print(f"📄 JSON Data: {results['json_data']}")
    print(f"🖼️  Plots: {len(results['plot_files'])} files generated")
    print("="*60)


if __name__ == "__main__":
    main()
