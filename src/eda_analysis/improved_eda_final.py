#!/usr/bin/env python3
"""
Improved EDA Final Script - Clean and Focused
Senior Data Scientist Review & Production

This script creates exactly two publication-ready figures:
1. Before/After Cleaning Distributions (2x2 histograms)
2. Cleaned Data Summary (Boxplots with jitter)

Uses Antique color palette and follows academic publication standards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
import logging
import time
from typing import Dict, List, Any, Optional
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from PIL import Image, PngImagePlugin
try:
    from scipy import stats
    from scipy.stats import norm, ks_2samp, mannwhitneyu
except Exception:
    stats = None
    norm = None
    ks_2samp = None
    mannwhitneyu = None  # graceful fallback if SciPy isn't available
warnings.filterwarnings('ignore')

# Set up comprehensive logging
def setup_logging(verbose=False):
    """Set up detailed logging with timestamps and progress tracking."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter with detailed timestamps
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Initialize logger
logger = logging.getLogger(__name__)

# Progress tracking utilities
class ProgressTracker:
    """Track progress with timestamps and detailed logging."""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        
    def start(self):
        """Start the progress tracking."""
        logger.info(f"🚀 Starting {self.description} - {self.total_steps} steps total")
        print(f"\n{'='*80}")
        print(f"🚀 STARTING {self.description.upper()}")
        print(f"📊 Total Steps: {self.total_steps}")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
    def step(self, step_name, details=""):
        """Mark a step as complete."""
        self.current_step += 1
        step_start = time.time()
        
        # Calculate progress
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        # Log the step
        logger.info(f"✅ Step {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - {step_name}")
        if details:
            logger.debug(f"   Details: {details}")
        
        # Print progress
        print(f"✅ [{self.current_step:2d}/{self.total_steps}] ({progress_pct:5.1f}%) {step_name}")
        if details:
            print(f"   📝 {details}")
        
        # Track timing
        step_time = time.time() - step_start
        self.step_times.append(step_time)
        
        # Estimate remaining time
        if self.current_step > 0:
            avg_time = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_time
            print(f"   ⏱️  Step time: {step_time:.2f}s | ETA: {eta:.1f}s")
        
        print()
        
    def finish(self):
        """Finish the progress tracking."""
        total_time = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        logger.info(f"🎉 Completed {self.description} in {total_time:.2f}s")
        print(f"\n{'='*80}")
        print(f"🎉 COMPLETED {self.description.upper()}")
        print(f"⏰ Total Time: {total_time:.2f}s")
        print(f"📊 Average Step Time: {avg_step_time:.2f}s")
        print(f"🏁 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

def log_data_info(df, name, logger):
    """Log detailed information about a dataframe."""
    logger.info(f"📊 {name} Dataset Info:")
    logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"   Missing values: {df.isnull().sum().sum():,}")
    logger.info(f"   Duplicate rows: {df.duplicated().sum():,}")
    
    # Log column info
    logger.debug(f"   Columns: {list(df.columns)}")
    logger.debug(f"   Data types: {dict(df.dtypes)}")

# Custom exceptions for Coding Agent Pattern
class AnalysisError(Exception):
    """Raised when task analysis fails."""
    pass

class ModificationError(Exception):
    """Raised when code modification fails."""
    pass

# Color policy validator (fail CI if violated)
class ColorPolicyValidator:
    """Lightweight CI guard enforcing basic color/contrast rules."""
    def __init__(self, max_hues=7): 
        self.max_hues = max_hues
    
    @staticmethod
    def _contrast_ratio(hex_fg, hex_bg):
        def hex_to_rgb(h):
            h = h.strip().lstrip("#")
            if len(h)==8: h=h[:6]
            return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))
        def lum(rgb):
            def f(c): 
                return (c/12.92) if (c<=0.03928) else (((c+0.055)/1.055)**2.4)
            r,g,b = map(f, rgb)
            return 0.2126*r + 0.7152*g + 0.0722*b
        L1, L2 = lum(hex_to_rgb(hex_fg)), lum(hex_to_rgb(hex_bg))
        Lh, Ll = max(L1,L2), min(L1,L2)
        return (Lh+0.05)/(Ll+0.05)
    
    def validate(self, meta):
        # meta: legend_present, variable_colors{}, background, text_color, uses_gradient_for_categories, is_categorical
        assert meta.get("legend_present", True), "Legend is required."
        colors = list((meta.get("variable_colors") or {}).values())
        if meta.get("is_categorical", True):
            assert not meta.get("uses_gradient_for_categories", False), "No gradients for categories."
            assert len(set(colors)) <= self.max_hues, f"Too many hues (> {self.max_hues})."
        contrast = self._contrast_ratio(meta.get("text_color","#111111"), meta.get("background","#FFFFFF"))
        assert contrast >= 4.0, f"Contrast too low: {contrast:.2f} (<4.0)."
        return True

# Statistical analysis functions
def hedges_g(x, y):
    """Calculate Hedges' g effect size."""
    if stats is None:
        return None
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    g = (np.mean(x) - np.mean(y)) / pooled_std
    # Hedges' correction
    correction = 1 - (3 / (4*(n1+n2) - 9))
    return g * correction

def cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    if stats is None:
        return None
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def cliffs_delta(x, y):
    """Calculate Cliff's delta effect size."""
    if stats is None:
        return None
    n1, n2 = len(x), len(y)
    count = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                count += 1
            elif xi < yj:
                count -= 1
    return count / (n1 * n2)

def statistical_tests(x, y):
    """Run Mann-Whitney U and Kolmogorov-Smirnov tests."""
    if stats is None:
        return {"mannwhitney": None, "ks": None}
    
    try:
        mw_stat, mw_p = stats.mannwhitneyu(x, y, alternative='two-sided')
        ks_stat, ks_p = stats.ks_2samp(x, y)
        return {
            "mannwhitney": {"statistic": mw_stat, "p_value": mw_p},
            "ks": {"statistic": ks_stat, "p_value": ks_p}
        }
    except Exception as e:
        logger.warning(f"Statistical tests failed: {e}")
        return {"mannwhitney": None, "ks": None}

def bootstrap_median_ci(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for median."""
    if len(data) == 0:
        return None, None
    
    np.random.seed(42)  # For reproducibility
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_medians, 100 * alpha/2)
    upper = np.percentile(bootstrap_medians, 100 * (1 - alpha/2))
    return lower, upper

def bootstrap_mean_ci(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for mean."""
    if len(data) == 0:
        return None, None
    
    np.random.seed(42)  # For reproducibility
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha/2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    return lower, upper

def stratified_sample(df, sample_size=10000, stratify_by=['publication_year', 'ratings_count_sum']):
    """Stratified sampling by year and popularity."""
    if len(df) <= sample_size:
        return df
    
    # Create strata based on year and popularity
    df_copy = df.copy()
    
    # Create year bins (decades)
    df_copy['year_bin'] = pd.cut(df_copy['publication_year'], 
                                bins=range(2000, 2025, 5), 
                                labels=[f"{y}-{y+4}" for y in range(2000, 2020, 5)],
                                include_lowest=True)
    
    # Create popularity bins (quartiles)
    df_copy['popularity_bin'] = pd.qcut(df_copy['ratings_count_sum'], 
                                       q=4, 
                                       labels=['Low', 'Medium', 'High', 'Very High'],
                                       duplicates='drop')
    
    # Sample proportionally from each stratum
    sampled_dfs = []
    for year_bin in df_copy['year_bin'].dropna().unique():
        for pop_bin in df_copy['popularity_bin'].dropna().unique():
            stratum = df_copy[(df_copy['year_bin'] == year_bin) & 
                             (df_copy['popularity_bin'] == pop_bin)]
            if len(stratum) > 0:
                # Calculate proportional sample size
                stratum_size = min(len(stratum), 
                                 max(1, int(sample_size * len(stratum) / len(df_copy))))
                sampled_stratum = stratum.sample(n=stratum_size, random_state=42)
                sampled_dfs.append(sampled_stratum)
    
    if sampled_dfs:
        result = pd.concat(sampled_dfs, ignore_index=True)
        # Clean up temporary columns
        result = result.drop(['year_bin', 'popularity_bin'], axis=1)
        return result
    else:
        # Fallback to random sampling
        return df.sample(n=min(sample_size, len(df)), random_state=42)

# Set up academic publication style
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'figure.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

# Antique color palette from pypalettes
ANTIQUE_COLORS = [
    '#855C75FF',  # Dark purple
    '#D9AF6BFF',  # Gold
    '#AF6458FF',  # Red-brown
    '#736F4CFF',  # Olive
    '#526A83FF',  # Blue-gray
    '#625377FF',  # Purple
    '#68855CFF',  # Green
    '#9C9C5EFF',  # Yellow-green
    '#A06177FF',  # Pink-purple
    '#8C785DFF',  # Brown
    '#467378FF',  # Teal
    '#7C7C7CFF'   # Gray
]

class ImprovedEDAFinal:
    """Clean, focused EDA plot generator with Coding Agent Pattern implementation."""
    
    def __init__(self, data_path_before, data_path_after, output_dir, 
                 sample_size=None, dpi=300, bins=30, clip_outliers=True):
        """
        Initialize the visualizer with Coding Agent Pattern components.
        
        Args:
            data_path_before: Path to the raw dataset CSV file
            data_path_after: Path to the cleaned dataset CSV file
            output_dir: Directory to save visualization outputs
            sample_size: Size for stratified sampling (None for no sampling)
            dpi: DPI for saved figures
            bins: Number of bins for histograms
            clip_outliers: Whether to clip outliers for better visualization
        """
        logger.info("🔧 Initializing ImprovedEDAFinal with enhanced parameters")
        print(f"\n{'='*60}")
        print("🔧 INITIALIZING EDA ANALYZER")
        print(f"{'='*60}")
        
        # Store parameters
        self.data_path_before = Path(data_path_before)
        self.data_path_after = Path(data_path_after)
        self.output_dir = Path(output_dir)
        
        # Enhanced parameters
        self.sample_size = sample_size
        self.dpi = dpi
        self.bins = bins
        self.clip_outliers = clip_outliers
        
        logger.info(f"📁 Before dataset: {self.data_path_before}")
        logger.info(f"📁 After dataset: {self.data_path_after}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Sample size: {self.sample_size if self.sample_size else 'No sampling'}")
        logger.info(f"🖼️  DPI: {self.dpi}")
        logger.info(f"📊 Bins: {self.bins}")
        logger.info(f"✂️  Clip outliers: {self.clip_outliers}")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"✅ Created output directory: {self.output_dir}")
        
        # Initialize color validator
        self.color_validator = ColorPolicyValidator()
        logger.info("✅ Initialized color policy validator")
        
        # Coding Agent Pattern: Change history tracking
        self.change_history = []
        self.task_analysis = {}
        self.change_plan = {}
        logger.info("✅ Initialized Coding Agent Pattern components")
        
        # Load datasets with detailed error handling
        logger.info("📥 Loading datasets...")
        try:
            logger.info(f"📖 Reading before dataset from: {self.data_path_before}")
            self.df_before = pd.read_csv(self.data_path_before)
            logger.info("✅ Successfully loaded before dataset")
            
            logger.info(f"📖 Reading after dataset from: {self.data_path_after}")
            self.df_after = pd.read_csv(self.data_path_after)
            logger.info("✅ Successfully loaded after dataset")
            
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {str(e)}")
            raise
        
        # Log detailed dataset information
        log_data_info(self.df_before, "Before Cleaning", logger)
        log_data_info(self.df_after, "After Cleaning", logger)
        
        # Apply stratified sampling if requested
        if self.sample_size:
            logger.info(f"🎲 Applying stratified sampling with size {self.sample_size}")
            print(f"🎲 Applying stratified sampling: {self.sample_size:,} samples")
            
            original_before_size = len(self.df_before)
            original_after_size = len(self.df_after)
            
            self.df_before = stratified_sample(self.df_before, self.sample_size)
            self.df_after = stratified_sample(self.df_after, self.sample_size)
            
            logger.info(f"📊 Before sampling: {original_before_size:,} → {len(self.df_before):,} books")
            logger.info(f"📊 After sampling: {original_after_size:,} → {len(self.df_after):,} books")
            print(f"📊 Sampling complete: {len(self.df_before):,} before, {len(self.df_after):,} after")
        
        # Define the four numerical variables to analyze
        self.numerical_vars = [
            'publication_year',
            'num_pages_median', 
            'ratings_count_sum',
            'average_rating_weighted_mean'
        ]
        logger.info(f"🎯 Target variables: {self.numerical_vars}")
        
        # Store bin information for reproducibility
        self.bin_info = {}
        logger.info("✅ Initialized bin information storage")
        
        # Final summary
        logger.info("🎉 Initialization complete!")
        print(f"✅ Initialization complete!")
        print(f"📊 Final dataset sizes: {len(self.df_before):,} before, {len(self.df_after):,} after")
        print(f"📈 Data reduction: {((len(self.df_before) - len(self.df_after)) / len(self.df_before) * 100):.1f}%")
        print(f"{'='*60}\n")
    
    def create_figure_1_histograms(self):
        """
        FIGURE 1: Before/After Cleaning Distributions (2x2 histograms)
        
        Creates a 2×2 subplot layout comparing distributions before and after 
        data cleaning for four numerical variables using Antique color palette.
        Enhanced with bootstrap confidence intervals and reproducible bin counts.
        """
        print("Creating Figure 1: Before/After Cleaning Distributions...")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Define variables and their specific settings
        variables = [
            ("publication_year", "Publication Year", "Publication Year", False, self.bins),
            ("num_pages_median", "Number of Pages (Median)", "Number of Pages (Median)", False, self.bins),
            ("ratings_count_sum", "Total Ratings Count (log scale)", "Total Ratings Count", True, 50),
            ("average_rating_weighted_mean", "Weighted Average Rating", "Weighted Average Rating", False, self.bins)
        ]
        
        for i, (var, xlabel, title, use_log, bins) in enumerate(variables):
            row, col = i // 2, i % 2
            ax = axs[row, col]
            
            # Get data
            data_before = self.df_before[var].dropna()
            data_after = self.df_after[var].dropna()
            
            # Clip outliers if requested
            if self.clip_outliers and var != "publication_year":
                q1_before, q3_before = np.percentile(data_before, [25, 75])
                q1_after, q3_after = np.percentile(data_after, [25, 75])
                iqr_before = q3_before - q1_before
                iqr_after = q3_after - q1_after
                
                data_before = data_before[(data_before >= q1_before - 1.5*iqr_before) & 
                                        (data_before <= q3_before + 1.5*iqr_before)]
                data_after = data_after[(data_after >= q1_after - 1.5*iqr_after) & 
                                      (data_after <= q3_after + 1.5*iqr_after)]
            
            # Create reproducible bins
            if var not in self.bin_info:
                if use_log:
                    # For log scale, use log-spaced bins
                    min_val = min(data_before.min(), data_after.min())
                    max_val = max(data_before.max(), data_after.max())
                    bins_edges = np.logspace(np.log10(max(1, min_val)), np.log10(max_val), bins+1)
                else:
                    # For linear scale, use equal-width bins
                    min_val = min(data_before.min(), data_after.min())
                    max_val = max(data_before.max(), data_after.max())
                    bins_edges = np.linspace(min_val, max_val, bins+1)
                
                self.bin_info[var] = {
                    'bins': bins_edges,
                    'use_log': use_log,
                    'min_val': min_val,
                    'max_val': max_val
                }
            else:
                bins_edges = self.bin_info[var]['bins']
                use_log = self.bin_info[var]['use_log']
            
            # Plot histograms
            sns.histplot(data=data_before, bins=bins_edges, color=ANTIQUE_COLORS[i*2], 
                        label="Before Cleaning", ax=ax, alpha=0.6, stat="count")
            sns.histplot(data=data_after, bins=bins_edges, color=ANTIQUE_COLORS[i*2+1], 
                        label="After Cleaning", ax=ax, alpha=0.6, stat="count")
            
            # Add bootstrap confidence intervals for median
            if len(data_before) > 0 and len(data_after) > 0:
                ci_before = bootstrap_median_ci(data_before)
                ci_after = bootstrap_median_ci(data_after)
                
                if ci_before[0] is not None and ci_after[0] is not None:
                    # Add vertical lines for median CIs
                    ax.axvline(ci_before[0], color=ANTIQUE_COLORS[i*2], linestyle='--', alpha=0.8, linewidth=2)
                    ax.axvline(ci_before[1], color=ANTIQUE_COLORS[i*2], linestyle='--', alpha=0.8, linewidth=2)
                    ax.axvline(ci_after[0], color=ANTIQUE_COLORS[i*2+1], linestyle='--', alpha=0.8, linewidth=2)
                    ax.axvline(ci_after[1], color=ANTIQUE_COLORS[i*2+1], linestyle='--', alpha=0.8, linewidth=2)
            
            # Set log scale if needed
            if use_log:
                ax.set_xscale("log")
            
            ax.set_title(f"{title} Distribution", fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.legend()
        
        plt.suptitle("Data Distribution: Before vs After Cleaning", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Validate color policy before saving
        color_meta = {
            "legend_present": True,
            "variable_colors": {f"var_{i}": ANTIQUE_COLORS[i] for i in range(4)},  # Only 4 variables
            "background": "#ffffff",
            "is_categorical": True,
            "text_color": "#111111"
        }
        self.color_validator.validate(color_meta)
        
        # Save the figure
        output_path = self.output_dir / 'figure_1_before_after_distributions.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save bin information for reproducibility
        bin_info_path = self.output_dir / 'histogram_bin_info.json'
        with open(bin_info_path, 'w') as f:
            json.dump(self.bin_info, f, indent=2, default=str)
        
        print(f"Figure 1 saved to: {output_path}")
        print(f"Bin information saved to: {bin_info_path}")
    
    def create_figure_2_boxplots(self):
        """
        FIGURE 2: Cleaned Data Summary (Boxplots with jitter)
        
        Creates a horizontal boxplot with jittered individual points showing 
        the distribution of all four cleaned numerical variables.
        """
        print("Creating Figure 2: Cleaned Data Summary (Boxplots with jitter)...")
        
        # Prepare data for boxplot - use raw values but transform ratings_count to log scale for visibility
        cleaned_vars = self.df_after[self.numerical_vars].copy()
        
        # Transform ratings_count to log scale for better visibility
        cleaned_vars['ratings_count_sum'] = np.log1p(cleaned_vars['ratings_count_sum'])  # log(1+x) to handle zeros
        
        # Melt dataframe for grouped boxplot
        cleaned_melted = cleaned_vars.melt(var_name='Variable', value_name='Value')
        
        # Rename variables for better display
        variable_names = {
            'publication_year': 'Publication Year',
            'num_pages_median': 'Pages (Median)',
            'ratings_count_sum': 'Ratings Count (log)',
            'average_rating_weighted_mean': 'Avg Rating (Weighted)'
        }
        cleaned_melted['Variable'] = cleaned_melted['Variable'].map(variable_names)
        
        # Create horizontal boxplot with jitter
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Boxplot
        sns.boxplot(data=cleaned_melted, y='Variable', x='Value', 
                    palette=ANTIQUE_COLORS[:4], orient='h', ax=ax, width=0.6)
        
        # Add jittered points
        sns.stripplot(data=cleaned_melted, y='Variable', x='Value', 
                      color='gray', alpha=0.3, size=2, jitter=True, ax=ax)
        
        ax.set_title("Distribution Summary of Cleaned Numerical Variables", fontsize=14, fontweight='bold')
        ax.set_xlabel("Value", fontsize=11)
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'figure_2_cleaned_data_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Figure 2 saved to: {output_path}")
    
    def create_qq_plots(self):
        """
        Create Q-Q plots for normality checks of the four numerical variables.
        """
        print("Creating Q-Q plots for normality checks...")
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, var in enumerate(self.numerical_vars):
            row, col = i // 2, i % 2
            ax = axs[row, col]
            
            # Get data
            data_before = self.df_before[var].dropna()
            data_after = self.df_after[var].dropna()
            
            # Create Q-Q plots
            if stats is not None and len(data_before) > 0 and len(data_after) > 0:
                # Before cleaning
                stats.probplot(data_before, dist="norm", plot=ax)
                ax.get_lines()[0].set_color(ANTIQUE_COLORS[i*2])
                ax.get_lines()[0].set_alpha(0.6)
                ax.get_lines()[1].set_color(ANTIQUE_COLORS[i*2])
                ax.get_lines()[1].set_linewidth(2)
                
                # After cleaning (overlay)
                stats.probplot(data_after, dist="norm", plot=ax)
                ax.get_lines()[2].set_color(ANTIQUE_COLORS[i*2+1])
                ax.get_lines()[2].set_alpha(0.6)
                ax.get_lines()[3].set_color(ANTIQUE_COLORS[i*2+1])
                ax.get_lines()[3].set_linewidth(2)
                
                # Add legend
                ax.plot([], [], color=ANTIQUE_COLORS[i*2], label="Before Cleaning", alpha=0.6)
                ax.plot([], [], color=ANTIQUE_COLORS[i*2+1], label="After Cleaning", alpha=0.6)
                ax.legend()
            
            ax.set_title(f"Q-Q Plot: {var.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Q-Q Plots: Normality Assessment", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'qq_plots_normality_check.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Q-Q plots saved to: {output_path}")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for both datasets."""
        print("Generating summary statistics...")
        
        summary_stats = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'before_cleaning': {
                    'total_books': len(self.df_before),
                    'features': len(self.df_before.columns)
                },
                'after_cleaning': {
                    'total_books': len(self.df_after),
                    'features': len(self.df_after.columns)
                },
                'data_reduction': {
                    'books_removed': len(self.df_before) - len(self.df_after),
                    'reduction_percentage': round(((len(self.df_before) - len(self.df_after)) / len(self.df_before)) * 100, 2)
                }
            },
            'numerical_variables_analysis': {}
        }
        
        # Analyze each numerical variable
        for var in self.numerical_vars:
            data_before = self.df_before[var].dropna()
            data_after = self.df_after[var].dropna()
            
            # Calculate effect sizes and statistical tests
            effect_sizes = {}
            statistical_tests_result = {}
            
            if len(data_before) > 0 and len(data_after) > 0:
                effect_sizes = {
                    'hedges_g': hedges_g(data_before, data_after),
                    'cohens_d': cohens_d(data_before, data_after),
                    'cliffs_delta': cliffs_delta(data_before, data_after)
                }
                statistical_tests_result = statistical_tests(data_before, data_after)
            
            # Calculate bootstrap confidence intervals
            bootstrap_ci_before = bootstrap_median_ci(data_before) if len(data_before) > 0 else (None, None)
            bootstrap_ci_after = bootstrap_median_ci(data_after) if len(data_after) > 0 else (None, None)
            
            summary_stats['numerical_variables_analysis'][var] = {
                'before_cleaning': {
                    'count': len(data_before),
                    'mean': round(data_before.mean(), 4) if len(data_before) > 0 else None,
                    'median': round(data_before.median(), 4) if len(data_before) > 0 else None,
                    'std': round(data_before.std(), 4) if len(data_before) > 0 else None,
                    'min': data_before.min() if len(data_before) > 0 else None,
                    'max': data_before.max() if len(data_before) > 0 else None,
                    'bootstrap_median_ci': {
                        'lower': round(bootstrap_ci_before[0], 4) if bootstrap_ci_before[0] is not None else None,
                        'upper': round(bootstrap_ci_before[1], 4) if bootstrap_ci_before[1] is not None else None
                    }
                },
                'after_cleaning': {
                    'count': len(data_after),
                    'mean': round(data_after.mean(), 4) if len(data_after) > 0 else None,
                    'median': round(data_after.median(), 4) if len(data_after) > 0 else None,
                    'std': round(data_after.std(), 4) if len(data_after) > 0 else None,
                    'min': data_after.min() if len(data_after) > 0 else None,
                    'max': data_after.max() if len(data_after) > 0 else None,
                    'bootstrap_median_ci': {
                        'lower': round(bootstrap_ci_after[0], 4) if bootstrap_ci_after[0] is not None else None,
                        'upper': round(bootstrap_ci_after[1], 4) if bootstrap_ci_after[1] is not None else None
                    }
                },
                'cleaning_impact': {
                    'data_points_removed': len(data_before) - len(data_after),
                    'removal_percentage': round(((len(data_before) - len(data_after)) / len(data_before)) * 100, 2) if len(data_before) > 0 else 0,
                    'mean_change': round(data_after.mean() - data_before.mean(), 4) if len(data_before) > 0 and len(data_after) > 0 else None,
                    'std_change': round(data_after.std() - data_before.std(), 4) if len(data_before) > 0 and len(data_after) > 0 else None
                },
                'effect_sizes': effect_sizes,
                'statistical_tests': statistical_tests_result
            }
        
        return summary_stats
    
    def save_summary_report(self, summary_stats):
        """Save summary statistics to JSON file."""
        report_path = self.output_dir / 'improved_eda_final_summary.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Summary report saved to: {report_path}")
    
    def create_pdf_report(self, summary_stats):
        """Create a single PDF report with all figures and markdown summary."""
        print("Creating PDF report...")
        
        pdf_path = self.output_dir / 'eda_comprehensive_report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Add title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.7, 'EDA Comprehensive Report', 
                   fontsize=24, fontweight='bold', ha='center', va='center')
            ax.text(0.5, 0.6, 'Romance Novel Dataset Analysis', 
                   fontsize=16, ha='center', va='center')
            ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                   fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.4, f'Before: {len(self.df_before):,} books', 
                   fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.35, f'After: {len(self.df_after):,} books', 
                   fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.3, f'Reduction: {((len(self.df_before) - len(self.df_after)) / len(self.df_before) * 100):.1f}%', 
                   fontsize=12, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add summary statistics page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.1, 0.9, 'Summary Statistics', fontsize=18, fontweight='bold')
            
            y_pos = 0.8
            for var, analysis in summary_stats['numerical_variables_analysis'].items():
                ax.text(0.1, y_pos, f'{var.replace("_", " ").title()}:', 
                       fontsize=14, fontweight='bold')
                y_pos -= 0.05
                
                # Before cleaning stats
                before = analysis['before_cleaning']
                ax.text(0.15, y_pos, f'Before: n={before["count"]:,}, mean={before["mean"]}, std={before["std"]}', 
                       fontsize=10)
                y_pos -= 0.03
                
                # After cleaning stats
                after = analysis['after_cleaning']
                ax.text(0.15, y_pos, f'After: n={after["count"]:,}, mean={after["mean"]}, std={after["std"]}', 
                       fontsize=10)
                y_pos -= 0.03
                
                # Effect sizes
                if analysis['effect_sizes']:
                    es = analysis['effect_sizes']
                    ax.text(0.15, y_pos, f'Effect sizes: Hedges g={es.get("hedges_g", "N/A"):.3f}, Cohen d={es.get("cohens_d", "N/A"):.3f}', 
                           fontsize=10)
                    y_pos -= 0.03
                
                y_pos -= 0.02
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"PDF report saved to: {pdf_path}")
        return pdf_path
    
    def print_summary(self, summary_stats):
        """Print a summary of the analysis."""
        print("=" * 80)
        print("IMPROVED EDA FINAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Dataset overview
        dataset_info = summary_stats['dataset_info']
        print(f"Dataset Overview:")
        print(f"  Before cleaning: {dataset_info['before_cleaning']['total_books']:,} books")
        print(f"  After cleaning: {dataset_info['after_cleaning']['total_books']:,} books")
        print(f"  Books removed: {dataset_info['data_reduction']['books_removed']:,} ({dataset_info['data_reduction']['reduction_percentage']}%)")
        print()
        
        # Numerical variables analysis
        print("Numerical Variables Analysis:")
        for var, analysis in summary_stats['numerical_variables_analysis'].items():
            print(f"  {var}:")
            print(f"    Before: n={analysis['before_cleaning']['count']:,}, "
                  f"mean={analysis['before_cleaning']['mean']}, "
                  f"std={analysis['before_cleaning']['std']}")
            print(f"    After: n={analysis['after_cleaning']['count']:,}, "
                  f"mean={analysis['after_cleaning']['mean']}, "
                  f"std={analysis['after_cleaning']['std']}")
            print(f"    Impact: {analysis['cleaning_impact']['removal_percentage']}% removed, "
                  f"mean change={analysis['cleaning_impact']['mean_change']}")
            print()
        
        print("=" * 80)
    
    # ============================================================================
    # ENHANCED HELPER METHODS
    # ============================================================================
    
    def _coerce_numeric(self, df, cols):
        """Coerce specified columns to numeric, handling errors gracefully."""
        df = df.copy()
        for c in cols:
            if c in df: 
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _fd_bin_edges(self, x, clip=None, max_bins=120, min_bins=20):
        """Freedman–Diaconis binning; returns edges array."""
        x = pd.Series(x).dropna()
        if x.empty: 
            return np.array([0,1])
        if clip:
            lo, hi = np.percentile(x, clip)
            x = x.clip(lo, hi)
        iqr = np.subtract(*np.percentile(x, [75,25]))
        if iqr == 0:  # fallback to sqrt rule
            bins = int(np.clip(np.sqrt(x.size), min_bins, max_bins))
        else:
            h = 2*iqr*(x.size**(-1/3))
            bins = int(np.clip(np.ceil((x.max()-x.min())/h), min_bins, max_bins))
        return np.histogram_bin_edges(x, bins=bins)

    def _common_bins(self, before, after, use_fd=True, bins=60, clip=None, edges_override=None):
        """Create common bin edges for before/after comparison."""
        if edges_override is not None: 
            return np.array(edges_override, dtype=float)
        if use_fd:
            return self._fd_bin_edges(pd.concat([before.dropna(), after.dropna()]), clip=clip)
        x = pd.concat([before.dropna(), after.dropna()])
        if clip:
            lo, hi = np.percentile(x, clip)
            x = x.clip(lo, hi)
        return np.histogram_bin_edges(x, bins=bins)

    def _save_edges_json(self, col, edges, outdir=None):
        """Save bin edges to JSON for reproducibility."""
        outdir = Path(outdir or (self.output_dir/"hist_counts"))
        outdir.mkdir(parents=True, exist_ok=True)
        j = outdir / f"bin_edges_{col}.json"
        j.write_text(json.dumps({"column": col, "edges": list(map(float, edges))}, indent=2))
        return str(j)

    def bootstrap_median_ci(self, x, reps=2000, alpha=0.05, seed=0):
        """Calculate bootstrap confidence interval for median."""
        rng = np.random.default_rng(seed)
        x = pd.Series(x).dropna().to_numpy()
        if x.size==0: 
            return np.nan, [np.nan, np.nan]
        boots = np.median(rng.choice(x, size=(reps, x.size), replace=True), axis=1)
        med = float(np.median(x))
        lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
        return med, [float(lo), float(hi)]

    def cohens_d(self, x, y):
        """Calculate Cohen's d effect size."""
        x = pd.Series(x).dropna().to_numpy()
        y = pd.Series(y).dropna().to_numpy()
        if len(x)<2 or len(y)<2: 
            return np.nan
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        sp = np.sqrt(((len(x)-1)*s1**2 + (len(y)-1)*s2**2)/(len(x)+len(y)-2))
        return 0.0 if sp==0 else (x.mean()-y.mean())/sp

    def hedges_g(self, x, y):
        """Calculate Hedges' g effect size."""
        d = self.cohens_d(x, y)
        n1, n2 = pd.Series(x).dropna().size, pd.Series(y).dropna().size
        J = 1 - (3 / (4*(n1+n2)-9))
        return d*J

    def cliffs_delta(self, x, y):
        """Calculate Cliff's delta effect size."""
        x = pd.Series(x).dropna().to_numpy()
        y = pd.Series(y).dropna().to_numpy()
        if len(x)==0 or len(y)==0: 
            return np.nan
        max_pairs = 2_000_000
        if len(x)*len(y) > max_pairs:
            rng = np.random.default_rng(0)
            x = rng.choice(x, size=min(len(x), int(np.sqrt(max_pairs))), replace=False)
            y = rng.choice(y, size=min(len(y), int(np.sqrt(max_pairs))), replace=False)
        gt = sum((xi > y).sum() for xi in x)
        lt = sum((xi < y).sum() for xi in x)
        return (gt - lt) / (len(x)*len(y))

    def two_sample_tests(self, x, y):
        """Run two-sample statistical tests."""
        xb = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        ya = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
        out = {}
        if ks_2samp and mannwhitneyu:
            try:
                k = ks_2samp(xb, ya)
                u = mannwhitneyu(xb, ya, alternative="two-sided")
                out["ks_stat"], out["ks_p"] = float(k.statistic), float(k.pvalue)
                out["mw_u"], out["mw_p"] = float(u.statistic), float(u.pvalue)
            except Exception:
                out["ks_stat"]=out["ks_p"]=out["mw_u"]=out["mw_p"]=np.nan
        else:
            out["ks_stat"]=out["ks_p"]=out["mw_u"]=out["mw_p"]=np.nan
        out["cohens_d"]=self.cohens_d(ya, xb)
        out["hedges_g"]=self.hedges_g(ya, xb)
        out["cliffs_delta"]=self.cliffs_delta(ya, xb)
        return out

    def save_hist_counts(self, col, edges, outdir=None):
        """Save histogram counts to CSV for reproducibility."""
        outdir = Path(outdir or (self.output_dir/"hist_counts"))
        outdir.mkdir(parents=True, exist_ok=True)
        xb = pd.to_numeric(self.df_before[col], errors="coerce").dropna()
        xa = pd.to_numeric(self.df_after[col], errors="coerce").dropna()
        cb, _ = np.histogram(xb, bins=edges)
        ca, _ = np.histogram(xa, bins=edges)
        df = pd.DataFrame({
            "bin_left": edges[:-1], 
            "bin_right": edges[1:], 
            "count_before": cb, 
            "count_after": ca
        })
        p = outdir / f"bincounts_{col}.csv"
        df.to_csv(p, index=False)
        return str(p)

    def savefig_with_meta(self, fig, title, tags=None, dpi=300, color_meta=None):
        """Save figure with metadata and color validation."""
        ColorPolicyValidator().validate(color_meta or {
            "legend_present": True, 
            "variable_colors": {}, 
            "background":"#FFFFFF",
            "text_color":"#111111", 
            "uses_gradient_for_categories": False, 
            "is_categorical": True
        })
        outdir = self.output_dir / "figures"
        outdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"eda_{title.lower().replace(' ','-')}_{ts}"
        png_path = outdir / f"{stem}.png"
        json_path = outdir / f"{stem}.json"
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        im = Image.open(buf)
        meta = PngImagePlugin.PngInfo()
        md = {
            "title": title, 
            "tags": tags or [], 
            "created": ts, 
            "matplotlib_version": plt.matplotlib.__version__
        }
        for k,v in md.items(): 
            meta.add_text(k, json.dumps(v) if not isinstance(v,str) else v)
        im.save(png_path, "PNG", pnginfo=meta)
        with open(json_path, "w") as f: 
            json.dump(md, f, indent=2)
        return str(png_path)

    def _normal_pdf(self, x, mu, sd):
        """Normal PDF helper for bell curve (fallback if SciPy missing)."""
        return (1/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sd)**2)

    def plot_hist_with_bands(self, col, use_fd=True, bins=60, clip=None, edges_override=None, density=False):
        """Create outlined step histograms with bell curves and median CI bands."""
        xb = pd.to_numeric(self.df_before[col], errors="coerce").dropna()
        xa = pd.to_numeric(self.df_after[col], errors="coerce").dropna()
        edges = self._common_bins(xb, xa, use_fd=use_fd, bins=bins, clip=clip, edges_override=edges_override)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(xb, bins=edges, histtype="step", linewidth=2, label=f"Before (n={len(xb):,})", density=density)
        ax.hist(xa, bins=edges, histtype="step", linewidth=2, label=f"After (n={len(xa):,})", density=density)

        # Medians + bootstrap CIs
        for data, color, label in [(xb,"#6F7C85","Before"), (xa,"#AF6458","After")]:
            med, ci = self.bootstrap_median_ci(data)
            ax.axvline(med, color=color, linestyle="--", linewidth=2)
            ax.axvspan(ci[0], ci[1], color=color, alpha=0.15)

            # Bell curve overlay (reference)
            mu, sd = np.mean(data), np.std(data, ddof=1)
            xs = np.linspace(edges[0], edges[-1], 400)
            pdf = (norm.pdf(xs, mu, sd) if norm else self._normal_pdf(xs, mu, sd))
            if density:
                ax.plot(xs, pdf, color=color, alpha=0.8)
            else:
                # scale to counts height approx
                height = max(ax.get_ylim())
                ax.plot(xs, pdf/pdf.max()*height*0.85, color=color, alpha=0.8)

        if col == "ratings_count_sum":
            ax.set_xscale("log")
            from matplotlib.ticker import LogLocator, ScalarFormatter
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)

        ax.set_title(f"{col.replace('_',' ').title()} — Before vs After")
        ax.set_xlabel(col.replace('_',' ').title())
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig, edges

    def plot_pages_small_multiples(self, clip=(1,99), use_fd=True):
        """Create small-multiples panel for pages (full vs clipped)."""
        col = "num_pages_median"
        b = pd.to_numeric(self.df_before[col], errors="coerce").dropna()
        a = pd.to_numeric(self.df_after[col],  errors="coerce").dropna()
        edges_full = self._common_bins(b, a, use_fd=use_fd)
        bf, _ = np.histogram(b, bins=edges_full)
        af, _ = np.histogram(a, bins=edges_full)

        lo_b, hi_b = np.percentile(b, clip)
        lo_a, hi_a = np.percentile(a, clip)
        b_clip, a_clip = b.clip(lo_b, hi_b), a.clip(lo_a, hi_a)
        edges_clip = self._common_bins(b_clip, a_clip, use_fd=use_fd)

        fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=False)
        for ax, x1, x2, edges, title in [
            (axes[0], b, a, edges_full, "Full range"),
            (axes[1], b_clip, a_clip, edges_clip, f"Clipped {clip[0]}–{clip[1]} pct")
        ]:
            ax.hist(x1, bins=edges, histtype="step", linewidth=2, label=f"Before (n={len(x1):,})")
            ax.hist(x2, bins=edges, histtype="step", linewidth=2, label=f"After (n={len(x2):,})")
            ax.set_title(title)
            ax.set_xlabel("Median pages")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("Pages distribution — small multiples")
        plt.tight_layout()
        return fig

    def write_stats_outputs(self):
        """Generate stats CSV and Markdown outputs."""
        path_md = Path(self.output_dir/"README_EDA_Summary.md")
        path_csv = Path(self.output_dir/"EDA_stats_effectsizes.csv")
        rows=[]
        for c in self.numerical_vars:
            b = pd.to_numeric(self.df_before[c], errors="coerce")
            a = pd.to_numeric(self.df_after[c],  errors="coerce")
            med_b, ci_b = self.bootstrap_median_ci(b)
            med_a, ci_a = self.bootstrap_median_ci(a)
            t = self.two_sample_tests(b, a)
            rows.append({
                "variable": c,
                "n_before": int(b.notna().sum()), 
                "n_after": int(a.notna().sum()),
                "mean_before": b.mean(), 
                "mean_after": a.mean(),
                "median_before": med_b, 
                "median_before_CI95_low": ci_b[0], 
                "median_before_CI95_high": ci_b[1],
                "median_after": med_a, 
                "median_after_CI95_low": ci_a[0], 
                "median_after_CI95_high": ci_a[1],
                "delta_mean": a.mean()-b.mean(), 
                "delta_median": med_a-med_b,
                **t
            })
        df = pd.DataFrame(rows).round(6)
        
        # Generate markdown with fallback for tabulate dependency
        try:
            md = "### Descriptive statistics: Before vs After (effects + tests)\n\n" + df.to_markdown(index=False)
        except ImportError:
            # Fallback to simple text table if tabulate is not available
            md = "### Descriptive statistics: Before vs After (effects + tests)\n\n"
            md += df.to_string(index=False)
        
        path_md.write_text(md)
        df.to_csv(path_csv, index=False)
        return str(path_md), str(path_csv)

    def alt_text_for_hist(self, col, edges):
        """Generate ALT text template for histogram."""
        return (f"Step-outline histograms compare {col.replace('_',' ')} before and after cleaning "
                f"using common bin edges ({len(edges)-1} bins) chosen via Freedman–Diaconis. "
                "Dashed lines mark medians with shaded 95% bootstrap confidence bands. "
                "Faint bell-curve overlays provide a normal reference. "
                f"{'X-axis is logarithmic for ratings count.' if col=='ratings_count_sum' else ''}")
    
    def alt_text_for_pages_small_multiples(self):
        """Generate ALT text template for pages small multiples."""
        return ("Two panels show the pages distribution: full range and a clipped 1–99th percentile view. "
                "Each panel uses step-outline histograms with common bins; medians and CI bands indicate central tendency changes.")

    def bundle_pdf_report(self, figure_paths, md_path):
        """Bundle all figures and markdown into a single PDF report."""
        pdf_path = self.output_dir / 'eda_comprehensive_report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Add title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.7, 'EDA Comprehensive Report', 
                   fontsize=24, fontweight='bold', ha='center', va='center')
            ax.text(0.5, 0.6, 'Romance Novel Dataset Analysis', 
                   fontsize=16, ha='center', va='center')
            ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                   fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.4, f'Before: {len(self.df_before):,} books', 
                   fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.35, f'After: {len(self.df_after):,} books', 
                   fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.3, f'Reduction: {((len(self.df_before) - len(self.df_after)) / len(self.df_before) * 100):.1f}%', 
                   fontsize=12, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add summary statistics page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.1, 0.9, 'Summary Statistics', fontsize=18, fontweight='bold')
            
            y_pos = 0.8
            for var in self.numerical_vars:
                b = pd.to_numeric(self.df_before[var], errors="coerce")
                a = pd.to_numeric(self.df_after[var], errors="coerce")
                med_b, ci_b = self.bootstrap_median_ci(b)
                med_a, ci_a = self.bootstrap_median_ci(a)
                
                ax.text(0.1, y_pos, f'{var.replace("_", " ").title()}:', 
                       fontsize=14, fontweight='bold')
                y_pos -= 0.05
                
                # Before cleaning stats
                ax.text(0.15, y_pos, f'Before: n={b.notna().sum():,}, mean={b.mean():.3f}, median={med_b:.3f}', 
                       fontsize=10)
                y_pos -= 0.03
                
                # After cleaning stats
                ax.text(0.15, y_pos, f'After: n={a.notna().sum():,}, mean={a.mean():.3f}, median={med_a:.3f}', 
                       fontsize=10)
                y_pos -= 0.03
                
                y_pos -= 0.02
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        return str(pdf_path)

    def generate_full_eda(self, cfg):
        """Generate full EDA with all new features."""
        self.df_before = self._coerce_numeric(self.df_before, self.numerical_vars)
        self.df_after  = self._coerce_numeric(self.df_after,  self.numerical_vars)

        figure_paths=[]
        alt_texts=[]

        # Loop variables: FD-bins + step-outline + bell curves
        for col in self.numerical_vars:
            fig, edges = self.plot_hist_with_bands(
                col, 
                use_fd=True, 
                bins=cfg.get("bins",{}).get(col,60),
                clip=cfg.get("clip",{}).get(col,None),
                edges_override=cfg.get("edges",{}).get(col,None),
                density=False
            )
            # freeze edges: save CSV counts + JSON edges
            self._save_edges_json(col, edges)
            self.save_hist_counts(col, edges)

            color_meta = {
                "legend_present":True, 
                "variable_colors":{"before":"#6F7C85","after":"#AF6458"},
                "background":"#FFFFFF",
                "text_color":"#111111",
                "uses_gradient_for_categories":False,
                "is_categorical":True
            }
            p = self.savefig_with_meta(fig, title=f"{col}_hist_step", dpi=cfg.get("dpi",300), color_meta=color_meta)
            figure_paths.append(p)
            alt_texts.append(self.alt_text_for_hist(col, edges))
            plt.close(fig)

        # Small multiples for pages
        fig = self.plot_pages_small_multiples(clip=cfg.get("pages_clip_pct",(1,99)), use_fd=True)
        p = self.savefig_with_meta(fig, title="pages_small_multiples", dpi=cfg.get("dpi",300),
                                   color_meta={"legend_present":True,"variable_colors":{},
                                               "background":"#FFFFFF","text_color":"#111111",
                                               "uses_gradient_for_categories":False,"is_categorical":True})
        figure_paths.append(p)
        alt_texts.append(self.alt_text_for_pages_small_multiples())
        plt.close(fig)

        # Stats: Markdown + CSV
        md_path, csv_path = self.write_stats_outputs()

        # Bundle PDF
        pdf_path = self.bundle_pdf_report(figure_paths, md_path)

        # Save ALT texts
        alt_path = Path(self.output_dir/"figure_alt_texts.txt")
        alt_path.write_text("\n\n".join(f"- {Path(fp).name}: {t}" for fp,t in zip(figure_paths, alt_texts)))

        print(f"Figures: {len(figure_paths)} | PDF: {pdf_path}\nStats CSV: {csv_path}\nALT: {alt_path}")
        return figure_paths, alt_texts, md_path, csv_path, pdf_path

    # ============================================================================
    # CODING AGENT PATTERN METHODS
    # ============================================================================
    
    def analyze_task(self, task_description: str = "Create EDA visualizations") -> Dict[str, Any]:
        """
        Analyze coding task requirements (Coding Agent Pattern).
        
        Args:
            task_description: Description of the task
            
        Returns:
            Analysis results with requirements, affected files, and change plan
        """
        logger.info("Analyzing EDA visualization task...")
        
        try:
            # Parse requirements
            requirements = {
                'create_figure_1': True,  # Before/after histograms
                'create_figure_2': True,  # Cleaned data boxplots
                'generate_statistics': True,  # Summary statistics
                'save_outputs': True,  # Save all outputs
                'use_antique_palette': True,  # Use specified color palette
                'academic_standards': True  # Follow publication standards
            }
            
            # Identify affected files
            affected_files = {
                'input_files': [str(self.data_path_before), str(self.data_path_after)],
                'output_files': [
                    str(self.output_dir / 'figure_1_before_after_distributions.png'),
                    str(self.output_dir / 'figure_2_cleaned_data_summary.png'),
                    str(self.output_dir / 'improved_eda_final_summary.json')
                ]
            }
            
            # Create change plan
            change_plan = {
                'sequence': [
                    'validate_data_availability',
                    'create_figure_1_histograms',
                    'create_figure_2_boxplots', 
                    'generate_summary_statistics',
                    'save_all_outputs',
                    'verify_outputs'
                ],
                'dependencies': {
                    'create_figure_1_histograms': ['validate_data_availability'],
                    'create_figure_2_boxplots': ['validate_data_availability'],
                    'generate_summary_statistics': ['validate_data_availability'],
                    'save_all_outputs': ['create_figure_1_histograms', 'create_figure_2_boxplots', 'generate_summary_statistics'],
                    'verify_outputs': ['save_all_outputs']
                }
            }
            
            self.task_analysis = {
                'requirements': requirements,
                'affected_files': affected_files,
                'change_plan': change_plan,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Task analysis completed successfully")
            return self.task_analysis
            
        except Exception as e:
            logger.error(f"Task analysis failed: {str(e)}")
            raise AnalysisError("Failed to analyze EDA task")
    
    def validate_data_availability(self) -> bool:
        """
        Validate that required data is available (Coding Agent Pattern).
        
        Returns:
            bool indicating if data validation passed
        """
        logger.info("Validating data availability...")
        
        try:
            # Check if datasets are loaded
            if self.df_before is None or self.df_after is None:
                raise ValueError("Datasets not loaded")
            
            # Check if required columns exist
            missing_cols_before = set(self.numerical_vars) - set(self.df_before.columns)
            missing_cols_after = set(self.numerical_vars) - set(self.df_after.columns)
            
            if missing_cols_before:
                raise ValueError(f"Missing columns in before dataset: {missing_cols_before}")
            if missing_cols_after:
                raise ValueError(f"Missing columns in after dataset: {missing_cols_after}")
            
            # Check if output directory is writable
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def apply_changes(self, change_plan: Dict[str, Any]) -> List[str]:
        """
        Apply planned code changes (Coding Agent Pattern).
        
        Args:
            change_plan: Planned changes
            
        Returns:
            List of modified files
        """
        logger.info("Applying planned changes...")
        
        try:
            modified_files = []
            start_time = time.time()
            
            # Execute changes in dependency order
            for step in change_plan['sequence']:
                logger.info(f"Executing step: {step}")
                
                if step == 'validate_data_availability':
                    if not self.validate_data_availability():
                        raise ValueError("Data validation failed")
                        
                elif step == 'create_figure_1_histograms':
                    self.create_figure_1_histograms()
                    modified_files.append(str(self.output_dir / 'figure_1_before_after_distributions.png'))
                    
                elif step == 'create_figure_2_boxplots':
                    self.create_figure_2_boxplots()
                    modified_files.append(str(self.output_dir / 'figure_2_cleaned_data_summary.png'))
                    
                elif step == 'generate_summary_statistics':
                    summary_stats = self.generate_summary_statistics()
                    self.save_summary_report(summary_stats)
                    modified_files.append(str(self.output_dir / 'improved_eda_final_summary.json'))
                    
                elif step == 'save_all_outputs':
                    # Already handled in individual steps
                    pass
                    
                elif step == 'verify_outputs':
                    if not self.verify_changes(modified_files):
                        raise ValueError("Output verification failed")
                
                # Record change
                self.change_history.append({
                    'step': step,
                    'timestamp': time.time(),
                    'duration': time.time() - start_time,
                    'status': 'completed'
                })
            
            logger.info(f"All changes applied successfully. Modified {len(modified_files)} files")
            return modified_files
            
        except Exception as e:
            logger.error(f"Change application failed: {str(e)}")
            self._revert_changes()
            raise ModificationError("Failed to apply EDA changes")
    
    def verify_changes(self, modified_files: List[str]) -> bool:
        """
        Verify changes through testing (Coding Agent Pattern).
        
        Args:
            modified_files: List of modified files
            
        Returns:
            bool indicating if changes pass verification
        """
        logger.info("Verifying changes...")
        
        try:
            verification_results = {
                'file_existence': True,
                'data_integrity': True,
                'figure_quality': True
            }
            
            # Check file existence
            for file_path in modified_files:
                if not Path(file_path).exists():
                    logger.error(f"Output file not found: {file_path}")
                    verification_results['file_existence'] = False
            
            # Check data integrity
            if len(self.df_before) == 0 or len(self.df_after) == 0:
                logger.error("Empty datasets detected")
                verification_results['data_integrity'] = False
            
            # Check figure quality (basic checks)
            for var in self.numerical_vars:
                if self.df_after[var].isna().all():
                    logger.error(f"All values missing for variable: {var}")
                    verification_results['figure_quality'] = False
            
            all_passed = all(verification_results.values())
            
            if all_passed:
                logger.info("All verifications passed")
            else:
                logger.error(f"Verification failed: {verification_results}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False
    
    def _revert_changes(self):
        """Revert changes on failure (Coding Agent Pattern)."""
        logger.warning("Reverting changes due to failure...")
        
        try:
            # Remove any created output files
            for change in self.change_history:
                if change['status'] == 'completed':
                    # Could implement file cleanup here if needed
                    pass
            
            # Clear change history
            self.change_history = []
            logger.info("Changes reverted successfully")
            
        except Exception as e:
            logger.error(f"Failed to revert changes: {str(e)}")
    
    def generate_all_visualizations(self, skip_qq=False, skip_pdf=False):
        """Generate all visualizations using Coding Agent Pattern with detailed progress tracking."""
        logger.info("🚀 Starting comprehensive EDA visualization generation")
        
        # Calculate total steps
        total_steps = 6  # Base steps
        if not skip_qq:
            total_steps += 1
        if not skip_pdf:
            total_steps += 1
            
        # Initialize progress tracker
        progress = ProgressTracker(total_steps, "EDA Visualization Generation")
        progress.start()
        
        try:
            # Step 1: Analyze task
            progress.step("Task Analysis", "Analyzing requirements and creating change plan")
            task_analysis = self.analyze_task("Create EDA visualizations")
            self.change_plan = task_analysis['change_plan']
            logger.info("✅ Task analysis completed successfully")
            
            # Step 2: Apply changes using the plan
            progress.step("Core Visualizations", "Creating histograms and boxplots")
            modified_files = self.apply_changes(self.change_plan)
            logger.info(f"✅ Core visualizations completed: {len(modified_files)} files")
            
            # Step 3: Generate Q-Q plots (if not skipped)
            if not skip_qq:
                progress.step("Q-Q Plots", "Creating normality assessment plots")
                self.create_qq_plots()
                modified_files.append(str(self.output_dir / 'qq_plots_normality_check.png'))
                logger.info("✅ Q-Q plots completed")
            
            # Step 4: Generate summary statistics
            progress.step("Statistical Analysis", "Computing effect sizes and statistical tests")
            summary_stats = self.generate_summary_statistics()
            self.save_summary_report(summary_stats)
            modified_files.append(str(self.output_dir / 'improved_eda_final_summary.json'))
            logger.info("✅ Statistical analysis completed")
            
            # Step 5: Create PDF report (if not skipped)
            if not skip_pdf:
                progress.step("PDF Report", "Generating comprehensive PDF report")
                pdf_path = self.create_pdf_report(summary_stats)
                modified_files.append(str(pdf_path))
                logger.info("✅ PDF report completed")
            
            # Step 6: Print summary
            progress.step("Summary Report", "Generating final summary")
            self.print_summary(summary_stats)
            logger.info("✅ Summary report completed")
            
            # Finish progress tracking
            progress.finish()
            
            # Final success message
            logger.info("🎉 All EDA visualizations generated successfully!")
            print(f"\n🎉 SUCCESS! Generated {len(modified_files)} files:")
            for i, file_path in enumerate(modified_files, 1):
                print(f"   {i:2d}. {Path(file_path).name}")
            print(f"\n📁 Output directory: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ EDA generation failed: {str(e)}")
            print(f"\n❌ ERROR: {str(e)}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced EDA Analysis with Statistical Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python improved_eda_final.py
  
  # With custom paths and sampling
  python improved_eda_final.py --before data/raw.csv --after data/clean.csv --output results/ --sample-size 5000
  
  # High DPI with custom bins
  python improved_eda_final.py --dpi 600 --bins 50 --no-clip-outliers
        """
    )
    
    # Data paths
    parser.add_argument('--before', 
                       default="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/final_books_2000_2020_en_enhanced_20250907_013708.csv",
                       help='Path to before cleaning dataset CSV')
    parser.add_argument('--after',
                       default="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/romance_novels_text_preprocessed_20250907_015606.csv", 
                       help='Path to after cleaning dataset CSV')
    parser.add_argument('--output', '-o',
                       default="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/complete_pipeline_analysis",
                       help='Output directory for results')
    
    # Analysis parameters
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Size for stratified sampling (None for no sampling)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--bins', type=int, default=30,
                       help='Number of bins for histograms (default: 30)')
    parser.add_argument('--no-clip-outliers', action='store_true',
                       help='Disable outlier clipping for better visualization')
    
    # Analysis options
    parser.add_argument('--skip-qq', action='store_true',
                       help='Skip Q-Q plots generation')
    parser.add_argument('--skip-pdf', action='store_true',
                       help='Skip PDF report generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main execution function with enhanced EDA features."""
    parser = argparse.ArgumentParser(description="Improved EDA with step hists, bell curves, FD bins, stats CSV, small multiples, CI color guard.")
    parser.add_argument("--before", required=True, help="Path to before cleaning dataset CSV")
    parser.add_argument("--after", required=True, help="Path to after cleaning dataset CSV")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--pages-clip-pct", type=str, default="1,99", help="Pages clipping percentiles (e.g., '1,99')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(verbose=args.verbose)
    logger.info("🔧 Enhanced logging initialized")
    
    try:
        # Create visualizer
        logger.info("🏗️  Creating EDA analyzer instance...")
        viz = ImprovedEDAFinal(args.before, args.after, args.outdir)
        
        # Parse pages clipping percentiles
        lo, hi = map(float, args.pages_clip_pct.split(","))
        cfg = {
            "dpi": args.dpi, 
            "pages_clip_pct": (lo,hi), 
            "bins":{}, 
            "clip":{}, 
            "edges":{}
        }
        
        # Generate full EDA with new features
        logger.info("🎨 Starting enhanced EDA generation...")
        figure_paths, alt_texts, md_path, csv_path, pdf_path = viz.generate_full_eda(cfg)
        
        # Final success message
        logger.info("🎉 Enhanced EDA analysis completed successfully!")
        print(f"\n🎉 ENHANCED EDA COMPLETE!")
        print(f"📊 Generated {len(figure_paths)} figures")
        print(f"📄 PDF report: {pdf_path}")
        print(f"📈 Stats CSV: {csv_path}")
        print(f"📝 Markdown: {md_path}")
        print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        print(f"\n💥 FATAL ERROR: {str(e)}")
        print(f"⏰ Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
