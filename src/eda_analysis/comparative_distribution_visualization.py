#!/usr/bin/env python3
"""
Comparative Distribution Visualization Script
Senior Data Scientist Review & Production

This script creates two publication-ready figures comparing numerical variables
before and after data cleaning using the Antique color palette.

Figure 1: Before vs. After Distribution Histograms (2x2 grid)
Figure 2: Comparative Boxplots of Cleaned Data (2x2 grid)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up academic publication style
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.figsize': (12, 8),
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

class ComparativeDistributionVisualizer:
    """Create comparative distribution visualizations for before/after data cleaning."""
    
    def __init__(self, data_path_before, data_path_after, output_dir):
        """
        Initialize the visualizer.
        
        Args:
            data_path_before: Path to the raw dataset CSV file
            data_path_after: Path to the cleaned dataset CSV file
            output_dir: Directory to save visualization outputs
        """
        self.data_path_before = Path(data_path_before)
        self.data_path_after = Path(data_path_after)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load datasets
        self.df_before = pd.read_csv(self.data_path_before)
        self.df_after = pd.read_csv(self.data_path_after)
        
        print(f"Loaded before dataset: {self.df_before.shape[0]:,} books, {self.df_before.shape[1]} features")
        print(f"Loaded after dataset: {self.df_after.shape[0]:,} books, {self.df_after.shape[1]} features")
        print(f"Output directory: {self.output_dir}")
        
        # Define the four numerical variables to analyze
        self.numerical_vars = [
            'publication_year',
            'num_pages_median', 
            'ratings_count_sum',
            'average_rating_weighted_mean'
        ]
        
        # Define clean titles for plots
        self.var_titles = [
            'Publication Year Distribution',
            'Median Page Count Distribution',
            'Total Ratings Count Distribution',
            'Average Rating Distribution'
        ]
        
        # Define x-axis labels
        self.var_labels = [
            'Publication Year',
            'Number of Pages',
            'Number of Ratings',
            'Average Rating (1-5)'
        ]
    
    def create_figure_1_histograms(self):
        """
        Create Figure 1: Before vs. After Distribution Histograms (2x2 grid).
        
        Shows overlaid histograms for each numerical variable comparing
        raw data vs cleaned data distributions.
        """
        print("Creating Figure 1: Before vs. After Distribution Histograms...")
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()
        
        for i, (var, title, xlabel) in enumerate(zip(self.numerical_vars, self.var_titles, self.var_labels)):
            ax = axs[i]
            
            # Get data for both datasets
            data_before = self.df_before[var].dropna()
            data_after = self.df_after[var].dropna()
            
            # Create overlaid histograms
            sns.histplot(data=data_before, alpha=0.6, color=ANTIQUE_COLORS[0], 
                        label='Before Cleaning', ax=ax, kde=False)
            sns.histplot(data=data_after, alpha=0.6, color=ANTIQUE_COLORS[1], 
                        label='After Cleaning', ax=ax, kde=False)
            
            # Special handling for average_rating_weighted_mean - add KDE
            if var == 'average_rating_weighted_mean':
                sns.histplot(data=data_after, alpha=0.3, color=ANTIQUE_COLORS[1], 
                            kde=True, ax=ax, label='After Cleaning (KDE)')
            
            # Special handling for ratings_count_sum - use log scale if heavily skewed
            if var == 'ratings_count_sum':
                # Check if data is heavily skewed (skewness > 2)
                skewness = data_after.skew()
                if skewness > 2:
                    ax.set_xscale('log')
                    ax.set_xlabel(f'{xlabel} (log scale)')
                else:
                    ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(xlabel)
            
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add sample size annotations
            ax.text(0.02, 0.98, 
                   f'Before: n={len(data_before):,}\nAfter: n={len(data_after):,}', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
        
        plt.suptitle('Distribution Comparison: Before vs After Data Cleaning', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'figure_1_before_after_histograms.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Figure 1 saved to: {output_path}")
    
    def create_figure_2_boxplots(self):
        """
        Create Figure 2: Comparative Boxplots of Cleaned Data (2x2 grid).
        
        Shows horizontal boxplots with jitter for all four numerical variables
        from the cleaned dataset, each on its own subplot with appropriate scaling.
        """
        print("Creating Figure 2: Comparative Boxplots of Cleaned Data...")
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        axs = axs.flatten()
        
        for i, (var, title, xlabel) in enumerate(zip(self.numerical_vars, self.var_titles, self.var_labels)):
            ax = axs[i]
            
            # Get cleaned data
            data_clean = self.df_after[var].dropna()
            
            # Create horizontal boxplot
            sns.boxplot(data=self.df_after, x=var, ax=ax, color=ANTIQUE_COLORS[i], width=0.4)
            
            # Add jitter/stripplot overlay
            sns.stripplot(data=self.df_after, x=var, ax=ax, color='grey', alpha=0.3, size=3)
            
            # Set labels and title
            ax.set_title(f'Distribution of {title.split()[0]} {title.split()[1]} (Cleaned)')
            ax.set_xlabel('Value')
            ax.set_ylabel('')  # No y-label needed for single horizontal boxplot
            
            # Add sample size annotation
            ax.text(0.02, 0.98, f'n = {len(data_clean):,}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
            
            # Add basic statistics
            mean_val = data_clean.mean()
            median_val = data_clean.median()
            std_val = data_clean.std()
            
            stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
            ax.text(0.98, 0.98, stats_text, 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   verticalalignment='top', horizontalalignment='right')
        
        plt.suptitle('Summary of Numerical Variables After Cleaning', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'figure_2_cleaned_data_boxplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Figure 2 saved to: {output_path}")
    
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
        for var, title in zip(self.numerical_vars, self.var_titles):
            data_before = self.df_before[var].dropna()
            data_after = self.df_after[var].dropna()
            
            summary_stats['numerical_variables_analysis'][var] = {
                'variable_name': title,
                'before_cleaning': {
                    'count': len(data_before),
                    'mean': round(data_before.mean(), 4),
                    'median': round(data_before.median(), 4),
                    'std': round(data_before.std(), 4),
                    'min': data_before.min(),
                    'max': data_before.max(),
                    'skewness': round(data_before.skew(), 4)
                },
                'after_cleaning': {
                    'count': len(data_after),
                    'mean': round(data_after.mean(), 4),
                    'median': round(data_after.median(), 4),
                    'std': round(data_after.std(), 4),
                    'min': data_after.min(),
                    'max': data_after.max(),
                    'skewness': round(data_after.skew(), 4)
                },
                'cleaning_impact': {
                    'data_points_removed': len(data_before) - len(data_after),
                    'removal_percentage': round(((len(data_before) - len(data_after)) / len(data_before)) * 100, 2),
                    'mean_change': round(data_after.mean() - data_before.mean(), 4),
                    'std_change': round(data_after.std() - data_before.std(), 4)
                }
            }
        
        return summary_stats
    
    def save_summary_report(self, summary_stats):
        """Save summary statistics to JSON file."""
        report_path = self.output_dir / 'comparative_distribution_summary.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Summary report saved to: {report_path}")
    
    def print_summary(self, summary_stats):
        """Print a summary of the analysis."""
        print("=" * 80)
        print("COMPARATIVE DISTRIBUTION ANALYSIS SUMMARY")
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
            print(f"  {analysis['variable_name']}:")
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
    
    def generate_all_visualizations(self):
        """Generate all comparative distribution visualizations."""
        print("Generating comparative distribution visualizations...")
        print("=" * 60)
        
        # Create Figure 1: Before vs After Histograms
        self.create_figure_1_histograms()
        
        # Create Figure 2: Cleaned Data Boxplots
        self.create_figure_2_boxplots()
        
        # Generate and save summary statistics
        summary_stats = self.generate_summary_statistics()
        self.save_summary_report(summary_stats)
        self.print_summary(summary_stats)
        
        print("=" * 60)
        print("All comparative distribution visualizations generated successfully!")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main execution function."""
    # Set up paths - using proper before/after datasets
    data_path_before = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/final_books_2000_2020_en_enhanced_20250906_204009.csv"
    data_path_after = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/eda_with_cleaning/eda_cleaned_dataset_20250907_003348.csv"
    output_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/comparative_distribution_analysis"
    
    # Create visualizer
    visualizer = ComparativeDistributionVisualizer(data_path_before, data_path_after, output_dir)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
