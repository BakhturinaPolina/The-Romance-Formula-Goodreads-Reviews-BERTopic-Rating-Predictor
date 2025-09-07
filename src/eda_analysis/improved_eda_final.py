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
warnings.filterwarnings('ignore')

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
    """Clean, focused EDA plot generator with exactly two essential figures."""
    
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
    
    def create_figure_1_histograms(self):
        """
        FIGURE 1: Before/After Cleaning Distributions (2x2 histograms)
        
        Creates a 2×2 subplot layout comparing distributions before and after 
        data cleaning for four numerical variables using Antique color palette.
        """
        print("Creating Figure 1: Before/After Cleaning Distributions...")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Publication Year Distribution
        sns.histplot(data=self.df_before, x="publication_year", kde=False, color=ANTIQUE_COLORS[0], 
                     label="Before Cleaning", ax=axs[0,0], alpha=0.6, bins=30)
        sns.histplot(data=self.df_after, x="publication_year", kde=False, color=ANTIQUE_COLORS[1], 
                     label="After Cleaning", ax=axs[0,0], alpha=0.6, bins=30)
        axs[0,0].set_title("Publication Year Distribution", fontsize=12, fontweight='bold')
        axs[0,0].set_xlabel("Publication Year")
        axs[0,0].set_ylabel("Count")
        axs[0,0].legend()
        
        # 2. Median Page Count Distribution  
        sns.histplot(data=self.df_before, x="num_pages_median", kde=False, color=ANTIQUE_COLORS[2], 
                     label="Before Cleaning", ax=axs[0,1], alpha=0.6, bins=30)
        sns.histplot(data=self.df_after, x="num_pages_median", kde=False, color=ANTIQUE_COLORS[3], 
                     label="After Cleaning", ax=axs[0,1], alpha=0.6, bins=30)
        axs[0,1].set_title("Median Page Count Distribution", fontsize=12, fontweight='bold')
        axs[0,1].set_xlabel("Number of Pages (Median)")
        axs[0,1].set_ylabel("Count")
        axs[0,1].legend()
        
        # 3. Ratings Count Distribution (with log scale to show popular vs niche)
        sns.histplot(data=self.df_before, x="ratings_count_sum", kde=False, color=ANTIQUE_COLORS[4], 
                     label="Before Cleaning", ax=axs[1,0], alpha=0.6, bins=50)
        sns.histplot(data=self.df_after, x="ratings_count_sum", kde=False, color=ANTIQUE_COLORS[5], 
                     label="After Cleaning", ax=axs[1,0], alpha=0.6, bins=50)
        axs[1,0].set_title("Ratings Count Distribution (Popular vs Niche)", fontsize=12, fontweight='bold')
        axs[1,0].set_xlabel("Total Ratings Count (log scale)")
        axs[1,0].set_ylabel("Count")
        axs[1,0].set_xscale("log")  # Log scale to distinguish popular vs niche books
        axs[1,0].legend()
        
        # 4. Average Rating Distribution (with KDE to show skewness)
        sns.histplot(data=self.df_before, x="average_rating_weighted_mean", kde=True, color=ANTIQUE_COLORS[6], 
                     label="Before Cleaning", ax=axs[1,1], alpha=0.6, bins=30, stat="density")
        sns.histplot(data=self.df_after, x="average_rating_weighted_mean", kde=True, color=ANTIQUE_COLORS[7], 
                     label="After Cleaning", ax=axs[1,1], alpha=0.6, bins=30, stat="density")
        axs[1,1].set_title("Average Rating Distribution (Skewness Check)", fontsize=12, fontweight='bold')
        axs[1,1].set_xlabel("Weighted Average Rating")
        axs[1,1].set_ylabel("Density")
        axs[1,1].legend()
        
        plt.suptitle("Data Distribution: Before vs After Cleaning", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'figure_1_before_after_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Figure 1 saved to: {output_path}")
    
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
            
            summary_stats['numerical_variables_analysis'][var] = {
                'before_cleaning': {
                    'count': len(data_before),
                    'mean': round(data_before.mean(), 4),
                    'median': round(data_before.median(), 4),
                    'std': round(data_before.std(), 4),
                    'min': data_before.min(),
                    'max': data_before.max()
                },
                'after_cleaning': {
                    'count': len(data_after),
                    'mean': round(data_after.mean(), 4),
                    'median': round(data_after.median(), 4),
                    'std': round(data_after.std(), 4),
                    'min': data_after.min(),
                    'max': data_after.max()
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
        report_path = self.output_dir / 'improved_eda_final_summary.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Summary report saved to: {report_path}")
    
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
    
    def generate_all_visualizations(self):
        """Generate both essential figures."""
        print("Generating improved EDA final visualizations...")
        print("=" * 60)
        
        # Create Figure 1: Before/After Distributions
        self.create_figure_1_histograms()
        
        # Create Figure 2: Cleaned Data Summary
        self.create_figure_2_boxplots()
        
        # Generate and save summary statistics
        summary_stats = self.generate_summary_statistics()
        self.save_summary_report(summary_stats)
        self.print_summary(summary_stats)
        
        print("=" * 60)
        print("All improved EDA final visualizations generated successfully!")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main execution function."""
    # Set up paths - using proper before/after datasets
    data_path_before = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/romance_novels_text_preprocessed_20250906_213043.csv"
    data_path_after = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/archive/processed_data_20250109/cleaned_romance_novels_2000_2017_20250905_014541.csv"
    output_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/improved_eda_final"
    
    # Create visualizer
    visualizer = ImprovedEDAFinal(data_path_before, data_path_after, output_dir)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
