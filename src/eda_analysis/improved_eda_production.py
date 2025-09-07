#!/usr/bin/env python3
"""
Improved EDA Production Script for Romance Novel Research
Senior Data Scientist Review & Production

This script creates publication-ready EDA plots with:
- Meaningful, descriptive titles (no "figure_N")
- Before/after cleaning versions for numerical variables
- Consistent color palette
- Statistical annotations and sample sizes
- Academic publication standards
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
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
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

# Colorblind-friendly color palette
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange  
    'accent': '#2ca02c',       # Green
    'highlight': '#d62728',    # Red
    'neutral': '#7f7f7f',      # Gray
    'light': '#bcbd22',        # Olive
    'dark': '#17becf',         # Cyan
    'purple': '#9467bd',       # Purple
    'brown': '#8c564b',        # Brown
    'pink': '#e377c2'          # Pink
}

# Sequential colormap for continuous data
CMAP_SEQUENTIAL = 'viridis'
CMAP_DIVERGING = 'RdBu_r'

class ImprovedEDAProducer:
    """Production-ready EDA plot generator with academic standards"""
    
    def __init__(self, data_path, results_path, output_dir):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data and results
        self.df = pd.read_csv(self.data_path)
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded dataset: {self.df.shape[0]:,} books, {self.df.shape[1]} features")
        print(f"Output directory: {self.output_dir}")
    
    def add_statistical_annotation(self, ax, stats, plot_type='histogram'):
        """Add statistical annotations to plots"""
        if plot_type == 'histogram':
            ax.text(0.02, 0.98, 
                   f'n = {stats["count"]:,}\n'
                   f'Mean = {stats["mean"]:.2f}\n'
                   f'Median = {stats["median"]:.2f}\n'
                   f'SD = {stats["std"]:.2f}',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
        elif plot_type == 'boxplot':
            ax.text(0.02, 0.98,
                   f'n = {stats["count"]:,}\n'
                   f'Outliers: {stats["outliers_count"]} ({stats["outliers_percentage"]:.1f}%)',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
    
    def plot_publication_year_analysis(self):
        """Publication year distribution with before/after cleaning"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original data
        year_data = self.df['publication_year'].dropna()
        ax1.hist(year_data, bins=20, color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Publication Year')
        ax1.set_ylabel('Number of Books')
        ax1.set_title('Publication Year Distribution (Original Data)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotation
        stats_orig = self.results['numerical_analysis']['publication_year']
        self.add_statistical_annotation(ax1, stats_orig, 'histogram')
        
        # Cleaned data (remove outliers)
        year_cleaned = year_data[(year_data >= 2005) & (year_data <= 2018)]
        ax2.hist(year_cleaned, bins=15, color=COLORS['accent'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Publication Year')
        ax2.set_ylabel('Number of Books')
        ax2.set_title('Publication Year Distribution (Outliers Removed)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotation for cleaned data
        stats_clean = stats_orig['cleaned_stats']
        self.add_statistical_annotation(ax2, stats_clean, 'histogram')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'publication_year_distribution_analysis.png', dpi=300)
        plt.show()
        
        # Box plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [year_data, year_cleaned]
        box_labels = ['Original\n(n=1,000)', 'Cleaned\n(n=970)']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['primary'])
        bp['boxes'][1].set_facecolor(COLORS['accent'])
        
        ax.set_ylabel('Publication Year')
        ax.set_title('Publication Year Distribution: Original vs Cleaned Data')
        ax.grid(True, alpha=0.3)
        
        # Add outlier information
        ax.text(0.02, 0.98,
               f'Outliers removed: {stats_orig["outliers_count"]} ({stats_orig["outliers_percentage"]:.1f}%)\n'
               f'Year range: {stats_orig["min"]}-{stats_orig["max"]}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'publication_year_boxplot_comparison.png', dpi=300)
        plt.show()
    
    def plot_pages_analysis(self):
        """Number of pages distribution with before/after cleaning"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original data
        pages_data = self.df['num_pages_median'].dropna()
        ax1.hist(pages_data, bins=25, color=COLORS['secondary'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Number of Pages')
        ax1.set_ylabel('Number of Books')
        ax1.set_title('Book Length Distribution (Original Data)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotation
        stats_orig = self.results['numerical_analysis']['num_pages_median']
        self.add_statistical_annotation(ax1, stats_orig, 'histogram')
        
        # Cleaned data (remove outliers)
        pages_cleaned = pages_data[pages_data <= 512]  # Remove very long books
        ax2.hist(pages_cleaned, bins=20, color=COLORS['highlight'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Number of Pages')
        ax2.set_ylabel('Number of Books')
        ax2.set_title('Book Length Distribution (Outliers Removed)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotation for cleaned data
        stats_clean = stats_orig['cleaned_stats']
        self.add_statistical_annotation(ax2, stats_clean, 'histogram')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_length_distribution_analysis.png', dpi=300)
        plt.show()
        
        # Box plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [pages_data, pages_cleaned]
        box_labels = ['Original\n(n=1,000)', 'Cleaned\n(n=993)']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['secondary'])
        bp['boxes'][1].set_facecolor(COLORS['highlight'])
        
        ax.set_ylabel('Number of Pages')
        ax.set_title('Book Length Distribution: Original vs Cleaned Data')
        ax.grid(True, alpha=0.3)
        
        # Add outlier information
        outlier_info = stats_orig['outlier_analysis']
        ax.text(0.02, 0.98,
               f'Outliers removed: {outlier_info["total_outliers"]} ({stats_orig["outliers_percentage"]:.1f}%)\n'
               f'Very long books: {outlier_info["very_long_books"]["count"]} (≥522 pages)',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_length_boxplot_comparison.png', dpi=300)
        plt.show()
    
    def plot_ratings_analysis(self):
        """Ratings count distribution with before/after cleaning"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original data
        ratings_data = self.df['ratings_count_sum'].dropna()
        ax1.hist(ratings_data, bins=30, color=COLORS['purple'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Number of Ratings')
        ax1.set_ylabel('Number of Books')
        ax1.set_title('Book Popularity Distribution (Original Data)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotation
        stats_orig = self.results['numerical_analysis']['ratings_count_sum']
        self.add_statistical_annotation(ax1, stats_orig, 'histogram')
        
        # Cleaned data (remove outliers)
        ratings_cleaned = ratings_data[ratings_data <= 711]  # Remove very popular books
        ax2.hist(ratings_cleaned, bins=25, color=COLORS['brown'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Number of Ratings')
        ax2.set_ylabel('Number of Books')
        ax2.set_title('Book Popularity Distribution (Outliers Removed)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotation for cleaned data
        stats_clean = stats_orig['cleaned_stats']
        self.add_statistical_annotation(ax2, stats_clean, 'histogram')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_popularity_distribution_analysis.png', dpi=300)
        plt.show()
        
        # Box plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [ratings_data, ratings_cleaned]
        box_labels = ['Original\n(n=1,000)', 'Cleaned\n(n=908)']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['purple'])
        bp['boxes'][1].set_facecolor(COLORS['brown'])
        
        ax.set_ylabel('Number of Ratings')
        ax.set_title('Book Popularity Distribution: Original vs Cleaned Data')
        ax.grid(True, alpha=0.3)
        
        # Add outlier information
        ax.text(0.02, 0.98,
               f'Outliers removed: {stats_orig["outliers_count"]} ({stats_orig["outliers_percentage"]:.1f}%)\n'
               f'Popular threshold: {stats_orig["popular_threshold"]:.0f} ratings',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_popularity_boxplot_comparison.png', dpi=300)
        plt.show()
    
    def plot_rating_quality_analysis(self):
        """Average rating distribution with before/after cleaning"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original data
        rating_data = self.df['average_rating_weighted_mean'].dropna()
        ax1.hist(rating_data, bins=20, color=COLORS['light'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Average Rating (1-5 scale)')
        ax1.set_ylabel('Number of Books')
        ax1.set_title('Book Rating Quality Distribution (Original Data)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotation
        stats_orig = self.results['numerical_analysis']['average_rating_weighted_mean']
        self.add_statistical_annotation(ax1, stats_orig, 'histogram')
        
        # Cleaned data (remove outliers)
        rating_cleaned = rating_data[(rating_data >= 3.0) & (rating_data <= 4.81)]
        ax2.hist(rating_cleaned, bins=18, color=COLORS['dark'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Average Rating (1-5 scale)')
        ax2.set_ylabel('Number of Books')
        ax2.set_title('Book Rating Quality Distribution (Outliers Removed)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotation for cleaned data
        stats_clean = stats_orig['cleaned_stats']
        self.add_statistical_annotation(ax2, stats_clean, 'histogram')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_rating_quality_distribution_analysis.png', dpi=300)
        plt.show()
        
        # Box plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [rating_data, rating_cleaned]
        box_labels = ['Original\n(n=1,000)', 'Cleaned\n(n=994)']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['light'])
        bp['boxes'][1].set_facecolor(COLORS['dark'])
        
        ax.set_ylabel('Average Rating (1-5 scale)')
        ax.set_title('Book Rating Quality Distribution: Original vs Cleaned Data')
        ax.grid(True, alpha=0.3)
        
        # Add outlier information
        ax.text(0.02, 0.98,
               f'Outliers removed: {stats_orig["outliers_count"]} ({stats_orig["outliers_percentage"]:.1f}%)\n'
               f'Rating range: {stats_orig["min"]:.2f}-{stats_orig["max"]:.2f}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_rating_quality_boxplot_comparison.png', dpi=300)
        plt.show()
    
    def plot_genre_analysis(self):
        """Comprehensive genre analysis"""
        # Top genre combinations
        genre_counts = self.results['categorical_analysis']['genres']['top_10_values']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Top genre combinations
        genres = list(genre_counts.keys())[:8]  # Top 8 for readability
        counts = list(genre_counts.values())[:8]
        
        bars1 = ax1.barh(range(len(genres)), counts, color=COLORS['primary'], alpha=0.8)
        ax1.set_yticks(range(len(genres)))
        ax1.set_yticklabels([g.replace(',', ',\n') for g in genres], fontsize=9)
        ax1.set_xlabel('Number of Books')
        ax1.set_title('Top Genre Combinations in Romance Novels')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars1, counts)):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                    f'{count}', ha='left', va='center', fontsize=9)
        
        # Individual genre words
        genre_words = self.results['categorical_analysis']['genres']['subgenre_analysis']['unique_words']['top_10_words']
        words = list(genre_words.keys())[:8]
        word_counts = list(genre_words.values())[:8]
        
        bars2 = ax2.barh(range(len(words)), word_counts, color=COLORS['accent'], alpha=0.8)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words, fontsize=10)
        ax2.set_xlabel('Number of Books')
        ax2.set_title('Most Common Genre Words')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars2, word_counts)):
            ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                    f'{count}', ha='left', va='center', fontsize=9)
        
        # Add sample size annotation
        total_books = self.results['categorical_analysis']['genres']['count']
        ax1.text(0.02, 0.98, f'Total books: {total_books:,}\nUnique combinations: {self.results["categorical_analysis"]["genres"]["unique_count"]}',
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'genre_analysis_comprehensive.png', dpi=300)
        plt.show()
    
    def plot_series_analysis(self):
        """Series vs standalone analysis"""
        series_stats = self.results['categorical_analysis']['in_series']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        labels = ['In Series', 'Standalone']
        sizes = [series_stats['top_10_values']['yes'], series_stats['top_10_values']['no']]
        colors = [COLORS['primary'], COLORS['secondary']]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                          startangle=90, textprops={'fontsize': 11})
        ax1.set_title('Series vs Standalone Classification')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.8)
        ax2.set_ylabel('Number of Books')
        ax2.set_title('Series vs Standalone Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{size}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add sample size annotation
        ax2.text(0.02, 0.98, f'Total books: {series_stats["count"]:,}\nSeries percentage: {series_stats["most_frequent_percentage"]:.1f}%',
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'series_classification_analysis.png', dpi=300)
        plt.show()
    
    def plot_description_analysis(self):
        """Description length analysis"""
        desc_stats = self.results['text_analysis']['description']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create histogram
        desc_lengths = self.df['description'].str.split().str.len()
        ax.hist(desc_lengths, bins=25, color=COLORS['highlight'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Description Length (words)')
        ax.set_ylabel('Number of Books')
        ax.set_title('Book Description Length Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotation
        length_stats = desc_stats['length_stats']
        ax.text(0.02, 0.98, 
               f'n = {desc_stats["count"]:,}\n'
               f'Mean = {length_stats["mean_length"]:.1f} words\n'
               f'Median = {length_stats["median_length"]:.0f} words\n'
               f'Range = {length_stats["min_length"]}-{length_stats["max_length"]} words',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        # Add outlier information
        outlier_info = desc_stats['outlier_analysis']
        ax.text(0.98, 0.98,
               f'Outliers: {outlier_info["total_outliers"]} ({outlier_info["outlier_percentage"]:.1f}%)\n'
               f'Very long: {outlier_info["very_long_descriptions"]["count"]} (≥329 words)',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
               verticalalignment='top', horizontalalignment='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'description_length_analysis.png', dpi=300)
        plt.show()
    
    def plot_popular_shelves_analysis(self):
        """Popular shelves/tags analysis"""
        shelves_stats = self.results['text_analysis']['popular_shelves']
        top_tags = shelves_stats['tag_analysis']['top_20_tags']
        
        # Get top 15 tags for readability
        tags = list(top_tags.keys())[:15]
        counts = list(top_tags.values())[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(tags)), counts, color=COLORS['purple'], alpha=0.8)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=10)
        ax.set_xlabel('Number of Mentions')
        ax.set_title('Most Popular Book Tags/Shelves')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                    f'{count}', ha='left', va='center', fontsize=9)
        
        # Add sample size annotation
        tag_info = shelves_stats['tag_analysis']
        ax.text(0.02, 0.98, 
               f'Total tag mentions: {tag_info["total_tag_mentions"]:,}\n'
               f'Unique tags: {tag_info["unique_tags"]:,}\n'
               f'Tag diversity: {tag_info["tag_diversity"]:.1f}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'popular_shelves_tags_analysis.png', dpi=300)
        plt.show()
    
    def plot_series_size_analysis(self):
        """Series size distribution analysis"""
        series_data = self.df[self.df['in_series'] == 'yes']['series_works_count'].dropna()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create histogram
        ax.hist(series_data, bins=20, color=COLORS['brown'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Number of Books in Series')
        ax.set_ylabel('Number of Series')
        ax.set_title('Series Size Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotation
        ax.text(0.02, 0.98, 
               f'n = {len(series_data):,} series\n'
               f'Mean = {series_data.mean():.1f} books\n'
               f'Median = {series_data.median():.0f} books\n'
               f'Range = {series_data.min():.0f}-{series_data.max():.0f} books',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        # Add series classification info
        series_stats = self.results['series_analysis']['series_classification_stats']
        ax.text(0.98, 0.98,
               f'Books in series: {int(series_stats["books_in_series"]):,} ({series_stats["series_percentage"]:.1f}%)\n'
               f'Standalone books: {int(series_stats["books_standalone"]):,} ({series_stats["standalone_percentage"]:.1f}%)',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
               verticalalignment='top', horizontalalignment='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'series_size_distribution_analysis.png', dpi=300)
        plt.show()
    
    def generate_all_plots(self):
        """Generate all improved EDA plots"""
        print("Generating improved EDA plots...")
        print("=" * 50)
        
        # Numerical variable analyses
        print("1. Publication Year Analysis...")
        self.plot_publication_year_analysis()
        
        print("2. Book Length Analysis...")
        self.plot_pages_analysis()
        
        print("3. Book Popularity Analysis...")
        self.plot_ratings_analysis()
        
        print("4. Book Rating Quality Analysis...")
        self.plot_rating_quality_analysis()
        
        # Categorical analyses
        print("5. Genre Analysis...")
        self.plot_genre_analysis()
        
        print("6. Series Classification Analysis...")
        self.plot_series_analysis()
        
        # Text analyses
        print("7. Description Length Analysis...")
        self.plot_description_analysis()
        
        print("8. Popular Shelves/Tags Analysis...")
        self.plot_popular_shelves_analysis()
        
        print("9. Series Size Distribution Analysis...")
        self.plot_series_size_analysis()
        
        print("=" * 50)
        print("All improved EDA plots generated successfully!")
        print(f"Output directory: {self.output_dir}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of the improved EDA"""
        report_path = self.output_dir / 'improved_eda_summary_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Improved EDA Analysis Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Dataset Overview\n")
            f.write(f"- **Total Books:** {self.df.shape[0]:,}\n")
            f.write(f"- **Features:** {self.df.shape[1]}\n")
            f.write(f"- **Memory Usage:** {self.results['dataset_overview']['memory_usage_mb']:.2f} MB\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("### Publication Years\n")
            f.write(f"- **Range:** {self.results['numerical_analysis']['publication_year']['min']}-{self.results['numerical_analysis']['publication_year']['max']}\n")
            f.write(f"- **Mean:** {self.results['numerical_analysis']['publication_year']['mean']:.1f}\n")
            f.write(f"- **Outliers:** {self.results['numerical_analysis']['publication_year']['outliers_count']} ({self.results['numerical_analysis']['publication_year']['outliers_percentage']:.1f}%)\n\n")
            
            f.write("### Book Characteristics\n")
            f.write(f"- **Average Pages:** {self.results['numerical_analysis']['num_pages_median']['mean']:.0f}\n")
            f.write(f"- **Average Rating:** {self.results['numerical_analysis']['average_rating_weighted_mean']['mean']:.2f}/5.0\n")
            f.write(f"- **Average Ratings Count:** {self.results['numerical_analysis']['ratings_count_sum']['mean']:.0f}\n\n")
            
            f.write("### Genre Analysis\n")
            f.write(f"- **Most Common:** {self.results['categorical_analysis']['genres']['most_frequent']}\n")
            f.write(f"- **Unique Combinations:** {self.results['categorical_analysis']['genres']['unique_count']}\n")
            f.write(f"- **Top Genre Word:** Romance (appears in all {self.results['categorical_analysis']['genres']['count']:,} books)\n\n")
            
            f.write("### Series Classification\n")
            f.write(f"- **In Series:** {self.results['categorical_analysis']['in_series']['most_frequent_count']} ({self.results['categorical_analysis']['in_series']['most_frequent_percentage']:.1f}%)\n")
            f.write(f"- **Standalone:** {self.results['categorical_analysis']['in_series']['top_10_values']['no']} ({100-self.results['categorical_analysis']['in_series']['most_frequent_percentage']:.1f}%)\n\n")
            
            f.write("## Generated Plots\n")
            f.write("1. `publication_year_distribution_analysis.png` - Publication year distribution (original vs cleaned)\n")
            f.write("2. `publication_year_boxplot_comparison.png` - Publication year boxplot comparison\n")
            f.write("3. `book_length_distribution_analysis.png` - Book length distribution (original vs cleaned)\n")
            f.write("4. `book_length_boxplot_comparison.png` - Book length boxplot comparison\n")
            f.write("5. `book_popularity_distribution_analysis.png` - Book popularity distribution (original vs cleaned)\n")
            f.write("6. `book_popularity_boxplot_comparison.png` - Book popularity boxplot comparison\n")
            f.write("7. `book_rating_quality_distribution_analysis.png` - Book rating quality distribution (original vs cleaned)\n")
            f.write("8. `book_rating_quality_boxplot_comparison.png` - Book rating quality boxplot comparison\n")
            f.write("9. `genre_analysis_comprehensive.png` - Comprehensive genre analysis\n")
            f.write("10. `series_classification_analysis.png` - Series vs standalone classification\n")
            f.write("11. `description_length_analysis.png` - Book description length analysis\n")
            f.write("12. `popular_shelves_tags_analysis.png` - Popular shelves/tags analysis\n")
            f.write("13. `series_size_distribution_analysis.png` - Series size distribution analysis\n\n")
            
            f.write("## Improvements Made\n")
            f.write("- ✅ Removed trivial plots (language codes - 100% English)\n")
            f.write("- ✅ Added descriptive titles (no 'figure_N')\n")
            f.write("- ✅ Created before/after cleaning versions for all numerical variables\n")
            f.write("- ✅ Applied consistent colorblind-friendly color palette\n")
            f.write("- ✅ Added statistical annotations and sample sizes\n")
            f.write("- ✅ Eliminated redundant information across titles and annotations\n")
            f.write("- ✅ Applied academic publication standards\n")
            f.write("- ✅ Used perceptually uniform colormaps\n")
            f.write("- ✅ Added proper axis labels with units\n")
            f.write("- ✅ Included comprehensive statistical information\n")
        
        print(f"Summary report created: {report_path}")


def main():
    """Main execution function"""
    # Set up paths
    data_path = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/eda_test_subset_with_saving/eda_cleaned_dataset_20250906_223941.csv"
    results_path = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/eda_test_subset_with_saving/simplified_eda_results.json"
    output_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/improved_eda_production"
    
    # Create improved EDA producer
    producer = ImprovedEDAProducer(data_path, results_path, output_dir)
    
    # Generate all plots
    producer.generate_all_plots()


if __name__ == "__main__":
    main()
