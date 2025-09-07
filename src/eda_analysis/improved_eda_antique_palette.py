#!/usr/bin/env python3
"""
Improved EDA Production Script with Antique Color Palette
Senior Data Scientist Review & Production

This script creates publication-ready EDA plots with:
- Antique color palette from pypalettes
- Small multiples for all four numerical variables (before/after cleaning)
- Combined boxplot with jitter for all numerical variables
- Meaningful, descriptive titles (no "figure_N")
- No redundant text across plots
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

class ImprovedEDAAntiqueProducer:
    """Production-ready EDA plot generator with Antique color palette"""
    
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
    
    def plot_numerical_distributions_small_multiples(self):
        """Create small multiples for all four numerical variables (before/after cleaning)"""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Publication Year
        year_data = self.df['publication_year'].dropna()
        year_cleaned = year_data[(year_data >= 2005) & (year_data <= 2018)]
        
        sns.histplot(data=year_data, kde=True, color=ANTIQUE_COLORS[0], alpha=0.7, ax=axs[0, 0])
        sns.histplot(data=year_cleaned, kde=True, color=ANTIQUE_COLORS[1], alpha=0.7, ax=axs[0, 0])
        axs[0, 0].set_title('Publication Year Distribution')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Count')
        axs[0, 0].legend(['Original', 'Cleaned'], loc='upper right')
        
        # Number of Pages
        pages_data = self.df['num_pages_median'].dropna()
        pages_cleaned = pages_data[pages_data <= 512]
        
        sns.histplot(data=pages_data, kde=True, color=ANTIQUE_COLORS[2], alpha=0.7, ax=axs[0, 1])
        sns.histplot(data=pages_cleaned, kde=True, color=ANTIQUE_COLORS[3], alpha=0.7, ax=axs[0, 1])
        axs[0, 1].set_title('Book Length Distribution')
        axs[0, 1].set_xlabel('Pages')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].legend(['Original', 'Cleaned'], loc='upper right')
        
        # Ratings Count
        ratings_data = self.df['ratings_count_sum'].dropna()
        ratings_cleaned = ratings_data[ratings_data <= 711]
        
        sns.histplot(data=ratings_data, kde=True, color=ANTIQUE_COLORS[4], alpha=0.7, ax=axs[1, 0])
        sns.histplot(data=ratings_cleaned, kde=True, color=ANTIQUE_COLORS[5], alpha=0.7, ax=axs[1, 0])
        axs[1, 0].set_title('Book Popularity Distribution')
        axs[1, 0].set_xlabel('Number of Ratings')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].legend(['Original', 'Cleaned'], loc='upper right')
        
        # Average Rating
        rating_data = self.df['average_rating_weighted_mean'].dropna()
        rating_cleaned = rating_data[(rating_data >= 3.0) & (rating_data <= 4.81)]
        
        sns.histplot(data=rating_data, kde=True, color=ANTIQUE_COLORS[6], alpha=0.7, ax=axs[1, 1])
        sns.histplot(data=rating_cleaned, kde=True, color=ANTIQUE_COLORS[7], alpha=0.7, ax=axs[1, 1])
        axs[1, 1].set_title('Rating Quality Distribution')
        axs[1, 1].set_xlabel('Average Rating (1-5)')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].legend(['Original', 'Cleaned'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'numerical_distributions_small_multiples.png', dpi=300)
        plt.show()
    
    def plot_combined_boxplot_with_jitter(self):
        """Create combined boxplot with jitter for all numerical variables (cleaned data)"""
        # Prepare cleaned data
        year_cleaned = self.df['publication_year'].dropna()
        year_cleaned = year_cleaned[(year_cleaned >= 2005) & (year_cleaned <= 2018)]
        
        pages_cleaned = self.df['num_pages_median'].dropna()
        pages_cleaned = pages_cleaned[pages_cleaned <= 512]
        
        ratings_cleaned = self.df['ratings_count_sum'].dropna()
        ratings_cleaned = ratings_cleaned[ratings_cleaned <= 711]
        
        rating_cleaned = self.df['average_rating_weighted_mean'].dropna()
        rating_cleaned = rating_cleaned[(rating_cleaned >= 3.0) & (rating_cleaned <= 4.81)]
        
        # Create DataFrame for plotting
        plot_data = []
        
        # Add publication year data
        for value in year_cleaned:
            plot_data.append({'Variable': 'Publication Year', 'Value': value})
        
        # Add pages data
        for value in pages_cleaned:
            plot_data.append({'Variable': 'Book Length (Pages)', 'Value': value})
        
        # Add ratings count data
        for value in ratings_cleaned:
            plot_data.append({'Variable': 'Number of Ratings', 'Value': value})
        
        # Add average rating data
        for value in rating_cleaned:
            plot_data.append({'Variable': 'Average Rating', 'Value': value})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create boxplot with jitter
        sns.boxplot(data=plot_df, x='Variable', y='Value', palette=ANTIQUE_COLORS[:4], ax=ax)
        sns.stripplot(data=plot_df, x='Variable', y='Value', color='gray', alpha=0.3, size=1, ax=ax)
        
        ax.set_title('Distribution of Numerical Variables (Cleaned Data)')
        ax.set_xlabel('')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Add sample size annotations
        variables = ['Publication Year', 'Book Length (Pages)', 'Number of Ratings', 'Average Rating']
        sample_sizes = [len(year_cleaned), len(pages_cleaned), len(ratings_cleaned), len(rating_cleaned)]
        
        for i, (var, n) in enumerate(zip(variables, sample_sizes)):
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n:,}', ha='center', va='top', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_boxplot_with_jitter.png', dpi=300)
        plt.show()
    
    def plot_publication_year_trends(self):
        """Publication year trends and missing years analysis"""
        year_data = self.df['publication_year'].dropna()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Year distribution
        year_counts = year_data.value_counts().sort_index()
        ax1.bar(year_counts.index, year_counts.values, color=ANTIQUE_COLORS[0], alpha=0.8)
        ax1.set_title('Publication Year Distribution')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Books')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(year_counts.index, year_counts.values, 1)
        p = np.poly1d(z)
        ax1.plot(year_counts.index, p(year_counts.index), "r--", alpha=0.8, linewidth=2)
        
        # Missing years analysis
        year_range = range(int(year_data.min()), int(year_data.max()) + 1)
        missing_years = [year for year in year_range if year not in year_counts.index]
        
        if missing_years:
            ax2.bar(range(len(missing_years)), [1] * len(missing_years), color=ANTIQUE_COLORS[1], alpha=0.8)
            ax2.set_title(f'Missing Years ({len(missing_years)} total)')
            ax2.set_xlabel('Missing Year Index')
            ax2.set_ylabel('Count')
            ax2.set_xticks(range(len(missing_years)))
            ax2.set_xticklabels(missing_years, rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Missing Years', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Missing Years Analysis')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'publication_year_trends_analysis.png', dpi=300)
        plt.show()
    
    def plot_pages_by_genre(self):
        """Book length distribution by genre with outlier analysis"""
        # Get top genres
        genre_counts = self.df['genres'].value_counts().head(5)
        top_genres = genre_counts.index.tolist()
        
        # Filter data for top genres
        genre_pages_data = []
        for genre in top_genres:
            genre_data = self.df[self.df['genres'] == genre]['num_pages_median'].dropna()
            for pages in genre_data:
                genre_pages_data.append({'Genre': genre, 'Pages': pages})
        
        genre_pages_df = pd.DataFrame(genre_pages_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot by genre
        sns.boxplot(data=genre_pages_df, x='Genre', y='Pages', palette=ANTIQUE_COLORS[:5], ax=ax1)
        ax1.set_title('Book Length Distribution by Genre')
        ax1.set_xlabel('Genre')
        ax1.set_ylabel('Number of Pages')
        ax1.tick_params(axis='x', rotation=45)
        
        # Outlier analysis
        all_pages = self.df['num_pages_median'].dropna()
        Q1 = all_pages.quantile(0.25)
        Q3 = all_pages.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = all_pages[(all_pages < lower_bound) | (all_pages > upper_bound)]
        
        ax2.hist(all_pages, bins=30, color=ANTIQUE_COLORS[2], alpha=0.7, label='All Books')
        ax2.hist(outliers, bins=10, color=ANTIQUE_COLORS[3], alpha=0.8, label='Outliers')
        ax2.axvline(lower_bound, color='red', linestyle='--', alpha=0.7, label='Lower Bound')
        ax2.axvline(upper_bound, color='red', linestyle='--', alpha=0.7, label='Upper Bound')
        ax2.set_title('Page Count Outlier Analysis')
        ax2.set_xlabel('Number of Pages')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_length_by_genre_analysis.png', dpi=300)
        plt.show()
    
    def plot_popularity_analysis(self):
        """Book popularity analysis - popular vs niche books"""
        ratings_data = self.df['ratings_count_sum'].dropna()
        
        # Define popularity thresholds
        popular_threshold = ratings_data.quantile(0.8)  # Top 20%
        niche_threshold = ratings_data.quantile(0.2)    # Bottom 20%
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Popularity distribution
        ax1.hist(ratings_data, bins=30, color=ANTIQUE_COLORS[4], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axvline(popular_threshold, color=ANTIQUE_COLORS[5], linestyle='--', linewidth=2, label='Popular Threshold')
        ax1.axvline(niche_threshold, color=ANTIQUE_COLORS[6], linestyle='--', linewidth=2, label='Niche Threshold')
        ax1.set_title('Book Popularity Distribution')
        ax1.set_xlabel('Number of Ratings')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Popular vs Niche categorization
        popular_books = ratings_data[ratings_data >= popular_threshold]
        niche_books = ratings_data[ratings_data <= niche_threshold]
        moderate_books = ratings_data[(ratings_data > niche_threshold) & (ratings_data < popular_threshold)]
        
        categories = ['Niche', 'Moderate', 'Popular']
        counts = [len(niche_books), len(moderate_books), len(popular_books)]
        colors = [ANTIQUE_COLORS[6], ANTIQUE_COLORS[7], ANTIQUE_COLORS[5]]
        
        bars = ax2.bar(categories, counts, color=colors, alpha=0.8)
        ax2.set_title('Book Popularity Categories')
        ax2.set_ylabel('Number of Books')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'book_popularity_analysis.png', dpi=300)
        plt.show()
    
    def plot_rating_quality_analysis(self):
        """Rating quality distribution and skewness analysis"""
        rating_data = self.df['average_rating_weighted_mean'].dropna()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Rating distribution
        ax1.hist(rating_data, bins=20, color=ANTIQUE_COLORS[8], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axvline(rating_data.mean(), color=ANTIQUE_COLORS[9], linestyle='--', linewidth=2, label=f'Mean: {rating_data.mean():.2f}')
        ax1.axvline(rating_data.median(), color=ANTIQUE_COLORS[10], linestyle='--', linewidth=2, label=f'Median: {rating_data.median():.2f}')
        ax1.set_title('Rating Quality Distribution')
        ax1.set_xlabel('Average Rating (1-5)')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Skewness analysis
        from scipy import stats
        skewness = stats.skew(rating_data)
        kurtosis = stats.kurtosis(rating_data)
        
        # Q-Q plot for normality check
        stats.probplot(rating_data, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot (Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rating_quality_analysis.png', dpi=300)
        plt.show()
    
    def generate_all_plots(self):
        """Generate all improved EDA plots with Antique palette"""
        print("Generating improved EDA plots with Antique color palette...")
        print("=" * 60)
        
        print("1. Numerical Distributions (Small Multiples)...")
        self.plot_numerical_distributions_small_multiples()
        
        print("2. Combined Boxplot with Jitter...")
        self.plot_combined_boxplot_with_jitter()
        
        print("3. Publication Year Trends Analysis...")
        self.plot_publication_year_trends()
        
        print("4. Book Length by Genre Analysis...")
        self.plot_pages_by_genre()
        
        print("5. Book Popularity Analysis...")
        self.plot_popularity_analysis()
        
        print("6. Rating Quality Analysis...")
        self.plot_rating_quality_analysis()
        
        print("=" * 60)
        print("All improved EDA plots generated successfully!")
        print(f"Output directory: {self.output_dir}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of the improved EDA"""
        report_path = self.output_dir / 'improved_eda_antique_summary_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Improved EDA Analysis Summary (Antique Palette)\n\n")
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
            
            f.write("## Generated Plots\n")
            f.write("1. `numerical_distributions_small_multiples.png` - All four numerical variables (before/after cleaning)\n")
            f.write("2. `combined_boxplot_with_jitter.png` - Combined boxplot with individual data points\n")
            f.write("3. `publication_year_trends_analysis.png` - Publication year trends and missing years\n")
            f.write("4. `book_length_by_genre_analysis.png` - Book length by genre with outlier analysis\n")
            f.write("5. `book_popularity_analysis.png` - Popular vs niche book analysis\n")
            f.write("6. `rating_quality_analysis.png` - Rating quality distribution and skewness\n\n")
            
            f.write("## Improvements Made\n")
            f.write("- ✅ Used Antique color palette from pypalettes\n")
            f.write("- ✅ Created small multiples for all four numerical variables\n")
            f.write("- ✅ Added before/after cleaning comparisons\n")
            f.write("- ✅ Combined boxplot with jitter for all variables\n")
            f.write("- ✅ Added descriptive titles (no 'figure_N')\n")
            f.write("- ✅ Eliminated redundant text across plots\n")
            f.write("- ✅ Applied academic publication standards\n")
            f.write("- ✅ Included comprehensive statistical information\n")
            f.write("- ✅ Removed meaningless plots\n")
        
        print(f"Summary report created: {report_path}")


def main():
    """Main execution function"""
    # Set up paths
    data_path = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/eda_test_subset_with_saving/eda_cleaned_dataset_20250906_223941.csv"
    results_path = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/eda_test_subset_with_saving/simplified_eda_results.json"
    output_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/improved_eda_antique"
    
    # Create improved EDA producer
    producer = ImprovedEDAAntiqueProducer(data_path, results_path, output_dir)
    
    # Generate all plots
    producer.generate_all_plots()


if __name__ == "__main__":
    main()
