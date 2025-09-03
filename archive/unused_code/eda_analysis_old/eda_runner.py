#!/usr/bin/env python3
"""
EDA Runner Module for Romance Novel NLP Research
Consolidates EDA analysis scripts into a single module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class EDARunner:
    """
    Comprehensive EDA runner for romance novel datasets.
    
    Consolidates all EDA analysis into a single, organized module.
    """
    
    def __init__(self, dataset_path: str, output_dir: str = "outputs/eda"):
        """
        Initialize the EDA runner.
        
        Args:
            dataset_path: Path to the dataset CSV file
            output_dir: Directory for EDA outputs
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.df = self._load_dataset()
        
        # Analysis results storage
        self.analysis_results = {}
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load the dataset for analysis."""
        print(f"📊 Loading dataset from {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    
    def run_complete_eda(self) -> Dict[str, Any]:
        """
        Run the complete EDA analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🔍 Starting comprehensive EDA analysis...")
        
        # Run all analysis steps
        self.analysis_results = {
            'dataset_overview': self.analyze_dataset_overview(),
            'missing_values': self.analyze_missing_values(),
            'title_analysis': self.analyze_titles(),
            'series_patterns': self.analyze_series_patterns(),
            'author_analysis': self.analyze_authors(),
            'description_analysis': self.analyze_descriptions(),
            'publication_analysis': self.analyze_publication_trends(),
            'text_analysis': self.analyze_text_columns()
        }
        
        # Generate summary report
        self._generate_eda_summary()
        
        print("✅ Complete EDA analysis finished!")
        return self.analysis_results
    
    def analyze_dataset_overview(self) -> Dict[str, Any]:
        """Analyze basic dataset characteristics."""
        print("\n📊 Analyzing dataset overview...")
        
        results = {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'dtypes': self.df.dtypes.value_counts().to_dict(),
            'columns': list(self.df.columns)
        }
        
        print(f"  - Shape: {results['shape']}")
        print(f"  - Memory usage: {results['memory_usage']:.1f} MB")
        print(f"  - Data types: {dict(results['dtypes'])}")
        
        return results
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        print("\n🔍 Analyzing missing values...")
        
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        results = {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
        
        print(f"  - Columns with missing values: {len(results['columns_with_missing'])}")
        print(f"  - Most missing: {missing_counts.idxmax()} ({missing_percentages.max():.1f}%)")
        
        return results
    
    def analyze_titles(self) -> Dict[str, Any]:
        """Analyze book titles."""
        print("\n📚 Analyzing book titles...")
        
        title_lengths = self.df['title'].str.len()
        title_word_counts = self.df['title'].str.split().str.len()
        
        results = {
            'title_length_stats': {
                'mean': title_lengths.mean(),
                'median': title_lengths.median(),
                'std': title_lengths.std(),
                'min': title_lengths.min(),
                'max': title_lengths.max()
            },
            'title_word_stats': {
                'mean': title_word_counts.mean(),
                'median': title_word_counts.median(),
                'std': title_word_counts.std(),
                'min': title_word_counts.min(),
                'max': title_word_counts.max()
            },
            'sample_titles': self.df['title'].head(10).tolist()
        }
        
        print(f"  - Average title length: {results['title_length_stats']['mean']:.1f} characters")
        print(f"  - Average word count: {results['title_word_stats']['mean']:.1f} words")
        
        return results
    
    def analyze_series_patterns(self) -> Dict[str, Any]:
        """Analyze series patterns in titles."""
        print("\n🔗 Analyzing series patterns...")
        
        # Check series coverage
        series_coverage = self.df['series_id'].notna().sum()
        series_percentage = (series_coverage / len(self.df)) * 100
        
        # Look for series patterns in titles
        series_patterns = [
            r'\b(Book|Volume|Part)\s+\d+',
            r'\b\d+\s*[:\-]',
            r'\b\d+\s*(?:st|nd|rd|th)',
            r'\s#\d+\s*$'
        ]
        
        pattern_matches = {}
        for pattern in series_patterns:
            matches = self.df['title'].str.contains(pattern, regex=True, na=False).sum()
            pattern_matches[pattern] = matches
        
        results = {
            'series_coverage': series_coverage,
            'series_percentage': series_percentage,
            'pattern_matches': pattern_matches,
            'sample_series_titles': self.df[self.df['series_id'].notna()]['title'].head(5).tolist()
        }
        
        print(f"  - Series coverage: {series_coverage:,} ({series_percentage:.1f}%)")
        print(f"  - Pattern matches found: {sum(pattern_matches.values()):,}")
        
        return results
    
    def analyze_authors(self) -> Dict[str, Any]:
        """Analyze author information."""
        print("\n👤 Analyzing author information...")
        
        unique_authors = self.df['author_name'].nunique()
        books_per_author = self.df.groupby('author_name').size()
        
        results = {
            'unique_authors': unique_authors,
            'books_per_author_stats': {
                'mean': books_per_author.mean(),
                'median': books_per_author.median(),
                'std': books_per_author.std(),
                'max': books_per_author.max()
            },
            'top_authors': books_per_author.nlargest(10).to_dict(),
            'author_name_lengths': self.df['author_name'].str.len().describe().to_dict()
        }
        
        print(f"  - Unique authors: {unique_authors:,}")
        print(f"  - Average books per author: {results['books_per_author_stats']['mean']:.1f}")
        
        return results
    
    def analyze_descriptions(self) -> Dict[str, Any]:
        """Analyze book descriptions."""
        print("\n📖 Analyzing book descriptions...")
        
        description_lengths = self.df['description'].str.len()
        description_word_counts = self.df['description'].str.split().str.len()
        
        results = {
            'description_length_stats': {
                'mean': description_lengths.mean(),
                'median': description_lengths.median(),
                'std': description_lengths.std(),
                'min': description_lengths.min(),
                'max': description_lengths.max()
            },
            'description_word_stats': {
                'mean': description_word_counts.mean(),
                'median': description_word_counts.median(),
                'std': description_word_counts.std(),
                'min': description_word_counts.min(),
                'max': description_word_counts.max()
            },
            'missing_descriptions': self.df['description'].isnull().sum(),
            'short_descriptions': (description_lengths < 50).sum()
        }
        
        print(f"  - Average description length: {results['description_length_stats']['mean']:.1f} characters")
        print(f"  - Missing descriptions: {results['missing_descriptions']:,}")
        
        return results
    
    def analyze_publication_trends(self) -> Dict[str, Any]:
        """Analyze publication trends and popularity."""
        print("\n📅 Analyzing publication trends...")
        
        year_distribution = self.df['publication_year'].value_counts().sort_index()
        ratings_stats = self.df['average_rating_weighted_mean'].describe()
        
        results = {
            'year_distribution': year_distribution.to_dict(),
            'year_range': {
                'min': self.df['publication_year'].min(),
                'max': self.df['publication_year'].max()
            },
            'ratings_stats': ratings_stats.to_dict(),
            'popular_years': year_distribution.nlargest(5).to_dict()
        }
        
        print(f"  - Publication years: {results['year_range']['min']} - {results['year_range']['max']}")
        print(f"  - Average rating: {results['ratings_stats']['mean']:.2f}")
        
        return results
    
    def analyze_text_columns(self) -> Dict[str, Any]:
        """Analyze text columns like popular_shelves."""
        print("\n📝 Analyzing text columns...")
        
        # Analyze popular_shelves
        popular_shelves = self.df['popular_shelves'].dropna()
        
        if len(popular_shelves) > 0:
            # Sample analysis
            sample_shelves = popular_shelves.head(3).tolist()
            
            # Count common patterns
            all_shelves = []
            for shelves in popular_shelves:
                if isinstance(shelves, str):
                    shelf_list = [s.strip() for s in shelves.split(',')]
                    all_shelves.extend(shelf_list)
            
            shelf_counts = pd.Series(all_shelves).value_counts()
            
            results = {
                'sample_shelves': sample_shelves,
                'total_shelves': len(all_shelves),
                'unique_shelves': len(shelf_counts),
                'top_shelves': shelf_counts.head(10).to_dict(),
                'shelf_format': 'comma_separated'
            }
        else:
            results = {
                'sample_shelves': [],
                'total_shelves': 0,
                'unique_shelves': 0,
                'top_shelves': {},
                'shelf_format': 'none'
            }
        
        print(f"  - Total shelf entries: {results['total_shelves']:,}")
        print(f"  - Unique shelves: {results['unique_shelves']:,}")
        
        return results
    
    def _generate_eda_summary(self) -> None:
        """Generate a summary report of the EDA analysis."""
        summary_path = self.output_dir / "eda_summary_report.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ROMANCE NOVEL DATASET EDA SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_path.name}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Records: {len(self.df):,}\n")
            f.write(f"Columns: {len(self.df.columns)}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Shape: {self.analysis_results['dataset_overview']['shape']}\n")
            f.write(f"Memory Usage: {self.analysis_results['dataset_overview']['memory_usage']:.1f} MB\n\n")
            
            # Missing Values
            f.write("MISSING VALUES:\n")
            f.write("-" * 20 + "\n")
            missing_analysis = self.analysis_results['missing_values']
            f.write(f"Columns with missing values: {len(missing_analysis['columns_with_missing'])}\n")
            if missing_analysis['columns_with_missing']:
                f.write("Top missing columns:\n")
                for col in missing_analysis['columns_with_missing'][:5]:
                    percentage = missing_analysis['missing_percentages'][col]
                    f.write(f"  - {col}: {percentage:.1f}%\n")
            f.write("\n")
            
            # Title Analysis
            f.write("TITLE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            title_analysis = self.analysis_results['title_analysis']
            f.write(f"Average title length: {title_analysis['title_length_stats']['mean']:.1f} characters\n")
            f.write(f"Average word count: {title_analysis['title_word_stats']['mean']:.1f} words\n\n")
            
            # Series Analysis
            f.write("SERIES ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            series_analysis = self.analysis_results['series_patterns']
            f.write(f"Series coverage: {series_analysis['series_coverage']:,} ({series_analysis['series_percentage']:.1f}%)\n")
            f.write(f"Pattern matches in titles: {sum(series_analysis['pattern_matches'].values()):,}\n\n")
            
            # Author Analysis
            f.write("AUTHOR ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            author_analysis = self.analysis_results['author_analysis']
            f.write(f"Unique authors: {author_analysis['unique_authors']:,}\n")
            f.write(f"Average books per author: {author_analysis['books_per_author_stats']['mean']:.1f}\n\n")
            
            # Description Analysis
            f.write("DESCRIPTION ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            desc_analysis = self.analysis_results['description_analysis']
            f.write(f"Average description length: {desc_analysis['description_length_stats']['mean']:.1f} characters\n")
            f.write(f"Missing descriptions: {desc_analysis['missing_descriptions']:,}\n\n")
            
            # Publication Analysis
            f.write("PUBLICATION ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            pub_analysis = self.analysis_results['publication_analysis']
            f.write(f"Publication years: {pub_analysis['year_range']['min']} - {pub_analysis['year_range']['max']}\n")
            f.write(f"Average rating: {pub_analysis['ratings_stats']['mean']:.2f}\n\n")
            
            # Text Analysis
            f.write("TEXT COLUMN ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            text_analysis = self.analysis_results['text_analysis']
            f.write(f"Total shelf entries: {text_analysis['total_shelves']:,}\n")
            f.write(f"Unique shelves: {text_analysis['unique_shelves']:,}\n")
            f.write(f"Format: {text_analysis['shelf_format']}\n")
        
        print(f"📄 EDA summary report saved to: {summary_path}")
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get the analysis results."""
        return self.analysis_results.copy()

def main():
    """Main function to run EDA analysis."""
    # Default dataset path
    dataset_path = "data/processed/final_books_2000_2020_en_20250901_024106.csv"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Initialize and run EDA
    eda_runner = EDARunner(dataset_path)
    results = eda_runner.run_complete_eda()
    
    print("\n🎉 EDA analysis completed successfully!")
    print(f"📁 Results saved to: {eda_runner.output_dir}")

if __name__ == "__main__":
    main()
