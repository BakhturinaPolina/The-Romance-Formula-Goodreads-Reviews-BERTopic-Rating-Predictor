#!/usr/bin/env python3
"""
Comprehensive Data Analysis and Cleaning Script
Addresses multiple data quality issues in the romance novel dataset.

Issues to address:
1. Remove books outside 2000-2017 range (122 books)
2. Delete books with missing descriptions
3. Exclude books with missing pages
4. Verify work_id/book_id edition grouping and statistics
5. Analyze authors with low ratings count
6. Check publication_year values for non-English books
7. Fix negative work count calculation error
8. Analyze and suggest cuts for ratings_count_sum and text_reviews_count_sum
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDataAnalyzer:
    """Comprehensive data analysis and cleaning for romance novel dataset."""
    
    def __init__(self, data_path: str = None):
        """Initialize the analyzer with data path."""
        if data_path is None:
            # Try to find the most recent processed data
            processed_dir = Path("data/processed")
            csv_files = list(processed_dir.glob("final_books_2000_2020_en_enhanced_*.csv"))
            if csv_files:
                # Get the most recent file
                data_path = str(max(csv_files, key=lambda x: x.stat().st_mtime))
                logger.info(f"Using most recent CSV file: {data_path}")
            else:
                raise FileNotFoundError("No processed CSV files found in data/processed/")
        
        self.data_path = Path(data_path)
        self.df = None
        self.analysis_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Try to load from pickle first (faster and preserves data types)
            pickle_path = self.data_path.with_suffix('.pkl')
            if pickle_path.exists():
                logger.info("Loading from pickle file for better performance")
                with open(pickle_path, 'rb') as f:
                    self.df = pickle.load(f)
            else:
                logger.info("Loading from CSV file")
                self.df = pd.read_csv(self.data_path)
                
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_publication_years(self) -> dict:
        """Analyze publication years and identify books outside 2000-2017 range."""
        logger.info("Analyzing publication years...")
        
        results = {
            'total_books': len(self.df),
            'year_range': {
                'min': self.df['publication_year'].min(),
                'max': self.df['publication_year'].max()
            },
            'year_distribution': self.df['publication_year'].value_counts().sort_index().to_dict(),
            'books_outside_2000_2017': None,
            'books_in_2018_2020': None
        }
        
        # Find books outside 2000-2017 range
        outside_range = self.df[~self.df['publication_year'].between(2000, 2017, inclusive='both')]
        results['books_outside_2000_2017'] = {
            'count': len(outside_range),
            'years': outside_range['publication_year'].value_counts().sort_index().to_dict()
        }
        
        # Specifically find books in 2018-2020
        books_2018_2020 = self.df[self.df['publication_year'].between(2018, 2020, inclusive='both')]
        results['books_in_2018_2020'] = {
            'count': len(books_2018_2020),
            'years': books_2018_2020['publication_year'].value_counts().sort_index().to_dict()
        }
        
        logger.info(f"Books outside 2000-2017: {results['books_outside_2000_2017']['count']}")
        logger.info(f"Books in 2018-2020: {results['books_in_2018_2020']['count']}")
        
        return results
    
    def analyze_missing_descriptions(self) -> dict:
        """Analyze books with missing descriptions."""
        logger.info("Analyzing missing descriptions...")
        
        results = {
            'total_books': len(self.df),
            'missing_descriptions': {
                'count': self.df['description'].isna().sum(),
                'percentage': (self.df['description'].isna().sum() / len(self.df)) * 100
            },
            'empty_descriptions': {
                'count': (self.df['description'] == '').sum(),
                'percentage': ((self.df['description'] == '').sum() / len(self.df)) * 100
            },
            'books_to_remove': None
        }
        
        # Books with missing or empty descriptions
        missing_or_empty = self.df[
            self.df['description'].isna() | (self.df['description'] == '')
        ]
        results['books_to_remove'] = {
            'count': len(missing_or_empty),
            'percentage': (len(missing_or_empty) / len(self.df)) * 100
        }
        
        logger.info(f"Books with missing descriptions: {results['missing_descriptions']['count']}")
        logger.info(f"Books with empty descriptions: {results['empty_descriptions']['count']}")
        logger.info(f"Total books to remove (missing/empty descriptions): {results['books_to_remove']['count']}")
        
        return results
    
    def analyze_missing_pages(self) -> dict:
        """Analyze books with missing page counts."""
        logger.info("Analyzing missing page counts...")
        
        results = {
            'total_books': len(self.df),
            'missing_pages': {
                'count': self.df['num_pages_median'].isna().sum(),
                'percentage': (self.df['num_pages_median'].isna().sum() / len(self.df)) * 100
            },
            'zero_pages': {
                'count': (self.df['num_pages_median'] == 0).sum(),
                'percentage': ((self.df['num_pages_median'] == 0).sum() / len(self.df)) * 100
            },
            'books_to_remove': None
        }
        
        # Books with missing or zero pages
        missing_or_zero = self.df[
            self.df['num_pages_median'].isna() | (self.df['num_pages_median'] == 0)
        ]
        results['books_to_remove'] = {
            'count': len(missing_or_zero),
            'percentage': (len(missing_or_zero) / len(self.df)) * 100
        }
        
        logger.info(f"Books with missing pages: {results['missing_pages']['count']}")
        logger.info(f"Books with zero pages: {results['zero_pages']['count']}")
        logger.info(f"Total books to remove (missing/zero pages): {results['books_to_remove']['count']}")
        
        return results
    
    def analyze_work_id_book_id_grouping(self) -> dict:
        """Analyze work_id/book_id edition grouping and statistics."""
        logger.info("Analyzing work_id/book_id edition grouping...")
        
        results = {
            'total_works': self.df['work_id'].nunique(),
            'total_books': len(self.df),
            'editions_per_work': {},
            'book_id_list_analysis': {},
            'statistics_consistency': {}
        }
        
        # Analyze editions per work
        editions_per_work = self.df.groupby('work_id').size()
        results['editions_per_work'] = {
            'mean': editions_per_work.mean(),
            'median': editions_per_work.median(),
            'max': editions_per_work.max(),
            'min': editions_per_work.min(),
            'distribution': editions_per_work.value_counts().sort_index().to_dict()
        }
        
        # Analyze book_id_list_en
        book_id_list_lengths = self.df['book_id_list_en'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        results['book_id_list_analysis'] = {
            'mean_length': book_id_list_lengths.mean(),
            'median_length': book_id_list_lengths.median(),
            'max_length': book_id_list_lengths.max(),
            'min_length': book_id_list_lengths.min(),
            'empty_lists': (book_id_list_lengths == 0).sum(),
            'distribution': book_id_list_lengths.value_counts().sort_index().to_dict()
        }
        
        # Check for consistency between editions_per_work and book_id_list_en
        consistency_issues = []
        for work_id in self.df['work_id'].unique()[:100]:  # Sample check
            work_data = self.df[self.df['work_id'] == work_id]
            if len(work_data) > 0:
                book_id_list = work_data.iloc[0]['book_id_list_en']
                if isinstance(book_id_list, list):
                    if len(book_id_list) != len(work_data):
                        consistency_issues.append({
                            'work_id': work_id,
                            'editions_in_df': len(work_data),
                            'book_ids_in_list': len(book_id_list)
                        })
        
        results['statistics_consistency'] = {
            'consistency_issues_found': len(consistency_issues),
            'sample_issues': consistency_issues[:10]  # First 10 issues
        }
        
        logger.info(f"Total works: {results['total_works']}")
        logger.info(f"Total books: {results['total_books']}")
        logger.info(f"Average editions per work: {results['editions_per_work']['mean']:.2f}")
        logger.info(f"Consistency issues found: {results['statistics_consistency']['consistency_issues_found']}")
        
        return results
    
    def analyze_authors_low_ratings(self) -> dict:
        """Analyze authors with low ratings count."""
        logger.info("Analyzing authors with low ratings...")
        
        results = {
            'total_authors': self.df['author_id'].nunique(),
            'author_ratings_analysis': {},
            'low_rating_authors': {},
            'recommendations': {}
        }
        
        # Analyze author ratings distribution
        author_stats = self.df.groupby('author_id').agg({
            'author_ratings_count': 'first',
            'author_average_rating': 'first',
            'ratings_count_sum': 'sum',
            'text_reviews_count_sum': 'sum',
            'work_id': 'count'
        }).rename(columns={'work_id': 'works_count'})
        
        results['author_ratings_analysis'] = {
            'ratings_count_stats': {
                'mean': author_stats['author_ratings_count'].mean(),
                'median': author_stats['author_ratings_count'].median(),
                'min': author_stats['author_ratings_count'].min(),
                'max': author_stats['author_ratings_count'].max(),
                'q25': author_stats['author_ratings_count'].quantile(0.25),
                'q75': author_stats['author_ratings_count'].quantile(0.75)
            },
            'average_rating_stats': {
                'mean': author_stats['author_average_rating'].mean(),
                'median': author_stats['author_average_rating'].median(),
                'min': author_stats['author_average_rating'].min(),
                'max': author_stats['author_average_rating'].max()
            }
        }
        
        # Identify low-rating authors
        low_ratings_threshold = author_stats['author_ratings_count'].quantile(0.1)  # Bottom 10%
        low_rating_threshold = 3.0  # Below 3.0 average rating
        
        low_rating_authors = author_stats[
            (author_stats['author_ratings_count'] <= low_ratings_threshold) &
            (author_stats['author_average_rating'] <= low_rating_threshold)
        ]
        
        results['low_rating_authors'] = {
            'count': len(low_rating_authors),
            'percentage': (len(low_rating_authors) / len(author_stats)) * 100,
            'total_works': low_rating_authors['works_count'].sum(),
            'sample_authors': low_rating_authors.head(10).to_dict('index')
        }
        
        # Generate recommendations
        results['recommendations'] = {
            'threshold_ratings_count': low_ratings_threshold,
            'threshold_average_rating': low_rating_threshold,
            'suggested_action': 'Consider removing authors with both low ratings count and low average rating',
            'impact_analysis': f"Removing {len(low_rating_authors)} authors would affect {low_rating_authors['works_count'].sum()} works"
        }
        
        logger.info(f"Total authors: {results['total_authors']}")
        logger.info(f"Low-rating authors: {results['low_rating_authors']['count']}")
        logger.info(f"Works affected by removal: {results['low_rating_authors']['total_works']}")
        
        return results
    
    def check_publication_year_values(self) -> dict:
        """Check publication_year values for non-English books."""
        logger.info("Checking publication_year values for non-English books...")
        
        results = {
            'unique_publication_years': sorted(self.df['publication_year'].unique().tolist()),
            'year_range_analysis': {},
            'potential_non_english_indicators': {}
        }
        
        # Analyze year distribution
        year_counts = self.df['publication_year'].value_counts().sort_index()
        results['year_range_analysis'] = {
            'total_unique_years': len(year_counts),
            'year_distribution': year_counts.to_dict(),
            'suspicious_years': []
        }
        
        # Look for suspicious years that might indicate non-English books
        suspicious_years = []
        for year in year_counts.index:
            if year < 1900 or year > 2020:
                suspicious_years.append(year)
        
        results['year_range_analysis']['suspicious_years'] = suspicious_years
        
        # Check for years that might indicate translation issues
        if 'language_codes_en' in self.df.columns:
            # Analyze language codes for books with suspicious years
            for year in suspicious_years:
                year_books = self.df[self.df['publication_year'] == year]
                if len(year_books) > 0:
                    language_analysis = year_books['language_codes_en'].value_counts()
                    results['potential_non_english_indicators'][str(year)] = {
                        'book_count': len(year_books),
                        'language_distribution': language_analysis.to_dict()
                    }
        
        logger.info(f"Unique publication years: {len(results['unique_publication_years'])}")
        logger.info(f"Suspicious years found: {len(suspicious_years)}")
        
        return results
    
    def find_negative_work_count_error(self) -> dict:
        """Find and analyze negative work count calculation errors."""
        logger.info("Looking for negative work count errors...")
        
        results = {
            'negative_work_counts': {},
            'error_analysis': {},
            'code_locations': {}
        }
        
        # Check for negative values in series_works_count
        if 'series_works_count' in self.df.columns:
            negative_counts = self.df[self.df['series_works_count'] < 0]
            results['negative_work_counts'] = {
                'count': len(negative_counts),
                'values': negative_counts['series_works_count'].value_counts().to_dict(),
                'sample_records': negative_counts[['work_id', 'series_id', 'series_works_count']].head(10).to_dict('records')
            }
        
        # Check for other potential negative values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        negative_values = {}
        
        for col in numeric_columns:
            if col in ['work_id', 'book_id', 'author_id', 'series_id']:
                continue  # Skip ID columns
            negative_count = (self.df[col] < 0).sum()
            if negative_count > 0:
                negative_values[col] = {
                    'count': negative_count,
                    'values': self.df[self.df[col] < 0][col].value_counts().to_dict()
                }
        
        results['error_analysis'] = negative_values
        
        # Suggest code locations to check
        results['code_locations'] = {
            'series_works_count': 'Check series data loading and aggregation logic',
            'ratings_count_sum': 'Check ratings aggregation in _aggregate_english_editions method',
            'text_reviews_count_sum': 'Check text reviews aggregation in _aggregate_english_editions method',
            'num_pages_median': 'Check page count processing and median calculation'
        }
        
        logger.info(f"Negative series_works_count found: {results['negative_work_counts'].get('count', 0)}")
        logger.info(f"Other negative values found in columns: {list(negative_values.keys())}")
        
        return results
    
    def analyze_ratings_and_reviews_cuts(self) -> dict:
        """Analyze and suggest cuts for ratings_count_sum and text_reviews_count_sum."""
        logger.info("Analyzing ratings and reviews for potential cuts...")
        
        results = {
            'ratings_count_analysis': {},
            'text_reviews_analysis': {},
            'cut_recommendations': {}
        }
        
        # Analyze ratings_count_sum distribution
        ratings_stats = self.df['ratings_count_sum'].describe()
        results['ratings_count_analysis'] = {
            'statistics': ratings_stats.to_dict(),
            'percentiles': {
                'p10': self.df['ratings_count_sum'].quantile(0.1),
                'p25': self.df['ratings_count_sum'].quantile(0.25),
                'p50': self.df['ratings_count_sum'].quantile(0.5),
                'p75': self.df['ratings_count_sum'].quantile(0.75),
                'p90': self.df['ratings_count_sum'].quantile(0.9),
                'p95': self.df['ratings_count_sum'].quantile(0.95),
                'p99': self.df['ratings_count_sum'].quantile(0.99)
            },
            'zero_ratings': (self.df['ratings_count_sum'] == 0).sum(),
            'very_low_ratings': (self.df['ratings_count_sum'] <= 10).sum()
        }
        
        # Analyze text_reviews_count_sum distribution
        reviews_stats = self.df['text_reviews_count_sum'].describe()
        results['text_reviews_analysis'] = {
            'statistics': reviews_stats.to_dict(),
            'percentiles': {
                'p10': self.df['text_reviews_count_sum'].quantile(0.1),
                'p25': self.df['text_reviews_count_sum'].quantile(0.25),
                'p50': self.df['text_reviews_count_sum'].quantile(0.5),
                'p75': self.df['text_reviews_count_sum'].quantile(0.75),
                'p90': self.df['text_reviews_count_sum'].quantile(0.9),
                'p95': self.df['text_reviews_count_sum'].quantile(0.95),
                'p99': self.df['text_reviews_count_sum'].quantile(0.99)
            },
            'zero_reviews': (self.df['text_reviews_count_sum'] == 0).sum(),
            'very_low_reviews': (self.df['text_reviews_count_sum'] <= 5).sum()
        }
        
        # Generate cut recommendations
        results['cut_recommendations'] = {
            'conservative_cuts': {
                'ratings_count_min': 5,  # Remove books with < 5 ratings
                'text_reviews_min': 1,   # Remove books with < 1 text review
                'books_removed_ratings': (self.df['ratings_count_sum'] < 5).sum(),
                'books_removed_reviews': (self.df['text_reviews_count_sum'] < 1).sum()
            },
            'moderate_cuts': {
                'ratings_count_min': 10,  # Remove books with < 10 ratings
                'text_reviews_min': 2,    # Remove books with < 2 text reviews
                'books_removed_ratings': (self.df['ratings_count_sum'] < 10).sum(),
                'books_removed_reviews': (self.df['text_reviews_count_sum'] < 2).sum()
            },
            'aggressive_cuts': {
                'ratings_count_min': 25,  # Remove books with < 25 ratings
                'text_reviews_min': 5,    # Remove books with < 5 text reviews
                'books_removed_ratings': (self.df['ratings_count_sum'] < 25).sum(),
                'books_removed_reviews': (self.df['text_reviews_count_sum'] < 5).sum()
            }
        }
        
        logger.info(f"Books with zero ratings: {results['ratings_count_analysis']['zero_ratings']}")
        logger.info(f"Books with very low ratings (≤10): {results['ratings_count_analysis']['very_low_ratings']}")
        logger.info(f"Books with zero reviews: {results['text_reviews_analysis']['zero_reviews']}")
        logger.info(f"Books with very low reviews (≤5): {results['text_reviews_analysis']['very_low_reviews']}")
        
        return results
    
    def run_comprehensive_analysis(self) -> dict:
        """Run all analyses and return comprehensive results."""
        logger.info("Starting comprehensive data analysis...")
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.analysis_results = {
            'publication_years': self.analyze_publication_years(),
            'missing_descriptions': self.analyze_missing_descriptions(),
            'missing_pages': self.analyze_missing_pages(),
            'work_id_grouping': self.analyze_work_id_book_id_grouping(),
            'authors_low_ratings': self.analyze_authors_low_ratings(),
            'publication_year_values': self.check_publication_year_values(),
            'negative_work_count': self.find_negative_work_count_error(),
            'ratings_reviews_cuts': self.analyze_ratings_and_reviews_cuts()
        }
        
        # Generate summary
        self.analysis_results['summary'] = self._generate_summary()
        
        logger.info("Comprehensive analysis completed")
        return self.analysis_results
    
    def _generate_summary(self) -> dict:
        """Generate a summary of all findings."""
        summary = {
            'total_books_initial': len(self.df),
            'books_to_remove': {},
            'recommendations': []
        }
        
        # Calculate total books to remove
        books_to_remove = 0
        
        # Books outside 2000-2017
        outside_range = self.analysis_results['publication_years']['books_outside_2000_2017']['count']
        summary['books_to_remove']['outside_2000_2017'] = outside_range
        books_to_remove += outside_range
        
        # Books with missing descriptions
        missing_desc = self.analysis_results['missing_descriptions']['books_to_remove']['count']
        summary['books_to_remove']['missing_descriptions'] = missing_desc
        books_to_remove += missing_desc
        
        # Books with missing pages
        missing_pages = self.analysis_results['missing_pages']['books_to_remove']['count']
        summary['books_to_remove']['missing_pages'] = missing_pages
        books_to_remove += missing_pages
        
        summary['books_to_remove']['total'] = books_to_remove
        summary['books_remaining'] = len(self.df) - books_to_remove
        
        # Generate recommendations
        summary['recommendations'] = [
            f"Remove {outside_range} books outside 2000-2017 range",
            f"Remove {missing_desc} books with missing descriptions",
            f"Remove {missing_pages} books with missing pages",
            f"Consider removing {self.analysis_results['authors_low_ratings']['low_rating_authors']['count']} low-rating authors",
            f"Fix negative work count errors in {self.analysis_results['negative_work_count']['negative_work_counts'].get('count', 0)} records",
            f"Consider applying ratings/reviews cuts based on analysis"
        ]
        
        return summary
    
    def save_analysis_report(self, filename: str = None) -> str:
        """Save comprehensive analysis report."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_data_analysis_report_{timestamp}.json"
        
        output_path = Path("outputs") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy_types(obj)
        
        # Convert results for JSON serialization
        json_results = recursive_convert(self.analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        if not self.analysis_results:
            logger.warning("No analysis results available. Run run_comprehensive_analysis() first.")
            return
        
        summary = self.analysis_results['summary']
        
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 INITIAL DATASET:")
        print(f"  Total books: {summary['total_books_initial']:,}")
        
        print(f"\n🗑️  BOOKS TO REMOVE:")
        for reason, count in summary['books_to_remove'].items():
            if reason != 'total':
                print(f"  {reason}: {count:,}")
        print(f"  TOTAL TO REMOVE: {summary['books_to_remove']['total']:,}")
        
        print(f"\n✅ BOOKS REMAINING:")
        print(f"  After cleanup: {summary['books_remaining']:,}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Main function to run comprehensive analysis."""
    print("🔍 Starting Comprehensive Data Analysis...")
    
    analyzer = ComprehensiveDataAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    # Save report
    report_path = analyzer.save_analysis_report()
    
    # Print summary
    analyzer.print_summary()
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()
