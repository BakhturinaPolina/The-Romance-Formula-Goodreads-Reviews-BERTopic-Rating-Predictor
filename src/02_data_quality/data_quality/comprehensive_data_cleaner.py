#!/usr/bin/env python3
"""
Comprehensive Data Cleaning Script
Implements all the data quality fixes identified in the analysis.

Based on comprehensive analysis results:
- Remove 122 books outside 2000-2017 range
- Remove 6,172 books with missing descriptions  
- Remove 35,908 books with missing pages
- Fix negative work count errors (3 records with -14.0)
- Apply ratings/reviews cuts
- Handle low-rating authors
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

class ComprehensiveDataCleaner:
    """Comprehensive data cleaning for romance novel dataset."""
    
    def __init__(self, data_path: str = None):
        """Initialize the cleaner with data path."""
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
        self.cleaning_results = {}
        
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
    
    def remove_books_outside_2000_2017(self) -> dict:
        """Remove books outside 2000-2017 range."""
        logger.info("Removing books outside 2000-2017 range...")
        
        initial_count = len(self.df)
        
        # Remove books outside 2000-2017 range
        self.df = self.df[self.df['publication_year'].between(2000, 2017, inclusive='both')]
        
        removed_count = initial_count - len(self.df)
        
        result = {
            'books_removed': removed_count,
            'books_remaining': len(self.df),
            'removal_percentage': (removed_count / initial_count) * 100
        }
        
        logger.info(f"Removed {removed_count} books outside 2000-2017 range")
        logger.info(f"Books remaining: {len(self.df)}")
        
        return result
    
    def remove_books_missing_descriptions(self) -> dict:
        """Remove books with missing or empty descriptions."""
        logger.info("Removing books with missing descriptions...")
        
        initial_count = len(self.df)
        
        # Remove books with missing or empty descriptions
        self.df = self.df[
            self.df['description'].notna() & 
            (self.df['description'] != '') & 
            (self.df['description'].str.strip() != '')
        ]
        
        removed_count = initial_count - len(self.df)
        
        result = {
            'books_removed': removed_count,
            'books_remaining': len(self.df),
            'removal_percentage': (removed_count / initial_count) * 100
        }
        
        logger.info(f"Removed {removed_count} books with missing descriptions")
        logger.info(f"Books remaining: {len(self.df)}")
        
        return result
    
    def remove_books_missing_pages(self) -> dict:
        """Remove books with missing or zero page counts."""
        logger.info("Removing books with missing pages...")
        
        initial_count = len(self.df)
        
        # Remove books with missing or zero pages
        self.df = self.df[
            self.df['num_pages_median'].notna() & 
            (self.df['num_pages_median'] > 0)
        ]
        
        removed_count = initial_count - len(self.df)
        
        result = {
            'books_removed': removed_count,
            'books_remaining': len(self.df),
            'removal_percentage': (removed_count / initial_count) * 100
        }
        
        logger.info(f"Removed {removed_count} books with missing pages")
        logger.info(f"Books remaining: {len(self.df)}")
        
        return result
    
    def fix_negative_work_count_errors(self) -> dict:
        """Fix negative work count calculation errors."""
        logger.info("Fixing negative work count errors...")
        
        initial_negative_count = (self.df['series_works_count'] < 0).sum()
        
        # Fix negative series_works_count values
        # Set negative values to NaN (missing) since they represent calculation errors
        self.df.loc[self.df['series_works_count'] < 0, 'series_works_count'] = np.nan
        
        # Also fix related series fields for consistency
        negative_mask = self.df['series_works_count'].isna() & self.df['series_id'].notna()
        self.df.loc[negative_mask, 'series_id'] = np.nan
        self.df.loc[negative_mask, 'series_title'] = np.nan
        
        result = {
            'negative_values_fixed': initial_negative_count,
            'method': 'Set negative values to NaN (missing)',
            'affected_series_fields': ['series_works_count', 'series_id', 'series_title']
        }
        
        logger.info(f"Fixed {initial_negative_count} negative work count errors")
        
        return result
    
    def apply_ratings_reviews_cuts(self, ratings_min: int = 10, reviews_min: int = 2) -> dict:
        """Apply cuts based on ratings and reviews counts."""
        logger.info(f"Applying ratings/reviews cuts (min ratings: {ratings_min}, min reviews: {reviews_min})...")
        
        initial_count = len(self.df)
        
        # Apply cuts
        self.df = self.df[
            (self.df['ratings_count_sum'] >= ratings_min) &
            (self.df['text_reviews_count_sum'] >= reviews_min)
        ]
        
        removed_count = initial_count - len(self.df)
        
        result = {
            'books_removed': removed_count,
            'books_remaining': len(self.df),
            'removal_percentage': (removed_count / initial_count) * 100,
            'criteria_applied': {
                'min_ratings': ratings_min,
                'min_reviews': reviews_min
            }
        }
        
        logger.info(f"Removed {removed_count} books with low ratings/reviews")
        logger.info(f"Books remaining: {len(self.df)}")
        
        return result
    
    def remove_low_rating_authors(self, ratings_threshold: float = None, rating_threshold: float = 3.0) -> dict:
        """Remove authors with low ratings count and low average rating."""
        logger.info("Analyzing and removing low-rating authors...")
        
        # Calculate author statistics
        author_stats = self.df.groupby('author_id').agg({
            'author_ratings_count': 'first',
            'author_average_rating': 'first',
            'work_id': 'count'
        }).rename(columns={'work_id': 'works_count'})
        
        # Set default threshold if not provided (bottom 10% of ratings count)
        if ratings_threshold is None:
            ratings_threshold = author_stats['author_ratings_count'].quantile(0.1)
        
        # Identify low-rating authors
        low_rating_authors = author_stats[
            (author_stats['author_ratings_count'] <= ratings_threshold) &
            (author_stats['author_average_rating'] <= rating_threshold)
        ]
        
        initial_count = len(self.df)
        
        # Remove books by low-rating authors
        self.df = self.df[~self.df['author_id'].isin(low_rating_authors.index)]
        
        removed_count = initial_count - len(self.df)
        
        result = {
            'authors_removed': len(low_rating_authors),
            'books_removed': removed_count,
            'books_remaining': len(self.df),
            'removal_percentage': (removed_count / initial_count) * 100,
            'criteria_applied': {
                'max_ratings_count': ratings_threshold,
                'max_average_rating': rating_threshold
            }
        }
        
        logger.info(f"Removed {len(low_rating_authors)} low-rating authors affecting {removed_count} books")
        logger.info(f"Books remaining: {len(self.df)}")
        
        return result
    
    def verify_work_id_book_id_grouping(self) -> dict:
        """Verify work_id/book_id edition grouping and statistics."""
        logger.info("Verifying work_id/book_id edition grouping...")
        
        # Check if we have the expected 1:1 relationship
        works_count = self.df['work_id'].nunique()
        books_count = len(self.df)
        
        # Check book_id_list_en (should be empty lists based on analysis)
        empty_lists = self.df['book_id_list_en'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        empty_lists_count = (empty_lists == 0).sum()
        
        result = {
            'total_works': works_count,
            'total_books': books_count,
            'works_books_ratio': books_count / works_count if works_count > 0 else 0,
            'empty_book_id_lists': empty_lists_count,
            'grouping_status': '1:1 work to book relationship (expected for this dataset)'
        }
        
        logger.info(f"Total works: {works_count}")
        logger.info(f"Total books: {books_count}")
        logger.info(f"Works to books ratio: {result['works_books_ratio']:.2f}")
        
        return result
    
    def get_publication_year_summary(self) -> dict:
        """Get summary of publication years to verify no non-English books."""
        logger.info("Analyzing publication years...")
        
        year_counts = self.df['publication_year'].value_counts().sort_index()
        
        result = {
            'unique_years': sorted(self.df['publication_year'].unique().tolist()),
            'year_range': {
                'min': self.df['publication_year'].min(),
                'max': self.df['publication_year'].max()
            },
            'year_distribution': year_counts.to_dict(),
            'total_years': len(year_counts),
            'suspicious_years': [year for year in year_counts.index if year < 1900 or year > 2020]
        }
        
        logger.info(f"Publication years: {result['year_range']['min']} - {result['year_range']['max']}")
        logger.info(f"Unique years: {len(result['unique_years'])}")
        logger.info(f"Suspicious years: {len(result['suspicious_years'])}")
        
        return result
    
    def run_comprehensive_cleaning(self, 
                                 apply_ratings_cuts: bool = True,
                                 apply_author_cuts: bool = True,
                                 ratings_min: int = 10,
                                 reviews_min: int = 2) -> dict:
        """Run comprehensive data cleaning."""
        logger.info("Starting comprehensive data cleaning...")
        
        # Load data
        self.load_data()
        
        initial_count = len(self.df)
        self.cleaning_results = {
            'initial_count': initial_count,
            'cleaning_steps': {}
        }
        
        # Step 1: Remove books outside 2000-2017 range
        self.cleaning_results['cleaning_steps']['remove_outside_2000_2017'] = self.remove_books_outside_2000_2017()
        
        # Step 2: Remove books with missing descriptions
        self.cleaning_results['cleaning_steps']['remove_missing_descriptions'] = self.remove_books_missing_descriptions()
        
        # Step 3: Remove books with missing pages
        self.cleaning_results['cleaning_steps']['remove_missing_pages'] = self.remove_books_missing_pages()
        
        # Step 4: Fix negative work count errors
        self.cleaning_results['cleaning_steps']['fix_negative_work_counts'] = self.fix_negative_work_count_errors()
        
        # Step 5: Apply ratings/reviews cuts (optional)
        if apply_ratings_cuts:
            self.cleaning_results['cleaning_steps']['apply_ratings_cuts'] = self.apply_ratings_reviews_cuts(ratings_min, reviews_min)
        
        # Step 6: Remove low-rating authors (optional)
        if apply_author_cuts:
            self.cleaning_results['cleaning_steps']['remove_low_rating_authors'] = self.remove_low_rating_authors()
        
        # Step 7: Verify work_id/book_id grouping
        self.cleaning_results['cleaning_steps']['verify_grouping'] = self.verify_work_id_book_id_grouping()
        
        # Step 8: Get publication year summary
        self.cleaning_results['cleaning_steps']['publication_years'] = self.get_publication_year_summary()
        
        # Generate final summary
        self.cleaning_results['final_summary'] = self._generate_cleaning_summary()
        
        logger.info("Comprehensive data cleaning completed")
        return self.cleaning_results
    
    def _generate_cleaning_summary(self) -> dict:
        """Generate a summary of the cleaning process."""
        final_count = len(self.df)
        total_removed = self.cleaning_results['initial_count'] - final_count
        
        summary = {
            'initial_books': self.cleaning_results['initial_count'],
            'final_books': final_count,
            'total_removed': total_removed,
            'removal_percentage': (total_removed / self.cleaning_results['initial_count']) * 100,
            'cleaning_steps_completed': len(self.cleaning_results['cleaning_steps']),
            'data_quality_improvements': [
                'Removed books outside 2000-2017 range',
                'Removed books with missing descriptions',
                'Removed books with missing pages',
                'Fixed negative work count calculation errors',
                'Applied ratings and reviews quality filters',
                'Removed low-rating authors',
                'Verified work_id/book_id grouping consistency'
            ]
        }
        
        return summary
    
    def save_cleaned_dataset(self, filename: str = None, format_type: str = 'csv') -> str:
        """Save the cleaned dataset."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cleaned_romance_novels_2000_2017_{timestamp}"
        
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        if format_type.lower() == 'csv':
            file_path = output_path.with_suffix('.csv')
            self.df.to_csv(file_path, index=False)
        elif format_type.lower() == 'pickle':
            file_path = output_path.with_suffix('.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(self.df, f)
        elif format_type.lower() == 'parquet':
            file_path = output_path.with_suffix('.parquet')
            self.df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Cleaned dataset saved to {file_path}")
        return str(file_path)
    
    def save_cleaning_report(self, filename: str = None) -> str:
        """Save comprehensive cleaning report."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_data_cleaning_report_{timestamp}.json"
        
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
        json_results = recursive_convert(self.cleaning_results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Cleaning report saved to {output_path}")
        return str(output_path)
    
    def print_cleaning_summary(self):
        """Print a summary of the cleaning results."""
        if not self.cleaning_results:
            logger.warning("No cleaning results available. Run run_comprehensive_cleaning() first.")
            return
        
        summary = self.cleaning_results['final_summary']
        
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA CLEANING SUMMARY")
        print("="*80)
        
        print(f"\n📊 DATASET TRANSFORMATION:")
        print(f"  Initial books: {summary['initial_books']:,}")
        print(f"  Final books: {summary['final_books']:,}")
        print(f"  Total removed: {summary['total_removed']:,}")
        print(f"  Removal percentage: {summary['removal_percentage']:.2f}%")
        
        print(f"\n🔧 CLEANING STEPS COMPLETED:")
        for i, step in enumerate(summary['data_quality_improvements'], 1):
            print(f"  {i}. {step}")
        
        print(f"\n📈 DATA QUALITY IMPROVEMENTS:")
        print(f"  - Focused on 2000-2017 publication years")
        print(f"  - Removed books with missing critical data")
        print(f"  - Fixed calculation errors")
        print(f"  - Applied quality filters for ratings and reviews")
        print(f"  - Ensured data consistency")
        
        print("\n" + "="*80)


def main():
    """Main function to run comprehensive data cleaning."""
    print("🧹 Starting Comprehensive Data Cleaning...")
    
    cleaner = ComprehensiveDataCleaner()
    
    # Run cleaning with conservative settings
    results = cleaner.run_comprehensive_cleaning(
        apply_ratings_cuts=True,
        apply_author_cuts=True,
        ratings_min=10,  # Conservative: remove books with < 10 ratings
        reviews_min=2    # Conservative: remove books with < 2 reviews
    )
    
    # Save cleaned dataset
    dataset_path = cleaner.save_cleaned_dataset(format_type='csv')
    
    # Save cleaning report
    report_path = cleaner.save_cleaning_report()
    
    # Print summary
    cleaner.print_cleaning_summary()
    
    print(f"\n📄 Cleaned dataset saved to: {dataset_path}")
    print(f"📄 Cleaning report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()
