"""
Enhanced Final CSV Builder with Null Value Fix
Handles empty strings properly before numeric conversion to prevent null introduction.
"""

import pandas as pd
import json
import gzip
import re
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OptimizedFinalCSVBuilder:
    """
    Enhanced version of FinalCSVBuilder with proper null value handling.
    
    Key improvements:
    - Handles empty strings before numeric conversion
    - Enhanced data quality validation
    - Comprehensive logging of data transformations
    - Better error handling and fallback logic
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
        # English language regex pattern
        self.english_regex = re.compile(r'^(eng|en(?:-[A-Za-z]+)?)$', re.IGNORECASE)
        
        # Simple title cleaner for bracket stripping
        self.bracket_pattern = re.compile(r'\s*\([^)]*\)|\s*\[[^\]]*\]')
        
        # Data quality tracking
        self.quality_metrics = {
            'works_processed': 0,
            'works_skipped': 0,
            'works_with_ratings': 0,
            'validation_errors': [],
            'data_conversion_issues': {}
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for CSV building operations."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def safe_numeric_conversion(self, series: pd.Series, field_name: str, default_value: Optional[float] = None) -> pd.Series:
        """
        Safely convert series to numeric, handling empty strings and other issues.
        
        Args:
            series: Pandas series to convert
            field_name: Name of the field for logging
            default_value: Default value to use for nulls (optional)
            
        Returns:
            Converted numeric series
        """
        original_nulls = series.isnull().sum()
        
        # Handle empty strings and common non-numeric values
        series_cleaned = series.copy()
        
        # Replace empty strings with None
        series_cleaned = series_cleaned.replace('', None)
        
        # Handle other common non-numeric values based on field type
        if field_name == 'publication_year':
            series_cleaned = series_cleaned.replace(['Unknown', 'N/A', 'TBD', '0'], None)
        elif field_name == 'num_pages':
            series_cleaned = series_cleaned.replace(['Unknown', 'N/A', 'TBD', '0'], None)
        elif field_name in ['ratings_count', 'text_reviews_count']:
            series_cleaned = series_cleaned.replace(['Unknown', 'N/A'], None)
        elif field_name == 'average_rating':
            series_cleaned = series_cleaned.replace(['Unknown', 'N/A'], None)
        
        # Convert to numeric
        converted = pd.to_numeric(series_cleaned, errors='coerce')
        
        # Apply default value if specified
        if default_value is not None:
            converted = converted.fillna(default_value)
        
        # Track conversion issues
        new_nulls = converted.isnull().sum() - original_nulls
        if new_nulls > 0:
            self.quality_metrics['data_conversion_issues'][field_name] = {
                'original_nulls': original_nulls,
                'new_nulls': new_nulls,
                'total_nulls': converted.isnull().sum()
            }
            self.logger.warning(f"{field_name}: {new_nulls} nulls introduced during conversion")
            
            # Log examples of problematic values
            problematic = series[converted.isnull() & series.notnull()]
            if len(problematic) > 0:
                unique_problematic = problematic.unique()[:5]
                self.logger.debug(f"Problematic values in {field_name}: {unique_problematic}")
        
        return converted
    
    def is_english_language(self, language_code: str) -> bool:
        """Check if language code represents English."""
        if not language_code or not isinstance(language_code, str):
            return False
        return bool(self.english_regex.match(language_code.strip()))
    
    def clean_title_simple(self, title: str) -> Tuple[str, str]:
        """
        Simple title cleaning: just strip everything within brackets/parentheses.
        
        Args:
            title: Original title string
            
        Returns:
            Tuple of (cleaned_title, removed_content)
        """
        if not title or not isinstance(title, str):
            return title, ""
        
        original_title = title.strip()
        cleaned_title = original_title
        removed_parts = []
        
        # Find and remove all bracket content
        for match in self.bracket_pattern.finditer(original_title):
            removed_parts.append(match.group(0))
            cleaned_title = cleaned_title.replace(match.group(0), '')
        
        # Clean up extra whitespace
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
        cleaned_title = cleaned_title.strip()
        
        # If cleaning resulted in empty title, return original
        if not cleaned_title:
            return original_title, ""
        
        removed_content = '; '.join(removed_parts) if removed_parts else ""
        return cleaned_title, removed_content
    
    def get_title_with_fallback(self, work_info: Dict[str, Any], english_editions: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """
        Get title with comprehensive fallback logic.
        
        Args:
            work_info: Work-level information from works data
            english_editions: List of English edition records
            
        Returns:
            Tuple of (title, title_source, fallback_used)
        """
        # Priority 1: works.original_title
        title = work_info.get('original_title', '')
        if title and isinstance(title, str) and title.strip():
            return title.strip(), 'works.original_title', False
        
        # Priority 2: works.best_book_title
        title = work_info.get('best_book_title', '')
        if title and isinstance(title, str) and title.strip():
            return title.strip(), 'works.best_book_title', True
        
        # Priority 3: works.title
        title = work_info.get('title', '')
        if title and isinstance(title, str) and title.strip():
            return title.strip(), 'works.title', True
        
        # Priority 4: First English edition title
        if english_editions:
            for edition in english_editions:
                edition_title = edition.get('title', '')
                if edition_title and isinstance(edition_title, str) and edition_title.strip():
                    return edition_title.strip(), 'edition.title', True
        
        # Priority 5: Default fallback
        return 'Untitled', 'default', True
    
    def load_books_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load books data into a pandas DataFrame with enhanced data cleaning."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loading books data into DataFrame...")
        
        records = []
        records_processed = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            records.append(record)
                            
                            records_processed += 1
                            if records_processed % 50000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} records...")
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Creating DataFrame from {len(records):,} records...")
        df = pd.DataFrame(records)
        
        # Enhanced data type conversion with proper null handling
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Applying enhanced data type conversions...")
        
        if 'work_id' in df.columns:
            df['work_id'] = self.safe_numeric_conversion(df['work_id'], 'work_id')
        if 'book_id' in df.columns:
            df['book_id'] = self.safe_numeric_conversion(df['book_id'], 'book_id')
        if 'publication_year' in df.columns:
            df['publication_year'] = self.safe_numeric_conversion(df['publication_year'], 'publication_year')
        if 'num_pages' in df.columns:
            df['num_pages'] = self.safe_numeric_conversion(df['num_pages'], 'num_pages')
        if 'ratings_count' in df.columns:
            df['ratings_count'] = self.safe_numeric_conversion(df['ratings_count'], 'ratings_count')
        if 'text_reviews_count' in df.columns:
            df['text_reviews_count'] = self.safe_numeric_conversion(df['text_reviews_count'], 'text_reviews_count')
        if 'average_rating' in df.columns:
            df['average_rating'] = self.safe_numeric_conversion(df['average_rating'], 'average_rating')
        
        # Log data quality metrics
        self._log_data_quality_metrics(df, "After loading and conversion")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Books DataFrame created: {df.shape}")
        return df
    
    def _log_data_quality_metrics(self, df: pd.DataFrame, stage: str):
        """Log data quality metrics at different stages."""
        print(f"📊 Data Quality Metrics - {stage}:")
        print(f"  - Total records: {len(df):,}")
        print(f"  - Work ID nulls: {df.get('work_id', pd.Series()).isnull().sum()}")
        print(f"  - Book ID nulls: {df.get('book_id', pd.Series()).isnull().sum()}")
        print(f"  - Publication year nulls: {df.get('publication_year', pd.Series()).isnull().sum()}")
        print(f"  - Page count nulls: {df.get('num_pages', pd.Series()).isnull().sum()}")
        print(f"  - Ratings count nulls: {df.get('ratings_count', pd.Series()).isnull().sum()}")
        print(f"  - Average rating nulls: {df.get('average_rating', pd.Series()).isnull().sum()}")
    
    def load_json_data(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data from gzipped file."""
        data = {}
        records_processed = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            # Use appropriate ID field based on file type
                            if 'work_id' in record:
                                data[str(record['work_id'])] = record
                            elif 'book_id' in record:
                                data[str(record['book_id'])] = record
                            elif 'author_id' in record:
                                data[str(record['author_id'])] = record
                            elif 'series_id' in record:
                                data[str(record['series_id'])] = record
                            
                            records_processed += 1
                            if records_processed % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} records...")
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return {}
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loaded {len(data):,} records from {file_path.name}")
        return data
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality before saving.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Running enhanced data quality validation...")
        
        validation_errors = []
        
        # Check 1: No empty work_id or title
        empty_work_id = df['work_id'].isna().sum()
        empty_title = df['title'].isna().sum()
        
        if empty_work_id > 0:
            validation_errors.append(f"Validation failed: {empty_work_id} empty work_ids")
        
        if empty_title > 0:
            validation_errors.append(f"Validation failed: {empty_title} empty titles")
        
        # Check 2: book_id_list_en is non-empty for every kept row
        empty_book_lists = df['book_id_list_en'].apply(lambda x: len(x) == 0).sum()
        if empty_book_lists > 0:
            validation_errors.append(f"Validation failed: {empty_book_lists} rows with empty book_id_list_en")
        
        # Check 3: Publication year in range
        out_of_range = df[~df['publication_year'].between(2000, 2020, inclusive='both')]
        if len(out_of_range) > 0:
            validation_errors.append(f"Validation failed: {len(out_of_range)} rows outside 2000-2020 range")
        
        # Check 4: Title quality
        if 'title_source' in df.columns:
            untitled_count = (df['title'] == 'Untitled').sum()
            if untitled_count > 0:
                print(f"⚠️  Warning: {untitled_count} works have 'Untitled' as title")
        
        # Check 5: Data conversion issues
        if self.quality_metrics['data_conversion_issues']:
            print(f"⚠️  Data conversion issues detected:")
            for field, issues in self.quality_metrics['data_conversion_issues'].items():
                print(f"  - {field}: {issues['new_nulls']} nulls introduced")
        
        # Report validation results
        if validation_errors:
            print(f"❌ Data quality validation failed:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        else:
            print(f"✅ All data quality validation checks passed")
            return True
    
    def build_final_csv_optimized(self, sample_size: Optional[int] = None) -> str:
        """
        Build the final CSV using enhanced data processing with proper null handling.
        
        Args:
            sample_size: Optional sample size for testing
            
        Returns:
            Path to the generated CSV file
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏗️ Building final CSV with enhanced null handling...")
        
        # Load books data as DataFrame
        books_df = self.load_books_dataframe(Path("data/raw/goodreads_books_romance.json.gz"))
        
        if books_df.empty:
            raise ValueError("Failed to load books data")
        
        # Load other data sources
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loading additional data sources...")
        works_data = self.load_json_data(Path("data/raw/goodreads_book_works.json.gz"))
        authors_data = self.load_json_data(Path("data/raw/goodreads_book_authors.json.gz"))
        series_data = self.load_json_data(Path("data/raw/goodreads_book_series.json.gz"))
        genres_data = self.load_json_data(Path("data/raw/goodreads_book_genres_initial.json.gz"))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Data sources loaded")
        
        # Filter to English editions only
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌍 Filtering to English editions...")
        books_df['is_english'] = books_df['language_code'].apply(self.is_english_language)
        english_books_df = books_df[books_df['is_english']].copy()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 English editions: {len(english_books_df):,} out of {len(books_df):,}")
        
        # Group by work_id and aggregate
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Grouping by work_id...")
        work_groups = english_books_df.groupby('work_id')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Found {len(work_groups)} unique works")
        
        # Apply sampling if requested
        if sample_size and sample_size < len(work_groups):
            import random
            random.seed(42)
            sampled_work_ids = random.sample(list(work_groups.groups.keys()), sample_size)
            work_groups = english_books_df[english_books_df['work_id'].isin(sampled_work_ids)].groupby('work_id')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Sampled {len(work_groups)} works")
        
        # Process works in batches
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏗️ Processing works with enhanced data handling...")
        master_records = []
        processed_works = 0
        total_works = len(work_groups)
        
        for work_id, group in work_groups:
            try:
                # Convert group to list of dictionaries for processing
                editions = group.to_dict('records')
                
                # Get work-level information
                work_info = works_data.get(str(work_id), {})
                
                # Get publication year logic
                publication_year, median_publication_year = self._get_publication_year_logic(work_info, editions)
                
                # Skip works with no valid publication year
                if publication_year is None:
                    self.quality_metrics['works_skipped'] += 1
                    continue
                
                # Get title with comprehensive fallback logic
                title, title_source, fallback_used = self.get_title_with_fallback(work_info, editions)
                
                # Track basic processing metrics
                self.quality_metrics['works_processed'] += 1
                
                # Select primary author
                primary_author_id = self._select_primary_author(editions)
                
                # Get author information
                author_info = self._get_author_info(primary_author_id, authors_data)
                
                # Get series information
                series_info = self._get_series_info(work_id, series_data, books_df)
                
                # Aggregate English editions
                aggregations = self._aggregate_english_editions(editions)
                
                # Track rating diversity
                if aggregations['average_rating_weighted_mean'] is not None:
                    self.quality_metrics['works_with_ratings'] += 1
                
                # Get description and shelves
                description = self._get_longest_description(editions)
                popular_shelves = self._get_all_popular_shelves(editions)
                
                # Get genres from all books in the work
                genres = self._aggregate_genres_from_work(aggregations['book_id_list_en'], genres_data)
                
                # Clean title
                clean_title, removed_content = self.clean_title_simple(title)
                
                # Create master record
                master_record = {
                    'work_id': work_id,
                    'book_id_list_en': aggregations['book_id_list_en'],
                    'title': clean_title,
                    'publication_year': publication_year,
                    'language_codes_en': aggregations['language_codes_en'],
                    'num_pages_median': aggregations['num_pages_median'],
                    'description': description,
                    'popular_shelves': popular_shelves,
                    'genres': genres,
                    'author_id': author_info['author_id'],
                    'author_name': author_info['author_name'],
                    'author_average_rating': author_info['author_average_rating'],
                    'author_ratings_count': author_info['author_ratings_count'],
                    'series_id': series_info['series_id'],
                    'series_title': series_info['series_title'],
                    'series_works_count': series_info['series_works_count'],
                    'ratings_count_sum': aggregations['ratings_count_sum'],
                    'text_reviews_count_sum': aggregations['text_reviews_count_sum'],
                    'average_rating_weighted_mean': aggregations['average_rating_weighted_mean']
                }
                
                master_records.append(master_record)
                processed_works += 1
                
                # Progress tracking
                if processed_works % 1000 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {processed_works:,}/{total_works:,} works...")
                    
            except Exception as e:
                self.logger.warning(f"Error processing work {work_id}: {e}")
                self.quality_metrics['validation_errors'].append(f"Work {work_id}: {e}")
                continue
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Master table built with {len(master_records)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(master_records)
        
        # Apply final filters
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Applying final filters...")
        df = df[df['publication_year'].between(2000, 2020, inclusive='both')]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📅 After year filter: {len(df)} records")
        
        # Validate data quality
        if not self.validate_data_quality(df):
            raise ValueError("Data quality validation failed")
        
        # Sort by publication_year ascending, then ratings_count_sum descending, then work_id
        df = df.sort_values(['publication_year', 'ratings_count_sum', 'work_id'], 
                           ascending=[True, False, True])
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if sample_size:
            filename = f"final_books_2000_2020_en_enhanced_sampled_{sample_size}_{timestamp}.csv"
        else:
            filename = f"final_books_2000_2020_en_enhanced_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Save to CSV
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        
        # Log summary and quality metrics
        self._log_quality_summary(df, total_works, output_path)
        
        return str(output_path)
    
    def _log_quality_summary(self, df: pd.DataFrame, total_works: int, output_path: str) -> None:
        """Log comprehensive quality summary."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Final CSV Summary:")
        print(f"  - Total works scanned: {total_works:,}")
        print(f"  - Works processed: {self.quality_metrics['works_processed']:,}")
        print(f"  - Works skipped: {self.quality_metrics['works_skipped']:,}")
        print(f"  - Works with ratings: {self.quality_metrics['works_with_ratings']:,}")
        print(f"  - Works kept after year filter: {len(df):,}")
        
        # Data conversion issues summary
        if self.quality_metrics['data_conversion_issues']:
            print(f"\n🔧 Data Conversion Issues:")
            for field, issues in self.quality_metrics['data_conversion_issues'].items():
                print(f"  - {field}: {issues['new_nulls']} nulls introduced")
        
        # Data quality metrics
        print(f"\n🔍 Final Data Quality Metrics:")
        print(f"  - Null publication_years: {df['publication_year'].isna().sum()}")
        print(f"  - Null author_ids: {df['author_id'].isna().sum()}")
        print(f"  - Null series_ids: {df['series_id'].isna().sum()}")
        print(f"  - Untitled works: {(df['title'] == 'Untitled').sum()}")
        
        # Save quality report
        self._save_quality_report(output_path)
    
    def _save_quality_report(self, output_path: Path) -> None:
        """Save detailed quality report."""
        report_path = output_path.parent / f"{output_path.stem}_quality_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("OPTIMIZED CSV BUILDER QUALITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output file: {output_path.name}\n\n")
            
            f.write("QUALITY METRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in self.quality_metrics.items():
                if metric not in ['validation_errors', 'data_conversion_issues']:
                    f.write(f"{metric}: {value}\n")
            
            f.write(f"\nDATA CONVERSION ISSUES:\n")
            f.write("-" * 25 + "\n")
            for field, issues in self.quality_metrics['data_conversion_issues'].items():
                f.write(f"{field}: {issues['new_nulls']} nulls introduced\n")
            
            f.write(f"\nVALIDATION ERRORS: {len(self.quality_metrics['validation_errors'])}\n")
            if self.quality_metrics['validation_errors']:
                for error in self.quality_metrics['validation_errors']:
                    f.write(f"  - {error}\n")
        
        print(f"📋 Quality report saved to {report_path}")
    
    # Include all the helper methods from the original class
    def _get_publication_year_logic(self, work_info: Dict[str, Any], english_editions: List[Dict[str, Any]]) -> Tuple[int, Optional[int]]:
        """Get publication year logic."""
        # Prefer original_publication_year from works data
        original_pub_year = work_info.get('original_publication_year')
        if original_pub_year and str(original_pub_year).strip():
            try:
                original_year = int(original_pub_year)
                if 1800 <= original_year <= 2030:
                    return original_year, None
            except (ValueError, TypeError):
                pass
        
        # Fallback: calculate median from English editions
        english_years = []
        for edition in english_editions:
            pub_year = edition.get('publication_year')
            if pd.notna(pub_year):
                try:
                    year = int(pub_year)
                    if 1800 <= year <= 2030:
                        english_years.append(year)
                except (ValueError, TypeError):
                    continue
        
        if english_years:
            median_year = int(statistics.median(english_years))
            return median_year, median_year
        else:
            return None, None
    
    def _select_primary_author(self, english_editions: List[Dict[str, Any]]) -> Optional[str]:
        """Select primary author."""
        author_counts = {}
        author_ratings = {}
        
        for edition in english_editions:
            authors = edition.get('authors', [])
            if authors and isinstance(authors, list) and len(authors) > 0:
                author_id = authors[0].get('author_id')
                if author_id:
                    author_id = str(author_id)
                    author_counts[author_id] = author_counts.get(author_id, 0) + 1
                    
                    ratings_count = edition.get('ratings_count', 0)
                    if pd.notna(ratings_count):
                        try:
                            ratings_count = int(ratings_count)
                            author_ratings[author_id] = author_ratings.get(author_id, 0) + ratings_count
                        except (ValueError, TypeError):
                            pass
        
        if not author_counts:
            return None
        
        max_count = max(author_counts.values())
        most_frequent_authors = [aid for aid, count in author_counts.items() if count == max_count]
        
        if len(most_frequent_authors) == 1:
            return most_frequent_authors[0]
        
        max_ratings = max(author_ratings.get(aid, 0) for aid in most_frequent_authors)
        highest_rated_authors = [aid for aid in most_frequent_authors if author_ratings.get(aid, 0) == max_ratings]
        
        if len(highest_rated_authors) == 1:
            return highest_rated_authors[0]
        
        return min(highest_rated_authors)
    
    def _get_author_info(self, primary_author_id: Optional[str], authors_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get author information."""
        if primary_author_id and primary_author_id in authors_data:
            author_data = authors_data[primary_author_id]
            return {
                'author_id': primary_author_id,
                'author_name': author_data.get('name', ''),
                'author_average_rating': author_data.get('average_rating', 0.0),
                'author_ratings_count': author_data.get('ratings_count', 0)
            }
        else:
            return {
                'author_id': primary_author_id,
                'author_name': '',
                'author_average_rating': 0.0,
                'author_ratings_count': 0
            }
    
    def _get_series_info(self, work_id: str, series_data: Dict[str, Any], books_df: pd.DataFrame) -> Dict[str, Any]:
        """Get series information."""
        # Get series IDs from books data
        work_books = books_df[books_df['work_id'] == work_id]
        series_ids = []
        
        for _, book in work_books.iterrows():
            series = book.get('series', [])
            if series and isinstance(series, list):
                series_ids.extend([str(sid) for sid in series if sid])
        
        if not series_ids:
            return {'series_id': None, 'series_title': None, 'series_works_count': None}
        
        series_id = series_ids[0]
        
        if series_id in series_data:
            series_info = series_data[series_id]
            return {
                'series_id': series_id,
                'series_title': series_info.get('title'),
                'series_works_count': series_info.get('series_works_count')
            }
        else:
            return {
                'series_id': series_id,
                'series_title': None,
                'series_works_count': None
            }
    
    def _aggregate_english_editions(self, english_editions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across English editions."""
        if not english_editions:
            return {
                'ratings_count_sum': 0,
                'text_reviews_count_sum': 0,
                'average_rating_weighted_mean': None,
                'num_pages_median': None,
                'book_id_list_en': [],
                'language_codes_en': []
            }
        
        ratings_counts = []
        text_reviews_counts = []
        average_ratings = []
        ratings_weights = []
        num_pages = []
        book_ids = []
        language_codes = set()
        
        for edition in english_editions:
            ratings_count = edition.get('ratings_count', 0)
            text_reviews_count = edition.get('text_reviews_count', 0)
            average_rating = edition.get('average_rating')
            
            if pd.notna(ratings_count):
                try:
                    ratings_count = int(ratings_count)
                    ratings_counts.append(ratings_count)
                    ratings_weights.append(ratings_count)
                    
                    if pd.notna(average_rating):
                        try:
                            avg_rating = float(average_rating)
                            average_ratings.append(avg_rating)
                        except (ValueError, TypeError):
                            pass
                except (ValueError, TypeError):
                    pass
            
            if pd.notna(text_reviews_count):
                try:
                    text_reviews_counts.append(int(text_reviews_count))
                except (ValueError, TypeError):
                    pass
            
            pages = edition.get('num_pages')
            if pd.notna(pages):
                try:
                    pages = int(pages)
                    if pages > 0:
                        num_pages.append(pages)
                except (ValueError, TypeError):
                    pass
            
            book_id = edition.get('book_id')
            if pd.notna(book_id):
                book_ids.append(str(book_id))
            
            language_code = edition.get('language_code')
            if language_code:
                language_codes.add(str(language_code))
        
        ratings_count_sum = sum(ratings_counts) if ratings_counts else 0
        text_reviews_count_sum = sum(text_reviews_counts) if text_reviews_counts else 0
        
        # Calculate weighted mean
        average_rating_weighted_mean = None
        if average_ratings and ratings_weights and len(average_ratings) == len(ratings_weights):
            total_weight = sum(ratings_weights)
            if total_weight > 0:
                weighted_sum = sum(rating * weight for rating, weight in zip(average_ratings, ratings_weights))
                average_rating_weighted_mean = weighted_sum / total_weight
        
        num_pages_median = None
        if num_pages:
            num_pages_median = statistics.median(num_pages)
        
        # Sort book IDs by descending ratings_count, then by book_id
        book_id_ratings = []
        for edition in english_editions:
            book_id = edition.get('book_id')
            ratings_count = edition.get('ratings_count', 0)
            if pd.notna(book_id):
                try:
                    ratings_count = int(ratings_count) if pd.notna(ratings_count) else 0
                    book_id_ratings.append((str(book_id), ratings_count))
                except (ValueError, TypeError):
                    book_id_ratings.append((str(book_id), 0))
        
        book_id_ratings.sort(key=lambda x: (-x[1], x[0]))
        book_id_list_en = [bid for bid, _ in book_id_ratings]
        
        return {
            'ratings_count_sum': ratings_count_sum,
            'text_reviews_count_sum': text_reviews_count_sum,
            'average_rating_weighted_mean': average_rating_weighted_mean,
            'num_pages_median': num_pages_median,
            'book_id_list_en': book_id_list_en,
            'language_codes_en': sorted(list(language_codes))
        }
    
    def _get_longest_description(self, english_editions: List[Dict[str, Any]]) -> str:
        """Get the longest non-empty description."""
        descriptions = []
        for edition in english_editions:
            description = edition.get('description', '')
            if description and isinstance(description, str):
                description = description.strip()
                if description:
                    descriptions.append(description)
        
        if not descriptions:
            return ''
        
        return max(descriptions, key=len)
    
    def _get_all_popular_shelves(self, english_editions: List[Dict[str, Any]]) -> str:
        """Get union of all popular_shelves."""
        all_shelves = set()
        
        for edition in english_editions:
            shelves = edition.get('popular_shelves', [])
            if shelves and isinstance(shelves, list):
                for shelf in shelves:
                    if isinstance(shelf, dict) and 'name' in shelf:
                        all_shelves.add(shelf['name'])
        
        return ','.join(sorted(all_shelves))
    
    def _aggregate_genres_from_work(self, book_ids: List[str], genres_data: Dict[str, Any]) -> str:
        """
        Aggregate genres from all books in a work.
        
        Args:
            book_ids: List of book IDs that belong to the work
            genres_data: Dictionary mapping book_id to genres data
            
        Returns:
            Comma-separated string of unique genres
        """
        all_genres = set()
        
        for book_id in book_ids:
            if book_id in genres_data:
                book_genres = genres_data[book_id].get('genres', {})
                if isinstance(book_genres, dict):
                    # Add genre names to the set
                    all_genres.update(book_genres.keys())
        
        return ','.join(sorted(all_genres)) if all_genres else ''
