"""
Improved Final CSV Builder for Romance Novel Data Processing
Integrates title cleaning and uses best available title sources for improved dataset quality.
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

# Import the title cleaner
from .title_cleaner import TitleCleaner

logger = logging.getLogger(__name__)


class ImprovedFinalCSVBuilder:
    """
    Improved version of FinalCSVBuilder with title cleaning and better data quality.
    
    Key improvements:
    - Integrates TitleCleaner for series-free titles
    - Uses best available title sources (works.original_title > books.title_without_series > books.title)
    - Better data validation and quality checks
    - Enhanced logging and progress tracking
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
        # Initialize title cleaner
        self.title_cleaner = TitleCleaner()
        
        # English language regex pattern
        self.english_regex = re.compile(r'^(eng|en(?:-[A-Za-z]+)?)$', re.IGNORECASE)
        
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
    
    def is_english_language(self, language_code: str) -> bool:
        """Check if language code represents English."""
        if not language_code or not isinstance(language_code, str):
            return False
        return bool(self.english_regex.match(language_code.strip()))
    
    def get_best_title(self, work_info: Dict[str, Any], editions: List[Dict[str, Any]]) -> Tuple[str, str, dict]:
        """
        Get the best available title using priority order:
        1. works.original_title (cleanest, but may be empty)
        2. books.title_without_series (should be clean, but we verify)
        3. books.title (fallback, will be cleaned)
        
        Args:
            work_info: Work-level information from works data
            editions: List of book editions
            
        Returns:
            Tuple of (best_title, title_source, series_info)
        """
        # Priority 1: Try works.original_title first
        original_title = work_info.get('original_title', '').strip()
        if original_title:
            # Check if it's actually clean
            if not self.title_cleaner.is_series_title(original_title):
                return original_title, 'works.original_title', {}
            else:
                # Clean it if it has series info
                cleaned_title, removed_info = self.title_cleaner.clean_title(original_title)
                series_info = self.title_cleaner.extract_series_info(original_title)
                return cleaned_title, 'works.original_title_cleaned', series_info
        
        # Priority 2: Try title_without_series from editions
        for edition in editions:
            title_without_series = edition.get('title_without_series', '').strip()
            if title_without_series:
                # Verify it's actually clean
                if not self.title_cleaner.is_series_title(title_without_series):
                    return title_without_series, 'books.title_without_series', {}
                else:
                    # Clean it if it has series info
                    cleaned_title, removed_info = self.title_cleaner.clean_title(title_without_series)
                    series_info = self.title_cleaner.extract_series_info(title_without_series)
                    return cleaned_title, 'books.title_without_series_cleaned', series_info
        
        # Priority 3: Use regular title and clean it
        for edition in editions:
            title = edition.get('title', '').strip()
            if title:
                cleaned_title, removed_info = self.title_cleaner.clean_title(title)
                series_info = self.title_cleaner.extract_series_info(title)
                return cleaned_title, 'books.title_cleaned', series_info
        
        # Fallback: empty title
        return '', 'none', {}
    
    def load_books_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load books data into a pandas DataFrame for efficient processing."""
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
        
        # Convert key fields to proper types
        if 'work_id' in df.columns:
            df['work_id'] = pd.to_numeric(df['work_id'], errors='coerce')
        if 'book_id' in df.columns:
            df['book_id'] = pd.to_numeric(df['book_id'], errors='coerce')
        if 'publication_year' in df.columns:
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
        if 'num_pages' in df.columns:
            df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
        if 'ratings_count' in df.columns:
            df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
        if 'text_reviews_count' in df.columns:
            df['text_reviews_count'] = pd.to_numeric(df['text_reviews_count'], errors='coerce')
        if 'average_rating' in df.columns:
            df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Books DataFrame created: {df.shape}")
        return df
    
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
    
    def build_final_csv_improved(self, sample_size: Optional[int] = None) -> str:
        """
        Build the final CSV using improved title handling and data quality.
        
        Args:
            sample_size: Optional sample size for testing
            
        Returns:
            Path to the generated CSV file
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏗️ Building improved final CSV...")
        
        # Load books data as DataFrame
        books_df = self.load_books_dataframe(Path("data/raw/goodreads_books_romance.json.gz"))
        
        if books_df.empty:
            raise ValueError("Failed to load books data")
        
        # Load other data sources
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loading additional data sources...")
        works_data = self.load_json_data(Path("data/raw/goodreads_book_works.json.gz"))
        authors_data = self.load_json_data(Path("data/raw/goodreads_book_authors.json.gz"))
        series_data = self.load_json_data(Path("data/raw/goodreads_book_series.json.gz"))
        
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏗️ Processing works with improved title handling...")
        master_records = []
        processed_works = 0
        total_works = len(work_groups)
        
        # Track title sources for quality analysis
        title_sources = {
            'works.original_title': 0,
            'works.original_title_cleaned': 0,
            'books.title_without_series': 0,
            'books.title_without_series_cleaned': 0,
            'books.title_cleaned': 0,
            'none': 0
        }
        
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
                    continue
                
                # Get best available title
                best_title, title_source, series_info = self.get_best_title(work_info, editions)
                title_sources[title_source] += 1
                
                # Skip works with no valid title
                if not best_title:
                    continue
                
                # Select primary author
                primary_author_id = self._select_primary_author(editions)
                
                # Get author information
                author_info = self._get_author_info(primary_author_id, authors_data)
                
                # Get series information
                series_info_from_books = self._get_series_info(work_id, series_data, books_df)
                
                # Aggregate English editions
                aggregations = self._aggregate_english_editions(editions)
                
                # Get description and shelves
                description = self._get_longest_description(editions)
                popular_shelves = self._get_all_popular_shelves(editions)
                
                # Create master record with improved title handling
                master_record = {
                    'work_id': work_id,
                    'book_id_list_en': aggregations['book_id_list_en'],
                    'title': best_title,
                    'title_source': title_source,
                    'title_has_series_info': series_info.get('has_series', False),
                    'series_name_from_title': series_info.get('series_name', ''),
                    'series_number_from_title': series_info.get('series_number', ''),
                    'publication_year': publication_year,
                    'median_publication_year': median_publication_year,
                    'language_codes_en': aggregations['language_codes_en'],
                    'num_pages_median': aggregations['num_pages_median'],
                    'description': description,
                    'popular_shelves': popular_shelves,
                    'author_id': author_info['author_id'],
                    'author_name': author_info['author_name'],
                    'author_average_rating': author_info['author_average_rating'],
                    'author_ratings_count': author_info['author_ratings_count'],
                    'series_id': series_info_from_books['series_id'],
                    'series_title': series_info_from_books['series_title'],
                    'series_works_count': series_info_from_books['series_works_count'],
                    'ratings_count_sum': aggregations['ratings_count_sum'],
                    'text_reviews_count_sum': aggregations['text_reviews_count_sum'],
                    'average_rating_weighted_mean': aggregations['average_rating_weighted_mean'],
                    'average_rating_weighted_median': aggregations['average_rating_weighted_median']
                }
                
                master_records.append(master_record)
                processed_works += 1
                
                # Progress tracking
                if processed_works % 1000 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {processed_works:,}/{total_works:,} works...")
                    
            except Exception as e:
                self.logger.warning(f"Error processing work {work_id}: {e}")
                continue
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Master table built with {len(master_records)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(master_records)
        
        # Apply final filters
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Applying final filters...")
        df = df[df['publication_year'].between(2000, 2020, inclusive='both')]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📅 After year filter: {len(df)} records")
        
        # Validation checks
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Running validation checks...")
        
        # Check 1: No empty work_id or title
        empty_work_id = df['work_id'].isna().sum()
        empty_title = df['title'].isna().sum()
        
        if empty_work_id > 0 or empty_title > 0:
            raise ValueError(f"Validation failed: {empty_work_id} empty work_ids, {empty_title} empty titles")
        
        # Check 2: book_id_list_en is non-empty for every kept row
        empty_book_lists = df['book_id_list_en'].apply(lambda x: len(x) == 0).sum()
        if empty_book_lists > 0:
            raise ValueError(f"Validation failed: {empty_book_lists} rows with empty book_id_list_en")
        
        # Check 3: Publication year in range
        out_of_range = df[~df['publication_year'].between(2000, 2020, inclusive='both')]
        if len(out_of_range) > 0:
            raise ValueError(f"Validation failed: {len(out_of_range)} rows outside 2000-2020 range")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ All validation checks passed")
        
        # Sort by publication_year ascending, then ratings_count_sum descending, then work_id
        df = df.sort_values(['publication_year', 'ratings_count_sum', 'work_id'], 
                           ascending=[True, False, True])
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if sample_size:
            filename = f"final_books_2000_2020_en_improved_sampled_{sample_size}_{timestamp}.csv"
        else:
            filename = f"final_books_2000_2020_en_improved_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Save to CSV
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        
        # Log summary with title quality metrics
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Final CSV Summary:")
        print(f"  - Total works scanned: {total_works:,}")
        print(f"  - Works with ≥1 English edition: {len(master_records):,}")
        print(f"  - Works kept after year filter: {len(df):,}")
        print(f"  - Null publication_years: {df['publication_year'].isna().sum()}")
        print(f"  - Null author_ids: {df['author_id'].isna().sum()}")
        print(f"  - Null series_ids: {df['series_id'].isna().sum()}")
        
        print(f"\n📖 Title Quality Metrics:")
        for source, count in title_sources.items():
            if count > 0:
                percentage = (count / len(master_records)) * 100
                print(f"  - {source}: {count:,} ({percentage:.1f}%)")
        
        # Series information from titles
        titles_with_series = df['title_has_series_info'].sum()
        print(f"  - Titles with series info: {titles_with_series:,} ({titles_with_series/len(df)*100:.1f}%)")
        
        return str(output_path)
    
    # Reuse existing helper methods from the original class
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
                'average_rating_weighted_median': None,
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
        
        # Calculate weighted median
        average_rating_weighted_median = None
        if average_ratings and ratings_weights and len(average_ratings) == len(ratings_weights):
            total_weight = sum(ratings_weights)
            if total_weight > 0:
                weighted_data = [(rating, weight) for rating, weight in zip(average_ratings, ratings_weights)]
                weighted_data.sort(key=lambda x: x[0])  # Sort by rating value
                
                median_pos = total_weight / 2
                current_weight = 0
                prev_rating = None
                prev_weight_sum = 0
                
                for rating, weight in weighted_data:
                    current_weight += weight
                    
                    # If we've crossed the median position
                    if current_weight >= median_pos:
                        # If this is the first item or we're exactly at median position
                        if prev_rating is None or current_weight == median_pos:
                            average_rating_weighted_median = rating
                        else:
                            # Interpolate between prev_rating and current rating
                            weight_before_median = median_pos - prev_weight_sum
                            weight_after_median = current_weight - median_pos
                            
                            if weight_before_median == 0:
                                average_rating_weighted_median = prev_rating
                            elif weight_after_median == 0:
                                average_rating_weighted_median = rating
                            else:
                                # Linear interpolation
                                total_weight_in_interval = weight_before_median + weight_after_median
                                average_rating_weighted_median = (
                                    (prev_rating * weight_after_median + rating * weight_before_median) / 
                                    total_weight_in_interval
                                )
                        break
                    
                    prev_rating = rating
                    prev_weight_sum = current_weight
        
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
            'average_rating_weighted_median': average_rating_weighted_median,
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
