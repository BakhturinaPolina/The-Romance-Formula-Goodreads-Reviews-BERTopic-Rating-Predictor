"""
Final CSV Builder for Romance Novel Data Processing
Implements the exact specifications for building the final CSV with work-level aggregation.
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


class FinalCSVBuilder:
    """
    Builds the final CSV according to exact specifications.
    
    Key Requirements:
    - Work-level aggregation (not edition-level)
    - English editions only for aggregations
    - Proper publication year logic with fallback
    - Primary author selection with tie-breakers
    - Series integration
    - All required columns populated before filtering
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
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
        """
        Check if language code represents English.
        
        Args:
            language_code: Language code string
            
        Returns:
            True if English, False otherwise
        """
        if not language_code or not isinstance(language_code, str):
            return False
        return bool(self.english_regex.match(language_code.strip()))
    
    def load_json_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON data from gzipped file.
        
        Args:
            file_path: Path to JSON.gz file
            
        Returns:
            Dictionary mapping ID to data record
        """
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
    
    def get_publication_year_logic(self, work_info: Dict[str, Any], english_editions: List[Dict[str, Any]]) -> Tuple[int, Optional[int]]:
        """
        Implement publication year logic according to specifications.
        
        Args:
            work_info: Work-level information from works data
            english_editions: List of English editions for this work
            
        Returns:
            Tuple of (publication_year, median_publication_year)
        """
        # Prefer original_publication_year from works data
        original_pub_year = work_info.get('original_publication_year')
        if original_pub_year and str(original_pub_year).strip():
            try:
                original_year = int(original_pub_year)
                if 1800 <= original_year <= 2030:  # Reasonable range
                    return original_year, None
            except (ValueError, TypeError):
                pass
        
        # Fallback: calculate median from English editions
        english_years = []
        for edition in english_editions:
            pub_year = edition.get('publication_year')
            if pub_year is not None:
                try:
                    year = int(pub_year)
                    if 1800 <= year <= 2030:  # Reasonable range
                        english_years.append(year)
                except (ValueError, TypeError):
                    continue
        
        if english_years:
            median_year = int(statistics.median(english_years))
            return median_year, median_year
        else:
            return None, None
    
    def select_primary_author(self, english_editions: List[Dict[str, Any]]) -> Optional[str]:
        """
        Select primary author according to specifications.
        
        Args:
            english_editions: List of English editions for this work
            
        Returns:
            Primary author_id or None
        """
        # Count author occurrences
        author_counts = {}
        author_ratings = {}
        
        for edition in english_editions:
            # Extract author_id from authors field (list of dicts)
            authors = edition.get('authors', [])
            if authors and isinstance(authors, list) and len(authors) > 0:
                author_id = authors[0].get('author_id')
                if author_id:
                    author_id = str(author_id)
                    author_counts[author_id] = author_counts.get(author_id, 0) + 1
                    
                    # Sum ratings for tie-breaking
                    ratings_count = edition.get('ratings_count', 0)
                    if ratings_count:
                        try:
                            ratings_count = int(ratings_count)
                            author_ratings[author_id] = author_ratings.get(author_id, 0) + ratings_count
                        except (ValueError, TypeError):
                            pass
        
        if not author_counts:
            return None
        
        # Find author with highest count
        max_count = max(author_counts.values())
        most_frequent_authors = [aid for aid, count in author_counts.items() if count == max_count]
        
        if len(most_frequent_authors) == 1:
            return most_frequent_authors[0]
        
        # Tie-breaker 1: highest total ratings_count
        max_ratings = max(author_ratings.get(aid, 0) for aid in most_frequent_authors)
        highest_rated_authors = [aid for aid in most_frequent_authors if author_ratings.get(aid, 0) == max_ratings]
        
        if len(highest_rated_authors) == 1:
            return highest_rated_authors[0]
        
        # Tie-breaker 2: lexicographically smallest author_id
        return min(highest_rated_authors)
    
    def aggregate_english_editions(self, english_editions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across English editions only.
        
        Args:
            english_editions: List of English editions for this work
            
        Returns:
            Dictionary of aggregated metrics
        """
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
        
        # Collect metrics
        ratings_counts = []
        text_reviews_counts = []
        average_ratings = []
        ratings_weights = []
        num_pages = []
        book_ids = []
        language_codes = set()
        
        for edition in english_editions:
            # Ratings and reviews
            ratings_count = edition.get('ratings_count', 0)
            text_reviews_count = edition.get('text_reviews_count', 0)
            average_rating = edition.get('average_rating')
            
            if ratings_count:
                try:
                    ratings_count = int(ratings_count)
                    ratings_counts.append(ratings_count)
                    ratings_weights.append(ratings_count)
                    
                    if average_rating:
                        try:
                            avg_rating = float(average_rating)
                            average_ratings.append(avg_rating)
                        except (ValueError, TypeError):
                            pass
                except (ValueError, TypeError):
                    pass
            
            if text_reviews_count:
                try:
                    text_reviews_counts.append(int(text_reviews_count))
                except (ValueError, TypeError):
                    pass
            
            # Pages
            pages = edition.get('num_pages')
            if pages:
                try:
                    pages = int(pages)
                    if pages > 0:
                        num_pages.append(pages)
                except (ValueError, TypeError):
                    pass
            
            # Book ID and language
            book_id = edition.get('book_id')
            if book_id:
                book_ids.append(str(book_id))
            
            language_code = edition.get('language_code')
            if language_code:
                language_codes.add(str(language_code))
        
        # Calculate sums
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
                # Sort by rating, weighted by ratings_count
                weighted_data = [(rating, weight) for rating, weight in zip(average_ratings, ratings_weights)]
                weighted_data.sort(key=lambda x: x[0])
                
                # Find median position
                median_pos = total_weight / 2
                current_weight = 0
                
                for rating, weight in weighted_data:
                    current_weight += weight
                    if current_weight >= median_pos:
                        average_rating_weighted_median = rating
                        break
        
        # Calculate pages median
        num_pages_median = None
        if num_pages:
            num_pages_median = statistics.median(num_pages)
        
        # Sort book IDs by descending ratings_count, then by book_id
        book_id_ratings = []
        for edition in english_editions:
            book_id = edition.get('book_id')
            ratings_count = edition.get('ratings_count', 0)
            if book_id:
                try:
                    ratings_count = int(ratings_count) if ratings_count else 0
                    book_id_ratings.append((str(book_id), ratings_count))
                except (ValueError, TypeError):
                    book_id_ratings.append((str(book_id), 0))
        
        book_id_ratings.sort(key=lambda x: (-x[1], x[0]))  # Descending ratings, then ascending book_id
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
    
    def get_longest_description(self, english_editions: List[Dict[str, Any]]) -> str:
        """
        Get the longest non-empty description among English editions.
        
        Args:
            english_editions: List of English editions
            
        Returns:
            Longest description or empty string
        """
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
    
    def get_all_popular_shelves(self, english_editions: List[Dict[str, Any]]) -> str:
        """
        Get union of all popular_shelves across English editions.
        
        Args:
            english_editions: List of English editions
            
        Returns:
            Comma-separated string of unique shelves in alphabetical order
        """
        all_shelves = set()
        
        for edition in english_editions:
            shelves = edition.get('popular_shelves', [])
            if shelves and isinstance(shelves, list):
                # Extract shelf names from list of dictionaries
                for shelf in shelves:
                    if isinstance(shelf, dict) and 'name' in shelf:
                        all_shelves.add(shelf['name'])
        
        return ','.join(sorted(all_shelves))
    
    def get_series_info(self, work_id: str, series_data: Dict[str, Any], books_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get series information for a work.
        
        Args:
            work_id: Work ID
            series_data: Series data dictionary
            books_data: Books data dictionary
            
        Returns:
            Dictionary with series information
        """
        # First, check if any edition of this work has series information
        series_ids = []
        for book_id, book in books_data.items():
            if book.get('work_id') == work_id:
                series = book.get('series', [])
                if series and isinstance(series, list):
                    # Series is a list of series IDs (strings)
                    series_ids.extend([str(sid) for sid in series if sid])
        
        if not series_ids:
            return {'series_id': None, 'series_title': None, 'series_works_count': None}
        
        # Take the first series ID
        series_id = series_ids[0]
        
        # Look up the series in series_data
        if series_id in series_data:
            series_info = series_data[series_id]
            return {
                'series_id': series_id,
                'series_title': series_info.get('title'),
                'series_works_count': series_info.get('series_works_count')
            }
        else:
            # Return just the series ID if not found in series data
            return {
                'series_id': series_id,
                'series_title': None,
                'series_works_count': None
            }
    
    def build_final_csv(self, sample_size: Optional[int] = None) -> str:
        """
        Build the final CSV according to exact specifications.
        
        Args:
            sample_size: Optional sample size for testing
            
        Returns:
            Path to the generated CSV file
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏗️ Building final CSV according to specifications...")
        
        # Load all data sources
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loading data sources...")
        
        books_data = self.load_json_data(Path("data/raw/goodreads_books_romance.json.gz"))
        works_data = self.load_json_data(Path("data/raw/goodreads_book_works.json.gz"))
        authors_data = self.load_json_data(Path("data/raw/goodreads_book_authors.json.gz"))
        series_data = self.load_json_data(Path("data/raw/goodreads_book_series.json.gz"))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Data sources loaded")
        
        # Group books by work_id
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Grouping books by work_id...")
        work_groups = {}
        
        for book_id, book in books_data.items():
            work_id = book.get('work_id')
            if work_id:
                work_id = str(work_id)
                if work_id not in work_groups:
                    work_groups[work_id] = []
                work_groups[work_id].append(book)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Found {len(work_groups)} unique works")
        
        # Apply sampling if requested
        if sample_size and sample_size < len(work_groups):
            import random
            random.seed(42)  # For reproducibility
            sampled_works = random.sample(list(work_groups.keys()), sample_size)
            work_groups = {work_id: work_groups[work_id] for work_id in sampled_works}
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Sampled {len(work_groups)} works")
        
        # Build master table with all required columns
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏗️ Building master table...")
        master_records = []
        
        for work_id, editions in work_groups.items():
            try:
                # Filter to English editions only
                english_editions = [ed for ed in editions if self.is_english_language(ed.get('language_code', ''))]
                
                # Skip works with no English editions
                if not english_editions:
                    continue
                
                # Get work-level information
                work_info = works_data.get(work_id, {})
                
                # Get publication year logic
                publication_year, median_publication_year = self.get_publication_year_logic(work_info, english_editions)
                
                # Skip works with no valid publication year
                if publication_year is None:
                    continue
                
                # Select primary author
                primary_author_id = self.select_primary_author(english_editions)
                
                # Get author information
                author_info = {}
                if primary_author_id and primary_author_id in authors_data:
                    author_data = authors_data[primary_author_id]
                    author_info = {
                        'author_id': primary_author_id,
                        'author_name': author_data.get('name', ''),
                        'author_average_rating': author_data.get('average_rating', 0.0),
                        'author_ratings_count': author_data.get('ratings_count', 0)
                    }
                else:
                    author_info = {
                        'author_id': primary_author_id,
                        'author_name': '',
                        'author_average_rating': 0.0,
                        'author_ratings_count': 0
                    }
                
                # Get series information
                series_info = self.get_series_info(work_id, series_data, books_data)
                
                # Aggregate English editions
                aggregations = self.aggregate_english_editions(english_editions)
                
                # Get description and shelves
                description = self.get_longest_description(english_editions)
                popular_shelves = self.get_all_popular_shelves(english_editions)
                
                # Create master record with ALL required columns
                master_record = {
                    'work_id': work_id,
                    'book_id_list_en': aggregations['book_id_list_en'],
                    'title': work_info.get('original_title') or english_editions[0].get('title', ''),
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
                    'series_id': series_info['series_id'],
                    'series_title': series_info['series_title'],
                    'series_works_count': series_info['series_works_count'],
                    'ratings_count_sum': aggregations['ratings_count_sum'],
                    'text_reviews_count_sum': aggregations['text_reviews_count_sum'],
                    'average_rating_weighted_mean': aggregations['average_rating_weighted_mean'],
                    'average_rating_weighted_median': aggregations['average_rating_weighted_median']
                }
                
                master_records.append(master_record)
                
            except Exception as e:
                self.logger.warning(f"Error processing work {work_id}: {e}")
                continue
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Master table built with {len(master_records)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(master_records)
        
        # Apply final filters (only after all columns are populated)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Applying final filters...")
        
        # Filter 1: Publication year between 2000-2020
        df = df[df['publication_year'].between(2000, 2020, inclusive='both')]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📅 After year filter: {len(df)} records")
        
        # Filter 2: Must have at least one English edition (already enforced above)
        
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
            filename = f"final_books_2000_2020_en_sampled_{sample_size}_{timestamp}.csv"
        else:
            filename = f"final_books_2000_2020_en_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Save to CSV
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        
        # Log summary
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Final CSV Summary:")
        print(f"  - Total works scanned: {len(work_groups)}")
        print(f"  - Works with ≥1 English edition: {len(master_records)}")
        print(f"  - Works kept after year filter: {len(df)}")
        print(f"  - Null publication_years: {df['publication_year'].isna().sum()}")
        print(f"  - Null author_ids: {df['author_id'].isna().sum()}")
        print(f"  - Null series_ids: {df['series_id'].isna().sum()}")
        
        return str(output_path)
