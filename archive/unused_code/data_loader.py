"""
Data Loader for Processing Pipeline
Handles reading JSON files and extracting required fields based on configuration.
"""

import json
import gzip
import pandas as pd
import time
import os
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union, Iterator

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and processes JSON data files for the romance novel dataset."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing raw data files (defaults to DATA_DIR env var or "data/raw")
        """
        if data_dir is None:
            # Use DATA_DIR environment variable if set, otherwise default to "data/raw"
            data_dir = os.environ.get('DATA_DIR', 'data/raw')
            if data_dir != 'data/raw':
                # If DATA_DIR is set, append "/raw" to it
                data_dir = str(Path(data_dir) / "raw")
        
        self.data_dir = Path(data_dir)
        self.chunk_size = 1000
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for data loading operations."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _convert_data_types(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert string fields to appropriate data types based on schema analysis.
        Optimized for batch processing and memory efficiency.
        
        Args:
            data: List of dictionaries containing book/review data
            
        Returns:
            List of dictionaries with converted data types
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Converting data types for {len(data)} records...")
        
        converted_data = []
        conversion_errors = []
        batch_size = 10000  # Process in batches for memory efficiency
        
        # Pre-define conversion mappings for better performance
        numeric_conversions = {
            'text_reviews_count': int,
            'ratings_count': int,
            'num_pages': int,
            'publication_day': int,
            'publication_month': int,
            'publication_year': int,
            'book_id': int,
            'work_id': int,
            'average_rating': float,
            'user_id': int,    # For reviews data
            'rating': int,     # For reviews data
            'author_id': int,  # For author data
            'author_average_rating': float,  # For author data
            'author_ratings_count': int,     # For author data
        }
        
        string_fields = ['review_id', 'language_code', 'format', 'publisher']
        boolean_conversions = {'is_ebook': bool}
        
        # Process in batches for memory efficiency
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch_data = data[batch_start:batch_end]
            
            for i, record in enumerate(batch_data):
                global_index = batch_start + i
                try:
                    converted_record = record.copy()
                    
                    # Handle string fields that should remain as strings
                    for field in string_fields:
                        if field in converted_record and converted_record[field]:
                            value = converted_record[field]
                            if isinstance(value, str):
                                # Clean up string values
                                converted_record[field] = value.strip()
                            elif value is None or value == '':
                                converted_record[field] = None
                            else:
                                # Convert other types to string
                                converted_record[field] = str(value).strip()
                    
                    # Convert numeric fields
                    for field, target_type in numeric_conversions.items():
                        if field in converted_record and converted_record[field]:
                            value = converted_record[field]
                            if isinstance(value, str):
                                if value.strip():  # Only process non-empty strings
                                    try:
                                        if target_type == int:
                                            converted_record[field] = int(float(value))  # Handle "3.0" -> 3
                                        else:
                                            converted_record[field] = target_type(value)
                                    except (ValueError, TypeError) as e:
                                        # Log error and set to None
                                        conversion_errors.append({
                                            'record_index': global_index,
                                            'field': field,
                                            'value': str(value)[:50],  # Truncate for memory efficiency
                                            'error': str(e)[:100]      # Truncate error message
                                        })
                                        converted_record[field] = None
                                else:
                                    converted_record[field] = None
                            elif isinstance(value, (int, float)):
                                # Value is already numeric, just ensure correct type
                                try:
                                    if target_type == int:
                                        converted_record[field] = int(value)
                                    else:
                                        converted_record[field] = target_type(value)
                                except (ValueError, TypeError) as e:
                                    conversion_errors.append({
                                        'record_index': global_index,
                                        'field': field,
                                        'value': str(value)[:50],
                                        'error': str(e)[:100]
                                    })
                                    converted_record[field] = None
                            else:
                                # Handle None, empty, or other types
                                converted_record[field] = None
                    
                    # Convert boolean fields
                    for field, target_type in boolean_conversions.items():
                        if field in converted_record:
                            value = converted_record[field]
                            if isinstance(value, str):
                                if value.lower() == 'true':
                                    converted_record[field] = True
                                elif value.lower() == 'false':
                                    converted_record[field] = False
                                else:
                                    converted_record[field] = None
                    
                    converted_data.append(converted_record)
                    
                except Exception as e:
                    conversion_errors.append({
                        'record_index': global_index,
                        'field': 'record_processing',
                        'value': str(record)[:50] + '...' if len(str(record)) > 50 else str(record),
                        'error': str(e)[:100]
                    })
                    # Keep original record if conversion fails
                    converted_data.append(record)
            
            # Progress tracking and memory cleanup
            if (batch_end) % 50000 == 0 or batch_end == len(data):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Converted {batch_end}/{len(data)} records...")
                # Force garbage collection for large datasets
                if len(data) > 100000:
                    import gc
                    gc.collect()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data type conversion completed.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Conversion errors: {len(conversion_errors)}")
        
        if conversion_errors:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sample conversion errors:")
            for error in conversion_errors[:5]:
                print(f"  - Record {error['record_index']}, Field '{error['field']}', Value '{error['value']}': {error['error']}")
        
        return converted_data

    def load_books_data(self, variable_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Load books data with required fields.
        
        Args:
            variable_selection: Variable selection configuration
            
        Returns:
            List of book records with required fields
        """
        logger.info("Loading books data from goodreads_books_romance.json.gz")
        
        books_file = self.data_dir / "goodreads_books_romance.json.gz"
        if not books_file.exists():
            raise FileNotFoundError(f"Books file not found: {books_file}")
        
        # Get required fields from configuration
        book_metadata = variable_selection.get('book_metadata', {})
        essential_fields = book_metadata.get('essential_fields', [])
        optional_fields = book_metadata.get('optional_fields', [])
        
        # Create field mapping
        required_field_names = [field['name'] for field in essential_fields]
        optional_field_names = [field['name'] for field in optional_fields]
        all_field_names = required_field_names + optional_field_names
        
        books_data = []
        records_processed = 0
        start_time = time.time()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📖 Loading books from {books_file.name}...")
        
        try:
            with gzip.open(books_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            book = json.loads(line.strip())
                            
                            # Extract required fields
                            processed_book = self._extract_book_fields(book, all_field_names)
                            
                            if processed_book:
                                books_data.append(processed_book)
                            
                            records_processed += 1
                            
                            if records_processed % 50000 == 0:
                                elapsed = time.time() - start_time
                                rate = records_processed / elapsed if elapsed > 0 else 0
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} records ({rate:.0f} records/sec)")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading books file: {e}")
            raise
        
        total_time = time.time() - start_time
        logger.info(f"Loaded {len(books_data)} book records from {records_processed} total records in {total_time:.2f}s")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Loaded {len(books_data):,} books from {records_processed:,} records ({total_time:.2f}s)")
        
        # Convert data types
        books_data = self._convert_data_types(books_data)
        
        return books_data
    
    def load_authors_data(self, variable_selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Load authors data with required fields.
        
        Args:
            variable_selection: Variable selection configuration
            
        Returns:
            Dictionary mapping author_id to author data
        """
        logger.info("Loading authors data from goodreads_book_authors.json.gz")
        
        authors_file = self.data_dir / "goodreads_book_authors.json.gz"
        if not authors_file.exists():
            raise FileNotFoundError(f"Authors file not found: {authors_file}")
        
        # Get required fields from configuration
        author_metadata = variable_selection.get('author_metadata', {})
        essential_fields = author_metadata.get('essential_fields', [])
        
        # Create field mapping
        required_field_names = [field['name'] for field in essential_fields]
        
        authors_data = {}
        records_processed = 0
        
        try:
            with gzip.open(authors_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            author = json.loads(line.strip())
                            
                            # Extract required fields
                            processed_author = self._extract_author_fields(author, required_field_names)
                            
                            if processed_author and 'author_id' in processed_author:
                                authors_data[processed_author['author_id']] = processed_author
                            
                            records_processed += 1
                            
                            if records_processed % 10000 == 0:
                                logger.info(f"Processed {records_processed} author records")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading authors file: {e}")
            raise
        
        logger.info(f"Loaded {len(authors_data)} author records from {records_processed} total records")
        return authors_data
    
    def load_reviews_data(self, variable_selection: Dict[str, Any], 
                         book_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load reviews data with required fields.
        
        Args:
            variable_selection: Variable selection configuration
            book_ids: Optional list of book IDs to filter reviews
            
        Returns:
            List of review records with required fields
        """
        logger.info("Loading reviews data from goodreads_reviews_romance.json.gz")
        
        reviews_file = self.data_dir / "goodreads_reviews_romance.json.gz"
        if not reviews_file.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_file}")
        
        # Get required fields from configuration
        review_metadata = variable_selection.get('review_metadata', {})
        essential_fields = review_metadata.get('essential_fields', [])
        
        # Create field mapping
        required_field_names = [field['name'] for field in essential_fields]
        
        reviews_data = []
        records_processed = 0
        start_time = time.time()
        
        # Create set for faster lookup if filtering by book_ids
        book_ids_set = set(book_ids) if book_ids else None
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Loading reviews from {reviews_file.name}...")
        if book_ids_set:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Filtering for {len(book_ids_set):,} specific books...")
        
        try:
            with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            review = json.loads(line.strip())
                            
                            # Filter by book_id if specified
                            if book_ids_set and review.get('book_id') not in book_ids_set:
                                continue
                            
                            # Extract required fields
                            processed_review = self._extract_review_fields(review, required_field_names)
                            
                            if processed_review:
                                reviews_data.append(processed_review)
                            
                            records_processed += 1
                            
                            if records_processed % 50000 == 0:
                                elapsed = time.time() - start_time
                                rate = records_processed / elapsed if elapsed > 0 else 0
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} records ({rate:.0f} records/sec)")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading reviews file: {e}")
            raise
        
        total_time = time.time() - start_time
        logger.info(f"Loaded {len(reviews_data)} review records from {records_processed} total records in {total_time:.2f}s")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Loaded {len(reviews_data):,} reviews from {records_processed:,} records ({total_time:.2f}s)")
        
        # Convert data types
        reviews_data = self._convert_data_types(reviews_data)
        
        return reviews_data
    
    def load_genres_data(self, variable_selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Load genres data with required fields.
        
        Args:
            variable_selection: Variable selection configuration
            
        Returns:
            Dictionary mapping book_id to genres data
        """
        logger.info("Loading genres data from goodreads_book_genres_initial.json.gz")
        
        genres_file = self.data_dir / "goodreads_book_genres_initial.json.gz"
        if not genres_file.exists():
            raise FileNotFoundError(f"Genres file not found: {genres_file}")
        
        # Get required fields from configuration
        genre_metadata = variable_selection.get('genre_metadata', {})
        essential_fields = genre_metadata.get('essential_fields', [])
        
        # Create field mapping
        required_field_names = [field['name'] for field in essential_fields]
        
        genres_data = {}
        records_processed = 0
        
        try:
            with gzip.open(genres_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            genre_record = json.loads(line.strip())
                            
                            # Extract required fields
                            processed_genre = self._extract_genre_fields(genre_record, required_field_names)
                            
                            if processed_genre and 'book_id' in processed_genre:
                                genres_data[processed_genre['book_id']] = processed_genre
                            
                            records_processed += 1
                            
                            if records_processed % 10000 == 0:
                                logger.info(f"Processed {records_processed} genre records")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading genres file: {e}")
            raise
        
        logger.info(f"Loaded {len(genres_data)} genre records from {records_processed} total records")
        return genres_data
    
    def load_series_data(self, variable_selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Load series data with required fields.
        
        Args:
            variable_selection: Variable selection configuration
            
        Returns:
            Dictionary mapping series_id to series data
        """
        logger.info("Loading series data from goodreads_book_series.json.gz")
        
        series_file = self.data_dir / "goodreads_book_series.json.gz"
        if not series_file.exists():
            raise FileNotFoundError(f"Series file not found: {series_file}")
        
        # Get required fields from configuration
        series_metadata = variable_selection.get('series_metadata', {})
        optional_fields = series_metadata.get('optional_fields', [])
        
        # Create field mapping
        optional_field_names = [field['name'] for field in optional_fields]
        
        series_data = {}
        records_processed = 0
        
        try:
            with gzip.open(series_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            series = json.loads(line.strip())
                            
                            # Extract required fields
                            processed_series = self._extract_series_fields(series, optional_field_names)
                            
                            if processed_series and 'series_id' in processed_series:
                                series_data[processed_series['series_id']] = processed_series
                            
                            records_processed += 1
                            
                            if records_processed % 10000 == 0:
                                logger.info(f"Processed {records_processed} series records")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading series file: {e}")
            raise
        
        logger.info(f"Loaded {len(series_data)} series records from {records_processed} total records")
        return series_data
    
    def _extract_book_fields(self, book: Dict[str, Any], field_names: List[str]) -> Optional[Dict[str, Any]]:
        """Extract specified fields from book record."""
        extracted_book = {}
        
        for field_name in field_names:
            if field_name in book:
                extracted_book[field_name] = book[field_name]
            else:
                # Handle special cases
                if field_name == 'reviews_count':
                    extracted_book[field_name] = book.get('text_reviews_count', '')
                else:
                    extracted_book[field_name] = ''
        
        return extracted_book
    
    def _extract_author_fields(self, author: Dict[str, Any], field_names: List[str]) -> Optional[Dict[str, Any]]:
        """Extract specified fields from author record."""
        extracted_author = {}
        
        for field_name in field_names:
            if field_name in author:
                extracted_author[field_name] = author[field_name]
            else:
                extracted_author[field_name] = ''
        
        return extracted_author
    
    def _extract_review_fields(self, review: Dict[str, Any], field_names: List[str]) -> Optional[Dict[str, Any]]:
        """Extract specified fields from review record."""
        extracted_review = {}
        
        for field_name in field_names:
            if field_name in review:
                extracted_review[field_name] = review[field_name]
            else:
                extracted_review[field_name] = ''
        
        return extracted_review
    
    def _extract_genre_fields(self, genre_record: Dict[str, Any], field_names: List[str]) -> Optional[Dict[str, Any]]:
        """Extract specified fields from genre record."""
        extracted_genre = {}
        
        for field_name in field_names:
            if field_name in genre_record:
                extracted_genre[field_name] = genre_record[field_name]
            else:
                extracted_genre[field_name] = ''
        
        return extracted_genre
    
    def _extract_series_fields(self, series: Dict[str, Any], field_names: List[str]) -> Optional[Dict[str, Any]]:
        """Extract specified fields from series record."""
        extracted_series = {}
        
        for field_name in field_names:
            if field_name in series:
                extracted_series[field_name] = series[field_name]
            else:
                extracted_series[field_name] = ''
        
        return extracted_series

    def load_books_data_batch(self, variable_selection: Dict[str, Any], batch_size: int = 1000, max_batches: int = None) -> Iterator[List[Dict[str, Any]]]:
        """
        Load books data in batches to avoid memory issues.
        
        Args:
            variable_selection: Variable selection configuration
            batch_size: Number of records to process in each batch
            max_batches: Maximum number of batches to process (None for all)
            
        Yields:
            Batches of book records with required fields
        """
        logger.info(f"Loading books data in batches of {batch_size}")
        
        books_file = self.data_dir / "goodreads_books_romance.json.gz"
        if not books_file.exists():
            raise FileNotFoundError(f"Books file not found: {books_file}")
        
        # Get required fields from configuration
        book_metadata = variable_selection.get('book_metadata', {})
        essential_fields = book_metadata.get('essential_fields', [])
        optional_fields = book_metadata.get('optional_fields', [])
        
        # Create field mapping
        required_field_names = [field['name'] for field in essential_fields]
        optional_field_names = [field['name'] for field in optional_fields]
        all_field_names = required_field_names + optional_field_names
        
        batch = []
        records_processed = 0
        batches_yielded = 0
        start_time = time.time()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📖 Loading books in batches from {books_file.name}...")
        
        try:
            with gzip.open(books_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            book = json.loads(line.strip())
                            
                            # Extract required fields
                            processed_book = self._extract_book_fields(book, all_field_names)
                            
                            if processed_book:
                                batch.append(processed_book)
                            
                            records_processed += 1
                            
                            # Yield batch when it reaches the specified size
                            if len(batch) >= batch_size:
                                yield batch
                                batches_yielded += 1
                                batch = []  # Clear the batch to free memory
                                
                                elapsed = time.time() - start_time
                                rate = records_processed / elapsed if elapsed > 0 else 0
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Batch {batches_yielded}: {records_processed:,} records processed ({rate:.0f} records/sec)")
                                
                                # Check if we've reached max_batches
                                if max_batches and batches_yielded >= max_batches:
                                    logger.info(f"Reached maximum batches limit: {max_batches}")
                                    break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
            # Yield any remaining records in the final batch
            if batch:
                yield batch
                batches_yielded += 1
                
        except Exception as e:
            logger.error(f"Error reading books file: {e}")
            raise
        
        total_time = time.time() - start_time
        logger.info(f"Processed {records_processed} total records in {batches_yielded} batches in {total_time:.2f}s")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Processed {records_processed:,} records in {batches_yielded} batches ({total_time:.2f}s)")

    def load_reviews_data_batch(self, variable_selection: Dict[str, Any], book_ids: List[str] = None, 
                               batch_size: int = 1000, max_batches: int = None) -> Iterator[List[Dict[str, Any]]]:
        """
        Load reviews data in batches to avoid memory issues.
        
        Args:
            variable_selection: Variable selection configuration
            book_ids: Optional list of book IDs to filter reviews
            batch_size: Number of records to process in each batch
            max_batches: Maximum number of batches to process (None for all)
            
        Yields:
            Batches of review records with required fields
        """
        logger.info(f"Loading reviews data in batches of {batch_size}")
        
        reviews_file = self.data_dir / "goodreads_reviews_romance.json.gz"
        if not reviews_file.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_file}")
        
        # Get required fields from configuration
        review_metadata = variable_selection.get('review_metadata', {})
        essential_fields = review_metadata.get('essential_fields', [])
        
        # Create field mapping
        required_field_names = [field['name'] for field in essential_fields]
        
        batch = []
        records_processed = 0
        batches_yielded = 0
        start_time = time.time()
        
        # Create set for faster lookup if filtering by book_ids
        book_ids_set = set(book_ids) if book_ids else None
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Loading reviews in batches from {reviews_file.name}...")
        if book_ids_set:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Filtering for {len(book_ids_set):,} specific books...")
        
        try:
            with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            review = json.loads(line.strip())
                            
                            # Filter by book_id if specified
                            if book_ids_set and review.get('book_id') not in book_ids_set:
                                continue
                            
                            # Extract required fields
                            processed_review = self._extract_review_fields(review, required_field_names)
                            
                            if processed_review:
                                batch.append(processed_review)
                            
                            records_processed += 1
                            
                            # Yield batch when it reaches the specified size
                            if len(batch) >= batch_size:
                                yield batch
                                batches_yielded += 1
                                batch = []  # Clear the batch to free memory
                                
                                elapsed = time.time() - start_time
                                rate = records_processed / elapsed if elapsed > 0 else 0
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Batch {batches_yielded}: {records_processed:,} records processed ({rate:.0f} records/sec)")
                                
                                # Check if we've reached max_batches
                                if max_batches and batches_yielded >= max_batches:
                                    logger.info(f"Reached maximum batches limit: {max_batches}")
                                    break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                            
            # Yield any remaining records in the final batch
            if batch:
                yield batch
                batches_yielded += 1
                
        except Exception as e:
            logger.error(f"Error reading reviews file: {e}")
            raise
        
        total_time = time.time() - start_time
        logger.info(f"Processed {records_processed} total records in {batches_yielded} batches in {total_time:.2f}s")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Processed {records_processed:,} records in {batches_yielded} batches ({total_time:.2f}s)")

    def count_records(self, file_name: str) -> int:
        """
        Count total records in a JSON.gz file without loading them into memory.
        
        Args:
            file_name: Name of the file to count records in
            
        Returns:
            Total number of records in the file
        """
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Counting records in {file_name}")
        count = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
                        
                        if count % 100000 == 0:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Counted {count:,} records...")
                            
        except Exception as e:
            logger.error(f"Error counting records in {file_name}: {e}")
            raise
        
        logger.info(f"Total records in {file_name}: {count:,}")
        return count

    def sample_records(self, file_name: str, sample_size: int = 1000, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Sample records from a JSON.gz file without loading everything into memory.
        
        Args:
            file_name: Name of the file to sample from
            sample_size: Number of records to sample
            random_seed: Random seed for reproducible sampling
            
        Returns:
            List of sampled records
        """
        import random
        random.seed(random_seed)
        
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Sampling {sample_size} records from {file_name}")
        
        # First, count total records
        total_records = self.count_records(file_name)
        
        if sample_size >= total_records:
            logger.warning(f"Sample size ({sample_size}) >= total records ({total_records}), returning all records")
            return self._load_all_records(file_path)
        
        # Generate random indices for sampling
        sample_indices = set(random.sample(range(total_records), sample_size))
        
        sampled_records = []
        current_index = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        if current_index in sample_indices:
                            try:
                                record = json.loads(line.strip())
                                sampled_records.append(record)
                                
                                if len(sampled_records) % 100 == 0:
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Sampled {len(sampled_records)}/{sample_size} records...")
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON line: {e}")
                        
                        current_index += 1
                        
                        # Early exit if we've sampled enough records
                        if len(sampled_records) >= sample_size:
                            break
                            
        except Exception as e:
            logger.error(f"Error sampling records from {file_name}: {e}")
            raise
        
        logger.info(f"Successfully sampled {len(sampled_records)} records from {file_name}")
        return sampled_records

    def _load_all_records(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load all records from a file (used when sample size >= total records).
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of all records
        """
        records = []
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            records.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            
        except Exception as e:
            logger.error(f"Error loading records from {file_path}: {e}")
            raise
        
        return records
