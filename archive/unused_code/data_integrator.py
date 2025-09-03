"""
Data Integrator for Processing Pipeline
Handles joining different data sources and creating final datasets.
"""

import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
import statistics
from typing import Dict, List, Any, Optional, Tuple
import logging
import gzip

logger = logging.getLogger(__name__)


class DataIntegrator:
    """Integrates data from multiple sources into final datasets."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize the data integrator.
        
        Args:
            csv_schema: CSV schema configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for data integration operations."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _calculate_weighted_mean(self, values: List[float], weights: List[int]) -> float:
        """
        Calculate weighted mean of values.
        
        Args:
            values: List of numeric values
            weights: List of weights (must be same length as values)
            
        Returns:
            Weighted mean as float
        """
        if not values or not weights or len(values) != len(weights):
            return 0.0
        
        # Filter out None values and ensure numeric types
        valid_pairs = []
        for val, weight in zip(values, weights):
            if val is not None and weight is not None:
                try:
                    val_float = float(val) if not isinstance(val, (int, float)) else val
                    weight_int = int(weight) if not isinstance(weight, int) else weight
                    if weight_int > 0:  # Only include positive weights
                        valid_pairs.append((val_float, weight_int))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return 0.0
        
        total_weight = sum(weight for _, weight in valid_pairs)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(val * weight for val, weight in valid_pairs)
        return weighted_sum / total_weight

    def _calculate_weighted_median(self, values: List[float], weights: List[int]) -> float:
        """
        Calculate weighted median of values.
        
        Args:
            values: List of numeric values
            weights: List of weights (must be same length as values)
            
        Returns:
            Weighted median as float
        """
        if not values or not weights or len(values) != len(weights):
            return 0.0
        
        # Filter out None values and ensure numeric types
        valid_pairs = []
        for val, weight in zip(values, weights):
            if val is not None and weight is not None:
                try:
                    val_float = float(val) if not isinstance(val, (int, float)) else val
                    weight_int = int(weight) if not isinstance(weight, int) else weight
                    if weight_int > 0:  # Only include positive weights
                        valid_pairs.append((val_float, weight_int))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return 0.0
        
        # Sort by values
        valid_pairs.sort(key=lambda x: x[0])
        
        total_weight = sum(weight for _, weight in valid_pairs)
        if total_weight == 0:
            return 0.0
        
        # Find the median position
        median_pos = total_weight / 2
        current_weight = 0
        
        for val, weight in valid_pairs:
            current_weight += weight
            if current_weight >= median_pos:
                return val
        
        return valid_pairs[-1][0] if valid_pairs else 0.0

    def calculate_edition_aggregations(self, editions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate edition-level aggregations for a work.
        
        Args:
            editions: List of edition dictionaries for a work
            
        Returns:
            Dictionary with aggregated values
        """
        if not editions:
            return {
                'num_pages_median': None,
                'ratings_count_sum': 0,
                'text_reviews_count_sum': 0,
                'average_rating_weighted_mean': 0.0,
                'average_rating_weighted_median': 0.0
            }
        
        # Extract numeric values, handling None and type conversion
        pages_values = []
        ratings_counts = []
        text_reviews_counts = []
        avg_ratings = []
        ratings_weights = []
        
        for edition in editions:
            # Handle num_pages
            pages = edition.get('num_pages')
            if pages is not None:
                try:
                    pages_int = int(pages) if not isinstance(pages, int) else pages
                    if pages_int > 0:
                        pages_values.append(pages_int)
                except (ValueError, TypeError):
                    continue
            
            # Handle ratings_count
            ratings_count = edition.get('ratings_count')
            if ratings_count is not None:
                try:
                    ratings_int = int(ratings_count) if not isinstance(ratings_count, int) else ratings_count
                    if ratings_int > 0:
                        ratings_counts.append(ratings_int)
                        ratings_weights.append(ratings_int)
                except (ValueError, TypeError):
                    continue
            
            # Handle text_reviews_count
            text_reviews_count = edition.get('text_reviews_count')
            if text_reviews_count is not None:
                try:
                    text_reviews_int = int(text_reviews_count) if not isinstance(text_reviews_count, int) else text_reviews_count
                    if text_reviews_int >= 0:  # Allow 0 for text reviews
                        text_reviews_counts.append(text_reviews_int)
                except (ValueError, TypeError):
                    continue
            
            # Handle average_rating
            avg_rating = edition.get('average_rating')
            if avg_rating is not None:
                try:
                    avg_rating_float = float(avg_rating) if not isinstance(avg_rating, (int, float)) else avg_rating
                    if 0.0 <= avg_rating_float <= 5.0:  # Valid rating range
                        avg_ratings.append(avg_rating_float)
                except (ValueError, TypeError):
                    continue
        
        # Calculate aggregations
        num_pages_median = statistics.median(pages_values) if pages_values else None
        ratings_count_sum = sum(ratings_counts) if ratings_counts else 0
        text_reviews_count_sum = sum(text_reviews_counts) if text_reviews_counts else 0
        
        # FIX: If we have only one edition, ensure the sum equals the individual count
        if len(editions) == 1:
            single_edition = editions[0]
            # Use the individual edition's values if aggregation failed
            if text_reviews_count_sum == 0:
                text_reviews_count = single_edition.get('text_reviews_count')
                if text_reviews_count is not None:
                    try:
                        text_reviews_count_sum = int(text_reviews_count)
                    except (ValueError, TypeError):
                        text_reviews_count_sum = 0
            
            if ratings_count_sum == 0:
                ratings_count = single_edition.get('ratings_count')
                if ratings_count is not None:
                    try:
                        ratings_count_sum = int(ratings_count)
                    except (ValueError, TypeError):
                        ratings_count_sum = 0
        
        # Calculate weighted ratings (only if we have both ratings and weights)
        if avg_ratings and ratings_weights and len(avg_ratings) == len(ratings_weights):
            average_rating_weighted_mean = self._calculate_weighted_mean(avg_ratings, ratings_weights)
            average_rating_weighted_median = self._calculate_weighted_median(avg_ratings, ratings_weights)
        else:
            average_rating_weighted_mean = 0.0
            average_rating_weighted_median = 0.0
        
        return {
            'num_pages_median': num_pages_median,
            'ratings_count_sum': ratings_count_sum,
            'text_reviews_count_sum': text_reviews_count_sum,
            'average_rating_weighted_mean': average_rating_weighted_mean,
            'average_rating_weighted_median': average_rating_weighted_median
        }

    def integrate_works_data(self, books_data: List[Dict[str, Any]], 
                           variable_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Integrate works data to get better publication year information.
        
        Args:
            books_data: List of book records
            variable_selection: Variable selection configuration
            
        Returns:
            List of book records with improved publication year data
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Integrating works data for better publication years...")
        
        # Load works data
        works_file = Path("data/raw/goodreads_book_works.json.gz")
        if not works_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Works file not found: {works_file}")
            return books_data
        
        # Load works data
        works_data = {}
        records_processed = 0
        
        try:
            with gzip.open(works_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            work = json.loads(line.strip())
                            work_id = work.get('work_id')
                            if work_id:
                                works_data[work_id] = work
                            
                            records_processed += 1
                            if records_processed % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} work records...")
                                
                        except json.JSONDecodeError as e:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading works data: {e}")
            return books_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loaded {len(works_data):,} work records")
        
        # Integrate works data into books
        books_with_works = []
        improved_count = 0
        missing_work_count = 0
        
        for book in books_data:
            book_with_works = book.copy()
            work_id = book.get('work_id')
            
            if work_id and str(work_id) in works_data:
                work_info = works_data[str(work_id)]
                
                # Get original publication year from works data
                original_pub_year = work_info.get('original_publication_year')
                current_pub_year = book.get('publication_year')
                
                # Use original publication year if available and better
                if original_pub_year and original_pub_year.strip():
                    try:
                        original_year = int(original_pub_year)
                        if 1800 <= original_year <= 2030:  # Reasonable range
                            book_with_works['publication_year'] = original_year
                            book_with_works['original_publication_year'] = original_year
                            improved_count += 1
                        else:
                            book_with_works['original_publication_year'] = None
                    except (ValueError, TypeError):
                        book_with_works['original_publication_year'] = None
                else:
                    book_with_works['original_publication_year'] = None
            else:
                book_with_works['original_publication_year'] = None
                missing_work_count += 1
            
            books_with_works.append(book_with_works)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Works data integration completed:")
        print(f"  - Books with improved publication years: {improved_count:,}")
        print(f"  - Books missing works data: {missing_work_count:,}")
        print(f"  - Total books processed: {len(books_with_works):,}")
        
        return books_with_works

    def fix_problematic_publication_years(self, books_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix problematic publication years by calculating median year from all works 
        associated with the same book_id using best_book_id relationships.
        
        Args:
            books_data: List of book records
            
        Returns:
            List of book records with corrected publication years
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Fixing problematic publication years using median calculation...")
        
        # Load works data
        works_file = Path("data/raw/goodreads_book_works.json.gz")
        if not works_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Works file not found: {works_file}")
            return books_data
        
        # Load all works data and create mappings
        works_data = {}
        best_book_to_works = {}  # Maps best_book_id to list of work_ids
        work_to_best_book = {}   # Maps work_id to best_book_id
        records_processed = 0
        
        try:
            with gzip.open(works_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            work = json.loads(line.strip())
                            work_id = work.get('work_id')
                            best_book_id = work.get('best_book_id')
                            
                            if work_id:
                                works_data[work_id] = work
                                
                                # Create mappings for finding related works
                                if best_book_id:
                                    if best_book_id not in best_book_to_works:
                                        best_book_to_works[best_book_id] = []
                                    best_book_to_works[best_book_id].append(work_id)
                                    work_to_best_book[work_id] = best_book_id
                            
                            records_processed += 1
                            if records_processed % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} work records...")
                                
                        except json.JSONDecodeError as e:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading works data: {e}")
            return books_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loaded {len(works_data):,} work records")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Created {len(best_book_to_works):,} best_book_id mappings")
        
        # Identify books with problematic years
        problematic_books = []
        for book in books_data:
            pub_year = book.get('publication_year')
            # Ensure pub_year is an integer for comparison
            if pub_year is not None:
                try:
                    pub_year_int = int(pub_year) if not isinstance(pub_year, int) else pub_year
                    if pub_year_int < 1800 or pub_year_int > 2030 or pub_year_int < 10 or pub_year_int > 30000:
                        problematic_books.append(book)
                except (ValueError, TypeError):
                    # If we can't convert to int, it's problematic
                    problematic_books.append(book)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Found {len(problematic_books)} books with problematic publication years")
        
        # Calculate global median year from all valid works as fallback
        valid_works_years = []
        for work_id, work_info in works_data.items():
            original_pub_year = work_info.get('original_publication_year')
            if original_pub_year and original_pub_year.strip():
                try:
                    year = int(original_pub_year)
                    if 1800 <= year <= 2030:
                        valid_works_years.append(year)
                except (ValueError, TypeError):
                    continue
        
        if valid_works_years:
            global_median_year = int(statistics.median(valid_works_years))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Global median publication year from works: {global_median_year}")
        else:
            global_median_year = 2010  # Fallback
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  No valid works years found, using fallback: {global_median_year}")
        
        # Fix problematic books
        fixed_count = 0
        books_fixed = []
        
        for book in books_data:
            book_fixed = book.copy()
            pub_year = book.get('publication_year')
            
            # Check if this book has a problematic year
            if pub_year is not None:
                try:
                    pub_year_int = int(pub_year) if not isinstance(pub_year, int) else pub_year
                    is_problematic = pub_year_int < 1800 or pub_year_int > 2030 or pub_year_int < 10 or pub_year_int > 30000
                except (ValueError, TypeError):
                    is_problematic = True
            else:
                is_problematic = False
                
            if is_problematic:
                # Find related works using best_book_id relationships
                work_id = book.get('work_id')
                related_years = []
                
                if work_id and str(work_id) in work_to_best_book:
                    # Get the best_book_id for this work
                    best_book_id = work_to_best_book[str(work_id)]
                    
                    # Find all works associated with this best_book_id
                    if best_book_id in best_book_to_works:
                        related_work_ids = best_book_to_works[best_book_id]
                        
                        # Collect valid publication years from all related works
                        for related_work_id in related_work_ids:
                            if related_work_id in works_data:
                                work_info = works_data[related_work_id]
                                original_pub_year = work_info.get('original_publication_year')
                                if original_pub_year and original_pub_year.strip():
                                    try:
                                        year = int(original_pub_year)
                                        if 1800 <= year <= 2030:
                                            related_years.append(year)
                                    except (ValueError, TypeError):
                                        pass
                
                # If we found related years, use their median
                if related_years:
                    median_year = int(statistics.median(related_years))
                    book_fixed['publication_year'] = median_year
                    book_fixed['publication_year_source'] = 'median_from_related_works'
                    book_fixed['related_works_count'] = len(related_years)
                    fixed_count += 1
                else:
                    # Use global median as fallback
                    book_fixed['publication_year'] = global_median_year
                    book_fixed['publication_year_source'] = 'global_median_fallback'
                    book_fixed['related_works_count'] = 0
                    fixed_count += 1
            else:
                book_fixed['publication_year_source'] = 'original'
                book_fixed['related_works_count'] = None
            
            books_fixed.append(book_fixed)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Publication year fixing completed:")
        print(f"  - Books with problematic years fixed: {fixed_count:,}")
        print(f"  - Global median year used: {global_median_year}")
        print(f"  - Total books processed: {len(books_fixed):,}")
        
        return books_fixed

    def integrate_author_data(self, books_data: List[Dict[str, Any]], 
                            variable_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Integrate author data by loading authors file and joining with books data.
        
        Args:
            books_data: List of book dictionaries
            variable_selection: Variable selection configuration
            
        Returns:
            List of books with integrated author data
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Integrating author data...")
        
        # Load authors data
        authors_file = Path("data/raw/goodreads_book_authors.json.gz")
        if not authors_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Authors file not found: {authors_file}")
            return books_data
        
        # Load authors data
        authors_data = {}
        records_processed = 0
        
        try:
            with gzip.open(authors_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            author = json.loads(line.strip())
                            author_id = author.get('author_id')
                            if author_id:
                                authors_data[author_id] = author
                            
                            records_processed += 1
                            if records_processed % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} author records...")
                                
                        except json.JSONDecodeError as e:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading authors data: {e}")
            return books_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loaded {len(authors_data):,} author records")
        
        # Join author data with books
        books_with_authors = []
        joined_count = 0
        missing_author_count = 0
        
        for book in books_data:
            book_with_author = book.copy()
            
            # Extract author_id from book's authors field
            author_id = None
            authors = book.get('authors', [])
            if authors and isinstance(authors, list) and len(authors) > 0:
                author_id = authors[0].get('author_id')
            
            # If no author_id in authors field, try direct author_id field
            if not author_id:
                author_id = book.get('author_id')
            
            # Join with author data
            if author_id and str(author_id) in authors_data:
                author_data = authors_data[str(author_id)]
                book_with_author['author_id'] = author_id
                book_with_author['author_name'] = author_data.get('name', '')
                book_with_author['author_average_rating'] = author_data.get('average_rating', 0.0)
                book_with_author['author_ratings_count'] = author_data.get('ratings_count', 0)
                joined_count += 1
            else:
                # Set default values for missing author data
                book_with_author['author_id'] = author_id if author_id else None
                book_with_author['author_name'] = ''
                book_with_author['author_average_rating'] = 0.0
                book_with_author['author_ratings_count'] = 0
                missing_author_count += 1
            
            books_with_authors.append(book_with_author)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Author data integration completed:")
        print(f"  - Books with author data: {joined_count:,}")
        print(f"  - Books missing author data: {missing_author_count:,}")
        print(f"  - Total books processed: {len(books_with_authors):,}")
        
        return books_with_authors

    def integrate_books_data(self, books_data: List[Dict[str, Any]], 
                           variable_selection: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate books data with all required columns from multiple data sources.
        
        Args:
            books_data: List of book dictionaries with converted data types
            variable_selection: Variable selection configuration
            
        Returns:
            DataFrame with integrated books data including all required columns
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Integrating books data with all required columns...")
        
        # Group books by work_id to handle editions
        work_groups = {}
        for book in books_data:
            work_id = book.get('work_id')
            if work_id is not None:
                if work_id not in work_groups:
                    work_groups[work_id] = []
                work_groups[work_id].append(book)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Found {len(work_groups)} unique works")
        
        # Load additional data sources
        works_data = self._load_works_data()
        authors_data = self._load_authors_data()
        series_data = self._load_series_data()
        
        integrated_books = []
        processed_works = 0
        
        for work_id, editions in work_groups.items():
            try:
                # Get work-level information
                work_info = works_data.get(str(work_id), {})
                
                # Get author information
                author_info = self._get_author_info(editions, authors_data)
                
                # Get series information
                series_info = self._get_series_info(work_id, series_data)
                
                # Calculate edition aggregations
                aggregations = self.calculate_edition_aggregations(editions)
                
                # Create integrated book record with all required columns
                integrated_book = {
                    # Core book information
                    'work_id': work_id,
                    'book_id': editions[0].get('book_id'),  # Use first edition's book_id
                    
                    # Title from works data (preferred) or first edition
                    'title': work_info.get('title') or editions[0].get('title'),
                    
                    # Publication year logic: original_publication_year from works, fallback to median
                    'publication_year': self._get_publication_year(work_info, editions),
                    
                    # Language code (keep English editions only)
                    'language_code': editions[0].get('language_code'),
                    
                    # Description (longest across English editions)
                    'description': self._get_longest_description(editions),
                    
                    # Popular shelves (all across English editions)
                    'popular_shelves': self._get_all_popular_shelves(editions),
                    
                    # Author information
                    'author_id': author_info.get('author_id'),
                    'author_name': author_info.get('author_name'),
                    'author_average_rating': author_info.get('author_average_rating'),
                    'author_ratings_count': author_info.get('author_ratings_count'),
                    
                    # Series information
                    'series_id': series_info.get('series_id'),
                    'series_title': series_info.get('series_title'),
                    'series_works_count': series_info.get('series_works_count'),
                    
                    # Aggregated metrics
                    **aggregations,
                    
                    # Quality score (calculated from aggregated metrics)
                    'quality_score': self._calculate_quality_score(aggregations)
                }
                
                integrated_books.append(integrated_book)
                processed_works += 1
                
                # Progress tracking
                if processed_works % 10000 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {processed_works}/{len(work_groups)} works...")
                    
            except Exception as e:
                self.logger.warning(f"Error processing work {work_id}: {e}")
                continue
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Books integration completed: {len(integrated_books)} works")
        
        # Convert to DataFrame
        df = pd.DataFrame(integrated_books)
        
        # Ensure proper data types in DataFrame
        df = self._ensure_dataframe_types(df)
        
        return df

    def _load_works_data(self) -> Dict[str, Any]:
        """Load works data from goodreads_book_works.json.gz."""
        works_file = Path("data/raw/goodreads_book_works.json.gz")
        works_data = {}
        
        if not works_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Works file not found: {works_file}")
            return works_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loading works data...")
        
        try:
            with gzip.open(works_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            work = json.loads(line.strip())
                            work_id = work.get('work_id')
                            if work_id:
                                works_data[str(work_id)] = work
                            
                            if (line_num + 1) % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {line_num + 1:,} work records...")
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading works data: {e}")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Loaded {len(works_data):,} work records")
        return works_data
    
    def _load_authors_data(self) -> Dict[str, Any]:
        """Load authors data from goodreads_book_authors.json.gz."""
        authors_file = Path("data/raw/goodreads_book_authors.json.gz")
        authors_data = {}
        
        if not authors_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Authors file not found: {authors_file}")
            return authors_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 👥 Loading authors data...")
        
        try:
            with gzip.open(authors_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            author = json.loads(line.strip())
                            author_id = author.get('author_id')
                            if author_id:
                                authors_data[str(author_id)] = author
                            
                            if (line_num + 1) % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {line_num + 1:,} author records...")
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading authors data: {e}")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Loaded {len(authors_data):,} author records")
        return authors_data
    
    def _load_series_data(self) -> Dict[str, Any]:
        """Load series data from goodreads_book_series.json.gz."""
        series_file = Path("data/raw/goodreads_book_series.json.gz")
        series_data = {}
        
        if not series_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Series file not found: {series_file}")
            return series_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loading series data...")
        
        try:
            with gzip.open(series_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            series = json.loads(line.strip())
                            # Note: Series data doesn't have work_id, so we can't link it directly
                            # For now, we'll store series data by series_id for potential future use
                            series_id = series.get('series_id')
                            if series_id:
                                series_data[str(series_id)] = series
                            
                            if (line_num + 1) % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {line_num + 1:,} series records...")
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading series data: {e}")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Loaded {len(series_data):,} series records")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Note: Series data cannot be linked to works without additional mapping file")
        return series_data
    
    def _get_author_info(self, editions: List[Dict[str, Any]], authors_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get author information for a work."""
        # Look for author_id in editions
        for edition in editions:
            if 'authors' in edition and edition['authors']:
                author_id = edition['authors'][0].get('author_id')
                if author_id and str(author_id) in authors_data:
                    author = authors_data[str(author_id)]
                    return {
                        'author_id': author_id,
                        'author_name': author.get('name'),
                        'author_average_rating': author.get('average_rating'),
                        'author_ratings_count': author.get('ratings_count')
                    }
        
        return {}
    
    def _get_series_info(self, work_id: Any, series_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get series information for a work."""
        # Note: Series data cannot be linked to works without additional mapping file
        # For now, return empty dict - series fields will be None in the output
        return {}
    
    def _get_publication_year(self, work_info: Dict[str, Any], editions: List[Dict[str, Any]]) -> Any:
        """Get publication year with fallback logic."""
        # First try original_publication_year from works data
        if work_info and work_info.get('original_publication_year'):
            return work_info['original_publication_year']
        
        # Fallback: calculate median from editions (English editions only)
        english_years = []
        for edition in editions:
            lang_code = edition.get('language_code', '')
            if lang_code == 'eng' or lang_code.startswith('en-'):
                pub_year = edition.get('publication_year')
                if pub_year and pub_year != '':
                    try:
                        english_years.append(int(pub_year))
                    except (ValueError, TypeError):
                        continue
        
        if english_years:
            return int(statistics.median(english_years))
        
        return None
    
    def _get_longest_description(self, editions: List[Dict[str, Any]]) -> str:
        """Get the longest description across English editions."""
        longest_desc = ""
        for edition in editions:
            lang_code = edition.get('language_code', '')
            if lang_code == 'eng' or lang_code.startswith('en-'):
                desc = edition.get('description', '')
                if desc and len(desc) > len(longest_desc):
                    longest_desc = desc
        return longest_desc
    
    def _get_all_popular_shelves(self, editions: List[Dict[str, Any]]) -> List[str]:
        """Get all popular shelves across English editions."""
        all_shelves = []
        for edition in editions:
            lang_code = edition.get('language_code', '')
            if lang_code == 'eng' or lang_code.startswith('en-'):
                shelves = edition.get('popular_shelves', [])
                if isinstance(shelves, list):
                    for shelf in shelves:
                        if isinstance(shelf, str):
                            all_shelves.append(shelf)
                        elif isinstance(shelf, dict) and 'name' in shelf:
                            # Handle shelf dictionaries with 'name' field
                            all_shelves.append(shelf['name'])
                        elif isinstance(shelf, dict):
                            # Handle other shelf dictionaries by converting to string
                            all_shelves.append(str(shelf))
                elif isinstance(shelves, str):
                    all_shelves.append(shelves)
                elif isinstance(shelves, dict):
                    # Handle case where popular_shelves is a single dict
                    if 'name' in shelves:
                        all_shelves.append(shelves['name'])
                    else:
                        all_shelves.append(str(shelves))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_shelves = []
        for shelf in all_shelves:
            if shelf not in seen:
                seen.add(shelf)
                unique_shelves.append(shelf)
        
        return unique_shelves
    
    def _calculate_quality_score(self, aggregations: Dict[str, Any]) -> float:
        """Calculate quality score based on aggregated metrics."""
        try:
            rating = aggregations.get('average_rating_weighted_mean', 0)
            ratings_count = aggregations.get('ratings_count_sum', 0)
            reviews_count = aggregations.get('text_reviews_count_sum', 0)
            
            # Simple quality score: rating * log(ratings_count + 1) * log(reviews_count + 1)
            score = rating * (1 + 0.1 * (ratings_count ** 0.5)) * (1 + 0.1 * (reviews_count ** 0.5))
            return round(score, 3)
        except:
            return 0.0

    def _ensure_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame columns have appropriate data types.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with proper data types
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ensuring proper DataFrame data types...")
        
        # Define expected data types for each column
        type_mapping = {
            'book_id': 'Int64',  # pandas nullable integer
            'work_id': 'Int64',
            'num_pages': 'Int64',
            'publication_day': 'Int64',
            'publication_month': 'Int64',
            'publication_year': 'Int64',
            'num_pages_median': 'float64',
            'ratings_count_sum': 'Int64',
            'text_reviews_count_sum': 'Int64',
            'average_rating_weighted_mean': 'float64',
            'average_rating_weighted_median': 'float64',
            'quality_score': 'float64',
            'review_id': 'Int64',  # For reviews
            'user_id': 'Int64',    # For reviews
            'rating': 'Int64',     # For reviews
            'author_id': 'Int64',  # For author data
            'author_average_rating': 'float64',  # For author data
            'author_ratings_count': 'Int64',     # For author data
            'series_id': 'Int64',  # For series data
            'series_works_count': 'Int64',       # For series data
        }
        
        for column, dtype in type_mapping.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert column {column} to {dtype}: {e}")
        
        return df

    def _remove_individual_edition_fields(self, book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove individual edition-specific fields to avoid confusion with aggregated fields.
        
        Args:
            book: Book dictionary
            
        Returns:
            Book dictionary with individual edition fields removed
        """
        # Remove individual edition-specific fields
        fields_to_remove = [
            'average_rating',      # Individual edition rating
            'ratings_count',       # Individual edition ratings count
            'text_reviews_count'   # Individual edition reviews count
        ]
        
        for field in fields_to_remove:
            if field in book:
                del book[field]
        
        return book

    def _add_derived_fields(self, book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add derived fields to a book record with enhanced quality scoring.
        
        Args:
            book: Book dictionary
            
        Returns:
            Book dictionary with derived fields
        """
        quality_score = 0.0
        
        # Basic field presence (0.5 total)
        if book.get('title'):
            quality_score += 0.1
        if book.get('description'):
            quality_score += 0.1
        if book.get('publication_year') is not None:
            quality_score += 0.1
        if book.get('author_name'):
            quality_score += 0.1
        if book.get('average_rating_weighted_mean') is not None:  # Use aggregated rating instead
            quality_score += 0.1
        
        # Data quality indicators (0.5 total)
        # Description length
        description = book.get('description', '')
        if description and len(str(description).strip()) > 100:
            quality_score += 0.1
        
        # Rating quality
        avg_rating = book.get('average_rating')
        if avg_rating is not None:
            try:
                rating_float = float(avg_rating) if not isinstance(avg_rating, (int, float)) else avg_rating
                if 3.0 <= rating_float <= 5.0:
                    quality_score += 0.1
            except (ValueError, TypeError):
                pass
        
        # Review count quality
        ratings_count = book.get('ratings_count')
        if ratings_count is not None:
            try:
                count_int = int(ratings_count) if not isinstance(ratings_count, int) else ratings_count
                if count_int > 50:
                    quality_score += 0.1
            except (ValueError, TypeError):
                pass
        
        # Publication year recency
        pub_year = book.get('publication_year')
        if pub_year is not None:
            try:
                year_int = int(pub_year) if not isinstance(pub_year, int) else pub_year
                if 2000 <= year_int <= 2017:
                    quality_score += 0.1
            except (ValueError, TypeError):
                pass
        
        # Author information completeness
        author_name = book.get('author_name', '')
        if author_name and len(str(author_name).strip()) > 2:
            quality_score += 0.1
        
        book['quality_score'] = quality_score
        return book

    def sample_reviews_data(self, reviews_data: List[Dict[str, Any]], 
                          sampling_config: Dict[str, Any],
                          book_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Sample reviews data with proper data types and memory optimization.
        
        Args:
            reviews_data: List of review dictionaries with converted data types
            sampling_config: Sampling configuration
            book_ids: Optional list of book IDs to filter reviews (prevents orphaned reviews)
            
        Returns:
            DataFrame with sampled reviews data
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sampling reviews data...")
        
        # Filter reviews to only include those for books in our dataset (prevent orphaned reviews)
        if book_ids:
            book_ids_set = set(book_ids)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Filtering reviews for {len(book_ids_set):,} books...")
            filtered_reviews = [review for review in reviews_data if review.get('book_id') in book_ids_set]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Filtered from {len(reviews_data):,} to {len(filtered_reviews):,} reviews")
            reviews_data = filtered_reviews
        
        # Apply sampling before converting to DataFrame for memory efficiency
        sample_size = sampling_config.get('reviews_sample_size', 100000)
        
        if len(reviews_data) > sample_size:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sampling {sample_size:,} reviews from {len(reviews_data):,} total...")
            import numpy as np
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(len(reviews_data), size=sample_size, replace=False)
            sampled_reviews = [reviews_data[i] for i in sample_indices]
        else:
            sampled_reviews = reviews_data
        
        # Convert to DataFrame with optimized memory usage
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Converting to DataFrame...")
        df = pd.DataFrame(sampled_reviews)
        
        # Ensure proper data types
        df = self._ensure_dataframe_types(df)
        
        # Clear memory
        del sampled_reviews
        if len(reviews_data) > sample_size:
            del reviews_data
        import gc
        gc.collect()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Reviews sampling completed: {len(df)} reviews")
        return df

    def save_datasets(self, books_df: pd.DataFrame, reviews_df: pd.DataFrame, 
                     subgenre_df: pd.DataFrame) -> None:
        """
        Save datasets to CSV files with proper escaping and error handling.
        
        Args:
            books_df: Books DataFrame
            reviews_df: Reviews DataFrame  
            subgenre_df: Subgenre classification DataFrame
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving datasets to CSV...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Reorder and filter books DataFrame columns
            books_df_reordered = self._reorder_books_columns(books_df)
            
            # Save books dataset with error handling
            books_file = self.output_dir / f"romance_books_{timestamp}.csv"
            try:
                books_df_reordered.to_csv(books_file, index=False, quoting=1)  # QUOTE_ALL for proper escaping
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved books dataset: {books_file}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error saving books dataset: {e}")
                raise
            
            # Save reviews dataset with error handling
            reviews_file = self.output_dir / f"romance_reviews_{timestamp}.csv"
            try:
                reviews_df.to_csv(reviews_file, index=False, quoting=1)  # QUOTE_ALL for proper escaping
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved reviews dataset: {reviews_file}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error saving reviews dataset: {e}")
                raise
            
            # Save subgenre classification dataset with error handling
            subgenre_file = self.output_dir / f"subgenre_classification_details_{timestamp}.csv"
            try:
                subgenre_df.to_csv(subgenre_file, index=False, quoting=1)  # QUOTE_ALL for proper escaping
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved subgenre dataset: {subgenre_file}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error saving subgenre dataset: {e}")
                raise
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] All datasets saved successfully")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to save datasets: {e}")
            raise

    def integrate_series_data(self, books_data: List[Dict[str, Any]], 
                            variable_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Integrate series data by loading series file and joining with books data.
        
        Args:
            books_data: List of book dictionaries
            variable_selection: Variable selection configuration
            
        Returns:
            List of books with integrated series data
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Integrating series data...")
        
        # Load series data
        series_file = Path("data/raw/goodreads_book_series.json.gz")
        if not series_file.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Series file not found: {series_file}")
            return books_data
        
        # Load series data
        series_data = {}
        records_processed = 0
        
        try:
            with gzip.open(series_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            series_record = json.loads(line.strip())
                            series_id = series_record.get('series_id')
                            if series_id:
                                series_data[series_id] = series_record
                            
                            records_processed += 1
                            if records_processed % 100000 == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {records_processed:,} series records...")
                                
                        except json.JSONDecodeError as e:
                            continue
                            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error loading series data: {e}")
            return books_data
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Loaded {len(series_data):,} series records")
        
        # Join series data with books
        books_with_series = []
        joined_count = 0
        missing_series_count = 0
        
        for book in books_data:
            book_with_series = book.copy()
            
            # Extract series_id from book's series field (which is a list)
            series_ids = book.get('series', [])
            if isinstance(series_ids, str):
                # Handle case where series might be a string
                try:
                    series_ids = json.loads(series_ids)
                except (json.JSONDecodeError, TypeError):
                    series_ids = []
            
            # Use the first series_id if available
            series_id = series_ids[0] if series_ids else None
            
            # Join with series data
            if series_id and str(series_id) in series_data:
                series_info = series_data[str(series_id)]
                book_with_series['series_id'] = series_id
                book_with_series['series_title'] = series_info.get('title', '')
                book_with_series['series_works_count'] = series_info.get('series_works_count', 0)
                joined_count += 1
            else:
                # Add empty series fields for books not in series
                book_with_series['series_id'] = None
                book_with_series['series_title'] = None
                book_with_series['series_works_count'] = None
                missing_series_count += 1
            
            books_with_series.append(book_with_series)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Series integration completed:")
        print(f"   - Books with series: {joined_count}")
        print(f"   - Books without series: {missing_series_count}")
        print(f"   - Total books processed: {len(books_with_series)}")
        
        return books_with_series

    def integrate_reviews_data(self, books_data: List[Dict[str, Any]], 
                             reviews_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integrate reviews data to calculate aggregated metrics for books.
        
        Args:
            books_data: List of book dictionaries
            reviews_data: List of review dictionaries
            
        Returns:
            List of book dictionaries with aggregated metrics from reviews
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Integrating reviews data for aggregated metrics...")
        
        # Create a mapping from book_id to work_id for reviews
        book_to_work_mapping = {}
        for book in books_data:
            book_id = book.get('book_id')
            work_id = book.get('work_id')
            if book_id and work_id:
                book_to_work_mapping[str(book_id)] = work_id
        
        # Group reviews by work_id
        reviews_by_work = {}
        for review in reviews_data:
            book_id = str(review.get('book_id', ''))
            if book_id in book_to_work_mapping:
                work_id = book_to_work_mapping[book_id]
                if work_id not in reviews_by_work:
                    reviews_by_work[work_id] = []
                reviews_by_work[work_id].append(review)
        
        # Integrate reviews data into books
        books_with_reviews = []
        books_with_reviews_count = 0
        
        for book in books_data:
            book_with_reviews = book.copy()
            work_id = book.get('work_id')
            
            if work_id in reviews_by_work:
                reviews = reviews_by_work[work_id]
                
                # Calculate aggregated metrics from reviews
                ratings = [r.get('rating', 0) for r in reviews if r.get('rating') is not None]
                text_reviews = [r for r in reviews if r.get('review_text') and r.get('review_text').strip()]
                
                # Update aggregated metrics
                book_with_reviews['ratings_count_sum'] = len(ratings)
                book_with_reviews['text_reviews_count_sum'] = len(text_reviews)
                
                if ratings:
                    book_with_reviews['average_rating_weighted_mean'] = sum(ratings) / len(ratings)
                    book_with_reviews['average_rating_weighted_median'] = statistics.median(ratings)
                else:
                    book_with_reviews['average_rating_weighted_mean'] = 0.0
                    book_with_reviews['average_rating_weighted_median'] = 0.0
                
                # Recalculate quality score with new metrics
                book_with_reviews['quality_score'] = self._calculate_quality_score({
                    'average_rating_weighted_mean': book_with_reviews['average_rating_weighted_mean'],
                    'ratings_count_sum': book_with_reviews['ratings_count_sum'],
                    'text_reviews_count_sum': book_with_reviews['text_reviews_count_sum']
                })
                
                books_with_reviews_count += 1
            else:
                # No reviews found, keep existing aggregated values
                pass
            
            books_with_reviews.append(book_with_reviews)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Reviews integration completed:")
        print(f"   - Books with reviews: {books_with_reviews_count}")
        print(f"   - Books without reviews: {len(books_with_reviews) - books_with_reviews_count}")
        print(f"   - Total books processed: {len(books_with_reviews)}")
        
        return books_with_reviews

    def _reorder_books_columns(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder and filter books DataFrame columns according to specified requirements.
        
        Args:
            books_df: Original books DataFrame
            
        Returns:
            DataFrame with reordered and filtered columns
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Reordering and filtering books columns...")
        
        # Define the exact column order as specified
        desired_column_order = [
            'author_id',
            'author_average_rating', 
            'author_ratings_count',
            'author_name',
            'book_id',
            'work_id',
            'title',
            'series_id',
            'series_title',
            'series_works_count',
            'publication_year',
            'language_code',
            'num_pages',
            'description',
            'average_rating',
            'ratings_count',
            'text_reviews_count',
            'popular_shelves',
            'quality_score'
        ]
        
        # Columns to remove
        columns_to_remove = ['authors', 'series', 'format', 'publisher']
        
        # Get current columns
        current_columns = list(books_df.columns)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Current columns ({len(current_columns)}): {current_columns}")
        
        # Remove specified columns
        columns_after_removal = [col for col in current_columns if col not in columns_to_remove]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] After removing columns ({len(columns_after_removal)}): {columns_after_removal}")
        
        # Find columns that are in desired order
        ordered_columns = [col for col in desired_column_order if col in columns_after_removal]
        
        # Find additional columns not in desired order (keep them at the end)
        additional_columns = [col for col in columns_after_removal if col not in desired_column_order]
        
        # Create final column order: desired order + additional columns
        final_column_order = ordered_columns + additional_columns
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Final column order ({len(final_column_order)}): {final_column_order}")
        
        # Reorder DataFrame
        books_df_reordered = books_df[final_column_order]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Column reordering completed")
        
        return books_df_reordered

    def filter_reviews_for_books(self, reviews_data: List[Dict[str, Any]], book_ids: List[int]) -> pd.DataFrame:
        """
        Filter reviews to include only those for the specified books.
        
        Args:
            reviews_data: List of review dictionaries
            book_ids: List of book IDs to include
            
        Returns:
            DataFrame with filtered reviews
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Filtering reviews for {len(book_ids):,} books...")
        
        book_ids_set = set(book_ids)
        filtered_reviews = []
        
        for review in reviews_data:
            if review.get('book_id') in book_ids_set:
                filtered_reviews.append(review)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Found {len(filtered_reviews):,} reviews for {len(book_ids):,} books")
        
        # Convert to DataFrame
        reviews_df = pd.DataFrame(filtered_reviews)
        
        # Ensure proper data types
        if not reviews_df.empty:
            reviews_df = self._ensure_dataframe_types(reviews_df)
        
        return reviews_df

    def save_full_datasets(self, books_df: pd.DataFrame, reviews_df: pd.DataFrame, subgenre_df: pd.DataFrame) -> None:
        """
        Save full cleaned datasets (all books that pass quality thresholds).
        
        Args:
            books_df: Full cleaned books DataFrame
            reviews_df: Full reviews DataFrame for cleaned books
            subgenre_df: Subgenre classification DataFrame
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saving full cleaned datasets...")
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure output directories exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        # Save full cleaned books dataset
        books_filename = f"romance_books_full_cleaned_{timestamp}.csv"
        books_path = Path("data/processed") / books_filename
        books_df.to_csv(books_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved full cleaned books: {books_path} ({len(books_df):,} records)")
        
        # Save full reviews dataset
        reviews_filename = f"romance_reviews_full_cleaned_{timestamp}.csv"
        reviews_path = Path("data/processed") / reviews_filename
        reviews_df.to_csv(reviews_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved full cleaned reviews: {reviews_path} ({len(reviews_df):,} records)")
        
        # Save subgenre classification
        subgenre_filename = f"subgenre_classification_full_cleaned_{timestamp}.csv"
        subgenre_path = Path("data/processed") / subgenre_filename
        subgenre_df.to_csv(subgenre_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved subgenre classification: {subgenre_path} ({len(subgenre_df):,} records)")
        
        # Save processing summary
        summary = {
            "timestamp": timestamp,
            "dataset_type": "full_cleaned",
            "books_count": len(books_df),
            "reviews_count": len(reviews_df),
            "subgenre_count": len(subgenre_df),
            "processing_steps": [
                "quality_filtering",
                "edition_aggregation", 
                "author_integration",
                "series_integration",
                "works_integration",
                "publication_year_fixing"
            ]
        }
        
        summary_filename = f"processing_summary_full_cleaned_{timestamp}.json"
        summary_path = Path("logs") / summary_filename
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved processing summary: {summary_path}")

    def save_sampled_datasets(self, books_df: pd.DataFrame, reviews_df: pd.DataFrame, subgenre_df: pd.DataFrame, sample_size: int) -> None:
        """
        Save sampled datasets (subset of full cleaned datasets).
        
        Args:
            books_df: Sampled books DataFrame
            reviews_df: Sampled reviews DataFrame
            subgenre_df: Sampled subgenre classification DataFrame
            sample_size: Number of books in the sample
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saving sampled datasets...")
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure output directories exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        # Save sampled books dataset
        books_filename = f"romance_books_sampled_{sample_size}_{timestamp}.csv"
        books_path = Path("data/processed") / books_filename
        books_df.to_csv(books_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved sampled books: {books_path} ({len(books_df):,} records)")
        
        # Save sampled reviews dataset
        reviews_filename = f"romance_reviews_sampled_{sample_size}_{timestamp}.csv"
        reviews_path = Path("data/processed") / reviews_filename
        reviews_df.to_csv(reviews_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved sampled reviews: {reviews_path} ({len(reviews_df):,} records)")
        
        # Save sampled subgenre classification
        subgenre_filename = f"subgenre_classification_sampled_{sample_size}_{timestamp}.csv"
        subgenre_path = Path("data/processed") / subgenre_filename
        subgenre_df.to_csv(subgenre_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved sampled subgenre classification: {subgenre_path} ({len(subgenre_df):,} records)")
        
        # Save processing summary
        summary = {
            "timestamp": timestamp,
            "dataset_type": "sampled",
            "sample_size": sample_size,
            "books_count": len(books_df),
            "reviews_count": len(reviews_df),
            "subgenre_count": len(subgenre_df),
            "processing_steps": [
                "quality_filtering",
                "edition_aggregation",
                "author_integration", 
                "series_integration",
                "works_integration",
                "publication_year_fixing",
                "random_sampling",
                "review_sampling"
            ]
        }
        
        summary_filename = f"processing_summary_sampled_{sample_size}_{timestamp}.json"
        summary_path = Path("logs") / summary_filename
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Saved processing summary: {summary_path}")

    def save_datasets(self, books_df: pd.DataFrame, reviews_df: pd.DataFrame, subgenre_df: pd.DataFrame) -> None:
        """
        Legacy method for backward compatibility - now calls save_full_datasets.
        
        Args:
            books_df: Books DataFrame
            reviews_df: Reviews DataFrame
            subgenre_df: Subgenre classification DataFrame
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Using legacy save_datasets method - calling save_full_datasets...")
        self.save_full_datasets(books_df, reviews_df, subgenre_df)
