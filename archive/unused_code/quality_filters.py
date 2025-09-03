"""
Quality Filters for Data Processing Pipeline
Implements filtering logic based on sampling policy configuration.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import gzip
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityFilters:
    """Implements quality filtering for romance novel data."""
    
    def __init__(self, sampling_policy: Dict[str, Any]):
        """
        Initialize quality filters with sampling policy.
        
        Args:
            sampling_policy: Sampling policy configuration
        """
        self.policy = sampling_policy
        self.filters = sampling_policy.get('quality_filters', {})
        self.random_seed = sampling_policy.get('author_limits', {}).get('random_seed', 42)
        
    def filter_books(self, books_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply quality filters to books data.
        
        Args:
            books_data: List of book records
            
        Returns:
            Filtered list of books
        """
        logger.info(f"Applying quality filters to {len(books_data)} books")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Applying quality filters to {len(books_data):,} books...")
        
        start_time = time.time()
        filtered_books = []
        filter_stats = {
            'total_books': len(books_data),
            'filtered_out': 0,
            'filter_reasons': {}
        }
        
        for i, book in enumerate(books_data):
            is_valid, reason = self._validate_book(book)
            
            if is_valid:
                filtered_books.append(book)
            else:
                filter_stats['filtered_out'] += 1
                if reason not in filter_stats['filter_reasons']:
                    filter_stats['filter_reasons'][reason] = 0
                filter_stats['filter_reasons'][reason] += 1
            
            # Progress tracking every 10,000 books
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Processed {i + 1:,} books ({rate:.0f} books/sec)")
        
        total_time = time.time() - start_time
        logger.info(f"Quality filtering complete: {len(filtered_books)} books passed filters in {total_time:.2f}s")
        logger.info(f"Filter statistics: {filter_stats}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Quality filtering complete: {len(filtered_books):,} books passed ({total_time:.2f}s)")
        
        return filtered_books
    
    def _validate_book(self, book: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single book against quality filters.
        
        Args:
            book: Book record to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Publication year filter (2000-2020)
        if not self._check_publication_year(book):
            return False, "publication_year_out_of_range"
        
        # Language filter (keep English and en-* patterns)
        if not self._check_language(book):
            return False, "non_english_language"
        
        return True, None
    
    def _check_publication_year(self, book: Dict[str, Any]) -> bool:
        """Check if book meets publication year requirements (2000-2020)."""
        min_year = self.filters.get('publication_year_min', 2000)
        max_year = self.filters.get('publication_year_max', 2020)
        
        try:
            pub_year = book.get('publication_year')
            if not pub_year or pub_year == '':
                return False
            
            pub_year = int(pub_year)
            
            # Clean obviously wrong publication years
            cleaned_year = self._clean_publication_year(pub_year)
            if cleaned_year != pub_year:
                book['publication_year'] = cleaned_year
                pub_year = cleaned_year
            
            return min_year <= pub_year <= max_year
        except (ValueError, TypeError):
            return False
    
    def _clean_publication_year(self, year: int) -> int:
        """
        Clean publication year by fixing common data errors.
        Now more aggressive since we have better data from works dataset.
        
        Args:
            year: Raw publication year
            
        Returns:
            Cleaned publication year
        """
        # Handle obviously wrong years (likely data errors)
        if year < 100:
            # Years like 12, 15, 16 are likely typos for 2012, 2015, 2016
            if 10 <= year <= 99:
                return 2000 + year
            else:
                return 2000  # Default to 2000 for very small numbers
        
        # Handle specific problematic years we've seen
        elif year in [201, 293]:  # These are likely typos for 2001, 2003
            return 2000 + (year % 100)
        
        elif year > 2100:
            # Years like 2105, 2107 are likely typos for 2005, 2007
            if 2100 < year <= 2199:
                return 2000 + (year - 2100)
            # Handle years like 22012 (likely 2012)
            elif 22000 <= year <= 22099:
                return 2000 + (year - 22000)
            # Handle years like 65535 (likely data corruption, default to 2000)
            elif year > 30000:
                return 2000
            else:
                return 2020  # Default to 2020 for other large numbers
        
        elif 1800 <= year <= 2030:
            # Year is in reasonable range
            return year
        
        else:
            # Year is outside reasonable range but not obviously wrong
            # Default to 2000 for very old books, 2020 for future books
            if year < 1800:
                return 2000
            else:
                return 2020
    
    def _check_language(self, book: Dict[str, Any]) -> bool:
        """Check if book is in English (eng or en-* patterns)."""
        language_code = book.get('language_code', '')
        
        # Accept 'eng' or any pattern starting with 'en-'
        return language_code == 'eng' or language_code.startswith('en-')
    
    def _check_ratings_count(self, book: Dict[str, Any]) -> bool:
        """Check if book has sufficient ratings."""
        min_ratings = self.filters.get('ratings_count_min', 10)
        
        try:
            ratings_count = book.get('ratings_count')
            if not ratings_count or ratings_count == '':
                return False
            
            ratings_count = int(ratings_count)
            return ratings_count >= min_ratings
        except (ValueError, TypeError):
            return False
    
    def _check_description_completeness(self, book: Dict[str, Any]) -> bool:
        """Check if book has description."""
        description = book.get('description', '')
        return description and description.strip() != ''
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of applied filters."""
        return {
            'publication_year_min': self.filters.get('publication_year_min', 2000),
            'language_code': self.filters.get('language_code', 'eng'),
            'ratings_count_min': self.filters.get('ratings_count_min', 10),
            'description_required': True,
            'random_seed': self.random_seed
        }


class AuthorBalancer:
    """Implements author balancing to limit prolific authors."""
    
    def __init__(self, sampling_policy: Dict[str, Any]):
        """
        Initialize author balancer with sampling policy.
        
        Args:
            sampling_policy: Sampling policy configuration
        """
        self.policy = sampling_policy
        self.author_limits = sampling_policy.get('author_limits', {})
        self.max_books_per_author = self.author_limits.get('max_books_per_author', 1)
        self.random_seed = self.author_limits.get('random_seed', 42)
        
    def balance_authors(self, books_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Balance books by limiting books per author.
        
        Args:
            books_data: List of book records
            
        Returns:
            Balanced list of books
        """
        logger.info(f"Balancing authors for {len(books_data)} books")
        
        # Group books by author
        author_books = {}
        for book in books_data:
            author_id = self._extract_author_id(book)
            if author_id:
                if author_id not in author_books:
                    author_books[author_id] = []
                author_books[author_id].append(book)
        
        # Sample books per author
        balanced_books = []
        import random
        random.seed(self.random_seed)
        
        for author_id, books in author_books.items():
            if len(books) <= self.max_books_per_author:
                balanced_books.extend(books)
            else:
                # Randomly sample books from prolific authors
                sampled_books = random.sample(books, self.max_books_per_author)
                balanced_books.extend(sampled_books)
        
        logger.info(f"Author balancing complete: {len(balanced_books)} books after balancing")
        logger.info(f"Original authors: {len(author_books)}, Books per author limit: {self.max_books_per_author}")
        
        return balanced_books
    
    def _extract_author_id(self, book: Dict[str, Any]) -> Optional[str]:
        """Extract author ID from book record."""
        # Try to get author_id from authors field
        authors = book.get('authors', [])
        if authors and isinstance(authors, list) and len(authors) > 0:
            return authors[0].get('author_id')
        
        # Fallback to direct author_id field
        return book.get('author_id')


class DecadeStratifier:
    """Implements decade stratification for balanced temporal distribution."""
    
    def __init__(self, sampling_policy: Dict[str, Any]):
        """
        Initialize decade stratifier with sampling policy.
        
        Args:
            sampling_policy: Sampling policy configuration
        """
        self.policy = sampling_policy
        self.decade_config = sampling_policy.get('decade_stratification', {})
        self.decades = self.decade_config.get('decades', ['2000-2009', '2010-2017'])
        self.target_books_per_decade = self.decade_config.get('target_books_per_decade', 1000)
        self.random_seed = self.decade_config.get('random_seed', 42)
        
    def stratify_by_decade(self, books_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stratify books by decade for balanced temporal distribution.
        
        Args:
            books_data: List of book records
            
        Returns:
            Stratified list of books
        """
        logger.info(f"Stratifying {len(books_data)} books by decade")
        
        # Group books by decade
        decade_books = {'2000-2009': [], '2010-2017': []}
        
        for book in books_data:
            decade = self._get_book_decade(book)
            if decade in decade_books:
                decade_books[decade].append(book)
        
        # Sample from each decade
        stratified_books = []
        import random
        random.seed(self.random_seed)
        
        for decade, books in decade_books.items():
            if len(books) <= self.target_books_per_decade:
                stratified_books.extend(books)
            else:
                sampled_books = random.sample(books, self.target_books_per_decade)
                stratified_books.extend(sampled_books)
        
        logger.info(f"Decade stratification complete: {len(stratified_books)} books")
        for decade, books in decade_books.items():
            logger.info(f"  {decade}: {len(books)} original, {min(len(books), self.target_books_per_decade)} sampled")
        
        return stratified_books
    
    def _get_book_decade(self, book: Dict[str, Any]) -> str:
        """Get decade for a book based on publication year."""
        try:
            pub_year = book.get('publication_year')
            if not pub_year or pub_year == '':
                return '2010-2017'  # Default to newer decade
            
            pub_year = int(pub_year)
            if 2000 <= pub_year <= 2009:
                return '2000-2009'
            elif 2010 <= pub_year <= 2017:
                return '2010-2017'
            else:
                return '2010-2017'  # Default for out-of-range years
        except (ValueError, TypeError):
            return '2010-2017'  # Default for invalid years


def apply_quality_filters(books_data: List[Dict[str, Any]], 
                         sampling_policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply all quality filters to books data.
    
    Args:
        books_data: List of book records
        sampling_policy: Sampling policy configuration
        
    Returns:
        Filtered and balanced list of books
    """
    logger.info("Starting quality filtering pipeline")
    
    # Step 1: Apply quality filters
    quality_filter = QualityFilters(sampling_policy)
    filtered_books = quality_filter.filter_books(books_data)
    
    # Step 2: Balance authors
    author_balancer = AuthorBalancer(sampling_policy)
    balanced_books = author_balancer.balance_authors(filtered_books)
    
    # Step 3: Stratify by decade
    decade_stratifier = DecadeStratifier(sampling_policy)
    final_books = decade_stratifier.stratify_by_decade(balanced_books)
    
    logger.info(f"Quality filtering pipeline complete: {len(final_books)} books remaining")
    
    return final_books
