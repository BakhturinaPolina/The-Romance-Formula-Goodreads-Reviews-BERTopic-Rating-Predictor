"""
Book Matching Algorithm for Romance Novel Corpus Creation

This module implements fuzzy matching between Goodreads metadata and Anna's Archive
to identify the correct books for download.

### Coding Agent Pattern
**Intent**: Enable autonomous book matching and corpus creation
**Problem**: Complex matching across different metadata sources
**Solution**: Fuzzy matching with confidence scoring and format preferences
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path

@dataclass
class MatchResult:
    """Result of a book matching attempt"""
    work_id: str
    title: str
    author: str
    year: int
    match_found: bool
    annas_id: Optional[str] = None
    annas_title: Optional[str] = None
    annas_author: Optional[str] = None
    available_formats: List[str] = None
    best_format: Optional[str] = None
    confidence_score: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.available_formats is None:
            self.available_formats = []

class BookMatcher:
    """
    Handles matching between Goodreads books and Anna's Archive entries.

    Uses title, author, and publication year for matching with configurable
    fuzzy matching thresholds.
    """

    def __init__(self, annas_client):
        """
        Initialize the book matcher.

        Args:
            annas_client: Anna's Archive API client instance
        """
        self.annas_client = annas_client
        self.logger = logging.getLogger(__name__)

        # Matching configuration
        self.title_weight = 0.4
        self.author_weight = 0.4
        self.year_weight = 0.2
        self.min_confidence = 0.7

        # Format preferences (epub > HTML > PDF)
        self.format_priority = ['epub', 'html', 'pdf']

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using multiple metrics"""
        if not str1 or not str2:
            return 0.0

        # Simple case-insensitive comparison first
        if str1.lower() == str2.lower():
            return 1.0

        # Use sequence matcher for more complex comparison
        import difflib
        return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def match_book(self, work_id: str, title: str, author: str, year: int) -> MatchResult:
        """
        Match a single book against Anna's Archive.

        Args:
            work_id: Goodreads work ID
            title: Book title
            author: Book author
            year: Publication year

        Returns:
            MatchResult with matching information
        """
        self.logger.debug(f"Matching book {work_id}: '{title}' by '{author}' ({year})")
        
        result = MatchResult(
            work_id=work_id,
            title=title,
            author=author,
            year=year,
            match_found=False
        )

        try:
            # Search Anna's Archive for the book
            search_results = self.annas_client.search_books(
                title=title,
                author=author,
                year=year
            )

            self.logger.debug(f"Search returned {len(search_results)} results for {work_id}")

            if not search_results:
                result.error_message = "No search results found"
                self.logger.debug(f"No search results for {work_id}")
                return result

            # Find the best match
            best_match = self._find_best_match(search_results, title, author, year)

            if best_match and best_match['confidence'] >= self.min_confidence:
                result.match_found = True
                result.annas_id = best_match['id']
                result.annas_title = best_match['title']
                result.annas_author = best_match['author']
                result.available_formats = best_match.get('formats', [])
                result.best_format = self._select_best_format(result.available_formats)
                result.confidence_score = best_match['confidence']
                self.logger.debug(f"Match found for {work_id}: confidence={best_match['confidence']:.3f}, format={result.best_format}")
            else:
                confidence = best_match['confidence'] if best_match else 0.0
                result.error_message = f"No match found above confidence threshold {self.min_confidence} (best: {confidence:.3f})"
                self.logger.debug(f"No suitable match for {work_id}: best confidence {confidence:.3f} < {self.min_confidence}")

        except Exception as e:
            result.error_message = f"Matching error: {str(e)}"
            self.logger.error(f"Error matching book {work_id}: {e}")

        return result

    def _find_best_match(self, search_results: List[Dict], title: str, author: str, year: int) -> Optional[Dict]:
        """Find the best matching book from search results"""
        best_match = None
        best_score = 0.0

        self.logger.debug(f"Evaluating {len(search_results)} search results for best match")

        for i, result in enumerate(search_results):
            # Calculate similarity scores
            title_sim = self.calculate_similarity(title, result.get('title', ''))
            author_sim = self.calculate_similarity(author, result.get('author', ''))
            year_sim = 1.0 if abs(year - result.get('year', 0)) <= 1 else 0.0

            # Weighted confidence score
            confidence = (
                title_sim * self.title_weight +
                author_sim * self.author_weight +
                year_sim * self.year_weight
            )

            self.logger.debug(f"Result {i+1}: title_sim={title_sim:.3f}, author_sim={author_sim:.3f}, year_sim={year_sim:.3f}, confidence={confidence:.3f}")

            if confidence > best_score:
                best_score = confidence
                best_match = {
                    'id': result.get('id'),
                    'title': result.get('title'),
                    'author': result.get('author'),
                    'year': result.get('year'),
                    'formats': result.get('formats', []),
                    'confidence': confidence
                }

        self.logger.debug(f"Best match: confidence={best_score:.3f}")
        return best_match

    def _select_best_format(self, available_formats: List[str]) -> Optional[str]:
        """Select the best format based on preference order"""
        for preferred_format in self.format_priority:
            if preferred_format in available_formats:
                return preferred_format
        return None if not available_formats else available_formats[0]

    def match_books_batch(self, books_df: pd.DataFrame) -> List[MatchResult]:
        """
        Match multiple books in batch for efficiency.

        Args:
            books_df: DataFrame with columns [work_id, title, author_name, publication_year]

        Returns:
            List of MatchResult objects
        """
        results = []

        for _, book in books_df.iterrows():
            result = self.match_book(
                work_id=str(book['work_id']),
                title=book['title'],
                author=book['author_name'],
                year=int(book['publication_year'])
            )
            results.append(result)

            # Log progress
            if len(results) % 10 == 0:
                self.logger.info(f"Processed {len(results)}/{len(books_df)} books")

        return results

    def save_match_results(self, results: List[MatchResult], output_path: Path):
        """Save match results to CSV for analysis"""
        results_data = []

        for result in results:
            results_data.append({
                'work_id': result.work_id,
                'title': result.title,
                'author': result.author,
                'year': result.year,
                'match_found': result.match_found,
                'annas_id': result.annas_id,
                'annas_title': result.annas_title,
                'annas_author': result.annas_author,
                'available_formats': ','.join(result.available_formats) if result.available_formats else '',
                'best_format': result.best_format,
                'confidence_score': result.confidence_score,
                'error_message': result.error_message
            })

        df = pd.DataFrame(results_data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved match results to {output_path}")

    def get_match_statistics(self, results: List[MatchResult]) -> Dict:
        """Get statistics about the matching process"""
        total = len(results)
        found = sum(1 for r in results if r.match_found)
        not_found = total - found

        # Format distribution
        format_counts = {}
        for result in results:
            if result.match_found and result.best_format:
                format_counts[result.best_format] = format_counts.get(result.best_format, 0) + 1

        # Confidence distribution
        confidence_scores = [r.confidence_score for r in results if r.match_found]

        return {
            'total_books': total,
            'matches_found': found,
            'matches_not_found': not_found,
            'success_rate': found / total if total > 0 else 0,
            'format_distribution': format_counts,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'min_confidence': min(confidence_scores) if confidence_scores else 0,
            'max_confidence': max(confidence_scores) if confidence_scores else 0
        }
