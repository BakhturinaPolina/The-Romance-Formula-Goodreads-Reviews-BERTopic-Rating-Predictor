"""
External Data Extraction Module

This module contains tools for extracting and matching books from external datasets
(e.g., Hugging Face romance-books dataset, BookRix) to Goodreads metadata.
"""

from .extract_romance_books import Extraction, extract_one
from .bookrix_extractor import extract_from_url, process_dataset
from .book_matcher import BookMatcher, MatchResult

__all__ = [
    'Extraction',
    'extract_one',
    'extract_from_url',
    'process_dataset',
    'BookMatcher',
    'MatchResult'
]

