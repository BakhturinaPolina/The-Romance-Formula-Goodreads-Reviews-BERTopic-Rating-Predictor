"""
Text Matching Module

This module provides functionality for fuzzy matching between Goodreads books
and external text datasets using title and author similarity.

Main Components:
- match_goodreads_to_texts: Core matching functionality
- run_matcher: Easy-to-use runner script with predefined configurations
- MatchConfig: Configuration class for tuning matching parameters

Author: Research Assistant
Date: 2025-01-09
"""

from .match_goodreads_to_texts import (
    MatchConfig,
    match_goodreads_to_texts,
    main,
    normalize_title,
    normalize_author,
    composite_score
)

__all__ = [
    "MatchConfig",
    "match_goodreads_to_texts", 
    "main",
    "normalize_title",
    "normalize_author",
    "composite_score"
]

__version__ = "1.0.0"
