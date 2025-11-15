"""
Topic modeling module using BERTopic with OCTIS integration.

This module provides tools for:
- BERTopic topic modeling on review sentences
- OCTIS framework integration for hyperparameter optimization
- Topic model evaluation and analysis
"""

# Note: Data preparation utilities have been moved to 05_prepare_reviews_corpus_for_BERTopic
# This module focuses solely on BERTopic+OCTIS modeling

# Core functionality
from .core.load_raw_sentences import (
    load_raw_sentences_from_reviews,
    load_raw_sentences_from_parquet
)

__all__ = [
    'load_raw_sentences_from_reviews',
    'load_raw_sentences_from_parquet'
]
