"""
Review corpus preparation module for BERTopic topic modeling.

This module provides tools for:
- Loading and joining books and reviews data
- Computing review coverage statistics
- Preparing sentence-level datasets for BERTopic
- Splitting reviews into sentences using spaCy
"""

# Core functionality
from .core.prepare_bertopic_input import (
    load_spacy_model,
    extract_sentences_from_doc,
    split_reviews_to_sentences,
    clean_sentence_text,
    create_sentence_dataset,
    save_sentence_dataset,
    main
)

# Utility functions
from .utils.data_loading import (
    load_books,
    load_reviews,
    load_joined_reviews,
    load_book_id_to_work_id_mapping,
    validate_join_keys
)

from .utils.checks_coverage import (
    compute_review_counts,
    generate_coverage_table,
    summarize_coverage,
    save_coverage_table
)

__all__ = [
    # Core functions
    'load_spacy_model',
    'extract_sentences_from_doc',
    'split_reviews_to_sentences',
    'clean_sentence_text',
    'create_sentence_dataset',
    'save_sentence_dataset',
    'main',
    # Data loading utilities
    'load_books',
    'load_reviews',
    'load_joined_reviews',
    'load_book_id_to_work_id_mapping',
    'validate_join_keys',
    # Coverage utilities
    'compute_review_counts',
    'generate_coverage_table',
    'summarize_coverage',
    'save_coverage_table'
]

