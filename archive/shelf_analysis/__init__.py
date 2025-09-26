"""
Shelf Analysis Module

This module provides tools for normalizing Goodreads shelf tags and performing
topic modeling using BERTopic on the normalized shelf data.

Main components:
- ShelfNormalizer: Handles shelf parsing, normalization, and clustering
- TopicModeler: Wraps BERTopic for topic modeling on shelf data
- FastAPI integration for serving topic models
"""

from .goodreads_shelf_normalization_and_topics import (
    ShelfNormalizer,
    TopicModeler,
    NormalizationConfig,
    TopicConfig,
    run_normalization,
    run_topics,
    create_app,
)

__all__ = [
    'ShelfNormalizer',
    'TopicModeler', 
    'NormalizationConfig',
    'TopicConfig',
    'run_normalization',
    'run_topics',
    'create_app',
]
