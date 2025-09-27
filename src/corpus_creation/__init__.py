"""
Corpus Creation Pipeline for Romance Novel Research

This module provides automated corpus creation from Anna's Archive datasets,
designed for romance novel NLP research projects.

Features:
- Book matching using title/author/year
- Format preference (epub > HTML > PDF)
- Organized storage with consistent naming
- Error handling and flagging for missing books
- Scalable design for large datasets
"""

__version__ = "0.1.0"
__author__ = "Research Assistant"

from .book_matcher import BookMatcher
from .annas_client import AnnasArchiveClient
from .downloader import BookDownloader
from .pipeline import CorpusCreationPipeline
from .free_pipeline import FreeCorpusCreationPipeline

__all__ = [
    'BookMatcher',
    'AnnasArchiveClient',
    'BookDownloader',
    'CorpusCreationPipeline',
    'FreeCorpusCreationPipeline'
]
