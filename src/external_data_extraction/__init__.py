"""
External Data Extraction Module

This module provides tools for extracting structured data from external datasets,
specifically designed for romance book data extraction from Hugging Face datasets.
"""

from .extract_romance_books import main, extract_one, Extraction

__all__ = ["main", "extract_one", "Extraction"]
