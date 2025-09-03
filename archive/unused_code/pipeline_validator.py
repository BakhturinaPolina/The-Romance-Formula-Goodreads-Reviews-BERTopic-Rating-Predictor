#!/usr/bin/env python3
"""
Pipeline Validator for Data Processing Pipeline

This module provides validation checks that can be integrated into the data processing pipeline
to catch issues early and ensure data quality throughout the process.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PipelineValidator:
    """Validator for data processing pipeline stages."""
    
    def __init__(self):
        self.validation_results = []
        self.errors = []
        self.warnings = []
        
    def validate_books_data(self, books_data: List[Dict[str, Any]], stage: str) -> bool:
        """
        Validate books data at different pipeline stages.
        
        Args:
            books_data: List of book records
            stage: Pipeline stage name
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating books data at stage: {stage}")
        
        if not books_data:
            self._add_error(f"Books data is empty at {stage}")
            return False
        
        # Basic structure validation
        if not self._validate_books_structure(books_data, stage):
            return False
        
        # Content validation
        if not self._validate_books_content(books_data, stage):
            return False
        
        # Business logic validation
        if not self._validate_books_business_logic(books_data, stage):
            return False
        
        logger.info(f"Books data validation passed at {stage}")
        return True
    
    def validate_reviews_data(self, reviews_data: List[Dict[str, Any]], stage: str) -> bool:
        """
        Validate reviews data at different pipeline stages.
        
        Args:
            reviews_data: List of review records
            stage: Pipeline stage name
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating reviews data at stage: {stage}")
        
        if not reviews_data:
            self._add_error(f"Reviews data is empty at {stage}")
            return False
        
        # Basic structure validation
        if not self._validate_reviews_structure(reviews_data, stage):
            return False
        
        # Content validation
        if not self._validate_reviews_content(reviews_data, stage):
            return False
        
        logger.info(f"Reviews data validation passed at {stage}")
        return True
    
    def validate_cross_dataset_consistency(self, books_data: List[Dict[str, Any]], 
                                         reviews_data: List[Dict[str, Any]], 
                                         stage: str) -> bool:
        """
        Validate consistency between books and reviews datasets.
        
        Args:
            books_data: List of book records
            reviews_data: List of review records
            stage: Pipeline stage name
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating cross-dataset consistency at stage: {stage}")
        
        if not books_data or not reviews_data:
            self._add_error(f"Missing data for cross-dataset validation at {stage}")
            return False
        
        # Extract book IDs
        book_ids = {book['book_id'] for book in books_data if 'book_id' in book}
        review_book_ids = {review['book_id'] for review in reviews_data if 'book_id' in review}
        
        # Check for orphaned reviews
        orphaned_reviews = review_book_ids - book_ids
        if orphaned_reviews:
            self._add_error(f"Found {len(orphaned_reviews)} orphaned reviews at {stage}")
            return False
        
        # Check for books without reviews
        books_without_reviews = book_ids - review_book_ids
        if books_without_reviews:
            self._add_warning(f"Found {len(books_without_reviews)} books without reviews at {stage}")
        
        logger.info(f"Cross-dataset consistency validation passed at {stage}")
        return True
    
    def _validate_books_structure(self, books_data: List[Dict[str, Any]], stage: str) -> bool:
        """Validate books data structure."""
        # Core required fields that must always be present
        core_required_fields = ['book_id', 'title', 'work_id']
        
        # Rating fields - can be either individual or aggregated
        rating_fields = ['average_rating', 'average_rating_weighted_mean']
        count_fields = ['ratings_count', 'ratings_count_sum']
        review_count_fields = ['text_reviews_count', 'text_reviews_count_sum']
        
        for i, book in enumerate(books_data):
            # Check core required fields
            for field in core_required_fields:
                if field not in book:
                    self._add_error(f"Missing required field '{field}' in book {i} at {stage}")
                    return False
                if book[field] is None or book[field] == '':
                    self._add_error(f"Empty required field '{field}' in book {i} at {stage}")
                    return False
            
            # Check that at least one rating field is present
            if not any(field in book for field in rating_fields):
                self._add_error(f"Missing rating field (need one of {rating_fields}) in book {i} at {stage}")
                return False
            
            # Check that at least one count field is present
            if not any(field in book for field in count_fields):
                self._add_error(f"Missing ratings count field (need one of {count_fields}) in book {i} at {stage}")
                return False
            
            # Check that at least one review count field is present
            if not any(field in book for field in review_count_fields):
                self._add_error(f"Missing review count field (need one of {review_count_fields}) in book {i} at {stage}")
                return False
        
        return True
    
    def _validate_books_content(self, books_data: List[Dict[str, Any]], stage: str) -> bool:
        """Validate books data content."""
        for i, book in enumerate(books_data):
            # Check publication year
            if 'publication_year' in book:
                pub_year = book['publication_year']
                if pub_year is not None:  # Handle both string and integer types
                    try:
                        # Convert to string first if it's an integer, then check if not empty
                        pub_year_str = str(pub_year) if not isinstance(pub_year, str) else pub_year
                        if pub_year_str.strip():  # Only validate if not empty
                            year = int(pub_year_str)
                            # More reasonable range: 1800-2030 (allows for historical books and future releases)
                            if year < 1800 or year > 2030:
                                self._add_warning(f"Publication year {year} outside reasonable range (1800-2030) in book {i} at {stage}")
                            # Flag obviously wrong years (likely data errors) - but be more lenient since we have cleaning
                            elif year < 10 or year > 30000:
                                self._add_warning(f"Suspicious publication year {year} in book {i} at {stage}")
                    except (ValueError, TypeError):
                        self._add_error(f"Invalid publication year format '{pub_year}' in book {i} at {stage}")
                        return False
            
            # Check average rating
            if 'average_rating' in book:
                try:
                    rating = float(book['average_rating'])
                    if rating < 0 or rating > 5:
                        self._add_error(f"Invalid average rating {rating} in book {i} at {stage}")
                        return False
                except (ValueError, TypeError):
                    self._add_error(f"Invalid average rating format in book {i} at {stage}")
                    return False
            
            # Check counts are non-negative
            count_fields = ['ratings_count', 'text_reviews_count']
            for field in count_fields:
                if field in book:
                    try:
                        count = int(book[field])
                        if count < 0:
                            self._add_error(f"Negative {field} {count} in book {i} at {stage}")
                            return False
                    except (ValueError, TypeError):
                        self._add_error(f"Invalid {field} format in book {i} at {stage}")
                        return False
        
        return True
    
    def _validate_books_business_logic(self, books_data: List[Dict[str, Any]], stage: str) -> bool:
        """Validate books business logic."""
        for i, book in enumerate(books_data):
            # Check that counts are reasonable (text_reviews_count should not be unreasonably high)
            if 'text_reviews_count' in book and 'ratings_count' in book:
                try:
                    text_reviews = int(book['text_reviews_count'])
                    ratings = int(book['ratings_count'])
                    # Only flag if text reviews are more than 10x total ratings (likely data error)
                    if text_reviews > ratings * 10 and ratings > 0:
                        self._add_warning(f"Text reviews ({text_reviews}) significantly higher than total ratings ({ratings}) in book {i} at {stage}")
                except (ValueError, TypeError):
                    pass
        
        return True
    
    def _validate_reviews_structure(self, reviews_data: List[Dict[str, Any]], stage: str) -> bool:
        """Validate reviews data structure."""
        required_fields = ['review_id', 'book_id']
        
        for i, review in enumerate(reviews_data):
            for field in required_fields:
                if field not in review:
                    self._add_error(f"Missing required field '{field}' in review {i} at {stage}")
                    return False
                if review[field] is None or review[field] == '':
                    self._add_error(f"Empty required field '{field}' in review {i} at {stage}")
                    return False
        
        return True
    
    def _validate_reviews_content(self, reviews_data: List[Dict[str, Any]], stage: str) -> bool:
        """Validate reviews data content."""
        for i, review in enumerate(reviews_data):
            # Check rating
            if 'rating' in review:
                try:
                    rating = float(review['rating'])
                    if rating < 0 or rating > 5:
                        self._add_error(f"Invalid rating {rating} in review {i} at {stage}")
                        return False
                except (ValueError, TypeError):
                    self._add_error(f"Invalid rating format in review {i} at {stage}")
                    return False
            
            # Check review text length
            if 'review_text' in review:
                text = review['review_text']
                if text and len(text.strip()) > 50000:  # Only flag extremely long reviews (>50k chars) as potentially problematic
                    self._add_warning(f"Extremely long review text ({len(text)} chars) in review {i} at {stage}")
        
        return True
    
    def _add_error(self, message: str):
        """Add an error message."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'type': 'error'
        })
        logger.error(message)
    
    def _add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'type': 'warning'
        })
        logger.warning(message)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'status': 'FAIL' if self.errors else 'WARNING' if self.warnings else 'PASS',
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def save_validation_report(self, report_path: Path):
        """Save validation report to specified path."""
        report = self.get_validation_summary()
        
        # Ensure the directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def print_validation_summary(self):
        """Print validation summary."""
        summary = self.get_validation_summary()
        
        print(f"\nPipeline Validation Summary:")
        print(f"Status: {summary['status']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Warnings: {summary['total_warnings']}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  ❌ {error['message']}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning['message']}")
        
        if not self.errors and not self.warnings:
            print("  ✅ All validations passed!")
