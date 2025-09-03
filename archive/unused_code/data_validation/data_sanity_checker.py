#!/usr/bin/env python3
"""
Data Sanity Checker for Romance Novel NLP Research Dataset

This module provides comprehensive validation for the processed CSV datasets,
checking for data quality issues, anomalies, and potential problems.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataSanityChecker:
    """Comprehensive data sanity checker for CSV datasets."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.validation_results = {}
        self.issues_found = []
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation on all CSV files.
        
        Returns:
            Dictionary containing validation results and issues found
        """
        logger.info("Starting comprehensive data validation")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Starting comprehensive data validation...")
        
        validation_start = datetime.now()
        
        # Validate each dataset
        datasets = {
            'books': 'romance_books_cleaned.csv',
            'reviews': 'romance_reviews_cleaned.csv', 
            'subgenre': 'subgenre_classification_details.csv',
            'quality': 'data_quality_report.csv'
        }
        
        for dataset_name, filename in datasets.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                logger.info(f"Validating {dataset_name} dataset")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Validating {dataset_name}...")
                
                try:
                    df = pd.read_csv(filepath)
                    self.validation_results[dataset_name] = self._validate_dataset(
                        df, dataset_name, filename
                    )
                except Exception as e:
                    logger.error(f"Error validating {dataset_name}: {e}")
                    self.issues_found.append({
                        'dataset': dataset_name,
                        'issue_type': 'file_error',
                        'description': f"Could not read {filename}: {str(e)}",
                        'severity': 'critical'
                    })
            else:
                logger.warning(f"File not found: {filepath}")
                self.issues_found.append({
                    'dataset': dataset_name,
                    'issue_type': 'missing_file',
                    'description': f"Expected file not found: {filename}",
                    'severity': 'critical'
                })
        
        # Cross-dataset validation
        self._validate_cross_dataset_consistency()
        
        # Generate summary report
        validation_duration = datetime.now() - validation_start
        summary = self._generate_validation_summary(validation_duration)
        
        # Save validation report
        self._save_validation_report()
        
        logger.info(f"Validation complete in {validation_duration}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Validation complete ({validation_duration})")
        
        return {
            'summary': summary,
            'issues': self.issues_found,
            'detailed_results': self.validation_results
        }
    
    def _validate_dataset(self, df: pd.DataFrame, dataset_name: str, filename: str) -> Dict[str, Any]:
        """Validate a single dataset."""
        results = {
            'filename': filename,
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'basic_stats': {},
            'data_quality_issues': [],
            'anomalies': [],
            'warnings': []
        }
        
        # Basic structure validation
        self._validate_basic_structure(df, dataset_name, results)
        
        # Data type validation
        self._validate_data_types(df, dataset_name, results)
        
        # Content validation based on dataset type
        if dataset_name == 'books':
            self._validate_books_dataset(df, results)
        elif dataset_name == 'reviews':
            self._validate_reviews_dataset(df, results)
        elif dataset_name == 'subgenre':
            self._validate_subgenre_dataset(df, results)
        elif dataset_name == 'quality':
            self._validate_quality_dataset(df, results)
        
        # Statistical anomaly detection
        self._detect_statistical_anomalies(df, dataset_name, results)
        
        # Data consistency checks
        self._validate_data_consistency(df, dataset_name, results)
        
        return results
    
    def _validate_basic_structure(self, df: pd.DataFrame, dataset_name: str, results: Dict[str, Any]):
        """Validate basic dataset structure."""
        # Check for empty dataset
        if df.empty:
            self.issues_found.append({
                'dataset': dataset_name,
                'issue_type': 'empty_dataset',
                'description': f"{dataset_name} dataset is completely empty",
                'severity': 'critical'
            })
            results['data_quality_issues'].append('Dataset is empty')
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.issues_found.append({
                'dataset': dataset_name,
                'issue_type': 'duplicate_rows',
                'description': f"{duplicate_count} duplicate rows found in {dataset_name}",
                'severity': 'high'
            })
            results['data_quality_issues'].append(f'{duplicate_count} duplicate rows')
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            self.issues_found.append({
                'dataset': dataset_name,
                'issue_type': 'empty_columns',
                'description': f"Columns with all null values: {empty_columns}",
                'severity': 'high'
            })
            results['data_quality_issues'].append(f'Empty columns: {empty_columns}')
        
        # Check for empty titles in books dataset
        if dataset_name == 'books' and 'title' in df.columns:
            empty_titles = df[df['title'].isnull() | (df['title'] == '')]
            if len(empty_titles) > 0:
                self.issues_found.append({
                    'dataset': dataset_name,
                    'issue_type': 'empty_title',
                    'description': f"{len(empty_titles)} books with empty titles",
                    'severity': 'high'
                })
                results['data_quality_issues'].append(f'{len(empty_titles)} empty titles')
    
    def _validate_data_types(self, df: pd.DataFrame, dataset_name: str, results: Dict[str, Any]):
        """Validate data types and detect type mismatches."""
        for column in df.columns:
            # Check for mixed data types
            if df[column].dtype == 'object':
                # Check if numeric data is stored as strings
                try:
                    pd.to_numeric(df[column], errors='raise')
                    if not df[column].str.contains(r'[a-zA-Z]', na=False).any():
                        results['warnings'].append(f'Column {column} contains numeric data stored as strings')
                except (ValueError, TypeError):
                    pass
            
            # Check for unexpected null values in required fields
            if column in ['book_id', 'title', 'work_id'] and df[column].isnull().any():
                self.issues_found.append({
                    'dataset': dataset_name,
                    'issue_type': 'null_required_field',
                    'description': f"Null values found in required field: {column}",
                    'severity': 'high'
                })
    
    def _validate_books_dataset(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate books dataset specific content."""
        # Check for empty titles
        if 'title' in df.columns:
            empty_titles = df[df['title'].isnull() | (df['title'] == '')]
            if len(empty_titles) > 0:
                self.issues_found.append({
                    'dataset': 'books',
                    'issue_type': 'empty_title',
                    'description': f"{len(empty_titles)} books with empty titles",
                    'severity': 'high'
                })
                results['data_quality_issues'].append(f'{len(empty_titles)} empty titles')
        
        # Check publication years
        if 'publication_year' in df.columns:
            invalid_years = df[
                (df['publication_year'] < 1900) | 
                (df['publication_year'] > 2024) |
                (df['publication_year'].isnull())
            ]
            if len(invalid_years) > 0:
                self.issues_found.append({
                    'dataset': 'books',
                    'issue_type': 'invalid_publication_year',
                    'description': f"{len(invalid_years)} books with invalid publication years",
                    'severity': 'high'
                })
                results['data_quality_issues'].append(f'{len(invalid_years)} invalid publication years')
        
        # Check ratings (work-level aggregated)
        if 'average_rating_weighted_mean' in df.columns:
            invalid_ratings = df[
                (df['average_rating_weighted_mean'] < 0) | 
                (df['average_rating_weighted_mean'] > 5) |
                (df['average_rating_weighted_mean'].isnull())
            ]
            if len(invalid_ratings) > 0:
                self.issues_found.append({
                    'dataset': 'books',
                    'issue_type': 'invalid_rating',
                    'description': f"{len(invalid_ratings)} books with invalid weighted ratings",
                    'severity': 'medium'
                })
                results['data_quality_issues'].append(f'{len(invalid_ratings)} invalid weighted ratings')
        
        # Check for negative counts (work-level aggregated)
        count_columns = ['ratings_count_sum', 'text_reviews_count_sum']
        for col in count_columns:
            if col in df.columns:
                negative_counts = df[df[col] < 0]
                if len(negative_counts) > 0:
                    self.issues_found.append({
                        'dataset': 'books',
                        'issue_type': 'negative_count',
                        'description': f"{len(negative_counts)} books with negative {col}",
                        'severity': 'high'
                    })
                    results['data_quality_issues'].append(f'{len(negative_counts)} negative {col}')
        
        # Note: Individual edition fields have been removed, only work-level aggregated fields remain
        results['warnings'].append('Dataset now uses work-level aggregated fields only (no individual edition fields)')
    
    def _validate_reviews_dataset(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate reviews dataset specific content."""
        # Check ratings
        if 'rating' in df.columns:
            invalid_ratings = df[
                (df['rating'] < 0) | 
                (df['rating'] > 5) |
                (df['rating'].isnull())
            ]
            if len(invalid_ratings) > 0:
                self.issues_found.append({
                    'dataset': 'reviews',
                    'issue_type': 'invalid_review_rating',
                    'description': f"{len(invalid_ratings)} reviews with invalid ratings",
                    'severity': 'medium'
                })
                results['data_quality_issues'].append(f'{len(invalid_ratings)} invalid review ratings')
        
        # Check review text length
        if 'review_text' in df.columns:
            empty_reviews = df[df['review_text'].isnull() | (df['review_text'] == '')]
            if len(empty_reviews) > 0:
                self.issues_found.append({
                    'dataset': 'reviews',
                    'issue_type': 'empty_review_text',
                    'description': f"{len(empty_reviews)} reviews with empty text",
                    'severity': 'medium'
                })
                results['data_quality_issues'].append(f'{len(empty_reviews)} empty review texts')
            
            # Check for extremely long reviews (potential data corruption)
            if 'review_length' in df.columns:
                extremely_long = df[df['review_length'] > 10000]
                if len(extremely_long) > 0:
                    results['warnings'].append(f"{len(extremely_long)} extremely long reviews (>10k chars)")
        
        # Check date consistency
        if 'review_year' in df.columns:
            invalid_years = df[
                (df['review_year'] < 1990) | 
                (df['review_year'] > 2024) |
                (df['review_year'].isnull())
            ]
            if len(invalid_years) > 0:
                self.issues_found.append({
                    'dataset': 'reviews',
                    'issue_type': 'invalid_review_year',
                    'description': f"{len(invalid_years)} reviews with invalid years",
                    'severity': 'medium'
                })
    
    def _validate_subgenre_dataset(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate subgenre classification dataset."""
        # Check if subgenre fields are empty (critical issue)
        subgenre_fields = ['subgenre_primary', 'subgenre_keywords', 'subgenre_final']
        empty_subgenre_fields = []
        
        for field in subgenre_fields:
            if field in df.columns:
                if df[field].isnull().all():
                    empty_subgenre_fields.append(field)
        
        if empty_subgenre_fields:
            self.issues_found.append({
                'dataset': 'subgenre',
                'issue_type': 'empty_subgenre_classification',
                'description': f"Subgenre classification not implemented: {empty_subgenre_fields}",
                'severity': 'critical'
            })
            results['data_quality_issues'].append('Subgenre classification not implemented')
        
        # Check for empty or missing subgenre values
        for field in subgenre_fields:
            if field in df.columns:
                empty_values = df[df[field].isnull() | (df[field] == '')]
                if len(empty_values) > 0:
                    self.issues_found.append({
                        'dataset': 'subgenre',
                        'issue_type': 'empty_subgenre_values',
                        'description': f"{len(empty_values)} records with empty {field}",
                        'severity': 'high'
                    })
                    results['data_quality_issues'].append(f'{len(empty_values)} empty {field} values')
        
        # Check for empty subgenre values (not all null, but some empty strings)
        for field in subgenre_fields:
            if field in df.columns:
                empty_values = df[df[field].isnull() | (df[field] == '')]
                if len(empty_values) > 0:
                    self.issues_found.append({
                        'dataset': 'subgenre',
                        'issue_type': 'empty_subgenre_values',
                        'description': f"{len(empty_values)} records with empty {field}",
                        'severity': 'high'
                    })
                    results['data_quality_issues'].append(f'{len(empty_values)} empty {field} values')
        
        # Check confidence scores
        if 'confidence_score' in df.columns:
            if df['confidence_score'].eq(0).all():
                self.issues_found.append({
                    'dataset': 'subgenre',
                    'issue_type': 'zero_confidence_scores',
                    'description': "All confidence scores are 0.0 (classification not performed)",
                    'severity': 'critical'
                })
                results['data_quality_issues'].append('All confidence scores are zero')
            
            # Check for zero confidence scores (not all, but some)
            zero_confidence = df[df['confidence_score'] == 0.0]
            if len(zero_confidence) > 0:
                self.issues_found.append({
                    'dataset': 'subgenre',
                    'issue_type': 'some_zero_confidence_scores',
                    'description': f"{len(zero_confidence)} records with zero confidence scores",
                    'severity': 'medium'
                })
                results['data_quality_issues'].append(f'{len(zero_confidence)} zero confidence scores')
    
    def _validate_quality_dataset(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate quality report dataset."""
        # Check if quality metrics are reasonable
        if 'metric_value' in df.columns:
            # Check for negative metric values
            negative_metrics = df[df['metric_value'] < 0]
            if len(negative_metrics) > 0:
                self.issues_found.append({
                    'dataset': 'quality',
                    'issue_type': 'negative_quality_metrics',
                    'description': f"{len(negative_metrics)} negative quality metric values",
                    'severity': 'high'
                })
            
            # Check for unreasonably high values
            high_metrics = df[df['metric_value'] > 1000000]
            if len(high_metrics) > 0:
                results['warnings'].append(f"{len(high_metrics)} unusually high metric values")
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame, dataset_name: str, results: Dict[str, Any]):
        """Detect statistical anomalies using various methods."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in df.columns and not df[column].isnull().all():
                # Z-score outlier detection
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = df[z_scores > 3]
                
                if len(outliers) > 0:
                    results['anomalies'].append({
                        'column': column,
                        'outlier_count': len(outliers),
                        'method': 'z_score_3sigma'
                    })
                
                # Check for suspicious patterns
                if df[column].nunique() == 1:
                    results['warnings'].append(f'Column {column} has only one unique value')
                
                # Check for all zeros
                if (df[column] == 0).all():
                    results['warnings'].append(f'Column {column} contains only zeros')
    
    def _validate_data_consistency(self, df: pd.DataFrame, dataset_name: str, results: Dict[str, Any]):
        """Validate data consistency across columns."""
        # Check for logical inconsistencies
        if dataset_name == 'books':
            if all(col in df.columns for col in ['ratings_count', 'text_reviews_count']):
                # Text reviews should not exceed total ratings
                inconsistent = df[df['text_reviews_count'] > df['ratings_count']]
                if len(inconsistent) > 0:
                    self.issues_found.append({
                        'dataset': 'books',
                        'issue_type': 'inconsistent_review_counts',
                        'description': f"{len(inconsistent)} books where text_reviews_count > ratings_count",
                        'severity': 'high'
                    })
    
    def _validate_cross_dataset_consistency(self):
        """Validate consistency across different datasets."""
        try:
            books_df = pd.read_csv(self.output_dir / 'romance_books_cleaned.csv')
            reviews_df = pd.read_csv(self.output_dir / 'romance_reviews_cleaned.csv')
            
            # Check if all review book_ids exist in books dataset
            books_book_ids = set(books_df['book_id'])
            review_book_ids = set(reviews_df['book_id'])
            orphaned_reviews = review_book_ids - books_book_ids
            
            if orphaned_reviews:
                self.issues_found.append({
                    'dataset': 'cross_dataset',
                    'issue_type': 'orphaned_reviews',
                    'description': f"{len(orphaned_reviews)} reviews reference non-existent books",
                    'severity': 'high'
                })
            
            # Check if all books have at least one review
            books_without_reviews = books_book_ids - review_book_ids
            if books_without_reviews:
                self.issues_found.append({
                    'dataset': 'cross_dataset',
                    'issue_type': 'books_without_reviews',
                    'description': f"{len(books_without_reviews)} books have no reviews",
                    'severity': 'medium'
                })
                
        except Exception as e:
            logger.error(f"Error in cross-dataset validation: {e}")
    
    def _generate_validation_summary(self, duration) -> Dict[str, Any]:
        """Generate validation summary."""
        critical_issues = [issue for issue in self.issues_found if issue['severity'] == 'critical']
        high_issues = [issue for issue in self.issues_found if issue['severity'] == 'high']
        medium_issues = [issue for issue in self.issues_found if issue['severity'] == 'medium']
        
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'duration': str(duration),
            'total_issues': len(self.issues_found),
            'critical_issues': len(critical_issues),
            'high_issues': len(high_issues),
            'medium_issues': len(medium_issues),
            'datasets_validated': len(self.validation_results),
            'overall_status': 'FAIL' if critical_issues else 'PASS' if not high_issues else 'WARNING'
        }
    
    def _save_validation_report(self):
        """Save detailed validation report."""
        report = {
            'summary': self._generate_validation_summary(datetime.now() - datetime.now()),
            'issues': self.issues_found,
            'detailed_results': self.validation_results
        }
        
        report_path = self.output_dir / 'data_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def print_validation_summary(self):
        """Print a human-readable validation summary."""
        print("\n" + "="*80)
        print("DATA VALIDATION SUMMARY")
        print("="*80)
        
        summary = self._generate_validation_summary(datetime.now() - datetime.now())
        
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Datasets Validated: {summary['datasets_validated']}")
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"  - Critical: {summary['critical_issues']}")
        print(f"  - High: {summary['high_issues']}")
        print(f"  - Medium: {summary['medium_issues']}")
        
        if self.issues_found:
            print("\nISSUES FOUND:")
            print("-" * 40)
            for issue in self.issues_found:
                severity_icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'high' else "🟠"
                print(f"{severity_icon} [{issue['severity'].upper()}] {issue['dataset']}: {issue['description']}")
        
        print("\n" + "="*80)
