#!/usr/bin/env python3
"""
Deep Data Inspector for Invisible Issues
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from pathlib import Path
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DeepDataInspector:
    """Deep data inspector for detecting invisible issues."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.invisible_issues = []
        
    def run_deep_inspection(self) -> Dict[str, Any]:
        """Run comprehensive deep inspection."""
        logger.info("Starting deep data inspection")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔬 Starting deep data inspection...")
        
        # Load datasets
        datasets = self._load_datasets()
        
        # Inspect each dataset
        for name, df in datasets.items():
            self._inspect_dataset(df, name)
        
        # Cross-dataset checks
        self._cross_dataset_checks(datasets)
        
        # Generate report
        summary = self._generate_summary()
        self._save_report(summary)
        
        return {'summary': summary, 'issues': self.invisible_issues}
    
    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV datasets."""
        datasets = {}
        files = {
            'books': 'romance_books_cleaned.csv',
            'reviews': 'romance_reviews_cleaned.csv',
            'subgenre': 'subgenre_classification_details.csv'
        }
        
        for name, filename in files.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                datasets[name] = pd.read_csv(filepath)
        
        return datasets
    
    def _inspect_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Inspect a single dataset for invisible issues."""
        
        # 1. Data Type Mismatches
        self._check_data_type_mismatches(df, dataset_name)
        
        # 2. Data Integrity Issues
        self._check_data_integrity(df, dataset_name)
        
        # 3. Business Logic Violations
        self._check_business_logic(df, dataset_name)
        
        # 4. Data Quality Issues
        self._check_data_quality(df, dataset_name)
        
        # 5. Performance Issues
        self._check_performance_issues(df, dataset_name)
        
        # 6. Suspicious Patterns
        self._check_suspicious_patterns(df, dataset_name)
    
    def _check_data_type_mismatches(self, df: pd.DataFrame, dataset_name: str):
        """Check for data type mismatches."""
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check for numeric data stored as strings
                try:
                    pd.to_numeric(df[column], errors='raise')
                    self.invisible_issues.append({
                        'category': 'data_type_mismatch',
                        'dataset': dataset_name,
                        'issue_type': 'numeric_as_string',
                        'description': f"Column '{column}' contains numeric data as strings",
                        'severity': 'medium'
                    })
                except (ValueError, TypeError):
                    pass
                
                # Check for date-like strings
                if self._looks_like_date(df[column]):
                    self.invisible_issues.append({
                        'category': 'data_type_mismatch',
                        'dataset': dataset_name,
                        'issue_type': 'date_as_string',
                        'description': f"Column '{column}' contains date data as strings",
                        'severity': 'medium'
                    })
    
    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if series looks like dates."""
        sample = series.dropna().head(10)
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}']
        return any(sample.astype(str).str.match(pattern).any() for pattern in date_patterns)
    
    def _check_data_integrity(self, df: pd.DataFrame, dataset_name: str):
        """Check for data integrity issues."""
        # Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.invisible_issues.append({
                'category': 'data_integrity',
                'dataset': dataset_name,
                'issue_type': 'duplicate_rows',
                'description': f"{duplicates} duplicate rows found",
                'severity': 'high'
            })
        
        # Duplicate primary keys (only for datasets where book_id should be unique)
        if 'book_id' in df.columns and dataset_name == 'books':
            duplicate_ids = df['book_id'].duplicated().sum()
            if duplicate_ids > 0:
                self.invisible_issues.append({
                    'category': 'data_integrity',
                    'dataset': dataset_name,
                    'issue_type': 'duplicate_primary_key',
                    'description': f"{duplicate_ids} duplicate book_ids found",
                    'severity': 'critical'
                })
        
        # Check for duplicate review_ids in reviews dataset
        if 'review_id' in df.columns and dataset_name == 'reviews':
            duplicate_review_ids = df['review_id'].duplicated().sum()
            if duplicate_review_ids > 0:
                self.invisible_issues.append({
                    'category': 'data_integrity',
                    'dataset': dataset_name,
                    'issue_type': 'duplicate_review_id',
                    'description': f"{duplicate_review_ids} duplicate review_ids found",
                    'severity': 'critical'
                })
    
    def _check_business_logic(self, df: pd.DataFrame, dataset_name: str):
        """Check for business logic violations."""
        if dataset_name == 'books':
            # Text reviews > total ratings
            if all(col in df.columns for col in ['text_reviews_count', 'ratings_count']):
                violations = (df['text_reviews_count'] > df['ratings_count']).sum()
                if violations > 0:
                    self.invisible_issues.append({
                        'category': 'business_logic',
                        'dataset': dataset_name,
                        'issue_type': 'text_reviews_exceed_total',
                        'description': f"{violations} books have text_reviews_count > ratings_count",
                        'severity': 'high'
                    })
            
            # Negative counts
            count_cols = ['ratings_count', 'text_reviews_count']
            for col in count_cols:
                if col in df.columns:
                    negative = (df[col] < 0).sum()
                    if negative > 0:
                        self.invisible_issues.append({
                            'category': 'business_logic',
                            'dataset': dataset_name,
                            'issue_type': 'negative_count',
                            'description': f"{negative} books have negative {col}",
                            'severity': 'high'
                        })
            
            # Invalid publication years
            if 'publication_year' in df.columns:
                invalid_years = df[(df['publication_year'] < 1900) | (df['publication_year'] > 2024)]
                if len(invalid_years) > 0:
                    self.invisible_issues.append({
                        'category': 'business_logic',
                        'dataset': dataset_name,
                        'issue_type': 'invalid_publication_year',
                        'description': f"{len(invalid_years)} books have invalid publication years",
                        'severity': 'medium'
                    })
            
            # Impossible ratings
            if 'average_rating' in df.columns:
                impossible = df[(df['average_rating'] < 0) | (df['average_rating'] > 5)]
                if len(impossible) > 0:
                    self.invisible_issues.append({
                        'category': 'business_logic',
                        'dataset': dataset_name,
                        'issue_type': 'impossible_rating',
                        'description': f"{len(impossible)} books have impossible ratings",
                        'severity': 'high'
                    })
    
    def _check_data_quality(self, df: pd.DataFrame, dataset_name: str):
        """Check for data quality issues."""
        for column in df.columns:
            # Missing required fields
            if column in ['book_id', 'title'] and df[column].isnull().any():
                null_count = df[column].isnull().sum()
                self.invisible_issues.append({
                    'category': 'data_quality',
                    'dataset': dataset_name,
                    'issue_type': 'missing_required_field',
                    'description': f"Required field '{column}' has {null_count} null values",
                    'severity': 'high'
                })
            
            # Empty strings
            if df[column].dtype == 'object':
                empty_strings = df[column].astype(str).str.strip().eq('').sum()
                if empty_strings > 0:
                    self.invisible_issues.append({
                        'category': 'data_quality',
                        'dataset': dataset_name,
                        'issue_type': 'empty_strings',
                        'description': f"Column '{column}' has {empty_strings} empty strings",
                        'severity': 'medium'
                    })
            
            # Extreme outliers
            if df[column].dtype in ['int64', 'float64'] and not df[column].isnull().all():
                Q1, Q3 = df[column].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = df[(df[column] < Q1 - 3*IQR) | (df[column] > Q3 + 3*IQR)]
                if len(outliers) > 0:
                    self.invisible_issues.append({
                        'category': 'data_quality',
                        'dataset': dataset_name,
                        'issue_type': 'extreme_outliers',
                        'description': f"Column '{column}' has {len(outliers)} extreme outliers",
                        'severity': 'medium'
                    })
    
    def _check_performance_issues(self, df: pd.DataFrame, dataset_name: str):
        """Check for performance issues."""
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 100:
            self.invisible_issues.append({
                'category': 'performance',
                'dataset': dataset_name,
                'issue_type': 'large_memory_usage',
                'description': f"Dataset uses {memory_mb:.2f}MB of memory",
                'severity': 'low'
            })
    
    def _check_suspicious_patterns(self, df: pd.DataFrame, dataset_name: str):
        """Check for suspicious patterns."""
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64'] and not df[column].isnull().all():
                # All zeros
                if (df[column] == 0).all():
                    self.invisible_issues.append({
                        'category': 'suspicious_patterns',
                        'dataset': dataset_name,
                        'issue_type': 'all_zeros',
                        'description': f"Column '{column}' contains only zeros",
                        'severity': 'medium'
                    })
                
                # Single value
                elif df[column].nunique() == 1:
                    self.invisible_issues.append({
                        'category': 'suspicious_patterns',
                        'dataset': dataset_name,
                        'issue_type': 'single_value',
                        'description': f"Column '{column}' has only one unique value",
                        'severity': 'medium'
                    })
    
    def _cross_dataset_checks(self, datasets: Dict[str, pd.DataFrame]):
        """Perform cross-dataset checks."""
        if 'books' in datasets and 'reviews' in datasets:
            books_df = datasets['books']
            reviews_df = datasets['reviews']
            
            # Orphaned reviews
            books_ids = set(books_df['book_id'])
            review_ids = set(reviews_df['book_id'])
            orphaned = review_ids - books_ids
            
            if orphaned:
                self.invisible_issues.append({
                    'category': 'cross_dataset_inconsistency',
                    'dataset': 'cross_dataset',
                    'issue_type': 'orphaned_reviews',
                    'description': f"{len(orphaned)} reviews reference non-existent books",
                    'severity': 'high'
                })
            
            # Books without reviews
            books_without = books_ids - review_ids
            if books_without:
                self.invisible_issues.append({
                    'category': 'cross_dataset_inconsistency',
                    'dataset': 'cross_dataset',
                    'issue_type': 'books_without_reviews',
                    'description': f"{len(books_without)} books have no reviews",
                    'severity': 'medium'
                })
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate inspection summary."""
        critical = [i for i in self.invisible_issues if i['severity'] == 'critical']
        high = [i for i in self.invisible_issues if i['severity'] == 'high']
        medium = [i for i in self.invisible_issues if i['severity'] == 'medium']
        low = [i for i in self.invisible_issues if i['severity'] == 'low']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.invisible_issues),
            'critical': len(critical),
            'high': len(high),
            'medium': len(medium),
            'low': len(low),
            'status': 'FAIL' if critical else 'WARNING' if high else 'PASS'
        }
    
    def _save_report(self, summary: Dict[str, Any]):
        """Save inspection report."""
        report = {
            'summary': summary,
            'issues': self.invisible_issues
        }
        
        report_path = self.output_dir / 'deep_inspection_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_summary(self):
        """Print inspection summary."""
        print("\n" + "="*80)
        print("DEEP DATA INSPECTION SUMMARY")
        print("="*80)
        
        summary = self._generate_summary()
        print(f"Status: {summary['status']}")
        print(f"Total Issues: {summary['total_issues']}")
        print(f"  Critical: {summary['critical']}")
        print(f"  High: {summary['high']}")
        print(f"  Medium: {summary['medium']}")
        print(f"  Low: {summary['low']}")
        
        if self.invisible_issues:
            print("\nINVISIBLE ISSUES:")
            print("-" * 40)
            for issue in self.invisible_issues:
                icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'high' else "🟠" if issue['severity'] == 'medium' else "🟢"
                print(f"{icon} [{issue['severity'].upper()}] {issue['category']}/{issue['dataset']}: {issue['description']}")
        
        print("\n" + "="*80)
