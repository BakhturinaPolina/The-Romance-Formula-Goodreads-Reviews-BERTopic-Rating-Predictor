#!/usr/bin/env python3
"""
Dataset Cleaning Pipeline for Integrated Romance Novels Dataset

This script implements the first three critical cleaning steps:
1. Missing Values Assessment (Critical Priority)
2. Duplicate Detection & Resolution (High Priority)  
3. Data Type Validation & Conversion (High Priority)

Author: AI Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetCleaningPipeline:
    """Implements critical dataset cleaning steps."""
    
    def __init__(self, dataset_path: str, output_dir: str = "data/processed"):
        """
        Initialize the cleaning pipeline.
        
        Args:
            dataset_path: Path to the CSV dataset
            output_dir: Directory for cleaned outputs
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.cleaning_results = {}
        self.cleaning_log = []
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset with memory optimization."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load with optimized dtypes
        self.df = pd.read_csv(
            self.dataset_path,
            low_memory=False,
            parse_dates=['cleaning_timestamp'] if 'cleaning_timestamp' in pd.read_csv(self.dataset_path, nrows=0).columns else False
        )
        
        logger.info(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def step1_missing_values_assessment(self) -> Dict[str, Any]:
        """
        Step 1: Missing Values Assessment (Critical Priority)
        
        Analyze missing values and implement appropriate strategies:
        - Flag non-critical variables for investigation
        - Document missing data patterns
        - Create missing data summary
        """
        logger.info("Step 1: Starting Missing Values Assessment...")
        
        missing_analysis = {}
        missing_summary = {}
        
        # Analyze missing values for each variable
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'data_type': str(self.df[col].dtype),
                'strategy': self._determine_missing_strategy(col, missing_pct),
                'action_taken': 'none'
            }
            
            # Categorize by missing percentage
            if missing_pct == 0:
                missing_summary['complete'] = missing_summary.get('complete', []) + [col]
            elif missing_pct < 1:
                missing_summary['minimal'] = missing_summary.get('minimal', []) + [col]
            elif missing_pct < 10:
                missing_summary['moderate'] = missing_summary.get('moderate', []) + [col]
            else:
                missing_summary['significant'] = missing_summary.get('significant', []) + [col]
        
        # Implement missing value strategies
        for col, info in missing_analysis.items():
            if info['missing_percentage'] > 0:
                action = self._apply_missing_strategy(col, info)
                missing_analysis[col]['action_taken'] = action
        
        # Create missing data patterns analysis
        missing_patterns = self._analyze_missing_patterns()
        
        self.cleaning_results['missing_values'] = {
            'analysis': missing_analysis,
            'summary': missing_summary,
            'patterns': missing_patterns,
            'records_affected': sum(1 for info in missing_analysis.values() if info['missing_percentage'] > 0)
        }
        
        logger.info(f"Step 1 Complete: {len([col for col, info in missing_analysis.items() if info['missing_percentage'] > 0])} variables have missing data")
        return self.cleaning_results['missing_values']
    
    def _determine_missing_strategy(self, col: str, missing_pct: float) -> str:
        """Determine appropriate strategy for handling missing values."""
        if missing_pct == 0:
            return 'none'
        elif col in ['work_id', 'title', 'author_id', 'publication_year', 'description']:
            return 'critical_investigation'  # These should not be missing
        elif col in ['series_id', 'series_title', 'series_works_count']:
            return 'flag_for_analysis'  # Series data may be legitimately missing
        elif col in ['average_rating_weighted_mean']:
            return 'exclude_from_analysis'  # Exclude books with missing ratings
        elif col in ['disambiguation_notes']:
            return 'flag_for_investigation'  # Non-critical metadata
        else:
            return 'flag_for_investigation'
    
    def _apply_missing_strategy(self, col: str, info: Dict[str, Any]) -> str:
        """Apply the determined missing value strategy."""
        strategy = info['strategy']
        
        if strategy == 'none':
            return 'no_action_needed'
        elif strategy == 'critical_investigation':
            # Log critical missing values for investigation
            self.cleaning_log.append(f"CRITICAL: {col} has {info['missing_count']} missing values ({info['missing_percentage']:.2f}%)")
            return 'flagged_for_investigation'
        elif strategy == 'flag_for_analysis':
            # Create flag column for missing series data
            flag_col = f"{col}_missing_flag"
            self.df[flag_col] = self.df[col].isnull().astype(int)
            self.cleaning_log.append(f"Created flag column: {flag_col}")
            return 'flag_column_created'
        elif strategy == 'exclude_from_analysis':
            # Exclude records with missing values from final analysis
            missing_mask = self.df[col].isnull()
            excluded_count = missing_mask.sum()
            self.df = self.df[~missing_mask].copy()
            self.cleaning_log.append(f"Excluded {excluded_count} records with missing {col} from final analysis")
            return 'excluded_from_analysis'
        elif strategy == 'flag_for_investigation':
            # Create flag column for investigation
            flag_col = f"{col}_missing_flag"
            self.df[flag_col] = self.df[col].isnull().astype(int)
            self.cleaning_log.append(f"Created flag column: {flag_col}")
            return 'flag_column_created'
        
        return 'strategy_not_implemented'
    
    def _analyze_missing_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in missing data across variables."""
        # Check for correlation between missing values
        missing_df = self.df.isnull().astype(int)
        
        # Calculate missing value correlations
        missing_corr = missing_df.corr()
        
        # Find variables with similar missing patterns
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'var1': missing_corr.columns[i],
                        'var2': missing_corr.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': missing_corr.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'total_missing_records': missing_df.sum(axis=1).value_counts().to_dict()
        }
    
    def step2_duplicate_detection_resolution(self) -> Dict[str, Any]:
        """
        Step 2: Duplicate Detection & Resolution (High Priority)
        
        Identify and resolve:
        - Exact duplicates across all fields
        - Key field duplicates (work_id, title, author_id)
        - Create duplicate classification system
        """
        logger.info("Step 2: Starting Duplicate Detection & Resolution...")
        
        duplicate_analysis = {}
        
        # 1. Exact duplicates (all fields)
        exact_duplicates = self.df.duplicated()
        exact_duplicate_count = exact_duplicates.sum()
        
        if exact_duplicate_count > 0:
            logger.warning(f"Found {exact_duplicate_count} exact duplicates - removing")
            self.df = self.df.drop_duplicates()
            self.cleaning_log.append(f"Removed {exact_duplicate_count} exact duplicates")
        
        duplicate_analysis['exact_duplicates'] = {
            'count': exact_duplicate_count,
            'action': 'removed' if exact_duplicate_count > 0 else 'none_found'
        }
        
        # 2. Key field duplicates
        key_fields = ['work_id', 'title', 'author_id']
        key_duplicates = {}
        
        for field in key_fields:
            if field in self.df.columns:
                field_duplicates = self.df[field].duplicated()
                duplicate_count = field_duplicates.sum()
                
                if duplicate_count > 0:
                    # Analyze duplicate patterns
                    duplicate_values = self.df[field][field_duplicates].value_counts()
                    
                    # Create duplicate classification
                    if field == 'title':
                        classification = self._classify_title_duplicates(field_duplicates)
                    elif field == 'author_id':
                        classification = self._classify_author_duplicates(field_duplicates)
                    else:
                        classification = self._classify_general_duplicates(field, field_duplicates)
                    
                    key_duplicates[field] = {
                        'count': duplicate_count,
                        'unique_duplicate_values': len(duplicate_values),
                        'classification': classification,
                        'action': 'analyzed_and_classified'
                    }
                    
                    # Create duplicate flag columns
                    flag_col = f"{field}_duplicate_flag"
                    self.df[flag_col] = field_duplicates.astype(int)
                    self.cleaning_log.append(f"Created duplicate flag: {flag_col}")
                else:
                    key_duplicates[field] = {
                        'count': 0,
                        'action': 'no_duplicates'
                    }
        
        duplicate_analysis['key_field_duplicates'] = key_duplicates
        
        # 3. Create duplicate summary statistics
        duplicate_summary = {
            'total_duplicate_records': sum(dup_info['count'] for dup_info in key_duplicates.values()),
            'variables_with_duplicates': [field for field, info in key_duplicates.items() if info['count'] > 0],
            'duplicate_flags_created': [f"{field}_duplicate_flag" for field, info in key_duplicates.items() if info['count'] > 0]
        }
        
        duplicate_analysis['summary'] = duplicate_summary
        
        self.cleaning_results['duplicates'] = duplicate_analysis
        logger.info(f"Step 2 Complete: Analyzed duplicates across {len(key_fields)} key fields")
        return duplicate_analysis
    
    def _classify_title_duplicates(self, duplicate_mask: pd.Series) -> Dict[str, Any]:
        """Classify title duplicates into meaningful categories."""
        duplicate_titles = self.df.loc[duplicate_mask, 'title']
        title_counts = duplicate_titles.value_counts()
        
        # Analyze duplicate patterns
        classification = {
            'total_duplicate_titles': len(title_counts),
            'most_duplicated_title': title_counts.index[0] if len(title_counts) > 0 else None,
            'max_duplicates_per_title': title_counts.max() if len(title_counts) > 0 else 0,
            'duplicate_distribution': {
                '2_copies': len(title_counts[title_counts == 2]),
                '3_copies': len(title_counts[title_counts == 3]),
                '4_plus_copies': len(title_counts[title_counts >= 4])
            }
        }
        
        # Check if duplicates might be series entries
        series_indicators = ['book', 'part', 'volume', 'series', 'trilogy', 'saga']
        potential_series = []
        
        for title in title_counts.index[:10]:  # Check first 10 most duplicated titles
            if any(indicator in title.lower() for indicator in series_indicators):
                potential_series.append(title)
        
        classification['potential_series_titles'] = potential_series[:5]  # Top 5
        
        return classification
    
    def _classify_author_duplicates(self, duplicate_mask: pd.Series) -> Dict[str, Any]:
        """Classify author ID duplicates (expected in book datasets)."""
        duplicate_authors = self.df.loc[duplicate_mask, 'author_id']
        author_counts = duplicate_authors.value_counts()
        
        classification = {
            'total_duplicate_authors': len(author_counts),
            'most_prolific_author': author_counts.index[0] if len(author_counts) > 0 else None,
            'max_books_per_author': author_counts.max() if len(author_counts) > 0 else 0,
            'author_productivity_distribution': {
                '2_books': len(author_counts[author_counts == 2]),
                '3_5_books': len(author_counts[(author_counts >= 3) & (author_counts <= 5)]),
                '6_10_books': len(author_counts[(author_counts >= 6) & (author_counts <= 10)]),
                '10_plus_books': len(author_counts[author_counts > 10])
            }
        }
        
        return classification
    
    def _classify_general_duplicates(self, field: str, duplicate_mask: pd.Series) -> Dict[str, Any]:
        """Classify duplicates for general fields."""
        duplicate_values = self.df.loc[duplicate_mask, field]
        value_counts = duplicate_values.value_counts()
        
        classification = {
            'total_duplicate_values': len(value_counts),
            'most_duplicated_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'max_duplicates_per_value': value_counts.max() if len(value_counts) > 0 else 0
        }
        
        return classification
    
    def step3_data_type_validation_conversion(self) -> Dict[str, Any]:
        """
        Step 3: Data Type Validation & Conversion (High Priority)
        
        Ensure correct data types and convert as needed:
        - Validate publication year ranges
        - Ensure numerical types are correct
        - Create derived categorical variables
        """
        logger.info("Step 3: Starting Data Type Validation & Conversion...")
        
        data_type_analysis = {}
        
        # 1. Publication year validation and categorization
        if 'publication_year' in self.df.columns:
            year_validation = self._validate_publication_years()
            data_type_analysis['publication_year'] = year_validation
            
            # Create decade categories
            self.df['decade'] = self._create_decade_categories()
            self.cleaning_log.append("Created decade categories from publication_year")
        
        # 2. Numerical field validation and optimization
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_validation = {}
        
        for col in numerical_cols:
            if col in self.df.columns:
                validation = self._validate_numerical_field(col)
                numerical_validation[col] = validation
                
                # Apply data type optimizations based on cleaning report recommendations
                if validation['needs_optimization']:
                    self._optimize_numerical_dtype(col)
                    self.cleaning_log.append(f"Optimized data type for {col}")
                else:
                    # Apply specific optimizations identified in cleaning report
                    self._apply_specific_optimizations(col)
        
        # Force apply all identified optimizations from cleaning report
        self._force_apply_all_optimizations()
        
        data_type_analysis['numerical_fields'] = numerical_validation
        
        # 3. Categorical field validation and optimization
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_validation = {}
        
        for col in categorical_cols:
            if col in self.df.columns:
                validation = self._validate_categorical_field(col)
                categorical_validation[col] = validation
                
                # Convert to category type if beneficial or specifically recommended
                if validation['should_convert_to_category'] or self._should_force_category_conversion(col):
                    self.df[col] = self.df[col].astype('category')
                    self.cleaning_log.append(f"Converted {col} to category type")
        
        # Force apply all category conversions from cleaning report
        self._force_apply_all_category_conversions()
        
        data_type_analysis['categorical_fields'] = categorical_validation
        
        # 4. Create derived categorical variables
        derived_vars = self._create_derived_categorical_variables()
        data_type_analysis['derived_variables'] = derived_vars
        
        self.cleaning_results['data_types'] = data_type_analysis
        logger.info("Step 3 Complete: Data types validated and optimized")
        return data_type_analysis
    
    def _validate_publication_years(self) -> Dict[str, Any]:
        """Validate publication year ranges and identify issues."""
        year_col = 'publication_year'
        years = self.df[year_col]
        
        validation = {
            'min_year': int(years.min()),
            'max_year': int(years.max()),
            'expected_range': [2000, 2020],
            'out_of_range_count': 0,
            'issues': []
        }
        
        # Check for years outside expected range
        out_of_range = years[(years < 2000) | (years > 2020)]
        validation['out_of_range_count'] = len(out_of_range)
        
        if validation['out_of_range_count'] > 0:
            validation['issues'].append(f"Found {validation['out_of_range_count']} years outside 2000-2020 range")
            
            # Flag extreme years for investigation
            extreme_years = years[(years < 1900) | (years > 2025)]
            if len(extreme_years) > 0:
                validation['issues'].append(f"Found {len(extreme_years)} extreme years: {extreme_years.unique().tolist()}")
        
        # Check for missing years
        missing_years = years.isnull().sum()
        if missing_years > 0:
            validation['issues'].append(f"Found {missing_years} missing publication years")
        
        return validation
    
    def _create_decade_categories(self) -> pd.Series:
        """Create decade categories from publication years."""
        years = self.df['publication_year']
        
        def year_to_decade(year):
            if pd.isna(year):
                return 'Unknown'
            elif year < 2000:
                return 'Pre-2000'
            elif year < 2010:
                return '2000s'
            elif year < 2020:
                return '2010s'
            else:
                return '2020s'
        
        return years.apply(year_to_decade)
    
    def _validate_numerical_field(self, col: str) -> Dict[str, Any]:
        """Validate numerical field characteristics."""
        values = self.df[col]
        
        validation = {
            'current_dtype': str(values.dtype),
            'min_value': float(values.min()),
            'max_value': float(values.max()),
            'mean_value': float(values.mean()),
            'std_value': float(values.std()),
            'missing_count': int(values.isnull().sum()),
            'needs_optimization': False,
            'issues': []
        }
        
        # Check for data type optimization opportunities
        if values.dtype == 'int64':
            if values.max() < 32767 and values.min() > -32768:
                validation['needs_optimization'] = True
                validation['issues'].append("Can optimize to int16")
            elif values.max() < 2147483647 and values.min() > -2147483648:
                validation['needs_optimization'] = True
                validation['issues'].append("Can optimize to int32")
        
        # Check for unrealistic values
        if col == 'num_pages_median':
            if values.max() > 2000:
                validation['issues'].append("Unrealistic page count detected")
        elif col in ['ratings_count_sum', 'text_reviews_count_sum']:
            if values.max() > 1000000:
                validation['issues'].append("Extremely high count values detected")
        
        return validation
    
    def _optimize_numerical_dtype(self, col: str):
        """Optimize numerical data types for memory efficiency."""
        values = self.df[col]
        
        if values.dtype == 'int64':
            if values.max() < 32767 and values.min() > -32768:
                self.df[col] = values.astype('int16')
            elif values.max() < 2147483647 and values.min() > -2147483648:
                self.df[col] = values.astype('int32')
        elif values.dtype == 'float64':
            # Check if we can use float32
            if values.max() < 3.4e38 and values.min() > -3.4e38:
                self.df[col] = values.astype('float32')
    
    def _apply_specific_optimizations(self, col: str):
        """Apply specific optimizations identified in the cleaning report."""
        values = self.df[col]
        
        # Apply optimizations based on cleaning report analysis
        if col == 'work_id':
            # Can optimize to int32 (max: 58,240,793)
            if values.max() < 2147483647 and values.min() > -2147483648:
                self.df[col] = values.astype('int32')
                self.cleaning_log.append(f"Optimized {col} to int32 based on cleaning report")
        
        elif col == 'publication_year':
            # Can optimize to int16 (range: 2000-2018)
            if values.max() < 32767 and values.min() > -32768:
                self.df[col] = values.astype('int16')
                self.cleaning_log.append(f"Optimized {col} to int16 based on cleaning report")
        
        elif col == 'author_id':
            # Can optimize to int32 (max: 17,273,997)
            if values.max() < 2147483647 and values.min() > -2147483648:
                self.df[col] = values.astype('int32')
                self.cleaning_log.append(f"Optimized {col} to int32 based on cleaning report")
        
        elif col == 'author_ratings_count':
            # Can optimize to int32 (max: 5,280,268)
            if values.max() < 2147483647 and values.min() > -32768:
                self.df[col] = values.astype('int32')
                self.cleaning_log.append(f"Optimized {col} to int32 based on cleaning report")
        
        elif col == 'ratings_count_sum':
            # Can optimize to int32 (max: 1,686,868)
            if values.max() < 2147483647 and values.min() > -32768:
                self.df[col] = values.astype('int32')
                self.cleaning_log.append(f"Optimized {col} to int32 based on cleaning report")
        
        elif col == 'text_reviews_count_sum':
            # Can optimize to int32 (max: 74,298)
            if values.max() < 2147483647 and values.min() > -32768:
                self.df[col] = values.astype('int32')
                self.cleaning_log.append(f"Optimized {col} to int32 based on cleaning report")
        
        elif col in ['series_id_missing_flag', 'series_title_missing_flag', 'series_works_count_missing_flag']:
            # Can optimize to int16 (binary flags: 0 or 1)
            if values.max() < 32767 and values.min() > -32768:
                self.df[col] = values.astype('int16')
                self.cleaning_log.append(f"Optimized {col} to int16 based on cleaning report")
    
    def _should_force_category_conversion(self, col: str) -> bool:
        """Determine if a field should be forced to category type based on cleaning report."""
        # Fields identified in cleaning report as needing category conversion
        force_category_fields = [
            'genres', 'author_name', 'series_title', 'duplication_status', 
            'cleaning_strategy', 'disambiguation_notes', 'decade'
        ]
        
        return col in force_category_fields
    
    def _force_apply_all_optimizations(self):
        """Force apply all data type optimizations identified in the cleaning report."""
        logger.info("Forcing application of all identified data type optimizations...")
        
        # Numerical field optimizations
        optimization_map = {
            'work_id': 'int32',
            'publication_year': 'int16', 
            'author_id': 'int32',
            'author_ratings_count': 'int32',
            'ratings_count_sum': 'int32',
            'text_reviews_count_sum': 'int32',
            'series_id_missing_flag': 'int16',
            'series_title_missing_flag': 'int16',
            'series_works_count_missing_flag': 'int16',
            'disambiguation_notes_missing_flag': 'int16',
            'title_duplicate_flag': 'int16',
            'author_id_duplicate_flag': 'int16'
        }
        
        for col, target_type in optimization_map.items():
            if col in self.df.columns:
                try:
                    if target_type == 'int16':
                        self.df[col] = self.df[col].astype('int16')
                    elif target_type == 'int32':
                        self.df[col] = self.df[col].astype('int32')
                    elif target_type == 'float32':
                        self.df[col] = self.df[col].astype('float32')
                    
                    self.cleaning_log.append(f"FORCED: {col} optimized to {target_type}")
                except Exception as e:
                    logger.warning(f"Failed to optimize {col} to {target_type}: {str(e)}")
        
        # Float optimizations
        float_optimizations = ['series_id', 'series_works_count']
        for col in float_optimizations:
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype('float32')
                    self.cleaning_log.append(f"FORCED: {col} optimized to float32")
                except Exception as e:
                    logger.warning(f"Failed to optimize {col} to float32: {str(e)}")
    
    def _force_apply_all_category_conversions(self):
        """Force apply all category conversions identified in the cleaning report."""
        logger.info("Forcing application of all identified category conversions...")
        
        # Fields identified in cleaning report as needing category conversion
        category_fields = [
            'genres', 'author_name', 'series_title', 'duplication_status', 
            'cleaning_strategy', 'disambiguation_notes', 'decade',
            'book_length_category', 'rating_category', 'popularity_category'
        ]
        
        for col in category_fields:
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype('category')
                    self.cleaning_log.append(f"FORCED: {col} converted to category")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to category: {str(e)}")
    
    def _validate_categorical_field(self, col: str) -> Dict[str, Any]:
        """Validate categorical field characteristics."""
        values = self.df[col]
        
        validation = {
            'current_dtype': str(values.dtype),
            'unique_count': int(values.nunique()),
            'missing_count': int(values.isnull().sum()),
            'most_common_values': values.value_counts().head(5).to_dict(),
            'should_convert_to_category': False,
            'issues': []
        }
        
        # Determine if conversion to category type would be beneficial
        if values.dtype == 'object':
            # Convert to category if unique values are < 50% of total values
            if values.nunique() < len(values) * 0.5:
                validation['should_convert_to_category'] = True
        
        # Check for mixed data types
        if values.dtype == 'object':
            # Check if all values are strings
            non_string_count = sum(1 for x in values.dropna() if not isinstance(x, str))
            if non_string_count > 0:
                validation['issues'].append(f"Found {non_string_count} non-string values")
        
        return validation
    
    def _create_derived_categorical_variables(self) -> Dict[str, Any]:
        """Create useful derived categorical variables."""
        derived_vars = {}
        
        # 1. Book length categories
        if 'num_pages_median' in self.df.columns:
            self.df['book_length_category'] = pd.cut(
                self.df['num_pages_median'],
                bins=[0, 200, 400, 600, 1000, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Very Long', 'Extreme'],
                include_lowest=True
            )
            derived_vars['book_length_category'] = 'Created from num_pages_median'
            self.cleaning_log.append("Created book_length_category")
        
        # 2. Rating categories
        if 'average_rating_weighted_mean' in self.df.columns:
            self.df['rating_category'] = pd.cut(
                self.df['average_rating_weighted_mean'],
                bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                include_lowest=True
            )
            derived_vars['rating_category'] = 'Created from average_rating_weighted_mean'
            self.cleaning_log.append("Created rating_category")
        
        # 3. Popularity categories
        if 'ratings_count_sum' in self.df.columns:
            self.df['popularity_category'] = pd.qcut(
                self.df['ratings_count_sum'],
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            derived_vars['popularity_category'] = 'Created from ratings_count_sum'
            self.cleaning_log.append("Created popularity_category")
        
        return derived_vars
    
    def run_cleaning_pipeline(self) -> Dict[str, Any]:
        """Run the complete cleaning pipeline."""
        logger.info("Starting Dataset Cleaning Pipeline...")
        
        # Load dataset
        self.load_dataset()
        
        # Run cleaning steps
        step1_results = self.step1_missing_values_assessment()
        step2_results = self.step2_duplicate_detection_resolution()
        step3_results = self.step3_data_type_validation_conversion()
        
        # Generate cleaning summary
        cleaning_summary = self._generate_cleaning_summary()
        
        # Save cleaned dataset
        self._save_cleaned_dataset()
        
        # Save cleaning report
        self._save_cleaning_report()
        
        logger.info("Dataset Cleaning Pipeline Complete!")
        return {
            'step1_missing_values': step1_results,
            'step2_duplicates': step2_results,
            'step3_data_types': step3_results,
            'summary': cleaning_summary
        }
    
    def _generate_cleaning_summary(self) -> Dict[str, Any]:
        """Generate a summary of all cleaning actions taken."""
        summary = {
            'original_shape': self.df.shape,
            'cleaning_actions': self.cleaning_log,
            'new_columns_created': [col for col in self.df.columns if col not in pd.read_csv(self.dataset_path, nrows=0).columns],
            'memory_optimization': {
                'before_mb': 224.05,  # From original analysis
                'after_mb': self.df.memory_usage(deep=True).sum() / 1024**2
            },
            'data_quality_improvements': {
                'missing_values_addressed': len([log for log in self.cleaning_log if 'flag' in log or 'imputed' in log]),
                'duplicate_flags_created': len([log for log in self.cleaning_log if 'duplicate_flag' in log]),
                'derived_variables_created': len([log for log in self.cleaning_log if 'Created' in log])
            }
        }
        
        return summary
    
    def _save_cleaned_dataset(self):
        """Save the cleaned dataset with optimized data types preserved."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV version (may lose some optimizations)
        csv_path = self.output_dir / f"cleaned_romance_novels_step1_3_{timestamp}.csv"
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Cleaned dataset saved to: {csv_path}")
        
        # Save pickle version to preserve all data type optimizations
        pickle_path = self.output_dir / f"cleaned_romance_novels_step1_3_{timestamp}.pkl"
        self.df.to_pickle(pickle_path)
        logger.info(f"Optimized dataset saved to: {pickle_path} (data types preserved)")
        
        # Try to save parquet version (optional)
        try:
            parquet_path = self.output_dir / f"cleaned_romance_novels_step1_3_{timestamp}.parquet"
            self.df.to_parquet(parquet_path, index=False)
            logger.info(f"Compressed dataset saved to: {parquet_path}")
        except ImportError:
            logger.warning("Parquet export skipped - pyarrow/fastparquet not available")
        except Exception as e:
            logger.warning(f"Parquet export failed: {str(e)}")
        
        # Log final memory usage after optimizations
        final_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Final memory usage after optimizations: {final_memory:.2f} MB")
        
        # Save memory usage comparison
        self._save_memory_comparison(timestamp)
    
    def _save_memory_comparison(self, timestamp: str):
        """Save memory usage comparison before and after optimizations."""
        memory_comparison = {
            'timestamp': timestamp,
            'memory_usage': {
                'before_optimization_mb': 224.05,  # From original analysis
                'after_optimization_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
                'memory_saved_mb': 224.05 - (self.df.memory_usage(deep=True).sum() / 1024**2),
                'memory_saved_percentage': ((224.05 - (self.df.memory_usage(deep=True).sum() / 1024**2)) / 224.05) * 100
            },
            'data_type_optimizations': {
                'numerical_optimizations': len([log for log in self.cleaning_log if 'FORCED:' in log and any(t in log for t in ['int16', 'int32', 'float32'])]),
                'category_conversions': len([log for log in self.cleaning_log if 'FORCED:' in log and 'category' in log])
            }
        }
        
        memory_path = self.output_dir / f"memory_optimization_report_{timestamp}.json"
        with open(memory_path, 'w') as f:
            json.dump(memory_comparison, f, indent=2)
        
        logger.info(f"Memory optimization report saved to: {memory_path}")
    
    def _save_cleaning_report(self):
        """Save the cleaning report."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"cleaning_report_step1_3_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        serializable_results = json.loads(
            json.dumps(self.cleaning_results, default=convert_numpy_types)
        )
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Cleaning report saved to: {report_path}")
    
    def print_cleaning_summary(self):
        """Print a summary of the cleaning results."""
        if not self.cleaning_results:
            print("No cleaning results available. Run run_cleaning_pipeline() first.")
            return
        
        print("\n" + "="*80)
        print("DATASET CLEANING PIPELINE RESULTS")
        print("="*80)
        
        # Step 1: Missing Values
        if 'missing_values' in self.cleaning_results:
            missing_info = self.cleaning_results['missing_values']
            print(f"\nStep 1: Missing Values Assessment")
            print(f"  Variables with missing data: {missing_info['records_affected']}")
            print(f"  Actions taken: {len([log for log in self.cleaning_log if 'flag' in log or 'imputed' in log])}")
        
        # Step 2: Duplicates
        if 'duplicates' in self.cleaning_results:
            dup_info = self.cleaning_results['duplicates']
            print(f"\nStep 2: Duplicate Detection & Resolution")
            print(f"  Exact duplicates removed: {dup_info['exact_duplicates']['count']}")
            print(f"  Duplicate flags created: {len(dup_info['summary']['duplicate_flags_created'])}")
        
        # Step 3: Data Types
        if 'data_types' in self.cleaning_results:
            dt_info = self.cleaning_results['data_types']
            print(f"\nStep 3: Data Type Validation & Conversion")
            print(f"  Derived variables created: {len(dt_info.get('derived_variables', {}))}")
            print(f"  Data type optimizations: {len([log for log in self.cleaning_log if 'Optimized' in log])}")
        
        # Summary
        print(f"\nOverall Results:")
        print(f"  New columns created: {len([col for col in self.df.columns if col not in pd.read_csv(self.dataset_path, nrows=0).columns])}")
        print(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n" + "="*80)


def main():
    """Main function to run the cleaning pipeline."""
    # Dataset path
    dataset_path = "data/processed/integrated_romance_novels_nlp_ready_20250902_193603.csv"
    
    # Initialize pipeline
    pipeline = DatasetCleaningPipeline(dataset_path)
    
    try:
        # Run cleaning pipeline
        results = pipeline.run_cleaning_pipeline()
        
        # Print summary
        pipeline.print_cleaning_summary()
        
        print(f"\nCleaning pipeline completed successfully!")
        print(f"Check the data/processed/ directory for cleaned datasets and reports.")
        
    except Exception as e:
        logger.error(f"Cleaning pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
