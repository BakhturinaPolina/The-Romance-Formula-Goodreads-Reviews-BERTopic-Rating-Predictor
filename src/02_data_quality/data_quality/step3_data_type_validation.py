#!/usr/bin/env python3
"""
Step 3: Data Type Validation & Conversion (High Priority)

This script implements comprehensive data type validation and conversion for the romance novel dataset:
1. Data Type Analysis and Validation
2. Publication Year Range Validation
3. Numerical Field Optimization
4. Categorical Variable Creation
5. Derived Variable Generation
6. Data Type Conversion and Optimization

Author: Research Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTypeValidator:
    """
    Comprehensive data type validation and conversion system for romance novel dataset.
    Implements validation, optimization, and derived variable creation.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data type validator.
        
        Args:
            data_path: Path to the input dataset file (from Step 2)
        """
        self.data_path = data_path or "outputs/duplicate_detection/romance_novels_step2_duplicates_resolved_20250902_000000.pkl"
        self.df = None
        self.validation_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/data_type_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data type optimization mappings
        self.optimization_mappings = {
            'work_id': 'int32',
            'publication_year': 'int16',
            'author_id': 'int32',
            'author_ratings_count': 'int32',
            'ratings_count_sum': 'int32',
            'text_reviews_count_sum': 'int32',
            'series_id': 'float32',
            'series_works_count': 'float32',
            'average_rating_weighted_mean': 'float32',
            'author_average_rating': 'float32',
            'num_pages_median': 'float32'
        }
        
        # Categorical field mappings
        self.categorical_fields = [
            'genres', 'author_name', 'series_title', 'language_codes_en',
            'duplication_status', 'cleaning_strategy', 'disambiguation_notes'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset for data type validation.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading dataset from: {self.data_path}")
            
            if self.data_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.data_path)
                logger.info("Data loaded from pickle file - data types preserved")
            else:
                self.df = pd.read_csv(self.data_path)
                logger.info("Data loaded from CSV file")
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def analyze_data_types(self) -> Dict[str, Any]:
        """
        Analyze current data types and identify optimization opportunities.
        
        Returns:
            Dictionary with data type analysis results
        """
        logger.info("🔍 Analyzing current data types...")
        
        analysis = {
            'total_columns': len(self.df.columns),
            'current_memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_type_distribution': {},
            'optimization_opportunities': {},
            'validation_issues': {},
            'categorical_opportunities': []
        }
        
        # Analyze data type distribution
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            analysis['data_type_distribution'][str(dtype)] = int(count)
        
        # Identify optimization opportunities
        for col in self.df.columns:
            current_dtype = str(self.df[col].dtype)
            
            # Check for optimization opportunities
            if col in self.optimization_mappings:
                target_dtype = self.optimization_mappings[col]
                if current_dtype != target_dtype:
                    analysis['optimization_opportunities'][col] = {
                        'current': current_dtype,
                        'target': target_dtype,
                        'potential_savings': self._estimate_memory_savings(col, current_dtype, target_dtype)
                    }
            
            # Check for categorical opportunities
            if col in self.categorical_fields and current_dtype != 'category':
                unique_count = self.df[col].nunique()
                total_count = len(self.df[col])
                if unique_count < total_count * 0.5:  # Less than 50% unique values
                    analysis['categorical_opportunities'].append({
                        'column': col,
                        'unique_count': unique_count,
                        'total_count': total_count,
                        'uniqueness_ratio': unique_count / total_count
                    })
            
            # Validate data types
            validation_issues = self._validate_field_data_type(col)
            if validation_issues:
                analysis['validation_issues'][col] = validation_issues
        
        logger.info(f"Data type analysis completed:")
        logger.info(f"  • Total columns: {analysis['total_columns']}")
        logger.info(f"  • Current memory: {analysis['current_memory_mb']:.2f} MB")
        logger.info(f"  • Optimization opportunities: {len(analysis['optimization_opportunities'])}")
        logger.info(f"  • Categorical opportunities: {len(analysis['categorical_opportunities'])}")
        logger.info(f"  • Validation issues: {len(analysis['validation_issues'])}")
        
        return analysis
    
    def _estimate_memory_savings(self, col: str, current_dtype: str, target_dtype: str) -> float:
        """
        Estimate memory savings from data type conversion.
        
        Args:
            col: Column name
            current_dtype: Current data type
            target_dtype: Target data type
            
        Returns:
            Estimated memory savings in MB
        """
        # Rough estimates for memory savings
        dtype_sizes = {
            'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1,
            'float64': 8, 'float32': 4,
            'object': 8, 'category': 1
        }
        
        current_size = dtype_sizes.get(current_dtype, 8)
        target_size = dtype_sizes.get(target_dtype, 8)
        
        if current_size > target_size:
            savings_per_value = current_size - target_size
            total_savings = savings_per_value * len(self.df[col]) / 1024 / 1024
            return total_savings
        
        return 0.0
    
    def _validate_field_data_type(self, col: str) -> List[str]:
        """
        Validate data type for a specific field.
        
        Args:
            col: Column name
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Publication year validation
        if col == 'publication_year':
            if self.df[col].dtype not in ['int64', 'int32', 'int16']:
                issues.append(f"Publication year should be integer type, got {self.df[col].dtype}")
            
            # Check for reasonable year range
            years = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(years) > 0:
                min_year, max_year = years.min(), years.max()
                if min_year < 1800 or max_year > 2030:
                    issues.append(f"Publication year range suspicious: {min_year}-{max_year}")
        
        # Rating validation
        elif col in ['average_rating_weighted_mean', 'author_average_rating']:
            if self.df[col].dtype not in ['float64', 'float32']:
                issues.append(f"Rating field should be float type, got {self.df[col].dtype}")
            
            # Check for reasonable rating range
            ratings = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(ratings) > 0:
                min_rating, max_rating = ratings.min(), ratings.max()
                if min_rating < 0 or max_rating > 5:
                    issues.append(f"Rating range suspicious: {min_rating}-{max_rating}")
        
        # Count validation
        elif col in ['ratings_count_sum', 'text_reviews_count_sum', 'author_ratings_count']:
            if self.df[col].dtype not in ['int64', 'int32', 'int16']:
                issues.append(f"Count field should be integer type, got {self.df[col].dtype}")
            
            # Check for negative values
            counts = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(counts) > 0 and counts.min() < 0:
                issues.append(f"Count field has negative values: {counts.min()}")
        
        return issues
    
    def validate_publication_years(self) -> Dict[str, Any]:
        """
        Validate publication year data and create derived variables.
        
        Returns:
            Dictionary with publication year validation results
        """
        logger.info("🔍 Validating publication years...")
        
        if 'publication_year' not in self.df.columns:
            return {'error': 'Publication year field not found'}
        
        years = pd.to_numeric(self.df['publication_year'], errors='coerce')
        
        validation = {
            'total_records': len(self.df),
            'valid_years': len(years.dropna()),
            'invalid_years': len(years) - len(years.dropna()),
            'year_range': {
                'min': int(years.min()) if not years.isna().all() else None,
                'max': int(years.max()) if not years.isna().all() else None
            },
            'decade_distribution': {},
            'validation_issues': []
        }
        
        # Check for reasonable year range
        if not years.isna().all():
            min_year, max_year = years.min(), years.max()
            if min_year < 1800:
                validation['validation_issues'].append(f"Very old publication year: {min_year}")
            if max_year > 2030:
                validation['validation_issues'].append(f"Future publication year: {max_year}")
        
        # Create decade distribution
        valid_years = years.dropna()
        if len(valid_years) > 0:
            decades = {}
            for year in valid_years:
                decade = (int(year) // 10) * 10
                decades[decade] = decades.get(decade, 0) + 1
            validation['decade_distribution'] = decades
        
        logger.info(f"Publication year validation completed:")
        logger.info(f"  • Valid years: {validation['valid_years']}")
        logger.info(f"  • Invalid years: {validation['invalid_years']}")
        logger.info(f"  • Year range: {validation['year_range']['min']}-{validation['year_range']['max']}")
        
        return validation
    
    def create_derived_variables(self) -> Dict[str, Any]:
        """
        Create derived variables for analysis.
        
        Returns:
            Dictionary with derived variable creation results
        """
        logger.info("🔧 Creating derived variables...")
        
        creation_results = {
            'variables_created': 0,
            'creation_details': {},
            'validation_issues': []
        }
        
        # 1. Create decade categories
        if 'publication_year' in self.df.columns:
            try:
                years = pd.to_numeric(self.df['publication_year'], errors='coerce')
                self.df['decade'] = (years // 10) * 10
                self.df['decade'] = self.df['decade'].astype('category')
                
                creation_results['variables_created'] += 1
                creation_results['creation_details']['decade'] = {
                    'type': 'categorical',
                    'unique_values': self.df['decade'].nunique(),
                    'description': 'Publication decade (e.g., 2000, 2010)'
                }
                
                logger.info(f"  ✅ Created decade variable: {self.df['decade'].nunique()} unique decades")
            except Exception as e:
                creation_results['validation_issues'].append(f"Failed to create decade: {str(e)}")
        
        # 2. Create book length categories
        if 'num_pages_median' in self.df.columns:
            try:
                pages = pd.to_numeric(self.df['num_pages_median'], errors='coerce')
                
                # Create length categories
                length_categories = pd.cut(
                    pages,
                    bins=[0, 200, 400, 600, float('inf')],
                    labels=['Short', 'Medium', 'Long', 'Very Long'],
                    include_lowest=True
                )
                
                self.df['book_length_category'] = length_categories.astype('category')
                
                creation_results['variables_created'] += 1
                creation_results['creation_details']['book_length_category'] = {
                    'type': 'categorical',
                    'unique_values': self.df['book_length_category'].nunique(),
                    'description': 'Book length categories based on page count'
                }
                
                logger.info(f"  ✅ Created book_length_category variable")
            except Exception as e:
                creation_results['validation_issues'].append(f"Failed to create book length category: {str(e)}")
        
        # 3. Create rating categories
        if 'average_rating_weighted_mean' in self.df.columns:
            try:
                ratings = pd.to_numeric(self.df['average_rating_weighted_mean'], errors='coerce')
                
                # Create rating categories
                rating_categories = pd.cut(
                    ratings,
                    bins=[0, 2, 3, 4, 5],
                    labels=['Poor', 'Fair', 'Good', 'Excellent'],
                    include_lowest=True
                )
                
                self.df['rating_category'] = rating_categories.astype('category')
                
                creation_results['variables_created'] += 1
                creation_results['creation_details']['rating_category'] = {
                    'type': 'categorical',
                    'unique_values': self.df['rating_category'].nunique(),
                    'description': 'Rating categories based on average rating'
                }
                
                logger.info(f"  ✅ Created rating_category variable")
            except Exception as e:
                creation_results['validation_issues'].append(f"Failed to create rating category: {str(e)}")
        
        # 4. Create popularity categories
        if 'ratings_count_sum' in self.df.columns:
            try:
                counts = pd.to_numeric(self.df['ratings_count_sum'], errors='coerce')
                
                # Create popularity categories based on quartiles
                popularity_categories = pd.qcut(
                    counts,
                    q=4,
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    duplicates='drop'
                )
                
                self.df['popularity_category'] = popularity_categories.astype('category')
                
                creation_results['variables_created'] += 1
                creation_results['creation_details']['popularity_category'] = {
                    'type': 'categorical',
                    'unique_values': self.df['popularity_category'].nunique(),
                    'description': 'Popularity categories based on rating count quartiles'
                }
                
                logger.info(f"  ✅ Created popularity_category variable")
            except Exception as e:
                creation_results['validation_issues'].append(f"Failed to create popularity category: {str(e)}")
        
        logger.info(f"Derived variable creation completed:")
        logger.info(f"  • Variables created: {creation_results['variables_created']}")
        logger.info(f"  • Validation issues: {len(creation_results['validation_issues'])}")
        
        return creation_results
    
    def apply_data_type_optimizations(self) -> Dict[str, Any]:
        """
        Apply data type optimizations.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("🔧 Applying data type optimizations...")
        
        optimization_results = {
            'optimizations_applied': 0,
            'optimizations_failed': 0,
            'memory_before_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'optimization_details': {}
        }
        
        # Apply numerical optimizations
        for col, target_dtype in self.optimization_mappings.items():
            if col not in self.df.columns:
                continue
            
            try:
                current_dtype = str(self.df[col].dtype)
                
                if current_dtype != target_dtype:
                    # Apply optimization
                    if target_dtype.startswith('int'):
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(target_dtype)
                    elif target_dtype.startswith('float'):
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(target_dtype)
                    
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['optimization_details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'success'
                    }
                    
                    logger.info(f"  ✅ {col}: {current_dtype} → {target_dtype}")
                else:
                    optimization_results['optimization_details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'already_optimized'
                    }
                    
            except Exception as e:
                optimization_results['optimizations_failed'] += 1
                optimization_results['optimization_details'][col] = {
                    'from': str(self.df[col].dtype),
                    'to': target_dtype,
                    'status': 'failed',
                    'error': str(e)
                }
                
                logger.error(f"  ❌ {col}: Failed to optimize to {target_dtype} - {str(e)}")
        
        # Apply categorical optimizations
        for col in self.categorical_fields:
            if col in self.df.columns and str(self.df[col].dtype) != 'category':
                try:
                    self.df[col] = self.df[col].astype('category')
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['optimization_details'][col] = {
                        'from': 'object',
                        'to': 'category',
                        'status': 'success',
                        'unique_values': self.df[col].nunique()
                    }
                    
                    logger.info(f"  ✅ {col}: object → category ({self.df[col].nunique()} unique values)")
                except Exception as e:
                    optimization_results['optimizations_failed'] += 1
                    optimization_results['optimization_details'][col] = {
                        'from': str(self.df[col].dtype),
                        'to': 'category',
                        'status': 'failed',
                        'error': str(e)
                    }
                    
                    logger.error(f"  ❌ {col}: Failed to convert to category - {str(e)}")
        
        optimization_results['memory_after_mb'] = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        optimization_results['memory_saved_mb'] = optimization_results['memory_before_mb'] - optimization_results['memory_after_mb']
        
        logger.info(f"Data type optimization completed:")
        logger.info(f"  • Optimizations applied: {optimization_results['optimizations_applied']}")
        logger.info(f"  • Optimizations failed: {optimization_results['optimizations_failed']}")
        logger.info(f"  • Memory saved: {optimization_results['memory_saved_mb']:.2f} MB")
        
        return optimization_results
    
    def validate_optimization_results(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the results of data type optimization.
        
        Args:
            optimization_results: Optimization results dictionary
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating data type optimization results...")
        
        validation = {
            'validation_passed': True,
            'optimization_validation': {},
            'validation_errors': [],
            'memory_efficiency': {}
        }
        
        # Validate each optimization
        for col, details in optimization_results['optimization_details'].items():
            if details['status'] == 'success':
                current_dtype = str(self.df[col].dtype)
                expected_dtype = details['to']
                
                if current_dtype == expected_dtype:
                    validation['optimization_validation'][col] = {
                        'status': '✅ Validated',
                        'expected': expected_dtype,
                        'actual': current_dtype
                    }
                else:
                    validation['optimization_validation'][col] = {
                        'status': '❌ Failed',
                        'expected': expected_dtype,
                        'actual': current_dtype
                    }
                    validation['validation_errors'].append(
                        f"Optimization failed for {col}: expected {expected_dtype}, got {current_dtype}"
                    )
                    validation['validation_passed'] = False
        
        # Calculate memory efficiency
        memory_saved = optimization_results['memory_saved_mb']
        memory_before = optimization_results['memory_before_mb']
        
        validation['memory_efficiency'] = {
            'memory_before_mb': memory_before,
            'memory_after_mb': optimization_results['memory_after_mb'],
            'memory_saved_mb': memory_saved,
            'efficiency_percentage': (memory_saved / memory_before) * 100 if memory_before > 0 else 0
        }
        
        logger.info(f"Optimization validation completed: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
        
        return validation
    
    def save_validated_dataset(self, validated_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the validated dataset.
        
        Args:
            validated_df: Validated DataFrame
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"romance_novels_step3_data_types_validated_{self.timestamp}.pkl"
        
        filepath = self.output_dir / filename
        
        # Save as pickle to preserve data types
        validated_df.to_pickle(filepath)
        
        logger.info(f"Validated dataset saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(filepath)
    
    def save_validation_report(self, analysis: Dict[str, Any], optimization_results: Dict[str, Any], 
                             validation: Dict[str, Any], derived_vars: Dict[str, Any], 
                             filename: str = None) -> str:
        """
        Save comprehensive validation report.
        
        Args:
            analysis: Data type analysis
            optimization_results: Optimization results
            validation: Validation results
            derived_vars: Derived variables results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"data_type_validation_report_step3_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            'validation_timestamp': self.timestamp,
            'data_type_analysis': analysis,
            'optimization_results': optimization_results,
            'validation_results': validation,
            'derived_variables': derived_vars,
            'summary': {
                'original_records': self.df.shape[0],
                'validated_records': self.df.shape[0],
                'optimizations_applied': optimization_results['optimizations_applied'],
                'optimizations_failed': optimization_results['optimizations_failed'],
                'derived_variables_created': derived_vars['variables_created'],
                'memory_saved_mb': optimization_results['memory_saved_mb'],
                'validation_passed': validation['validation_passed']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {filepath}")
        return str(filepath)
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete data type validation and conversion process.
        
        Returns:
            Dictionary with complete validation results
        """
        logger.info("🚀 Starting complete data type validation and conversion process...")
        start_time = datetime.now()
        
        # Initialize results
        self.validation_results = {
            'validation_timestamp': self.timestamp,
            'dataset_info': {},
            'data_type_analysis': {},
            'publication_year_validation': {},
            'derived_variables': {},
            'optimization_results': {},
            'validation_results': {},
            'final_dataset_info': {},
            'validation_summary': {}
        }
        
        # 1. Load dataset
        logger.info("📥 Loading dataset...")
        original_df = self.load_data()
        
        self.validation_results['dataset_info'] = {
            'original_shape': original_df.shape,
            'original_memory_mb': original_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'original_columns': list(original_df.columns)
        }
        
        # 2. Analyze data types
        logger.info("🔍 Analyzing data types...")
        analysis = self.analyze_data_types()
        self.validation_results['data_type_analysis'] = analysis
        
        # 3. Validate publication years
        logger.info("🔍 Validating publication years...")
        year_validation = self.validate_publication_years()
        self.validation_results['publication_year_validation'] = year_validation
        
        # 4. Create derived variables
        logger.info("🔧 Creating derived variables...")
        derived_vars = self.create_derived_variables()
        self.validation_results['derived_variables'] = derived_vars
        
        # 5. Apply optimizations
        logger.info("🔧 Applying data type optimizations...")
        optimization_results = self.apply_data_type_optimizations()
        self.validation_results['optimization_results'] = optimization_results
        
        # 6. Validate results
        logger.info("🔍 Validating optimization results...")
        validation = self.validate_optimization_results(optimization_results)
        self.validation_results['validation_results'] = validation
        
        # 7. Save validated dataset
        logger.info("💾 Saving validated dataset...")
        dataset_file = self.save_validated_dataset(self.df)
        
        # 8. Final dataset information
        self.validation_results['final_dataset_info'] = {
            'final_shape': self.df.shape,
            'final_memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'final_columns': list(self.df.columns),
            'data_type_preservation': 'Optimized and preserved'
        }
        
        # 9. Generate validation summary
        self.validation_results['validation_summary'] = {
            'total_optimizations_applied': optimization_results['optimizations_applied'],
            'total_optimizations_failed': optimization_results['optimizations_failed'],
            'derived_variables_created': derived_vars['variables_created'],
            'memory_saved_mb': optimization_results['memory_saved_mb'],
            'validation_passed': validation['validation_passed'],
            'data_integrity': 'Maintained - data types optimized and validated',
            'execution_time': str(datetime.now() - start_time)
        }
        
        # 10. Save validation report
        logger.info("📊 Saving validation report...")
        report_file = self.save_validation_report(analysis, optimization_results, validation, derived_vars)
        
        logger.info("✅ Complete data type validation and conversion process finished!")
        
        return self.validation_results
    
    def print_validation_summary(self):
        """Print a human-readable validation summary."""
        if not self.validation_results:
            print("No validation results available. Run run_complete_validation() first.")
            return
        
        print("\n" + "="*80)
        print("DATA TYPE VALIDATION & CONVERSION SUMMARY - STEP 3")
        print("="*80)
        
        # Original dataset info
        original_info = self.validation_results['dataset_info']
        print(f"Original Dataset: {original_info['original_shape'][0]:,} records × {original_info['original_shape'][1]} columns")
        print(f"Original Memory: {original_info['original_memory_mb']:.2f} MB")
        
        # Data type analysis
        analysis = self.validation_results['data_type_analysis']
        print(f"\n🔍 DATA TYPE ANALYSIS:")
        print(f"  • Optimization opportunities: {len(analysis['optimization_opportunities'])}")
        print(f"  • Categorical opportunities: {len(analysis['categorical_opportunities'])}")
        print(f"  • Validation issues: {len(analysis['validation_issues'])}")
        
        # Derived variables
        derived_vars = self.validation_results['derived_variables']
        print(f"\n🔧 DERIVED VARIABLES:")
        print(f"  • Variables created: {derived_vars['variables_created']}")
        print(f"  • Validation issues: {len(derived_vars['validation_issues'])}")
        
        # Optimization results
        optimization = self.validation_results['optimization_results']
        print(f"\n🔧 OPTIMIZATION RESULTS:")
        print(f"  • Optimizations applied: {optimization['optimizations_applied']}")
        print(f"  • Optimizations failed: {optimization['optimizations_failed']}")
        print(f"  • Memory saved: {optimization['memory_saved_mb']:.2f} MB")
        
        # Final dataset info
        final_info = self.validation_results['final_dataset_info']
        print(f"\n📊 FINAL DATASET:")
        print(f"  • Final records: {final_info['final_shape'][0]:,} records × {final_info['final_shape'][1]} columns")
        print(f"  • Final memory: {final_info['final_memory_mb']:.2f} MB")
        print(f"  • Data types: {final_info['data_type_preservation']}")
        
        # Validation summary
        summary = self.validation_results['validation_summary']
        print(f"\n🎯 VALIDATION SUMMARY:")
        print(f"  • Validation passed: {'✅ YES' if summary['validation_passed'] else '❌ NO'}")
        print(f"  • Data integrity: {summary['data_integrity']}")
        print(f"  • Execution time: {summary['execution_time']}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    
    print("🔧 STEP 3: DATA TYPE VALIDATION & CONVERSION")
    print("=" * 60)
    
    try:
        # Initialize data type validator
        validator = DataTypeValidator()
        
        # Run complete validation process
        results = validator.run_complete_validation()
        
        # Print summary
        validator.print_validation_summary()
        
        print("\n🎉 Step 3: Data Type Validation & Conversion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data type validation and conversion failed: {str(e)}", exc_info=True)
        print(f"\n💥 Data type validation and conversion failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
