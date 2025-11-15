#!/usr/bin/env python3
"""
Step 5: Data Type Optimization & Persistence (High Priority)

This script fixes the data type persistence issues from Step 3 by:
1. Applying comprehensive data type optimizations
2. Implementing parquet export for type preservation
3. Validating memory usage improvements
4. Ensuring all optimizations are properly persisted

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

class DataTypeOptimizer:
    """
    Comprehensive data type optimization and persistence system.
    Fixes issues from Step 3 where CSV export lost optimizations.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data type optimizer.
        
        Args:
            data_path: Path to the treated dataset from Step 4
        """
        self.data_path = data_path or "outputs/outlier_detection/cleaned_romance_novels_step4_treated_20250902_231021.pkl"
        self.df = None
        self.optimization_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/data_type_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define optimization mappings based on Step 3 analysis
        self.numerical_optimizations = {
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
        
        self.categorical_optimizations = {
            'genres': 'category',
            'author_name': 'category',
            'series_title': 'category',
            'duplication_status': 'category',
            'cleaning_strategy': 'category',
            'disambiguation_notes': 'category',
            'decade': 'category',
            'book_length_category': 'category',
            'rating_category': 'category',
            'popularity_category': 'category',
            'language_codes_en': 'category'
        }
        
        self.float_optimizations = {
            'average_rating_weighted_mean': 'float32',
            'author_average_rating': 'float32',
            'num_pages_median': 'float32',
            'series_id': 'float32',
            'series_works_count': 'float32'
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the treated dataset from Step 4.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading treated dataset from: {self.data_path}")
            
            if self.data_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.data_path)
                logger.info("Data loaded from pickle file - data types preserved")
            else:
                self.df = pd.read_csv(self.data_path)
                logger.info("Data loaded from CSV file")
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Display current data types
            logger.info("Current data types:")
            for col, dtype in self.df.dtypes.items():
                logger.info(f"  {col}: {dtype}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def analyze_current_data_types(self) -> Dict[str, Any]:
        """
        Analyze current data types and identify optimization opportunities.
        
        Returns:
            Dictionary with current data type analysis
        """
        logger.info("🔍 Analyzing current data types...")
        
        analysis = {
            'total_columns': len(self.df.columns),
            'current_memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_type_distribution': {},
            'optimization_opportunities': {},
            'already_optimized': []
        }
        
        # Analyze data type distribution
        for dtype in self.df.dtypes.value_counts().items():
            analysis['data_type_distribution'][str(dtype[0])] = int(dtype[1])
        
        # Identify optimization opportunities
        for col in self.df.columns:
            current_dtype = str(self.df[col].dtype)
            
            # Check numerical optimizations
            if col in self.numerical_optimizations:
                target_dtype = self.numerical_optimizations[col]
                if current_dtype != target_dtype:
                    analysis['optimization_opportunities'][col] = {
                        'current': current_dtype,
                        'target': target_dtype,
                        'type': 'numerical'
                    }
                else:
                    analysis['already_optimized'].append(f"{col} ({target_dtype})")
            
            # Check categorical optimizations
            elif col in self.categorical_optimizations:
                target_dtype = self.categorical_optimizations[col]
                if current_dtype != target_dtype:
                    analysis['optimization_opportunities'][col] = {
                        'current': current_dtype,
                        'target': target_dtype,
                        'type': 'categorical'
                    }
                else:
                    analysis['already_optimized'].append(f"{col} ({target_dtype})")
            
            # Check float optimizations
            elif col in self.float_optimizations:
                target_dtype = self.float_optimizations[col]
                if current_dtype != target_dtype:
                    analysis['optimization_opportunities'][col] = {
                        'current': current_dtype,
                        'target': target_dtype,
                        'type': 'float'
                    }
                else:
                    analysis['already_optimized'].append(f"{col} ({target_dtype})")
        
        logger.info(f"Data type analysis completed:")
        logger.info(f"  • Total columns: {analysis['total_columns']}")
        logger.info(f"  • Current memory: {analysis['current_memory_mb']:.2f} MB")
        logger.info(f"  • Optimization opportunities: {len(analysis['optimization_opportunities'])}")
        logger.info(f"  • Already optimized: {len(analysis['already_optimized'])}")
        
        return analysis
    
    def apply_numerical_optimizations(self) -> Dict[str, Any]:
        """
        Apply numerical data type optimizations.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("🔧 Applying numerical data type optimizations...")
        
        results = {
            'optimizations_applied': 0,
            'optimizations_failed': 0,
            'memory_before_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'details': {}
        }
        
        for col, target_dtype in self.numerical_optimizations.items():
            if col not in self.df.columns:
                continue
            
            try:
                current_dtype = str(self.df[col].dtype)
                
                if current_dtype != target_dtype:
                    # Apply optimization
                    if target_dtype == 'int16':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('int16')
                    elif target_dtype == 'int32':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('int32')
                    elif target_dtype == 'int64':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('int64')
                    
                    results['optimizations_applied'] += 1
                    results['details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'success'
                    }
                    
                    logger.info(f"  ✅ {col}: {current_dtype} → {target_dtype}")
                else:
                    results['details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'already_optimized'
                    }
                    
            except Exception as e:
                results['optimizations_failed'] += 1
                results['details'][col] = {
                    'from': str(self.df[col].dtype),
                    'to': target_dtype,
                    'status': 'failed',
                    'error': str(e)
                }
                
                logger.error(f"  ❌ {col}: Failed to optimize to {target_dtype} - {str(e)}")
        
        results['memory_after_mb'] = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        results['memory_saved_mb'] = results['memory_before_mb'] - results['memory_after_mb']
        
        logger.info(f"Numerical optimizations completed:")
        logger.info(f"  • Applied: {results['optimizations_applied']}")
        logger.info(f"  • Failed: {results['optimizations_failed']}")
        logger.info(f"  • Memory saved: {results['memory_saved_mb']:.2f} MB")
        
        return results
    
    def apply_categorical_optimizations(self) -> Dict[str, Any]:
        """
        Apply categorical data type optimizations.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("🔧 Applying categorical data type optimizations...")
        
        results = {
            'optimizations_applied': 0,
            'optimizations_failed': 0,
            'memory_before_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'details': {}
        }
        
        for col, target_dtype in self.categorical_optimizations.items():
            if col not in self.df.columns:
                continue
            
            try:
                current_dtype = str(self.df[col].dtype)
                
                if current_dtype != target_dtype:
                    # Apply optimization
                    self.df[col] = self.df[col].astype('category')
                    
                    results['optimizations_applied'] += 1
                    results['details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'success',
                        'unique_values': self.df[col].nunique()
                    }
                    
                    logger.info(f"  ✅ {col}: {current_dtype} → {target_dtype} ({self.df[col].nunique()} unique values)")
                else:
                    results['details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'already_optimized',
                        'unique_values': self.df[col].nunique()
                    }
                    
            except Exception as e:
                results['optimizations_failed'] += 1
                results['details'][col] = {
                    'from': str(self.df[col].dtype),
                    'to': target_dtype,
                    'status': 'failed',
                    'error': str(e)
                }
                
                logger.error(f"  ❌ {col}: Failed to optimize to {target_dtype} - {str(e)}")
        
        results['memory_after_mb'] = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        results['memory_saved_mb'] = results['memory_before_mb'] - results['memory_after_mb']
        
        logger.info(f"Categorical optimizations completed:")
        logger.info(f"  • Applied: {results['optimizations_applied']}")
        logger.info(f"  • Failed: {results['optimizations_failed']}")
        logger.info(f"  • Memory saved: {results['memory_saved_mb']:.2f} MB")
        
        return results
    
    def apply_float_optimizations(self) -> Dict[str, Any]:
        """
        Apply float data type optimizations.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("🔧 Applying float data type optimizations...")
        
        results = {
            'optimizations_applied': 0,
            'optimizations_failed': 0,
            'memory_before_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'details': {}
        }
        
        for col, target_dtype in self.float_optimizations.items():
            if col not in self.df.columns:
                continue
            
            try:
                current_dtype = str(self.df[col].dtype)
                
                if current_dtype != target_dtype:
                    # Apply optimization
                    if target_dtype == 'float32':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('float32')
                    elif target_dtype == 'float64':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('float64')
                    
                    results['optimizations_applied'] += 1
                    results['details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'success'
                    }
                    
                    logger.info(f"  ✅ {col}: {current_dtype} → {target_dtype}")
                else:
                    results['details'][col] = {
                        'from': current_dtype,
                        'to': target_dtype,
                        'status': 'already_optimized'
                    }
                    
            except Exception as e:
                results['optimizations_failed'] += 1
                results['details'][col] = {
                    'from': str(self.df[col].dtype),
                    'from': str(self.df[col].dtype),
                    'to': target_dtype,
                    'status': 'failed',
                    'error': str(e)
                }
                
                logger.error(f"  ❌ {col}: Failed to optimize to {target_dtype} - {str(e)}")
        
        results['memory_after_mb'] = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        results['memory_saved_mb'] = results['memory_before_mb'] - results['memory_after_mb']
        
        logger.info(f"Float optimizations completed:")
        logger.info(f"  • Applied: {results['optimizations_applied']}")
        logger.info(f"  • Failed: {results['optimizations_failed']}")
        logger.info(f"  • Memory saved: {results['memory_saved_mb']:.2f} MB")
        
        return results
    
    def save_optimized_dataset(self, format_type: str = 'parquet') -> str:
        """
        Save the optimized dataset in the specified format.
        
        Args:
            format_type: Format to save ('parquet', 'pickle', 'csv')
            
        Returns:
            Path to saved file
        """
        logger.info(f"💾 Saving optimized dataset in {format_type.upper()} format...")
        
        timestamp = self.timestamp
        filename = f"cleaned_romance_novels_step5_optimized_{timestamp}.{format_type}"
        filepath = self.output_dir / filename
        
        try:
            if format_type == 'parquet':
                # Parquet preserves data types and is efficient
                self.df.to_parquet(filepath, engine='pyarrow', compression='snappy')
                logger.info("✅ Dataset saved as parquet with pyarrow engine")
                
            elif format_type == 'pickle':
                # Pickle preserves all data types
                self.df.to_pickle(filepath)
                logger.info("✅ Dataset saved as pickle")
                
            elif format_type == 'csv':
                # CSV may lose some optimizations but is widely compatible
                self.df.to_csv(filepath, index=False)
                logger.info("⚠️  Dataset saved as CSV (some optimizations may be lost)")
            
            # Get file size
            file_size_mb = filepath.stat().st_size / 1024 / 1024
            logger.info(f"File saved: {filepath}")
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save dataset as {format_type}: {str(e)}")
            raise
    
    def validate_optimizations(self) -> Dict[str, Any]:
        """
        Validate that all optimizations were applied correctly.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating data type optimizations...")
        
        validation = {
            'total_columns': len(self.df.columns),
            'optimized_columns': 0,
            'failed_optimizations': 0,
            'validation_details': {},
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Check numerical optimizations
        for col, target_dtype in self.numerical_optimizations.items():
            if col in self.df.columns:
                current_dtype = str(self.df[col].dtype)
                if current_dtype == target_dtype:
                    validation['optimized_columns'] += 1
                    validation['validation_details'][col] = {
                        'expected': target_dtype,
                        'actual': current_dtype,
                        'status': '✅ Optimized'
                    }
                else:
                    validation['failed_optimizations'] += 1
                    validation['validation_details'][col] = {
                        'expected': target_dtype,
                        'actual': current_dtype,
                        'status': '❌ Failed'
                    }
        
        # Check categorical optimizations
        for col, target_dtype in self.categorical_optimizations.items():
            if col in self.df.columns:
                current_dtype = str(self.df[col].dtype)
                if current_dtype == target_dtype:
                    validation['optimized_columns'] += 1
                    validation['validation_details'][col] = {
                        'expected': target_dtype,
                        'actual': current_dtype,
                        'status': '✅ Optimized'
                    }
                else:
                    validation['failed_optimizations'] += 1
                    validation['validation_details'][col] = {
                        'expected': target_dtype,
                        'actual': current_dtype,
                        'status': '❌ Failed'
                    }
        
        # Check float optimizations
        for col, target_dtype in self.float_optimizations.items():
            if col in self.df.columns:
                current_dtype = str(self.df[col].dtype)
                if current_dtype == target_dtype:
                    validation['optimized_columns'] += 1
                    validation['validation_details'][col] = {
                        'expected': target_dtype,
                        'actual': current_dtype,
                        'status': '✅ Optimized'
                    }
                else:
                    validation['failed_optimizations'] += 1
                    validation['validation_details'][col] = {
                        'expected': target_dtype,
                        'actual': current_dtype,
                        'status': '❌ Failed'
                    }
        
        logger.info(f"Optimization validation completed:")
        logger.info(f"  • Total columns: {validation['total_columns']}")
        logger.info(f"  • Optimized columns: {validation['optimized_columns']}")
        logger.info(f"  • Failed optimizations: {validation['failed_optimizations']}")
        logger.info(f"  • Final memory usage: {validation['memory_usage_mb']:.2f} MB")
        
        return validation
    
    def run_complete_optimization(self) -> Dict[str, Any]:
        """
        Run complete data type optimization process.
        
        Returns:
            Dictionary with complete optimization results
        """
        logger.info("🚀 Starting complete data type optimization process...")
        start_time = datetime.now()
        
        # Initialize results
        self.optimization_results = {
            'optimization_timestamp': self.timestamp,
            'initial_analysis': {},
            'numerical_optimizations': {},
            'categorical_optimizations': {},
            'float_optimizations': {},
            'validation_results': {},
            'final_dataset_info': {},
            'optimization_summary': {}
        }
        
        # 1. Load and analyze dataset
        logger.info("📥 Loading and analyzing dataset...")
        original_df = self.load_data()
        
        initial_analysis = self.analyze_current_data_types()
        self.optimization_results['initial_analysis'] = initial_analysis
        
        # 2. Apply numerical optimizations
        logger.info("🔧 Applying numerical optimizations...")
        numerical_results = self.apply_numerical_optimizations()
        self.optimization_results['numerical_optimizations'] = numerical_results
        
        # 3. Apply categorical optimizations
        logger.info("🔧 Applying categorical optimizations...")
        categorical_results = self.apply_categorical_optimizations()
        self.optimization_results['categorical_optimizations'] = categorical_results
        
        # 4. Apply float optimizations
        logger.info("🔧 Applying float optimizations...")
        float_results = self.apply_float_optimizations()
        self.optimization_results['float_optimizations'] = float_results
        
        # 5. Validate optimizations
        logger.info("🔍 Validating optimizations...")
        validation_results = self.validate_optimizations()
        self.optimization_results['validation_results'] = validation_results
        
        # 6. Save optimized dataset in multiple formats
        logger.info("💾 Saving optimized dataset...")
        saved_files = {}
        
        # Save as parquet (primary format for type preservation)
        try:
            parquet_file = self.save_optimized_dataset('parquet')
            saved_files['parquet'] = parquet_file
        except Exception as e:
            logger.error(f"Failed to save parquet: {str(e)}")
            saved_files['parquet'] = None
        
        # Save as pickle (backup format)
        try:
            pickle_file = self.save_optimized_dataset('pickle')
            saved_files['pickle'] = pickle_file
        except Exception as e:
            logger.error(f"Failed to save pickle: {str(e)}")
            saved_files['pickle'] = None
        
        # 7. Final dataset information
        self.optimization_results['final_dataset_info'] = {
            'final_shape': self.df.shape,
            'final_memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'final_columns': list(self.df.columns),
            'saved_files': saved_files,
            'data_type_preservation': 'Optimized and preserved'
        }
        
        # 8. Generate optimization summary
        total_optimizations = (
            numerical_results['optimizations_applied'] +
            categorical_results['optimizations_applied'] +
            float_results['optimizations_applied']
        )
        
        total_failures = (
            numerical_results['optimizations_failed'] +
            categorical_results['optimizations_failed'] +
            float_results['optimizations_failed']
        )
        
        memory_saved = (
            initial_analysis['current_memory_mb'] -
            validation_results['memory_usage_mb']
        )
        
        self.optimization_results['optimization_summary'] = {
            'total_optimizations_applied': total_optimizations,
            'total_optimizations_failed': total_failures,
            'success_rate': (total_optimizations / (total_optimizations + total_failures)) * 100 if (total_optimizations + total_failures) > 0 else 0,
            'memory_saved_mb': memory_saved,
            'memory_reduction_percentage': (memory_saved / initial_analysis['current_memory_mb']) * 100 if initial_analysis['current_memory_mb'] > 0 else 0,
            'execution_time': str(datetime.now() - start_time),
            'recommendations': []
        }
        
        # Add recommendations
        if total_failures > 0:
            self.optimization_results['optimization_summary']['recommendations'].append(
                f"Review {total_failures} failed optimizations for data quality issues"
            )
        
        if memory_saved > 0:
            self.optimization_results['optimization_summary']['recommendations'].append(
                f"Memory usage reduced by {memory_saved:.2f} MB ({self.optimization_results['optimization_summary']['memory_reduction_percentage']:.1f}%)"
            )
        
        if saved_files.get('parquet'):
            self.optimization_results['optimization_summary']['recommendations'].append(
                "Use parquet format for type preservation and efficient storage"
            )
        
        logger.info("✅ Complete data type optimization process finished!")
        
        return self.optimization_results
    
    def save_optimization_report(self, filename: str = None) -> str:
        """
        Save comprehensive optimization report.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"data_type_optimization_report_step5_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.Index):
                return obj.tolist()
            return obj
        
        # Deep copy and convert results
        import copy
        results_copy = copy.deepcopy(self.optimization_results)
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy_types(obj)
        
        results_copy = recursive_convert(results_copy)
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to: {filepath}")
        return str(filepath)
    
    def print_optimization_summary(self):
        """Print a human-readable optimization summary."""
        if not self.optimization_results:
            print("No optimization results available. Run run_complete_optimization() first.")
            return
        
        print("\n" + "="*80)
        print("DATA TYPE OPTIMIZATION SUMMARY - STEP 5")
        print("="*80)
        
        # Initial analysis
        initial = self.optimization_results['initial_analysis']
        print(f"Original Dataset: {initial['total_columns']} columns")
        print(f"Original Memory: {initial['current_memory_mb']:.2f} MB")
        print(f"Optimization Opportunities: {len(initial['optimization_opportunities'])}")
        
        # Optimization results
        summary = self.optimization_results['optimization_summary']
        print(f"\n🔧 OPTIMIZATION RESULTS:")
        print(f"  • Total optimizations applied: {summary['total_optimizations_applied']}")
        print(f"  • Failed optimizations: {summary['total_optimizations_failed']}")
        print(f"  • Success rate: {summary['success_rate']:.1f}%")
        print(f"  • Memory saved: {summary['memory_saved_mb']:.2f} MB")
        print(f"  • Memory reduction: {summary['memory_reduction_percentage']:.1f}%")
        
        # Final dataset info
        final = self.optimization_results['final_dataset_info']
        print(f"\n📊 FINAL DATASET:")
        print(f"  • Final shape: {final['final_shape'][0]:,} records × {final['final_shape'][1]} columns")
        print(f"  • Final memory: {final['final_memory_mb']:.2f} MB")
        print(f"  • Data types: {final['data_type_preservation']}")
        
        # Saved files
        print(f"\n💾 SAVED FILES:")
        for format_type, filepath in final['saved_files'].items():
            if filepath:
                print(f"  • {format_type.upper()}: {Path(filepath).name}")
            else:
                print(f"  • {format_type.upper()}: Failed to save")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    
    print("🔧 STEP 5: DATA TYPE OPTIMIZATION & PERSISTENCE")
    print("=" * 60)
    
    try:
        # Initialize optimizer
        optimizer = DataTypeOptimizer()
        
        # Run complete optimization process
        results = optimizer.run_complete_optimization()
        
        # Save optimization report
        logger.info("📊 Saving optimization report...")
        report_file = optimizer.save_optimization_report()
        
        # Print summary
        optimizer.print_optimization_summary()
        
        print("\n🎉 Step 5: Data Type Optimization & Persistence completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data type optimization failed: {str(e)}", exc_info=True)
        print(f"\n💥 Data type optimization failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
