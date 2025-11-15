#!/usr/bin/env python3
"""
Step 1: Missing Values Treatment (Critical Priority)

This script implements comprehensive missing values treatment for the romance novel dataset:
1. Missing Values Assessment and Analysis
2. Strategic Treatment Implementation
3. Missing Value Flagging and Documentation
4. Data Completeness Validation

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

class MissingValuesCleaner:
    """
    Comprehensive missing values treatment system for romance novel dataset.
    Implements strategic treatment approaches based on data analysis.
    NO IMPUTATION is applied - only flagging and exclusion strategies.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the missing values cleaner.
        
        Args:
            data_path: Path to the input dataset file
        """
        self.data_path = data_path or "data/processed/romance_novels_integrated.csv"
        self.df = None
        self.cleaning_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/missing_values_cleaning")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Missing value treatment strategies
        self.treatment_strategies = {
            'series_id': 'flag_for_analysis',
            'series_title': 'flag_for_analysis', 
            'series_works_count': 'flag_for_analysis',
            'average_rating_weighted_mean': 'exclude_from_analysis',
            'disambiguation_notes': 'flag_for_investigation',
            'num_pages_median': 'exclude_from_analysis',
            'description': 'exclude_from_analysis'
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset for missing values treatment.
        
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
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Comprehensive missing values analysis.
        
        Returns:
            Dictionary with missing values analysis results
        """
        logger.info("🔍 Analyzing missing values across all variables...")
        
        analysis = {
            'total_variables': len(self.df.columns),
            'variables_with_missing': 0,
            'missing_summary': {},
            'missing_by_variable': {},
            'treatment_recommendations': {}
        }
        
        # Analyze each variable
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            analysis['missing_by_variable'][col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'data_type': str(self.df[col].dtype),
                'completeness_rate': 100 - missing_pct
            }
            
            if missing_count > 0:
                analysis['variables_with_missing'] += 1
                
                # Categorize by missing percentage
                if missing_pct < 1:
                    category = 'minimal'
                elif missing_pct < 10:
                    category = 'moderate'
                elif missing_pct < 50:
                    category = 'significant'
                else:
                    category = 'extensive'
                
                if category not in analysis['missing_summary']:
                    analysis['missing_summary'][category] = []
                analysis['missing_summary'][category].append({
                    'variable': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct
                })
                
                # Determine treatment strategy
                strategy = self._determine_treatment_strategy(col, missing_pct)
                analysis['treatment_recommendations'][col] = {
                    'strategy': strategy,
                    'rationale': self._get_strategy_rationale(strategy, missing_pct)
                }
        
        logger.info(f"Missing values analysis completed:")
        logger.info(f"  • Total variables: {analysis['total_variables']}")
        logger.info(f"  • Variables with missing values: {analysis['variables_with_missing']}")
        
        return analysis
    
    def _determine_treatment_strategy(self, variable: str, missing_pct: float) -> str:
        """
        Determine appropriate treatment strategy for missing values.
        
        Args:
            variable: Variable name
            missing_pct: Missing percentage
            
        Returns:
            Treatment strategy
        """
        # Use predefined strategies if available
        if variable in self.treatment_strategies:
            return self.treatment_strategies[variable]
        
        # Default strategies based on missing percentage - NO IMPUTATION
        if missing_pct < 1:
            return 'flag_for_analysis'  # Changed from imputation to flagging
        elif missing_pct < 10:
            return 'flag_for_analysis'
        elif missing_pct < 50:
            return 'flag_for_investigation'
        else:
            return 'exclude_from_analysis'
    
    def _get_strategy_rationale(self, strategy: str, missing_pct: float) -> str:
        """
        Get rationale for treatment strategy.
        
        Args:
            strategy: Treatment strategy
            missing_pct: Missing percentage
            
        Returns:
            Rationale string
        """
        rationales = {
            'flag_for_analysis': f'Low missing rate ({missing_pct:.1f}%) - flag for analysis (no imputation)',
            'flag_for_investigation': f'High missing rate ({missing_pct:.1f}%) - flag for investigation',
            'exclude_from_analysis': f'Very high missing rate ({missing_pct:.1f}%) - exclude from analysis'
        }
        return rationales.get(strategy, 'Unknown strategy')
    
    def apply_missing_value_treatment(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply missing value treatment strategies.
        
        Returns:
            Tuple of (treated_dataframe, treatment_results)
        """
        logger.info("🔧 Applying missing value treatment strategies...")
        
        treated_df = self.df.copy()
        treatment_results = {
            'treatments_applied': 0,
            'records_excluded': 0,
            'flags_created': 0,
            'treatment_details': {}
        }
        
        # Apply treatments based on analysis
        analysis = self.analyze_missing_values()
        
        for variable, recommendation in analysis['treatment_recommendations'].items():
            strategy = recommendation['strategy']
            missing_count = analysis['missing_by_variable'][variable]['missing_count']
            
            if strategy == 'exclude_from_analysis':
                # Exclude records with missing values
                before_count = len(treated_df)
                treated_df = treated_df.dropna(subset=[variable])
                excluded_count = before_count - len(treated_df)
                
                treatment_results['records_excluded'] += excluded_count
                treatment_results['treatment_details'][variable] = {
                    'strategy': strategy,
                    'records_excluded': excluded_count,
                    'action': f'Excluded {excluded_count} records with missing {variable}'
                }
                
                logger.info(f"  ✅ {variable}: Excluded {excluded_count} records")
                
            elif strategy in ['flag_for_analysis', 'flag_for_investigation']:
                # Create missing value flags
                flag_column = f"{variable}_missing_flag"
                treated_df[flag_column] = treated_df[variable].isnull().astype(int)
                
                treatment_results['flags_created'] += 1
                treatment_results['treatment_details'][variable] = {
                    'strategy': strategy,
                    'flag_created': flag_column,
                    'missing_count': missing_count,
                    'action': f'Created flag column {flag_column}'
                }
                
                logger.info(f"  ✅ {variable}: Created flag column {flag_column}")
                
            # Note: Imputation strategies removed - no imputation will be applied
        
        treatment_results['final_shape'] = treated_df.shape
        treatment_results['original_shape'] = self.df.shape
        
        logger.info(f"Missing value treatment completed:")
        logger.info(f"  • Treatments applied: {treatment_results['treatments_applied']}")
        logger.info(f"  • Records excluded: {treatment_results['records_excluded']}")
        logger.info(f"  • Flags created: {treatment_results['flags_created']}")
        
        return treated_df, treatment_results
    
    def validate_treatment_results(self, treated_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the results of missing value treatment.
        
        Args:
            treated_df: Treated DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating missing value treatment results...")
        
        validation = {
            'validation_passed': True,
            'remaining_missing': {},
            'validation_errors': [],
            'completeness_improvement': {}
        }
        
        # Check for remaining missing values
        for col in treated_df.columns:
            missing_count = treated_df[col].isnull().sum()
            if missing_count > 0:
                validation['remaining_missing'][col] = missing_count
                
                # Check if this is acceptable
                if col in self.treatment_strategies:
                    strategy = self.treatment_strategies[col]
                    if strategy == 'exclude_from_analysis' and missing_count > 0:
                        validation['validation_errors'].append(
                            f"Variable {col} should have no missing values after exclusion strategy"
                        )
                        validation['validation_passed'] = False
        
        # Calculate completeness improvement
        original_completeness = {}
        treated_completeness = {}
        
        for col in self.df.columns:
            if col in treated_df.columns:
                original_missing = self.df[col].isnull().sum()
                treated_missing = treated_df[col].isnull().sum()
                
                original_completeness[col] = ((len(self.df) - original_missing) / len(self.df)) * 100
                treated_completeness[col] = ((len(treated_df) - treated_missing) / len(treated_df)) * 100
                
                improvement = treated_completeness[col] - original_completeness[col]
                validation['completeness_improvement'][col] = {
                    'original': original_completeness[col],
                    'treated': treated_completeness[col],
                    'improvement': improvement
                }
        
        logger.info(f"Validation completed: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
        
        return validation
    
    def save_treated_dataset(self, treated_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the treated dataset.
        
        Args:
            treated_df: Treated DataFrame
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"romance_novels_step1_missing_values_treated_{self.timestamp}.pkl"
        
        filepath = self.output_dir / filename
        
        # Save as pickle to preserve data types
        treated_df.to_pickle(filepath)
        
        logger.info(f"Treated dataset saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(filepath)
    
    def save_treatment_report(self, analysis: Dict[str, Any], treatment_results: Dict[str, Any], 
                            validation: Dict[str, Any], filename: str = None) -> str:
        """
        Save comprehensive treatment report.
        
        Args:
            analysis: Missing values analysis
            treatment_results: Treatment results
            validation: Validation results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"missing_values_treatment_report_step1_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            'treatment_timestamp': self.timestamp,
            'missing_values_analysis': analysis,
            'treatment_results': treatment_results,
            'validation_results': validation,
            'summary': {
                'original_records': self.df.shape[0],
                'treated_records': treatment_results['final_shape'][0],
                'records_excluded': treatment_results['records_excluded'],
                'treatments_applied': treatment_results['treatments_applied'],
                'flags_created': treatment_results['flags_created'],
                'validation_passed': validation['validation_passed']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Treatment report saved to: {filepath}")
        return str(filepath)
    
    def run_complete_treatment(self) -> Dict[str, Any]:
        """
        Run complete missing values treatment process.
        
        Returns:
            Dictionary with complete treatment results
        """
        logger.info("🚀 Starting complete missing values treatment process...")
        start_time = datetime.now()
        
        # Initialize results
        self.cleaning_results = {
            'treatment_timestamp': self.timestamp,
            'dataset_info': {},
            'missing_values_analysis': {},
            'treatment_results': {},
            'validation_results': {},
            'final_dataset_info': {},
            'treatment_summary': {}
        }
        
        # 1. Load dataset
        logger.info("📥 Loading dataset...")
        original_df = self.load_data()
        
        self.cleaning_results['dataset_info'] = {
            'original_shape': original_df.shape,
            'original_memory_mb': original_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'original_columns': list(original_df.columns)
        }
        
        # 2. Analyze missing values
        logger.info("🔍 Analyzing missing values...")
        analysis = self.analyze_missing_values()
        self.cleaning_results['missing_values_analysis'] = analysis
        
        # 3. Apply treatment
        logger.info("🔧 Applying missing value treatment...")
        treated_df, treatment_results = self.apply_missing_value_treatment()
        self.cleaning_results['treatment_results'] = treatment_results
        
        # 4. Validate results
        logger.info("🔍 Validating treatment results...")
        validation = self.validate_treatment_results(treated_df)
        self.cleaning_results['validation_results'] = validation
        
        # 5. Save treated dataset
        logger.info("💾 Saving treated dataset...")
        dataset_file = self.save_treated_dataset(treated_df)
        
        # 6. Final dataset information
        self.cleaning_results['final_dataset_info'] = {
            'final_shape': treated_df.shape,
            'final_memory_mb': treated_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'final_columns': list(treated_df.columns),
            'data_type_preservation': 'Maintained (pickle format)'
        }
        
        # 7. Generate treatment summary
        self.cleaning_results['treatment_summary'] = {
            'total_treatments_applied': treatment_results['treatments_applied'],
            'total_records_excluded': treatment_results['records_excluded'],
            'total_flags_created': treatment_results['flags_created'],
            'validation_passed': validation['validation_passed'],
            'data_integrity': 'Maintained - strategic treatment applied',
            'execution_time': str(datetime.now() - start_time)
        }
        
        # 8. Save treatment report
        logger.info("📊 Saving treatment report...")
        report_file = self.save_treatment_report(analysis, treatment_results, validation)
        
        logger.info("✅ Complete missing values treatment process finished!")
        
        return self.cleaning_results
    
    def print_treatment_summary(self):
        """Print a human-readable treatment summary."""
        if not self.cleaning_results:
            print("No treatment results available. Run run_complete_treatment() first.")
            return
        
        print("\n" + "="*80)
        print("MISSING VALUES TREATMENT SUMMARY - STEP 1")
        print("="*80)
        
        # Original dataset info
        original_info = self.cleaning_results['dataset_info']
        print(f"Original Dataset: {original_info['original_shape'][0]:,} records × {original_info['original_shape'][1]} columns")
        print(f"Original Memory: {original_info['original_memory_mb']:.2f} MB")
        
        # Missing values analysis
        analysis = self.cleaning_results['missing_values_analysis']
        print(f"\n🔍 MISSING VALUES ANALYSIS:")
        print(f"  • Variables with missing values: {analysis['variables_with_missing']}")
        print(f"  • Total variables analyzed: {analysis['total_variables']}")
        
        # Treatment results
        treatment = self.cleaning_results['treatment_results']
        print(f"\n🔧 TREATMENT RESULTS:")
        print(f"  • Treatments applied: {treatment['treatments_applied']}")
        print(f"  • Records excluded: {treatment['records_excluded']}")
        print(f"  • Flags created: {treatment['flags_created']}")
        
        # Final dataset info
        final_info = self.cleaning_results['final_dataset_info']
        print(f"\n📊 FINAL DATASET:")
        print(f"  • Final records: {final_info['final_shape'][0]:,} records × {final_info['final_shape'][1]} columns")
        print(f"  • Final memory: {final_info['final_memory_mb']:.2f} MB")
        print(f"  • Data types: {final_info['data_type_preservation']}")
        
        # Treatment summary
        summary = self.cleaning_results['treatment_summary']
        print(f"\n🎯 TREATMENT SUMMARY:")
        print(f"  • Validation passed: {'✅ YES' if summary['validation_passed'] else '❌ NO'}")
        print(f"  • Data integrity: {summary['data_integrity']}")
        print(f"  • Execution time: {summary['execution_time']}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    
    print("🔧 STEP 1: MISSING VALUES TREATMENT")
    print("=" * 60)
    
    try:
        # Initialize missing values cleaner
        cleaner = MissingValuesCleaner()
        
        # Run complete treatment process
        results = cleaner.run_complete_treatment()
        
        # Print summary
        cleaner.print_treatment_summary()
        
        print("\n🎉 Step 1: Missing Values Treatment completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Missing values treatment failed: {str(e)}", exc_info=True)
        print(f"\n💥 Missing values treatment failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
