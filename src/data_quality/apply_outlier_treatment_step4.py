#!/usr/bin/env python3
"""
Step 4: Outlier Treatment Application

This script applies the chosen outlier treatment strategy:
- Conservative approach: Document outliers, maintain data integrity for most fields
- Moderate approach: Remove publication year anomalies outside 2000-2017 range

Author: Research Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutlierTreatmentApplier:
    """
    Applies outlier treatment strategies while maintaining data integrity.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the treatment applier.
        
        Args:
            data_path: Path to the cleaned dataset file
        """
        self.data_path = data_path or "data/processed/cleaned_romance_novels_step1_3_20250902_223102.pkl"
        self.df = None
        self.treatment_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/outlier_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Treatment configuration
        self.publication_year_bounds = {
            'min_year': 2000,
            'max_year': 2017
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading data from: {self.data_path}")
            
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
    
    def analyze_publication_year_distribution(self) -> Dict[str, Any]:
        """
        Analyze publication year distribution before treatment.
        
        Returns:
            Dictionary with year distribution analysis
        """
        logger.info("🔍 Analyzing publication year distribution before treatment...")
        
        if 'publication_year' not in self.df.columns:
            return {'error': 'publication_year field not found'}
        
        # Get valid years
        years = pd.to_numeric(self.df['publication_year'], errors='coerce').dropna()
        
        # Analyze distribution
        year_counts = years.value_counts().sort_index()
        
        # Identify books outside bounds
        books_before_2000 = years[years < self.publication_year_bounds['min_year']]
        books_after_2017 = years[years > self.publication_year_bounds['max_year']]
        
        results = {
            'total_books': len(self.df),
            'valid_years': len(years),
            'missing_years': self.df['publication_year'].isnull().sum(),
            'year_range': {
                'min': int(years.min()),
                'max': int(years.max()),
                'target_min': self.publication_year_bounds['min_year'],
                'target_max': self.publication_year_bounds['max_year']
            },
            'books_to_remove': {
                'before_2000': len(books_before_2000),
                'after_2017': len(books_after_2017),
                'total': len(books_before_2000) + len(books_after_2017)
            },
            'year_distribution': year_counts.to_dict()
        }
        
        logger.info(f"Publication year analysis:")
        logger.info(f"  • Total books: {results['total_books']:,}")
        logger.info(f"  • Books before 2000: {results['books_to_remove']['before_2000']:,}")
        logger.info(f"  • Books after 2017: {results['books_to_remove']['after_2017']:,}")
        logger.info(f"  • Books to remove: {results['books_to_remove']['total']:,}")
        
        return results
    
    def apply_publication_year_treatment(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply publication year treatment by removing books outside 2000-2017 range.
        
        Returns:
            Tuple of (treated_dataframe, treatment_results)
        """
        logger.info("🔧 Applying publication year treatment...")
        
        # Analyze current distribution
        analysis = self.analyze_publication_year_distribution()
        
        if 'error' in analysis:
            logger.error(f"Publication year analysis failed: {analysis['error']}")
            return self.df, {'error': analysis['error']}
        
        # Create treatment mask
        valid_years_mask = (
            (self.df['publication_year'] >= self.publication_year_bounds['min_year']) &
            (self.df['publication_year'] <= self.publication_year_bounds['max_year'])
        )
        
        # Apply treatment
        treated_df = self.df[valid_years_mask].copy()
        
        # Record treatment results
        treatment_results = {
            'treatment_type': 'publication_year_filtering',
            'original_shape': self.df.shape,
            'treated_shape': treated_df.shape,
            'books_removed': len(self.df) - len(treated_df),
            'removal_percentage': ((len(self.df) - len(treated_df)) / len(self.df)) * 100,
            'year_bounds_applied': self.publication_year_bounds,
            'treatment_timestamp': self.timestamp
        }
        
        logger.info(f"Publication year treatment applied:")
        logger.info(f"  • Original books: {treatment_results['original_shape'][0]:,}")
        logger.info(f"  • Treated books: {treatment_results['treated_shape'][0]:,}")
        logger.info(f"  • Books removed: {treatment_results['books_removed']:,}")
        logger.info(f"  • Removal percentage: {treatment_results['removal_percentage']:.2f}%")
        
        return treated_df, treatment_results
    
    def apply_conservative_treatment_to_other_fields(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply conservative treatment to other outlier fields (documentation only).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with conservative treatment results
        """
        logger.info("📋 Applying conservative treatment to other outlier fields...")
        
        # Fields with high outlier rates from Step 4 analysis
        outlier_fields = [
            'average_rating_weighted_mean',
            'ratings_count_sum', 
            'text_reviews_count_sum',
            'author_ratings_count'
        ]
        
        conservative_results = {
            'treatment_type': 'conservative_documentation',
            'fields_analyzed': [],
            'outliers_documented': {},
            'no_action_taken': 'Outliers documented but no data modification performed'
        }
        
        for field in outlier_fields:
            if field in df.columns:
                # Get basic statistics
                data = pd.to_numeric(df[field], errors='coerce').dropna()
                
                if len(data) > 0:
                    field_stats = {
                        'field': field,
                        'total_values': len(data),
                        'missing_values': df[field].isnull().sum(),
                        'statistics': {
                            'mean': float(data.mean()),
                            'median': float(data.median()),
                            'std': float(data.std()),
                            'min': float(data.min()),
                            'max': float(data.max())
                        }
                    }
                    
                    conservative_results['fields_analyzed'].append(field)
                    conservative_results['outliers_documented'][field] = field_stats
                    
                    logger.info(f"  • {field}: {len(data):,} values analyzed (conservative approach)")
        
        logger.info(f"Conservative treatment applied to {len(conservative_results['fields_analyzed'])} fields")
        logger.info("  → No data modification performed - outliers documented for analysis")
        
        return conservative_results
    
    def save_treated_dataset(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the treated dataset.
        
        Args:
            df: Treated DataFrame
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"cleaned_romance_novels_step4_treated_{self.timestamp}.pkl"
        
        filepath = self.output_dir / filename
        
        # Save as pickle to preserve data types
        df.to_pickle(filepath)
        
        logger.info(f"Treated dataset saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(filepath)
    
    def save_treatment_report(self, treatment_results: Dict[str, Any], filename: str = None) -> str:
        """
        Save comprehensive treatment report.
        
        Args:
            treatment_results: Treatment results dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"outlier_treatment_report_step4_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(treatment_results, f, indent=2, default=str)
        
        logger.info(f"Treatment report saved to: {filepath}")
        return str(filepath)
    
    def run_complete_treatment(self) -> Dict[str, Any]:
        """
        Run complete outlier treatment process.
        
        Returns:
            Dictionary with complete treatment results
        """
        logger.info("🚀 Starting complete outlier treatment process...")
        start_time = datetime.now()
        
        # Initialize results
        self.treatment_results = {
            'treatment_timestamp': self.timestamp,
            'dataset_info': {},
            'publication_year_treatment': {},
            'conservative_treatment': {},
            'final_dataset_info': {},
            'treatment_summary': {}
        }
        
        # 1. Load and analyze original dataset
        logger.info("📥 Loading and analyzing original dataset...")
        original_df = self.load_data()
        
        self.treatment_results['dataset_info'] = {
            'original_shape': original_df.shape,
            'original_memory_mb': original_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'original_columns': list(original_df.columns)
        }
        
        # 2. Apply publication year treatment (moderate approach)
        logger.info("🔧 Applying publication year treatment (moderate approach)...")
        treated_df, year_treatment = self.apply_publication_year_treatment()
        
        if 'error' in year_treatment:
            logger.error(f"Publication year treatment failed: {year_treatment['error']}")
            return self.treatment_results
        
        self.treatment_results['publication_year_treatment'] = year_treatment
        
        # 3. Apply conservative treatment to other fields
        logger.info("📋 Applying conservative treatment to other fields...")
        conservative_treatment = self.apply_conservative_treatment_to_other_fields(treated_df)
        self.treatment_results['conservative_treatment'] = conservative_treatment
        
        # 4. Final dataset information
        self.treatment_results['final_dataset_info'] = {
            'final_shape': treated_df.shape,
            'final_memory_mb': treated_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'final_columns': list(treated_df.columns),
            'data_type_preservation': 'Maintained (pickle format)'
        }
        
        # 5. Save treated dataset
        logger.info("💾 Saving treated dataset...")
        dataset_file = self.save_treated_dataset(treated_df)
        
        # 6. Generate treatment summary
        self.treatment_results['treatment_summary'] = {
            'total_books_removed': year_treatment['books_removed'],
            'removal_percentage': year_treatment['removal_percentage'],
            'final_books_retained': len(treated_df),
            'treatment_strategy': {
                'publication_year': 'moderate (removal)',
                'other_fields': 'conservative (documentation only)'
            },
            'data_integrity': 'Maintained - only publication year anomalies removed',
            'execution_time': str(datetime.now() - start_time)
        }
        
        # 7. Save treatment report
        logger.info("📊 Saving treatment report...")
        report_file = self.save_treatment_report(self.treatment_results)
        
        logger.info("✅ Complete outlier treatment process finished!")
        
        return self.treatment_results
    
    def print_treatment_summary(self):
        """Print a human-readable treatment summary."""
        if not self.treatment_results:
            print("No treatment results available. Run run_complete_treatment() first.")
            return
        
        print("\n" + "="*80)
        print("OUTLIER TREATMENT SUMMARY - STEP 4")
        print("="*80)
        
        # Original dataset info
        original_info = self.treatment_results['dataset_info']
        print(f"Original Dataset: {original_info['original_shape'][0]:,} records × {original_info['original_shape'][1]} columns")
        print(f"Original Memory: {original_info['original_memory_mb']:.2f} MB")
        
        # Publication year treatment
        year_treatment = self.treatment_results['publication_year_treatment']
        print(f"\n🔧 PUBLICATION YEAR TREATMENT (Moderate Approach):")
        print(f"  • Books removed: {year_treatment['books_removed']:,}")
        print(f"  • Removal percentage: {year_treatment['removal_percentage']:.2f}%")
        print(f"  • Year bounds applied: {year_treatment['year_bounds_applied']['min_year']}-{year_treatment['year_bounds_applied']['max_year']}")
        
        # Conservative treatment
        conservative = self.treatment_results['conservative_treatment']
        print(f"\n📋 CONSERVATIVE TREATMENT (Documentation Only):")
        print(f"  • Fields analyzed: {len(conservative['fields_analyzed'])}")
        print(f"  • Action taken: {conservative['no_action_taken']}")
        
        # Final dataset info
        final_info = self.treatment_results['final_dataset_info']
        print(f"\n📊 FINAL DATASET:")
        print(f"  • Final records: {final_info['final_shape'][0]:,} records × {final_info['final_shape'][1]} columns")
        print(f"  • Final memory: {final_info['final_memory_mb']:.2f} MB")
        print(f"  • Data types: {final_info['data_type_preservation']}")
        
        # Treatment summary
        summary = self.treatment_results['treatment_summary']
        print(f"\n🎯 TREATMENT STRATEGY:")
        print(f"  • Publication year: {summary['treatment_strategy']['publication_year']}")
        print(f"  • Other fields: {summary['treatment_strategy']['other_fields']}")
        print(f"  • Data integrity: {summary['data_integrity']}")
        print(f"  • Execution time: {summary['execution_time']}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    
    print("🔧 STEP 4: OUTLIER TREATMENT APPLICATION")
    print("=" * 60)
    
    try:
        # Initialize treatment applier
        applier = OutlierTreatmentApplier()
        
        # Run complete treatment process
        results = applier.run_complete_treatment()
        
        # Print summary
        applier.print_treatment_summary()
        
        print("\n🎉 Step 4: Outlier Treatment Application completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Treatment application failed: {str(e)}", exc_info=True)
        print(f"\n💥 Treatment application failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
