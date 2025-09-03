#!/usr/bin/env python3
"""
Dataset Cleaning Analysis for Integrated Romance Novels Dataset

This script analyzes the integrated_romance_novels_nlp_ready dataset and provides
comprehensive cleaning recommendations for each variable including:
- Missing values handling
- Duplicates & inconsistencies  
- Outliers detection & treatment
- Variable transformations
- Derived features
- Prioritized cleaning order

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

class DatasetCleaningAnalyzer:
    """Analyzes dataset and provides comprehensive cleaning recommendations."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the analyzer.
        
        Args:
            dataset_path: Path to the CSV dataset
        """
        self.dataset_path = Path(dataset_path)
        self.df = None
        self.analysis_results = {}
        
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
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values for each variable."""
        logger.info("Analyzing missing values...")
        
        missing_analysis = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'data_type': str(self.df[col].dtype),
                'unique_values': self.df[col].nunique() if self.df[col].dtype == 'object' else None
            }
        
        self.analysis_results['missing_values'] = missing_analysis
        return missing_analysis
    
    def analyze_duplicates_and_inconsistencies(self) -> Dict[str, Any]:
        """Analyze duplicates and inconsistencies."""
        logger.info("Analyzing duplicates and inconsistencies...")
        
        duplicate_analysis = {}
        
        # Check for exact duplicates
        exact_duplicates = self.df.duplicated().sum()
        
        # Check for key field duplicates
        key_duplicates = {}
        key_fields = ['work_id', 'title', 'author_id']
        
        for field in key_fields:
            if field in self.df.columns:
                field_duplicates = self.df[field].duplicated().sum()
                key_duplicates[field] = field_duplicates
        
        # Check for potential inconsistencies in categorical fields
        categorical_inconsistencies = {}
        categorical_cols = ['genres', 'language_codes_en', 'publication_year']
        
        for col in categorical_cols:
            if col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Check for mixed formats, extra spaces, etc.
                    sample_values = self.df[col].dropna().head(10).tolist()
                    categorical_inconsistencies[col] = {
                        'sample_values': sample_values,
                        'unique_count': self.df[col].nunique()
                    }
        
        duplicate_analysis = {
            'exact_duplicates': exact_duplicates,
            'key_field_duplicates': key_duplicates,
            'categorical_inconsistencies': categorical_inconsistencies
        }
        
        self.analysis_results['duplicates_and_inconsistencies'] = duplicate_analysis
        return duplicate_analysis
    
    def analyze_outliers(self) -> Dict[str, Any]:
        """Analyze outliers in numerical variables."""
        logger.info("Analyzing outliers...")
        
        outlier_analysis = {}
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
                outlier_analysis[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(self.df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min_value': self.df[col].min(),
                    'max_value': self.df[col].max(),
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std()
                }
        
        self.analysis_results['outliers'] = outlier_analysis
        return outlier_analysis
    
    def analyze_variable_characteristics(self) -> Dict[str, Any]:
        """Analyze characteristics of each variable for transformation recommendations."""
        logger.info("Analyzing variable characteristics...")
        
        variable_analysis = {}
        
        for col in self.df.columns:
            col_analysis = {
                'data_type': str(self.df[col].dtype),
                'unique_count': self.df[col].nunique(),
                'sample_values': None,
                'transformation_recommendations': []
            }
            
            if self.df[col].dtype == 'object':
                # Text/categorical analysis
                col_analysis['sample_values'] = self.df[col].dropna().head(5).tolist()
                
                # Check if it's text that needs preprocessing
                if col in ['title', 'description']:
                    col_analysis['transformation_recommendations'].extend([
                        'text_preprocessing',
                        'lowercase_conversion',
                        'special_character_removal',
                        'tokenization'
                    ])
                elif col in ['genres', 'popular_shelves']:
                    col_analysis['transformation_recommendations'].extend([
                        'categorical_encoding',
                        'multi_label_encoding',
                        'genre_normalization'
                    ])
                elif col in ['language_codes_en']:
                    col_analysis['transformation_recommendations'].extend([
                        'language_code_standardization',
                        'categorical_encoding'
                    ])
                    
            elif self.df[col].dtype in ['int64', 'float64']:
                # Numerical analysis
                if col in ['publication_year']:
                    col_analysis['transformation_recommendations'].extend([
                        'year_validation',
                        'decade_categorization',
                        'period_binning'
                    ])
                elif col in ['num_pages_median', 'ratings_count_sum', 'text_reviews_count_sum']:
                    col_analysis['transformation_recommendations'].extend([
                        'log_transformation',
                        'scaling',
                        'outlier_capping'
                    ])
                elif col in ['average_rating_weighted_mean', 'author_average_rating']:
                    col_analysis['transformation_recommendations'].extend([
                        'rating_validation',
                        'binning',
                        'scaling'
                    ])
            
            variable_analysis[col] = col_analysis
        
        self.analysis_results['variable_characteristics'] = variable_analysis
        return variable_analysis
    
    def suggest_derived_features(self) -> Dict[str, List[str]]:
        """Suggest derived features based on research questions."""
        logger.info("Suggesting derived features...")
        
        derived_features = {
            'book_metrics': [
                'rating_popularity_score',  # Combine rating and popularity
                'engagement_ratio',  # text_reviews / ratings_count
                'page_density_score',  # rating / pages
                'publication_recency',  # 2025 - publication_year
                'decade_category'  # 2000s, 2010s, 2020s
            ],
            'author_metrics': [
                'author_productivity',  # books per author
                'author_popularity_score',  # Combine author rating and book count
                'author_genre_diversity'  # Number of unique genres per author
            ],
            'series_metrics': [
                'series_length_category',  # Short (1-3), Medium (4-7), Long (8+)
                'series_popularity_score',  # Average rating across series
                'series_completion_rate'  # Completed vs ongoing
            ],
            'content_metrics': [
                'description_length',  # Character count
                'description_complexity',  # Word count, sentence count
                'genre_count',  # Number of genres per book
                'shelf_diversity'  # Number of unique shelves
            ]
        }
        
        self.analysis_results['derived_features'] = derived_features
        return derived_features
    
    def create_prioritized_cleaning_order(self) -> List[Dict[str, Any]]:
        """Create a prioritized cleaning order based on analysis."""
        logger.info("Creating prioritized cleaning order...")
        
        cleaning_steps = [
            {
                'step': 1,
                'priority': 'Critical',
                'task': 'Missing Values Assessment',
                'description': 'Evaluate and handle missing values based on criticality',
                'variables': ['work_id', 'title', 'author_id', 'publication_year', 'description'],
                'estimated_time': '30 minutes',
                'dependencies': []
            },
            {
                'step': 2,
                'priority': 'High',
                'task': 'Duplicate Detection & Resolution',
                'description': 'Identify and resolve exact and key field duplicates',
                'variables': ['work_id', 'title', 'author_id'],
                'estimated_time': '45 minutes',
                'dependencies': ['Missing Values Assessment']
            },
            {
                'step': 3,
                'priority': 'High',
                'task': 'Data Type Validation & Conversion',
                'description': 'Ensure correct data types and convert as needed',
                'variables': ['publication_year', 'num_pages_median', 'ratings_count_sum'],
                'estimated_time': '30 minutes',
                'dependencies': ['Duplicate Detection & Resolution']
            },
            {
                'step': 4,
                'priority': 'Medium',
                'task': 'Outlier Detection & Treatment',
                'description': 'Identify and handle outliers in numerical variables',
                'variables': ['num_pages_median', 'ratings_count_sum', 'text_reviews_count_sum'],
                'estimated_time': '60 minutes',
                'dependencies': ['Data Type Validation & Conversion']
            },
            {
                'step': 5,
                'priority': 'Medium',
                'task': 'Categorical Variable Standardization',
                'description': 'Clean and standardize categorical variables',
                'variables': ['genres', 'language_codes_en', 'popular_shelves'],
                'estimated_time': '90 minutes',
                'dependencies': ['Outlier Detection & Treatment']
            },
            {
                'step': 6,
                'priority': 'Medium',
                'task': 'Text Preprocessing',
                'description': 'Clean and preprocess text fields for NLP',
                'variables': ['title', 'description'],
                'estimated_time': '120 minutes',
                'dependencies': ['Categorical Variable Standardization']
            },
            {
                'step': 7,
                'priority': 'Low',
                'task': 'Derived Feature Creation',
                'description': 'Create new features for analysis',
                'variables': ['rating_popularity_score', 'engagement_ratio', 'decade_category'],
                'estimated_time': '60 minutes',
                'dependencies': ['Text Preprocessing']
            },
            {
                'step': 8,
                'priority': 'Low',
                'task': 'Final Validation & Quality Check',
                'description': 'Comprehensive data quality validation',
                'variables': 'All variables',
                'estimated_time': '45 minutes',
                'dependencies': ['Derived Feature Creation']
            }
        ]
        
        self.analysis_results['cleaning_order'] = cleaning_steps
        return cleaning_steps
    
    def generate_cleaning_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning recommendations for each variable."""
        logger.info("Generating comprehensive cleaning recommendations...")
        
        recommendations = {}
        
        for col in self.df.columns:
            col_recs = {
                'missing_values': {},
                'duplicates_inconsistencies': {},
                'outliers': {},
                'transformations': {},
                'notes': []
            }
            
            # Missing values recommendations
            if col in self.analysis_results.get('missing_values', {}):
                missing_info = self.analysis_results['missing_values'][col]
                if missing_info['missing_percentage'] > 0:
                    if col in ['work_id', 'title', 'author_id']:
                        col_recs['missing_values']['strategy'] = 'drop'
                        col_recs['missing_values']['reason'] = 'Critical identifier - cannot impute'
                    elif col in ['publication_year', 'num_pages_median']:
                        col_recs['missing_values']['strategy'] = 'impute_median'
                        col_recs['missing_values']['reason'] = 'Numerical variable - median imputation appropriate'
                    elif col in ['description']:
                        col_recs['missing_values']['strategy'] = 'drop'
                        col_recs['missing_values']['reason'] = 'Essential for NLP analysis'
                    else:
                        col_recs['missing_values']['strategy'] = 'flag'
                        col_recs['missing_values']['reason'] = 'Non-critical variable - flag for investigation'
                else:
                    col_recs['missing_values']['strategy'] = 'none'
                    col_recs['missing_values']['reason'] = 'No missing values'
            
            # Duplicates and inconsistencies recommendations
            if col in ['work_id', 'title', 'author_id']:
                col_recs['duplicates_inconsistencies']['strategy'] = 'investigate_resolve'
                col_recs['duplicates_inconsistencies']['reason'] = 'Critical identifier - must be unique'
            elif col in ['genres', 'popular_shelves']:
                col_recs['duplicates_inconsistencies']['strategy'] = 'standardize_format'
                col_recs['duplicates_inconsistencies']['reason'] = 'Categorical variable - needs consistent format'
            
            # Outlier recommendations
            if col in self.analysis_results.get('outliers', {}):
                outlier_info = self.analysis_results['outliers'][col]
                if outlier_info['outlier_percentage'] > 5:
                    if col in ['num_pages_median']:
                        col_recs['outliers']['strategy'] = 'cap_at_bounds'
                        col_recs['outliers']['reason'] = 'Realistic page limits - cap extreme values'
                    elif col in ['ratings_count_sum', 'text_reviews_count_sum']:
                        col_recs['outliers']['strategy'] = 'log_transformation'
                        col_recs['outliers']['reason'] = 'Highly skewed - log transformation recommended'
                    else:
                        col_recs['outliers']['strategy'] = 'investigate'
                        col_recs['outliers']['reason'] = 'Moderate outliers - investigate before treatment'
                else:
                    col_recs['outliers']['strategy'] = 'none'
                    col_recs['outliers']['reason'] = 'Low outlier percentage - no treatment needed'
            
            # Transformation recommendations
            if col in self.analysis_results.get('variable_characteristics', {}):
                var_info = self.analysis_results['variable_characteristics'][col]
                col_recs['transformations'] = var_info.get('transformation_recommendations', [])
            
            # Special notes
            if col == 'publication_year':
                col_recs['notes'].append('Validate year range (2000-2020)')
                col_recs['notes'].append('Consider decade categorization for analysis')
            elif col == 'description':
                col_recs['notes'].append('Remove HTML tags if present')
                col_recs['notes'].append('Standardize text length for NLP')
            elif col == 'genres':
                col_recs['notes'].append('Split multi-genre entries')
                col_recs['notes'].append('Create genre hierarchy')
            
            recommendations[col] = col_recs
        
        self.analysis_results['cleaning_recommendations'] = recommendations
        return recommendations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        logger.info("Starting complete dataset analysis...")
        
        # Load dataset
        self.load_dataset()
        
        # Run all analyses
        self.analyze_missing_values()
        self.analyze_duplicates_and_inconsistencies()
        self.analyze_outliers()
        self.analyze_variable_characteristics()
        self.suggest_derived_features()
        self.create_prioritized_cleaning_order()
        self.generate_cleaning_recommendations()
        
        logger.info("Complete analysis finished")
        return self.analysis_results
    
    def save_analysis_report(self, output_path: str):
        """Save the analysis report to a JSON file."""
        logger.info(f"Saving analysis report to {output_path}")
        
        # Convert numpy types to Python types for JSON serialization
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
        
        # Convert the analysis results
        serializable_results = json.loads(
            json.dumps(self.analysis_results, default=convert_numpy_types)
        )
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        print("\n" + "="*80)
        print("DATASET CLEANING ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.analysis_results:
            print("No analysis results available. Run run_complete_analysis() first.")
            return
        
        # Dataset overview
        print(f"\nDataset Overview:")
        print(f"  Total records: {len(self.df):,}")
        print(f"  Total variables: {len(self.df.columns)}")
        print(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values summary
        if 'missing_values' in self.analysis_results:
            missing_vars = [col for col, info in self.analysis_results['missing_values'].items() 
                          if info['missing_percentage'] > 0]
            print(f"\nMissing Values:")
            print(f"  Variables with missing data: {len(missing_vars)}")
            if missing_vars:
                print(f"  Affected variables: {', '.join(missing_vars[:5])}{'...' if len(missing_vars) > 5 else ''}")
        
        # Duplicates summary
        if 'duplicates_and_inconsistencies' in self.analysis_results:
            dup_info = self.analysis_results['duplicates_and_inconsistencies']
            print(f"\nDuplicates & Inconsistencies:")
            print(f"  Exact duplicates: {dup_info.get('exact_duplicates', 0):,}")
            if 'key_field_duplicates' in dup_info:
                for field, count in dup_info['key_field_duplicates'].items():
                    print(f"  {field} duplicates: {count:,}")
        
        # Outliers summary
        if 'outliers' in self.analysis_results:
            outlier_vars = [col for col, info in self.analysis_results['outliers'].items() 
                          if info['outlier_percentage'] > 5]
            print(f"\nOutliers:")
            print(f"  Variables with >5% outliers: {len(outlier_vars)}")
            if outlier_vars:
                print(f"  Affected variables: {', '.join(outlier_vars[:5])}{'...' if len(outlier_vars) > 5 else ''}")
        
        # Cleaning order
        if 'cleaning_order' in self.analysis_results:
            print(f"\nPrioritized Cleaning Order:")
            for step in self.analysis_results['cleaning_order'][:3]:  # Show first 3 steps
                print(f"  Step {step['step']}: {step['task']} ({step['priority']} priority)")
            if len(self.analysis_results['cleaning_order']) > 3:
                print(f"  ... and {len(self.analysis_results['cleaning_order']) - 3} more steps")
        
        print("\n" + "="*80)


def main():
    """Main function to run the analysis."""
    # Dataset path
    dataset_path = "data/processed/integrated_romance_novels_nlp_ready_20250902_193603.csv"
    
    # Initialize analyzer
    analyzer = DatasetCleaningAnalyzer(dataset_path)
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Print summary
        analyzer.print_summary()
        
        # Save detailed report
        output_path = "outputs/dataset_cleaning_analysis_report.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        analyzer.save_analysis_report(output_path)
        
        print(f"\nDetailed analysis report saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
