#!/usr/bin/env python3
"""
Step 4: Outlier Detection & Reporting (Medium Priority)

This script performs comprehensive outlier detection and reporting for the
romance novel dataset, focusing on:

1. Statistical Outlier Detection
   - Z-score method (3σ rule)
   - IQR method (1.5 * IQR rule)
   - Percentile-based detection
   - Multi-method consensus

2. Data Quality Anomaly Detection
   - Publication year anomalies
   - Rating and review count anomalies
   - Page count distribution issues
   - Author productivity outliers

3. Comprehensive Reporting
   - Outlier statistics and summaries
   - Field-by-field analysis
   - Treatment recommendations
   - Data quality impact assessment

Author: Research Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutlierDetectionReporter:
    """
    Comprehensive outlier detection and reporting system for romance novel dataset.
    Focuses on detection and reporting only - no data modification.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the outlier detection reporter.
        
        Args:
            data_path: Path to the cleaned dataset file (prefer pickle for data type preservation)
        """
        self.data_path = data_path or "data/processed/cleaned_romance_novels_step1_3_20250902_223102.pkl"
        self.df = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/outlier_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Define fields to analyze for outliers
        self.numerical_fields = [
            'publication_year',
            'average_rating_weighted_mean',
            'ratings_count_sum',
            'text_reviews_count_sum',
            'author_ratings_count',
            'num_pages'
        ]
        
        # Define categorical fields for distribution analysis
        self.categorical_fields = [
            'genres',
            'author_name',
            'series_title',
            'decade',
            'book_length_category',
            'rating_category',
            'popularity_category'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned dataset, preferring pickle format for data type preservation.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading data from: {self.data_path}")
            
            # Try pickle first, then CSV as fallback
            if self.data_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.data_path)
                logger.info("Data loaded from pickle file - data types preserved")
            else:
                self.df = pd.read_csv(self.data_path)
                logger.info("Data loaded from CSV file")
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            # Display data types
            logger.info("Data types:")
            for col, dtype in self.df.dtypes.items():
                logger.info(f"  {col}: {dtype}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def detect_statistical_outliers(self, field: str, methods: List[str] = None) -> Dict[str, Any]:
        """
        Detect outliers using multiple statistical methods.
        
        Args:
            field: Field name to analyze
            methods: List of methods to use ['zscore', 'iqr', 'percentile']
            
        Returns:
            Dictionary with outlier detection results
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'percentile']
        
        if field not in self.df.columns:
            return {'error': f'Field {field} not found in dataset'}
        
        # Get numeric data, handling missing values
        data = pd.to_numeric(self.df[field], errors='coerce').dropna()
        
        if len(data) == 0:
            return {'error': f'No valid numeric data in field {field}'}
        
        results = {
            'field': field,
            'total_values': len(data),
            'missing_values': self.df[field].isnull().sum(),
            'outliers': {},
            'statistics': {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'q1': float(data.quantile(0.25)),
                'q3': float(data.quantile(0.75))
            }
        }
        
        # Z-score method (3σ rule)
        if 'zscore' in methods:
            z_scores = np.abs((data - data.mean()) / data.std())
            zscore_outliers = data[z_scores > 3]
            results['outliers']['zscore'] = {
                'count': len(zscore_outliers),
                'percentage': (len(zscore_outliers) / len(data)) * 100,
                'indices': zscore_outliers.index.tolist(),
                'values': zscore_outliers.tolist()
            }
        
        # IQR method (1.5 * IQR rule)
        if 'iqr' in methods:
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            results['outliers']['iqr'] = {
                'count': len(iqr_outliers),
                'percentage': (len(iqr_outliers) / len(data)) * 100,
                'indices': iqr_outliers.index.tolist(),
                'values': iqr_outliers.tolist(),
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            }
        
        # Percentile method (1st and 99th percentiles)
        if 'percentile' in methods:
            p1 = data.quantile(0.01)
            p99 = data.quantile(0.99)
            
            percentile_outliers = data[(data < p1) | (data > p99)]
            results['outliers']['percentile'] = {
                'count': len(percentile_outliers),
                'percentage': (len(percentile_outliers) / len(data)) * 100,
                'indices': percentile_outliers.index.tolist(),
                'values': percentile_outliers.tolist(),
                'bounds': {'p1': float(p1), 'p99': float(p99)}
            }
        
        # Multi-method consensus
        if len(methods) > 1:
            all_outlier_indices = set()
            for method in methods:
                if method in results['outliers']:
                    all_outlier_indices.update(results['outliers'][method]['indices'])
            
            results['consensus'] = {
                'total_unique_outliers': len(all_outlier_indices),
                'consensus_percentage': (len(all_outlier_indices) / len(data)) * 100
            }
        
        return results
    
    def analyze_publication_year_outliers(self) -> Dict[str, Any]:
        """
        Specialized analysis for publication year outliers.
        
        Returns:
            Dictionary with publication year analysis results
        """
        logger.info("🔍 Analyzing publication year outliers...")
        
        if 'publication_year' not in self.df.columns:
            return {'error': 'publication_year field not found'}
        
        # Get valid years
        years = pd.to_numeric(self.df['publication_year'], errors='coerce').dropna()
        
        results = {
            'field': 'publication_year',
            'total_valid_years': len(years),
            'missing_years': self.df['publication_year'].isnull().sum(),
            'year_range': {
                'min': int(years.min()),
                'max': int(years.max()),
                'span': int(years.max() - years.min())
            },
            'decade_distribution': {},
            'anomalies': {}
        }
        
        # Analyze by decades
        decades = {
            '1800s': len(years[years < 1900]),
            '1900s': len(years[(years >= 1900) & (years < 2000)]),
            '2000s': len(years[(years >= 2000) & (years < 2010)]),
            '2010s': len(years[(years >= 2010) & (years < 2020)]),
            '2020s': len(years[years >= 2020])
        }
        
        results['decade_distribution'] = decades
        
        # Identify anomalies
        current_year = datetime.now().year
        
        # Future years (impossible)
        future_years = years[years > current_year]
        if len(future_years) > 0:
            results['anomalies']['future_years'] = {
                'count': len(future_years),
                'years': future_years.unique().tolist()
            }
        
        # Very old years (suspicious)
        very_old_years = years[years < 1800]
        if len(very_old_years) > 0:
            results['anomalies']['very_old_years'] = {
                'count': len(very_old_years),
                'years': very_old_years.unique().tolist()
            }
        
        # Zero or negative years
        invalid_years = years[years <= 0]
        if len(invalid_years) > 0:
            results['anomalies']['invalid_years'] = {
                'count': len(invalid_years),
                'years': invalid_years.unique().tolist()
            }
        
        return results
    
    def analyze_rating_outliers(self) -> Dict[str, Any]:
        """
        Specialized analysis for rating-related outliers.
        
        Returns:
            Dictionary with rating analysis results
        """
        logger.info("🔍 Analyzing rating outliers...")
        
        results = {
            'average_rating': {},
            'ratings_count': {},
            'text_reviews_count': {}
        }
        
        # Analyze average rating (should be 0-5)
        if 'average_rating_weighted_mean' in self.df.columns:
            ratings = pd.to_numeric(self.df['average_rating_weighted_mean'], errors='coerce').dropna()
            
            results['average_rating'] = {
                'total_valid': len(ratings),
                'missing': self.df['average_rating_weighted_mean'].isnull().sum(),
                'range': {'min': float(ratings.min()), 'max': float(ratings.max())},
                'anomalies': {}
            }
            
            # Check for impossible ratings
            impossible_ratings = ratings[(ratings < 0) | (ratings > 5)]
            if len(impossible_ratings) > 0:
                results['average_rating']['anomalies']['impossible_values'] = {
                    'count': len(impossible_ratings),
                    'values': impossible_ratings.unique().tolist()
                }
        
        # Analyze ratings count
        if 'ratings_count_sum' in self.df.columns:
            counts = pd.to_numeric(self.df['ratings_count_sum'], errors='coerce').dropna()
            
            results['ratings_count'] = {
                'total_valid': len(counts),
                'missing': self.df['ratings_count_sum'].isnull().sum(),
                'statistics': {
                    'mean': float(counts.mean()),
                    'median': float(counts.median()),
                    'std': float(counts.std())
                },
                'anomalies': {}
            }
            
            # Check for negative counts
            negative_counts = counts[counts < 0]
            if len(negative_counts) > 0:
                results['ratings_count']['anomalies']['negative_values'] = {
                    'count': len(negative_counts),
                    'values': negative_counts.unique().tolist()
                }
        
        # Analyze text reviews count
        if 'text_reviews_count_sum' in self.df.columns:
            text_counts = pd.to_numeric(self.df['text_reviews_count_sum'], errors='coerce').dropna()
            
            results['text_reviews_count'] = {
                'total_valid': len(text_counts),
                'missing': self.df['text_reviews_count_sum'].isnull().sum(),
                'statistics': {
                    'mean': float(text_counts.mean()),
                    'median': float(text_counts.median()),
                    'std': float(text_counts.std())
                },
                'anomalies': {}
            }
            
            # Check for negative counts
            negative_text_counts = text_counts[text_counts < 0]
            if len(negative_text_counts) > 0:
                results['text_reviews_count']['anomalies']['negative_values'] = {
                    'count': len(negative_text_counts),
                    'values': negative_text_counts.unique().tolist()
                }
        
        return results
    
    def analyze_page_count_outliers(self) -> Dict[str, Any]:
        """
        Specialized analysis for page count outliers.
        
        Returns:
            Dictionary with page count analysis results
        """
        logger.info("🔍 Analyzing page count outliers...")
        
        if 'num_pages' not in self.df.columns:
            return {'error': 'num_pages field not found'}
        
        pages = pd.to_numeric(self.df['num_pages'], errors='coerce').dropna()
        
        results = {
            'field': 'num_pages',
            'total_valid': len(pages),
            'missing': self.df['num_pages'].isnull().sum(),
            'statistics': {
                'mean': float(pages.mean()),
                'median': float(pages.median()),
                'std': float(pages.std()),
                'min': float(pages.min()),
                'max': float(pages.max())
            },
            'anomalies': {}
        }
        
        # Check for impossible page counts
        impossible_pages = pages[pages <= 0]
        if len(impossible_pages) > 0:
            results['anomalies']['impossible_values'] = {
                'count': len(impossible_pages),
                'values': impossible_pages.unique().tolist()
            }
        
        # Check for extremely long books (>2000 pages)
        extremely_long = pages[pages > 2000]
        if len(extremely_long) > 0:
            results['anomalies']['extremely_long'] = {
                'count': len(extremely_long),
                'values': extremely_long.unique().tolist()
            }
        
        # Check for extremely short books (<10 pages)
        extremely_short = pages[pages < 10]
        if len(extremely_short) > 0:
            results['anomalies']['extremely_short'] = {
                'count': len(extremely_short),
                'values': extremely_short.unique().tolist()
            }
        
        return results
    
    def analyze_categorical_distributions(self) -> Dict[str, Any]:
        """
        Analyze categorical field distributions for anomalies.
        
        Returns:
            Dictionary with categorical analysis results
        """
        logger.info("🔍 Analyzing categorical field distributions...")
        
        results = {}
        
        for field in self.categorical_fields:
            if field not in self.df.columns:
                continue
            
            # Get value counts
            value_counts = self.df[field].value_counts()
            
            results[field] = {
                'total_values': len(self.df[field]),
                'missing_values': self.df[field].isnull().sum(),
                'unique_values': len(value_counts),
                'top_values': value_counts.head(10).to_dict(),
                'anomalies': {}
            }
            
            # Check for fields with too many unique values (potential data quality issue)
            if len(value_counts) > 1000:
                results[field]['anomalies']['too_many_unique'] = {
                    'count': len(value_counts),
                    'threshold': 1000
                }
            
            # Check for fields with mostly missing values
            missing_pct = (self.df[field].isnull().sum() / len(self.df[field])) * 100
            if missing_pct > 80:
                results[field]['anomalies']['high_missing_rate'] = {
                    'missing_percentage': missing_pct,
                    'threshold': 80
                }
            
            # Check for fields with single dominant value (>90%)
            if len(value_counts) > 0:
                dominant_value_pct = (value_counts.iloc[0] / len(self.df[field])) * 100
                if dominant_value_pct > 90:
                    results[field]['anomalies']['single_dominant_value'] = {
                        'dominant_value': value_counts.index[0],
                        'percentage': dominant_value_pct,
                        'threshold': 90
                    }
        
        return results
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive outlier detection analysis.
        
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("🚀 Starting comprehensive outlier detection analysis...")
        start_time = time.time()
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Initialize results structure
        self.results = {
            'analysis_timestamp': self.timestamp,
            'dataset_info': {
                'shape': self.df.shape,
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
                'total_records': len(self.df)
            },
            'statistical_outliers': {},
            'specialized_analyses': {},
            'categorical_analysis': {},
            'summary': {}
        }
        
        # 1. Statistical outlier detection for numerical fields
        logger.info("📊 Detecting statistical outliers...")
        for field in self.numerical_fields:
            if field in self.df.columns:
                logger.info(f"  Analyzing field: {field}")
                self.results['statistical_outliers'][field] = self.detect_statistical_outliers(field)
        
        # 2. Specialized analyses
        logger.info("🔍 Running specialized analyses...")
        self.results['specialized_analyses']['publication_year'] = self.analyze_publication_year_outliers()
        self.results['specialized_analyses']['ratings'] = self.analyze_rating_outliers()
        self.results['specialized_analyses']['page_count'] = self.analyze_page_count_outliers()
        
        # 3. Categorical analysis
        logger.info("📈 Analyzing categorical distributions...")
        self.results['categorical_analysis'] = self.analyze_categorical_distributions()
        
        # 4. Generate summary
        self.results['summary'] = self._generate_summary()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self.results['execution_time_seconds'] = execution_time
        
        logger.info(f"✅ Analysis completed in {execution_time:.2f} seconds")
        
        return self.results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'total_outliers_detected': 0,
            'fields_with_outliers': 0,
            'critical_anomalies': 0,
            'data_quality_score': 0.0,
            'recommendations': []
        }
        
        # Count outliers across all methods
        outlier_counts = {}
        for field, results in self.results['statistical_outliers'].items():
            if 'error' not in results:
                total_field_outliers = 0
                for method, method_results in results.get('outliers', {}).items():
                    if 'count' in method_results:
                        total_field_outliers += method_results['count']
                
                if total_field_outliers > 0:
                    outlier_counts[field] = total_field_outliers
                    summary['total_outliers_detected'] += total_field_outliers
                    summary['fields_with_outliers'] += 1
        
        # Count critical anomalies
        critical_count = 0
        
        # Publication year anomalies
        pub_year = self.results['specialized_analyses'].get('publication_year', {})
        if 'anomalies' in pub_year:
            for anomaly_type, anomaly_data in pub_year['anomalies'].items():
                if anomaly_type in ['future_years', 'invalid_years']:
                    critical_count += anomaly_data.get('count', 0)
        
        # Rating anomalies
        ratings = self.results['specialized_analyses'].get('ratings', {})
        for rating_type, rating_data in ratings.items():
            if 'anomalies' in rating_data:
                for anomaly_type, anomaly_data in rating_data['anomalies'].items():
                    if anomaly_type in ['impossible_values', 'negative_values']:
                        critical_count += anomaly_data.get('count', 0)
        
        summary['critical_anomalies'] = critical_count
        
        # Calculate data quality score (0-100)
        total_records = self.results['dataset_info']['total_records']
        if total_records > 0:
            outlier_percentage = (summary['total_outliers_detected'] / total_records) * 100
            critical_percentage = (summary['critical_anomalies'] / total_records) * 100
            
            # Base score starts at 100, deduct points for issues
            base_score = 100
            outlier_penalty = min(outlier_percentage * 0.5, 30)  # Max 30 point penalty
            critical_penalty = min(critical_percentage * 2, 50)   # Max 50 point penalty
            
            summary['data_quality_score'] = max(0, base_score - outlier_penalty - critical_penalty)
        
        # Generate recommendations
        if summary['critical_anomalies'] > 0:
            summary['recommendations'].append("Critical anomalies detected - immediate review required")
        
        if summary['total_outliers_detected'] > total_records * 0.1:  # >10% outliers
            summary['recommendations'].append("High outlier rate - consider data quality investigation")
        
        if summary['data_quality_score'] < 70:
            summary['recommendations'].append("Data quality score below 70 - implement quality improvements")
        
        if not summary['recommendations']:
            summary['recommendations'].append("Data quality appears acceptable - continue with analysis")
        
        return summary
    
    def save_results(self, filename: str = None) -> str:
        """
        Save analysis results to JSON file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"outlier_detection_report_step4_{self.timestamp}.json"
        
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
        results_copy = copy.deepcopy(self.results)
        
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
        
        logger.info(f"Results saved to: {filepath}")
        return str(filepath)
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        if not self.results:
            print("No analysis results available. Run run_comprehensive_analysis() first.")
            return
        
        print("\n" + "="*80)
        print("OUTLIER DETECTION REPORT - STEP 4")
        print("="*80)
        
        # Dataset info
        info = self.results['dataset_info']
        print(f"Dataset: {info['shape'][0]:,} records × {info['shape'][1]} columns")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"Analysis Timestamp: {self.results['analysis_timestamp']}")
        
        # Summary
        summary = self.results['summary']
        print(f"\n📊 SUMMARY:")
        print(f"  Data Quality Score: {summary['data_quality_score']:.1f}/100")
        print(f"  Total Outliers Detected: {summary['total_outliers_detected']:,}")
        print(f"  Fields with Outliers: {summary['fields_with_outliers']}")
        print(f"  Critical Anomalies: {summary['critical_anomalies']:,}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        
        # Field-level summary
        print(f"\n🔍 FIELD-LEVEL OUTLIERS:")
        for field, results in self.results['statistical_outliers'].items():
            if 'error' not in results:
                total_outliers = 0
                for method, method_results in results.get('outliers', {}).items():
                    if 'count' in method_results:
                        total_outliers += method_results['count']
                
                if total_outliers > 0:
                    percentage = (total_outliers / info['total_records']) * 100
                    print(f"  • {field}: {total_outliers:,} outliers ({percentage:.1f}%)")
        
        print("\n" + "="*80)

# Import time module for performance monitoring
import time
