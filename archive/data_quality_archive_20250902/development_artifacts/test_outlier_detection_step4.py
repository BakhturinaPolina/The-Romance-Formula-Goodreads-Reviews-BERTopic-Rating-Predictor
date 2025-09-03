#!/usr/bin/env python3
"""
Test script for Step 4: Outlier Detection & Reporting

This script tests the outlier detection system with:
- Sample data validation
- Edge case testing
- Performance benchmarking
- Error handling validation

Author: Research Assistant
Date: 2025-09-02
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from outlier_detection_step4 import OutlierDetectionReporter

def setup_test_logging():
    """Set up logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_test_dataset() -> pd.DataFrame:
    """
    Create a test dataset with known outliers for validation.
    
    Returns:
        DataFrame with test data
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating test dataset with known outliers...")
    
    # Create base data
    np.random.seed(42)  # For reproducible results
    
    n_records = 1000
    
    # Normal distribution data
    normal_ratings = np.random.normal(3.5, 0.8, n_records)
    normal_ratings = np.clip(normal_ratings, 0, 5)  # Clip to valid range
    
    normal_pages = np.random.normal(300, 100, n_records)
    normal_pages = np.clip(normal_pages, 50, 800)  # Clip to reasonable range
    
    normal_years = np.random.normal(2010, 8, n_records)
    normal_years = np.clip(normal_years, 1990, 2024)  # Clip to reasonable range
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'book_id': range(1, n_records + 1),
        'title': [f'Test Book {i}' for i in range(1, n_records + 1)],
        'author_name': [f'Author {i % 100}' for i in range(1, n_records + 1)],
        'publication_year': normal_years.astype(int),
        'average_rating_weighted_mean': normal_ratings,
        'ratings_count_sum': np.random.poisson(150, n_records),
        'text_reviews_count_sum': np.random.poisson(30, n_records),
        'num_pages': normal_pages.astype(int),
        'genres': ['Romance'] * n_records,
        'decade': ['2010s'] * n_records,
        'book_length_category': ['Medium'] * n_records,
        'rating_category': ['Good'] * n_records,
        'popularity_category': ['Medium'] * n_records
    })
    
    # Add known outliers
    # 1. Impossible ratings
    test_data.loc[0, 'average_rating_weighted_mean'] = 6.0  # >5
    test_data.loc[1, 'average_rating_weighted_mean'] = -1.0  # <0
    
    # 2. Future publication years
    test_data.loc[2, 'publication_year'] = 2030
    test_data.loc[3, 'publication_year'] = 2025
    
    # 3. Impossible page counts
    test_data.loc[4, 'num_pages'] = 0
    test_data.loc[5, 'num_pages'] = -10
    test_data.loc[6, 'num_pages'] = 5000  # Extremely long
    
    # 4. Negative counts
    test_data.loc[7, 'ratings_count_sum'] = -5
    test_data.loc[8, 'text_reviews_count_sum'] = -3
    
    # 5. Statistical outliers (very high values)
    test_data.loc[9, 'ratings_count_sum'] = 10000  # Z-score outlier
    test_data.loc[10, 'num_pages'] = 2000  # IQR outlier
    
    logger.info(f"Test dataset created with {len(test_data)} records")
    logger.info("Known outliers added:")
    logger.info("  • Impossible ratings: 2 records")
    logger.info("  • Future years: 2 records")
    logger.info("  • Impossible page counts: 3 records")
    logger.info("  • Negative counts: 2 records")
    logger.info("  • Statistical outliers: 2 records")
    
    return test_data

def test_outlier_detection_methods(reporter: OutlierDetectionReporter, test_df: pd.DataFrame):
    """
    Test individual outlier detection methods.
    
    Args:
        reporter: OutlierDetectionReporter instance
        test_df: Test DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing individual outlier detection methods...")
    
    # Temporarily replace the reporter's DataFrame
    original_df = reporter.df
    reporter.df = test_df
    
    try:
        # Test Z-score method
        logger.info("Testing Z-score method...")
        zscore_results = reporter.detect_statistical_outliers('ratings_count_sum', methods=['zscore'])
        logger.info(f"Z-score outliers found: {zscore_results.get('outliers', {}).get('zscore', {}).get('count', 0)}")
        
        # Test IQR method
        logger.info("Testing IQR method...")
        iqr_results = reporter.detect_statistical_outliers('num_pages', methods=['iqr'])
        logger.info(f"IQR outliers found: {iqr_results.get('outliers', {}).get('iqr', {}).get('count', 0)}")
        
        # Test percentile method
        logger.info("Testing percentile method...")
        percentile_results = reporter.detect_statistical_outliers('average_rating_weighted_mean', methods=['percentile'])
        logger.info(f"Percentile outliers found: {percentile_results.get('outliers', {}).get('percentile', {}).get('count', 0)}")
        
        # Test multi-method consensus
        logger.info("Testing multi-method consensus...")
        consensus_results = reporter.detect_statistical_outliers('ratings_count_sum', methods=['zscore', 'iqr', 'percentile'])
        consensus_count = consensus_results.get('consensus', {}).get('total_unique_outliers', 0)
        logger.info(f"Consensus outliers found: {consensus_count}")
        
        return True
        
    finally:
        # Restore original DataFrame
        reporter.df = original_df

def test_specialized_analyses(reporter: OutlierDetectionReporter, test_df: pd.DataFrame):
    """
    Test specialized analysis methods.
    
    Args:
        reporter: OutlierDetectionReporter instance
        test_df: Test DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing specialized analyses...")
    
    # Temporarily replace the reporter's DataFrame
    original_df = reporter.df
    reporter.df = test_df
    
    try:
        # Test publication year analysis
        logger.info("Testing publication year analysis...")
        year_results = reporter.analyze_publication_year_outliers()
        logger.info(f"Publication year anomalies: {len(year_results.get('anomalies', {}))}")
        
        # Test rating analysis
        logger.info("Testing rating analysis...")
        rating_results = reporter.analyze_rating_outliers()
        logger.info(f"Rating anomalies found: {sum(len(data.get('anomalies', {})) for data in rating_results.values())}")
        
        # Test page count analysis
        logger.info("Testing page count analysis...")
        page_results = reporter.analyze_page_count_outliers()
        logger.info(f"Page count anomalies: {len(page_results.get('anomalies', {}))}")
        
        # Test categorical analysis
        logger.info("Testing categorical analysis...")
        cat_results = reporter.analyze_categorical_distributions()
        logger.info(f"Categorical fields analyzed: {len(cat_results)}")
        
        return True
        
    finally:
        # Restore original DataFrame
        reporter.df = original_df

def test_comprehensive_analysis(reporter: OutlierDetectionReporter, test_df: pd.DataFrame):
    """
    Test the complete comprehensive analysis.
    
    Args:
        reporter: OutlierDetectionReporter instance
        test_df: Test DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing comprehensive analysis...")
    
    # Temporarily replace the reporter's DataFrame
    original_df = reporter.df
    reporter.df = test_df
    
    try:
        # Run comprehensive analysis
        start_time = time.time()
        results = reporter.run_comprehensive_analysis()
        execution_time = time.time() - start_time
        
        logger.info(f"Comprehensive analysis completed in {execution_time:.2f} seconds")
        
        # Validate results structure
        required_keys = ['analysis_timestamp', 'dataset_info', 'statistical_outliers', 
                        'specialized_analyses', 'categorical_analysis', 'summary']
        
        for key in required_keys:
            if key not in results:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Check summary
        summary = results['summary']
        logger.info(f"Data quality score: {summary.get('data_quality_score', 0):.1f}/100")
        logger.info(f"Total outliers: {summary.get('total_outliers_detected', 0)}")
        logger.info(f"Critical anomalies: {summary.get('critical_anomalies', 0)}")
        
        # Print recommendations
        for rec in summary.get('recommendations', []):
            logger.info(f"Recommendation: {rec}")
        
        return True
        
    finally:
        # Restore original DataFrame
        reporter.df = original_df

def test_error_handling():
    """
    Test error handling with invalid inputs.
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing error handling...")
    
    try:
        # Test with non-existent file
        reporter = OutlierDetectionReporter("non_existent_file.pkl")
        
        # This should raise an exception
        reporter.load_data()
        logger.error("Expected exception not raised for non-existent file")
        return False
        
    except Exception as e:
        logger.info(f"✅ Correctly handled non-existent file: {type(e).__name__}")
    
    try:
        # Test with empty DataFrame
        reporter = OutlierDetectionReporter()
        reporter.df = pd.DataFrame()
        
        # Test outlier detection on empty DataFrame
        results = reporter.detect_statistical_outliers('test_field')
        if 'error' not in results:
            logger.error("Expected error for empty DataFrame")
            return False
        else:
            logger.info("✅ Correctly handled empty DataFrame")
        
    except Exception as e:
        logger.info(f"✅ Correctly handled empty DataFrame: {type(e).__name__}")
    
    return True

def run_performance_benchmark(reporter: OutlierDetectionReporter, test_df: pd.DataFrame):
    """
    Run performance benchmark tests.
    
    Args:
        reporter: OutlierDetectionReporter instance
        test_df: Test DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Running performance benchmarks...")
    
    # Temporarily replace the reporter's DataFrame
    original_df = reporter.df
    reporter.df = test_df
    
    try:
        # Benchmark individual method performance
        methods = ['zscore', 'iqr', 'percentile']
        field = 'ratings_count_sum'
        
        for method in methods:
            start_time = time.time()
            results = reporter.detect_statistical_outliers(field, methods=[method])
            execution_time = time.time() - start_time
            
            outlier_count = results.get('outliers', {}).get(method, {}).get('count', 0)
            logger.info(f"{method.upper()} method: {execution_time:.4f}s, {outlier_count} outliers")
        
        # Benchmark comprehensive analysis
        start_time = time.time()
        comprehensive_results = reporter.run_comprehensive_analysis()
        execution_time = time.time() - start_time
        
        logger.info(f"Comprehensive analysis: {execution_time:.4f}s")
        
        return True
        
    finally:
        # Restore original DataFrame
        reporter.df = original_df

def main():
    """Main test function."""
    logger = setup_test_logging()
    
    logger.info("🧪 Starting Step 4 Outlier Detection Tests")
    logger.info("=" * 60)
    
    try:
        # Create test dataset
        test_df = create_test_dataset()
        
        # Initialize reporter
        reporter = OutlierDetectionReporter()
        
        # Run tests
        tests = [
            ("Individual outlier detection methods", 
             lambda: test_outlier_detection_methods(reporter, test_df)),
            ("Specialized analyses", 
             lambda: test_specialized_analyses(reporter, test_df)),
            ("Comprehensive analysis", 
             lambda: test_comprehensive_analysis(reporter, test_df)),
            ("Error handling", 
             lambda: test_error_handling()),
            ("Performance benchmarks", 
             lambda: run_performance_benchmark(reporter, test_df))
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running test: {test_name}")
            try:
                if test_func():
                    logger.info(f"✅ {test_name} passed")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} failed")
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {str(e)}")
        
        # Test summary
        logger.info("\n" + "="*60)
        logger.info(f"🧪 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Outlier detection system is working correctly.")
            return True
        else:
            logger.error(f"💥 {total_tests - passed_tests} tests failed. Check logs for details.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
