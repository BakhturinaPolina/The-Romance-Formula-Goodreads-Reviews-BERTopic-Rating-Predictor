#!/usr/bin/env python3
"""
Enhanced Runner script for Step 3: Outlier Detection and Treatment

This script runs the comprehensive outlier detection analysis with:
- Detailed logging and timestamps
- Comprehensive variable validation
- Progress tracking
- Error prevention

Author: Research Assistant
Date: 2025-01-02
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from outlier_detection_analysis import OutlierDetectionAnalyzer

def setup_logging():
    """Set up enhanced logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/outlier_detection_enhanced_{timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def validate_results_structure(results: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Validate the complete results structure to prevent KeyError issues.
    
    Args:
        results: Analysis results dictionary
        logger: Logger instance
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    logger.info("🔍 Validating results structure...")
    
    # Define expected structure
    expected_structure = {
        'title_duplications': {
            'total_duplicate_titles': int,
            'total_duplicate_books': (int, str),
            'author_name_analysis': dict
        },
        'series_title_data': {
            'total_series_title_books': int,
            'unique_series_title': int,
            'series_title_works_count_analysis': dict,
            'missing_books_analysis': dict
        },
        'statistical_outliers': {
            'summary': {
                'total_outliers_detected': int,
                'fields_analyzed': list
            }
        },
        'treatment_recommendations': {
            'priority_actions': list,
            'research_impact': dict
        }
    }
    
    try:
        # Validate top-level keys
        for key, expected_type in expected_structure.items():
            if key not in results:
                logger.error(f"❌ Missing required key: {key}")
                return False
            logger.info(f"✅ Found key: {key}")
            
            # Validate nested structure
            if isinstance(expected_type, dict):
                for nested_key, nested_type in expected_type.items():
                    if nested_key not in results[key]:
                        logger.error(f"❌ Missing nested key: {key}.{nested_key}")
                        return False
                    logger.info(f"✅ Found nested key: {key}.{nested_key}")
                    
                    # Validate further nesting if needed
                    if isinstance(nested_type, dict):
                        for deep_key in nested_type.keys():
                            if deep_key not in results[key][nested_key]:
                                logger.error(f"❌ Missing deep key: {key}.{nested_key}.{deep_key}")
                                return False
                            logger.info(f"✅ Found deep key: {key}.{nested_key}.{deep_key}")
        
        logger.info("✅ Results structure validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Results structure validation failed: {str(e)}")
        return False

def print_enhanced_summary(results: Dict[str, Any], logger: logging.Logger, start_time: float):
    """
    Print enhanced summary with detailed information and timing.
    
    Args:
        results: Analysis results dictionary
        logger: Logger instance
        start_time: Analysis start time
    """
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*100)
    print("🎯 STEP 3: OUTLIER DETECTION AND TREATMENT ANALYSIS - COMPLETE")
    print("="*100)
    print(f"⏱️  Total Analysis Time: {duration:.2f} seconds")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    # Title Duplications
    if 'title_duplications' in results:
        dup_data = results['title_duplications']
        print(f"📚 TITLE DUPLICATIONS:")
        print(f"   • Total duplicate titles: {dup_data.get('total_duplicate_titles', 'N/A')}")
        print(f"   • Total duplicate books: {dup_data.get('total_duplicate_books', 'N/A')}")
        print(f"   • Analysis status: ✅ Complete")
    
    # Series Data
    if 'series_title_data' in results:
        series_data = results['series_title_data']
        print(f"\n📖 SERIES DATA:")
        print(f"   • Series books analyzed: {series_data.get('total_series_title_books', 'N/A')}")
        print(f"   • Unique series: {series_data.get('unique_series_title', 'N/A')}")
        print(f"   • Analysis status: ✅ Complete")
    
    # Statistical Outliers
    if 'statistical_outliers' in results:
        outlier_data = results['statistical_outliers']
        if 'summary' in outlier_data:
            summary = outlier_data['summary']
            print(f"\n📊 STATISTICAL OUTLIERS:")
            print(f"   • Total outliers detected: {summary.get('total_outliers_detected', 'N/A')}")
            print(f"   • Fields analyzed: {', '.join(summary.get('fields_analyzed', []))}")
            print(f"   • Analysis status: ✅ Complete")
    
    # Treatment Recommendations
    if 'treatment_recommendations' in results:
        rec_data = results['treatment_recommendations']
        priority_actions = rec_data.get('priority_actions', [])
        print(f"\n🎯 TREATMENT RECOMMENDATIONS:")
        print(f"   • High priority actions: {len(priority_actions)}")
        for i, action in enumerate(priority_actions[:3], 1):  # Show first 3
            print(f"     {i}. {action.get('issue', 'N/A')}")
        if len(priority_actions) > 3:
            print(f"     ... and {len(priority_actions) - 3} more")
        print(f"   • Analysis status: ✅ Complete")
    
    print("\n" + "="*100)
    print("📁 OUTPUT FILES:")
    print(f"   • Logs: logs/outlier_detection_enhanced_*.log")
    print(f"   • Results: outputs/outlier_detection/")
    print("="*100)

def main():
    """Enhanced main execution function with comprehensive logging and validation."""
    start_time = time.time()
    logger = setup_logging()
    
    try:
        logger.info("🚀 " + "="*80)
        logger.info("🚀 STEP 3: ENHANCED OUTLIER DETECTION AND TREATMENT ANALYSIS")
        logger.info("🚀 " + "="*80)
        logger.info(f"⏰ Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("🔍 Starting comprehensive outlier detection analysis...")
        
        # Initialize analyzer
        logger.info("🔧 Initializing OutlierDetectionAnalyzer...")
        analyzer = OutlierDetectionAnalyzer()
        logger.info("✅ Analyzer initialized successfully")
        
        # Run complete analysis
        logger.info("🔄 Running complete analysis...")
        results = analyzer.run_complete_analysis()
        logger.info("✅ Analysis completed successfully!")
        
        # Validate results structure
        if not validate_results_structure(results, logger):
            raise ValueError("Results structure validation failed")
        
        # Save results
        logger.info("💾 Saving results...")
        analyzer.save_results_json()
        logger.info("✅ Results saved successfully")
        
        # Print enhanced summary
        print_enhanced_summary(results, logger, start_time)
        
        # Log success summary
        logger.info("🎉 Analysis completed successfully!")
        logger.info(f"⏱️  Total duration: {time.time() - start_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"💥 Analysis failed after {duration:.2f} seconds: {str(e)}", exc_info=True)
        print(f"\n💥 ERROR: Analysis failed after {duration:.2f} seconds")
        print(f"💥 Error: {str(e)}")
        print("📋 Check logs for detailed error information.")
        print(f"📁 Log file: logs/outlier_detection_enhanced_*.log")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
