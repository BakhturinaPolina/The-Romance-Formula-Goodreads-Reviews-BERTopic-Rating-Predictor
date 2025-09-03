#!/usr/bin/env python3
"""
Runner script for Step 3: Outlier Detection and Treatment

This script runs the comprehensive outlier detection analysis and provides
a summary of results.

Author: Research Assistant
Date: 2025-01-02
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from outlier_detection_analysis import OutlierDetectionAnalyzer

def setup_logging():
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/outlier_detection_{timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main execution function."""
    logger = setup_logging()
    
    try:
        logger.info("=" * 80)
        logger.info("STEP 3: OUTLIER DETECTION AND TREATMENT ANALYSIS")
        logger.info("=" * 80)
        logger.info("Starting comprehensive outlier detection analysis...")
        
        # Initialize analyzer
        analyzer = OutlierDetectionAnalyzer()
        
        # Run complete analysis
        logger.info("Running complete analysis...")
        results = analyzer.run_complete_analysis()
        
        # Save results
        logger.info("Saving results...")
        analyzer.save_results_json()
        
        # Print summary
        print("\n" + "="*80)
        print("OUTLIER DETECTION ANALYSIS COMPLETE")
        print("="*80)
        print(f"Title duplications found: {results['title_duplications']['total_duplicate_titles']}")
        print(f"Series books analyzed: {results['series_title_data']['total_series_title_books']}")
        print(f"Total outliers detected: {results['statistical_outliers']['summary']['total_outliers_detected']}")
        print(f"High priority actions: {len(results['treatment_recommendations']['priority_actions'])}")
        print(f"\nReports saved to: {analyzer.output_dir}")
        print(f"Logs saved to: logs/")
        
        # Log summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Title duplications: {results['title_duplications']['total_duplicate_titles']}")
        logger.info(f"Series books: {results['series_title_data']['total_series_title_books']}")
        logger.info(f"Total outliers: {results['statistical_outliers']['summary']['total_outliers_detected']}")
        logger.info(f"High priority actions: {len(results['treatment_recommendations']['priority_actions'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"\nERROR: Analysis failed - {str(e)}")
        print("Check logs for detailed error information.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
