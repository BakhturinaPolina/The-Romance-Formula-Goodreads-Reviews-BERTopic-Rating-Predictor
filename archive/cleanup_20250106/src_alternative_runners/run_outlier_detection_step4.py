#!/usr/bin/env python3
"""
Runner script for Step 4: Outlier Detection & Reporting (Medium Priority)

This script executes the comprehensive outlier detection analysis with:
- Enhanced logging and progress tracking
- Error handling and validation
- Performance monitoring
- Comprehensive reporting

Author: Research Assistant
Date: 2025-09-02
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from outlier_detection_step4 import OutlierDetectionReporter

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
            logging.FileHandler(f"logs/outlier_detection_step4_{timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def validate_dataset_path(data_path: str) -> bool:
    """
    Validate that the dataset file exists and is accessible.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        bool: True if file exists and is accessible
    """
    path = Path(data_path)
    
    if not path.exists():
        print(f"❌ Dataset file not found: {data_path}")
        return False
    
    if not path.is_file():
        print(f"❌ Path is not a file: {data_path}")
        return False
    
    # Check file size
    file_size_mb = path.stat().st_size / 1024 / 1024
    print(f"✅ Dataset file found: {data_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    return True

def run_outlier_detection_analysis(data_path: str = None) -> Dict[str, Any]:
    """
    Run the complete outlier detection analysis.
    
    Args:
        data_path: Path to the cleaned dataset file
        
    Returns:
        Dictionary with analysis results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Step 4: Outlier Detection & Reporting")
    logger.info("=" * 80)
    
    try:
        # Initialize the outlier detection reporter
        logger.info("📋 Initializing OutlierDetectionReporter...")
        reporter = OutlierDetectionReporter(data_path)
        
        # Load the dataset
        logger.info("📥 Loading dataset...")
        df = reporter.load_data()
        logger.info(f"✅ Dataset loaded successfully: {df.shape[0]:,} records × {df.shape[1]} columns")
        
        # Run comprehensive analysis
        logger.info("🔍 Running comprehensive outlier detection analysis...")
        results = reporter.run_comprehensive_analysis()
        
        # Save results
        logger.info("💾 Saving analysis results...")
        report_file = reporter.save_results()
        logger.info(f"✅ Results saved to: {report_file}")
        
        # Print summary
        logger.info("📊 Generating analysis summary...")
        reporter.print_summary()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Outlier detection analysis failed: {str(e)}", exc_info=True)
        raise

def generate_execution_summary(results: Dict[str, Any], execution_time: float) -> str:
    """
    Generate a comprehensive execution summary.
    
    Args:
        results: Analysis results dictionary
        execution_time: Total execution time in seconds
        
    Returns:
        str: Formatted summary string
    """
    if not results:
        return "No results available"
    
    summary = results.get('summary', {})
    dataset_info = results.get('dataset_info', {})
    
    summary_text = f"""
{'='*80}
STEP 4: OUTLIER DETECTION & REPORTING - EXECUTION SUMMARY
{'='*80}

📊 ANALYSIS RESULTS:
  • Dataset: {dataset_info.get('shape', ['N/A', 'N/A'])[0]:,} records × {dataset_info.get('shape', ['N/A', 'N/A'])[1]} columns
  • Memory Usage: {dataset_info.get('memory_usage_mb', 0):.2f} MB
  • Data Quality Score: {summary.get('data_quality_score', 0):.1f}/100
  • Total Outliers Detected: {summary.get('total_outliers_detected', 0):,}
  • Fields with Outliers: {summary.get('fields_with_outliers', 0)}
  • Critical Anomalies: {summary.get('critical_anomalies', 0):,}

⏱️ PERFORMANCE:
  • Total Execution Time: {execution_time:.2f} seconds
  • Analysis Time: {results.get('execution_time_seconds', 0):.2f} seconds

💡 RECOMMENDATIONS:
"""
    
    for rec in summary.get('recommendations', []):
        summary_text += f"  • {rec}\n"
    
    summary_text += f"\n{'='*80}"
    
    return summary_text

def main():
    """Main execution function."""
    # Set up logging
    logger = setup_logging()
    
    print("🚀 STEP 4: OUTLIER DETECTION & REPORTING")
    print("=" * 60)
    
    # Configuration
    data_path = "data/processed/cleaned_romance_novels_step1_3_20250902_223102.pkl"
    
    # Validate dataset path
    if not validate_dataset_path(data_path):
        print("❌ Dataset validation failed. Exiting.")
        sys.exit(1)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run outlier detection analysis
        results = run_outlier_detection_analysis(data_path)
        
        # Calculate total execution time
        total_execution_time = time.time() - start_time
        
        # Generate and display execution summary
        execution_summary = generate_execution_summary(results, total_execution_time)
        print(execution_summary)
        
        # Save execution summary to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"outputs/outlier_detection/execution_summary_step4_{timestamp}.txt"
        
        # Ensure directory exists
        Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write(execution_summary)
        
        print(f"✅ Execution summary saved to: {summary_file}")
        print("\n🎉 Step 4: Outlier Detection & Reporting completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Step 4 execution failed: {str(e)}", exc_info=True)
        print(f"\n💥 Step 4 execution failed: {str(e)}")
        print("Check the logs for detailed error information.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
