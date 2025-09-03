#!/usr/bin/env python3
"""
Runner script for Title Duplication Cleaning - Priority: 🔴 CRITICAL

This script executes the comprehensive title duplication cleaning pipeline
with enhanced logging, progress tracking, and validation.

Author: Research Assistant
Date: 2025-01-02
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from title_duplication_cleaning import TitleDuplicationCleaner

def setup_logging():
    """Set up enhanced logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/title_cleaning_{timestamp}.log")
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Title Duplication Cleaning Pipeline Started")
    logger.info(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Working Directory: {Path.cwd()}")
    
    return logger

def validate_input_data(data_path: str) -> bool:
    """Validate that input data exists and is accessible."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Validating input data...")
    
    if not Path(data_path).exists():
        logger.error(f"❌ Input data not found: {data_path}")
        return False
    
    file_size = Path(data_path).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✅ Input data validated: {data_path}")
    logger.info(f"📊 File size: {file_size:.2f} MB")
    
    return True

def main():
    """Main execution function with enhanced error handling."""
    # Setup logging
    logger = setup_logging()
    
    # Configuration
    input_csv = "data/processed/final_books_2000_2020_en_cleaned_nlp_ready_20250902_161743.csv"
    output_dir = "outputs/title_cleaning"
    
    logger.info("⚙️ Configuration loaded:")
    logger.info(f"   📥 Input: {input_csv}")
    logger.info(f"   📤 Output: {output_dir}")
    
    try:
        # Validate input data
        if not validate_input_data(input_csv):
            logger.error("❌ Input validation failed. Exiting.")
            sys.exit(1)
        
        # Initialize cleaner
        logger.info("🔧 Initializing Title Duplication Cleaner...")
        cleaner = TitleDuplicationCleaner(input_csv)
        
        # Run complete cleaning pipeline
        logger.info("🚀 Starting cleaning pipeline...")
        start_time = time.time()
        
        results = cleaner.run_complete_cleaning(output_dir)
        
        pipeline_time = time.time() - start_time
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("🎯 TITLE DUPLICATION CLEANING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"✅ Pipeline Status: {results['pipeline_status']}")
        print(f"⏱️ Total Pipeline Time: {pipeline_time:.2f}s")
        print(f"📊 Records Cleaned: {results['cleaning_summary']['records_cleaned']:,}")
        print(f"🔍 Manual Review Priority: {results['manual_review_needed']}")
        print(f"📁 Outputs Generated: {len(results['outputs_generated'])} files")
        
        print("\n📋 Cleaning Summary:")
        print(f"   • Total Records: {results['cleaning_summary']['total_records']:,}")
        print(f"   • Unique Titles: {results['cleaning_summary']['unique_titles']:,}")
        print(f"   • Duplicate Titles Processed: {results['cleaning_summary']['duplicate_titles_processed']:,}")
        print(f"   • Records Cleaned: {results['cleaning_summary']['records_cleaned']:,}")
        
        print("\n📁 Generated Files:")
        for output_file in results['outputs_generated']:
            print(f"   • {Path(output_file).name}")
        
        print("\n🎯 Next Steps:")
        if results['manual_review_needed'] == 'HIGH':
            print("   🔴 CRITICAL: Manual review of identical duplicates required")
            print("   📋 Review the manual_review_needed list in the cleaning report")
            print("   ⏰ Estimated effort: 2-4 hours manual review")
        else:
            print("   🟢 All critical issues resolved automatically")
        
        print("   🟡 MEDIUM: Include disambiguation metadata in NLP preprocessing")
        print("   🟢 LOW: Monitor for new duplication patterns")
        
        print("="*80)
        
        # Log completion
        logger.info("🎉 Title duplication cleaning pipeline completed successfully!")
        logger.info(f"⏱️ Total pipeline time: {pipeline_time:.2f}s")
        logger.info(f"📊 Records cleaned: {results['cleaning_summary']['records_cleaned']:,}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        
        print("\n" + "="*80)
        print("❌ TITLE DUPLICATION CLEANING FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("Check the logs for detailed error information.")
        print("="*80)
        
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
