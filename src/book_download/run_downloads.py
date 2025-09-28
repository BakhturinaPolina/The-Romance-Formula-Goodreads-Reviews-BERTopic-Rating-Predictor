#!/usr/bin/env python3
"""
Book Download Research Component - Main Download Runner
Production script for downloading romance novels from Anna's Archive
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_manager import BookDownloadManager
from mcp_integration import AnnaMCPIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function for running book downloads"""
    
    parser = argparse.ArgumentParser(description='Download romance novels from Anna\'s Archive')
    parser.add_argument('--csv', 
                       default='/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/romance_subdataset_6000.csv',
                       help='Path to CSV file with book metadata')
    parser.add_argument('--sample', 
                       action='store_true',
                       help='Use sample CSV for testing (2-3 books)')
    parser.add_argument('--max-books', 
                       type=int, 
                       default=None,
                       help='Maximum number of books to process in this run')
    parser.add_argument('--daily-limit', 
                       type=int, 
                       default=999,
                       help='Daily download limit (default: 999)')
    parser.add_argument('--download-dir',
                       default='/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download',
                       help='Directory to save downloaded books')
    
    args = parser.parse_args()
    
    # Determine CSV path
    if args.sample:
        csv_path = '/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/sample_books_for_download.csv'
        logger.info("Using SAMPLE CSV for testing")
    else:
        csv_path = args.csv
        logger.info("Using FULL CSV for production")
    
    logger.info("=== ROMANCE NOVEL DOWNLOAD SYSTEM ===")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Download directory: {args.download_dir}")
    logger.info(f"Daily limit: {args.daily_limit}")
    logger.info(f"Max books this run: {args.max_books or 'No limit'}")
    logger.info(f"Start time: {datetime.now()}")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Initialize download manager
    try:
        manager = BookDownloadManager(
            csv_path=csv_path,
            download_dir=args.download_dir,
            daily_limit=args.daily_limit
        )
        logger.info("Download manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize download manager: {e}")
        sys.exit(1)
    
    # Test MCP integration
    try:
        mcp = AnnaMCPIntegration()
        logger.info("MCP integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP integration: {e}")
        sys.exit(1)
    
    # Run download batch
    try:
        logger.info("Starting download batch...")
        summary = manager.run_download_batch(max_books=args.max_books)
        
        logger.info("=== DOWNLOAD BATCH COMPLETED ===")
        logger.info(f"Status: {summary['status']}")
        logger.info(f"Books processed: {summary['processed']}")
        logger.info(f"Books downloaded: {summary['downloaded']}")
        logger.info(f"Books failed: {summary['failed']}")
        logger.info(f"Last row processed: {summary['last_row']}")
        logger.info(f"Daily downloads: {summary['daily_downloads']}/{args.daily_limit}")
        logger.info(f"Remaining today: {summary['remaining_today']}")
        
        # Save summary to file
        summary_file = os.path.join(args.download_dir, f"download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"Could not save summary: {e}")
        
        # Determine exit code
        if summary['status'] == 'completed':
            logger.info("Download batch completed successfully!")
            sys.exit(0)
        elif summary['status'] == 'daily_limit_reached':
            logger.info("Daily limit reached - run again tomorrow")
            sys.exit(0)
        else:
            logger.error("Download batch failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Download batch failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
