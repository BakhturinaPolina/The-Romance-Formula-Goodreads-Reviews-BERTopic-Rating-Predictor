#!/usr/bin/env python3
"""
Book Download Research Component - Production Runner
Run the full 6,000 book dataset with proper daily limits and monitoring
"""

import pandas as pd
import os
import sys
import logging
import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from book_download.download_manager import BookDownloadManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDownloadRunner:
    """Production runner for the full book download system"""
    
    def __init__(self, 
                 csv_path: str,
                 download_dir: str,
                 daily_limit: int = 50,
                 max_books: Optional[int] = None):
        """
        Initialize production download runner
        
        Args:
            csv_path: Path to the CSV file with book metadata
            download_dir: Directory to save downloaded books
            daily_limit: Maximum books to download per day
            max_books: Maximum books to process in this run (None for all)
        """
        self.csv_path = csv_path
        self.download_dir = download_dir
        self.daily_limit = daily_limit
        self.max_books = max_books
        
        # Initialize download manager
        self.manager = BookDownloadManager(
            csv_path=csv_path,
            download_dir=download_dir,
            daily_limit=daily_limit
        )
        
        # Create results directory
        self.results_dir = Path(download_dir) / "production_results"
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("=== PRODUCTION DOWNLOAD RUNNER INITIALIZED ===")
        logger.info(f"CSV Path: {csv_path}")
        logger.info(f"Download Directory: {download_dir}")
        logger.info(f"Daily Limit: {daily_limit}")
        logger.info(f"Max Books (this run): {max_books or 'All'}")
        logger.info(f"Results Directory: {self.results_dir}")
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        try:
            df = pd.read_csv(self.csv_path)
            stats = {
                'total_books': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'min_year': df['publication_year'].min(),
                    'max_year': df['publication_year'].max()
                },
                'popularity_distribution': df['pop_tier'].value_counts().to_dict(),
                'genre_distribution': df['genres_str'].value_counts().head(10).to_dict()
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting dataset stats: {e}")
            return {}
    
    def run_production_batch(self) -> Dict:
        """Run a production batch of downloads"""
        logger.info("=== STARTING PRODUCTION DOWNLOAD BATCH ===")
        
        # Get dataset statistics
        stats = self.get_dataset_stats()
        logger.info(f"Dataset contains {stats.get('total_books', 'unknown')} books")
        
        # Run download batch
        start_time = datetime.now()
        
        summary = self.manager.run_download_batch(max_books=self.max_books)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Add timing information
        summary['start_time'] = start_time.isoformat()
        summary['end_time'] = end_time.isoformat()
        summary['duration_seconds'] = duration.total_seconds()
        summary['duration_human'] = str(duration)
        
        # Save detailed results
        self._save_results(summary)
        
        # Log summary
        self._log_summary(summary)
        
        return summary
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _save_results(self, summary: Dict):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python native types
        summary_clean = self._convert_numpy_types(summary)
        progress_clean = self._convert_numpy_types(self.manager.progress)
        
        # Save summary
        summary_file = self.results_dir / f"download_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_clean, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")
        
        # Save detailed results
        if 'results' in summary_clean:
            results_file = self.results_dir / f"detailed_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(summary_clean['results'], f, indent=2)
            logger.info(f"Detailed results saved to: {results_file}")
        
        # Save progress tracking
        progress_file = self.results_dir / f"progress_{timestamp}.json"
        with open(progress_file, 'w') as f:
            json.dump(progress_clean, f, indent=2)
        logger.info(f"Progress tracking saved to: {progress_file}")
    
    def _log_summary(self, summary: Dict):
        """Log a comprehensive summary"""
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION DOWNLOAD BATCH SUMMARY")
        logger.info("="*60)
        logger.info(f"Status: {summary.get('status', 'unknown')}")
        logger.info(f"Books Processed: {summary.get('processed', 0)}")
        logger.info(f"Successful Downloads: {summary.get('downloaded', 0)}")
        logger.info(f"Failed Downloads: {summary.get('failed', 0)}")
        logger.info(f"Success Rate: {self._calculate_success_rate(summary):.1f}%")
        logger.info(f"Last Row Processed: {summary.get('last_row', 0)}")
        logger.info(f"Daily Downloads: {summary.get('daily_downloads', 0)}/{self.daily_limit}")
        logger.info(f"Remaining Today: {summary.get('remaining_today', 0)}")
        logger.info(f"Duration: {summary.get('duration_human', 'unknown')}")
        
        # Performance metrics
        if summary.get('duration_seconds'):
            duration = summary['duration_seconds']
            processed = summary.get('processed', 1)
            logger.info(f"Average Time per Book: {duration/processed:.1f} seconds")
            logger.info(f"Books per Hour: {processed/(duration/3600):.1f}")
        
        logger.info("="*60)
    
    def _calculate_success_rate(self, summary: Dict) -> float:
        """Calculate success rate percentage"""
        processed = summary.get('processed', 0)
        downloaded = summary.get('downloaded', 0)
        if processed == 0:
            return 0.0
        return (downloaded / processed) * 100
    
    def monitor_progress(self):
        """Monitor current progress and provide status"""
        logger.info("=== CURRENT PROGRESS STATUS ===")
        logger.info(f"Last Row: {self.manager.progress.get('last_row', 0)}")
        logger.info(f"Total Processed: {self.manager.progress.get('total_processed', 0)}")
        logger.info(f"Total Downloaded: {self.manager.progress.get('total_downloaded', 0)}")
        logger.info(f"Total Failed: {self.manager.progress.get('total_failed', 0)}")
        logger.info(f"Daily Downloads: {self.manager.progress.get('daily_downloads', 0)}/{self.daily_limit}")
        logger.info(f"Last Run Date: {self.manager.progress.get('last_run_date', 'Never')}")
        
        # Calculate overall success rate
        total_processed = self.manager.progress.get('total_processed', 0)
        total_downloaded = self.manager.progress.get('total_downloaded', 0)
        if total_processed > 0:
            success_rate = (total_downloaded / total_processed) * 100
            logger.info(f"Overall Success Rate: {success_rate:.1f}%")
        
        # Estimate completion
        if total_processed > 0:
            remaining = 6000 - self.manager.progress.get('last_row', 0)
            if remaining > 0:
                logger.info(f"Estimated Books Remaining: {remaining}")
                if self.manager.progress.get('daily_downloads', 0) > 0:
                    days_remaining = remaining / self.daily_limit
                    logger.info(f"Estimated Days to Complete: {days_remaining:.1f}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run production book downloads')
    parser.add_argument('--csv-path', 
                       default='data/processed/test_subdataset.csv',
                       help='Path to CSV file with book metadata')
    parser.add_argument('--download-dir',
                       default='organized_outputs/anna_archive_download',
                       help='Directory to save downloaded books')
    parser.add_argument('--daily-limit', type=int, default=50,
                       help='Maximum books to download per day')
    parser.add_argument('--max-books', type=int, default=None,
                       help='Maximum books to process in this run')
    parser.add_argument('--monitor-only', action='store_true',
                       help='Only show current progress status')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ProductionDownloadRunner(
        csv_path=args.csv_path,
        download_dir=args.download_dir,
        daily_limit=args.daily_limit,
        max_books=args.max_books
    )
    
    if args.monitor_only:
        runner.monitor_progress()
    else:
        # Run production batch
        summary = runner.run_production_batch()
        
        # Exit with appropriate code
        if summary.get('status') == 'completed':
            logger.info("Production batch completed successfully!")
            sys.exit(0)
        else:
            logger.error("Production batch failed or incomplete!")
            sys.exit(1)

if __name__ == "__main__":
    main()
