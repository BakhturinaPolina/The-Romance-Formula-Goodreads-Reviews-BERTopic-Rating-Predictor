#!/usr/bin/env python3
"""
Fully Automated Romance Book Discovery and Download Workflow
Complete automation from romance dataset to downloaded books
"""

import sys
import logging
import subprocess
from pathlib import Path
import argparse
import time
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fully_automated_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Run a shell command with logging
    
    Args:
        command: Command to run
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"✓ Completed: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def create_priority_lists(romance_csv: str, output_dir: str) -> bool:
    """
    Create priority lists for automated search
    
    Args:
        romance_csv: Path to romance books CSV
        output_dir: Output directory for priority lists
        
    Returns:
        True if successful
    """
    command = f"python utils/priority_book_selector.py --romance-csv {romance_csv} --output-dir {output_dir}"
    return run_command(command, "Creating priority book lists")


def run_automated_search(priority_list: str, output_csv: str, max_books: int = 100) -> bool:
    """
    Run automated search on Anna's Archive
    
    Args:
        priority_list: Path to priority list CSV
        output_csv: Output CSV for search results
        max_books: Maximum number of books to search
        
    Returns:
        True if successful
    """
    command = f"python utils/automated_search.py --romance-csv {priority_list} --output-csv {output_csv} --max-books {max_books}"
    return run_command(command, "Running automated Anna Archive search")


def run_batch_download(download_csv: str, output_dir: str) -> bool:
    """
    Run batch download of found books
    
    Args:
        download_csv: Path to download-ready CSV
        output_dir: Output directory for downloads
        
    Returns:
        True if successful
    """
    command = f"python ../../standalone_downloader.py --csv {download_csv} --output-dir {output_dir}"
    return run_command(command, "Running batch download of found books")


def generate_automation_report(results_csv: str, output_dir: str) -> None:
    """
    Generate comprehensive automation report
    
    Args:
        results_csv: Path to search results CSV
        output_dir: Output directory for reports
    """
    try:
        if not Path(results_csv).exists():
            logger.warning(f"Results CSV not found: {results_csv}")
            return
        
        results_df = pd.read_csv(results_csv)
        
        report_path = Path(output_dir) / "automation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Fully Automated Romance Book Discovery Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total books searched: {len(results_df)}\n")
            f.write(f"Books found: {len(results_df[results_df['md5_hash'].notna()])}\n")
            f.write(f"Success rate: {len(results_df[results_df['md5_hash'].notna()])/len(results_df)*100:.2f}%\n\n")
            
            if len(results_df) > 0:
                f.write("File Format Distribution:\n")
                all_formats = []
                for formats in results_df['file_formats'].dropna():
                    all_formats.extend(eval(formats) if isinstance(formats, str) else formats)
                
                format_counts = pd.Series(all_formats).value_counts()
                for fmt, count in format_counts.items():
                    f.write(f"  {fmt}: {count}\n")
                f.write("\n")
                
                f.write("Top 10 Found Books:\n")
                top_books = results_df.nlargest(10, 'match_quality')
                for idx, book in top_books.iterrows():
                    f.write(f"  {book['original_title']} by {book['original_author']} "
                           f"(quality: {book['match_quality']:.3f})\n")
                f.write("\n")
                
                f.write("Match Quality Distribution:\n")
                quality_stats = results_df['match_quality'].describe()
                for stat, value in quality_stats.items():
                    f.write(f"  {stat}: {value:.3f}\n")
        
        logger.info(f"Automation report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating automation report: {e}")


def main():
    """
    Main function for fully automated workflow
    """
    parser = argparse.ArgumentParser(description='Fully Automated Romance Book Workflow')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--output-dir', default='automated_outputs',
                       help='Path to output directory')
    parser.add_argument('--max-books', type=int, default=100,
                       help='Maximum number of books to search')
    parser.add_argument('--priority-list', default='top_rated',
                       help='Priority list to use (top_rated, most_reviewed, recent_popular, test_sample)')
    parser.add_argument('--skip-search', action='store_true',
                       help='Skip automated search (use existing results)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip batch download')
    parser.add_argument('--delay-min', type=float, default=1.0,
                       help='Minimum delay between requests (seconds)')
    parser.add_argument('--delay-max', type=float, default=3.0,
                       help='Maximum delay between requests (seconds)')
    
    args = parser.parse_args()
    
    logger.info("Starting Fully Automated Romance Book Workflow")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Create priority lists
        if not args.skip_search:
            logger.info("Step 1: Creating priority lists...")
            if not create_priority_lists(args.romance_csv, str(output_dir / "priority_lists")):
                logger.error("Failed to create priority lists. Exiting.")
                return 1
        
        # Step 2: Run automated search
        if not args.skip_search:
            logger.info("Step 2: Running automated search...")
            
            priority_list_path = output_dir / "priority_lists" / f"{args.priority_list}_popular_books.csv"
            if args.priority_list == "test_sample":
                priority_list_path = output_dir / "priority_lists" / "test_sample_50_books.csv"
            
            if not priority_list_path.exists():
                logger.error(f"Priority list not found: {priority_list_path}")
                return 1
            
            results_csv = output_dir / "automated_search_results.csv"
            
            if not run_automated_search(str(priority_list_path), str(results_csv), args.max_books):
                logger.error("Automated search failed. Exiting.")
                return 1
        else:
            logger.info("Skipping automated search (--skip-search)")
            results_csv = output_dir / "automated_search_results.csv"
        
        # Step 3: Run batch download
        if not args.skip_download:
            logger.info("Step 3: Running batch download...")
            
            download_csv = output_dir / "automated_search_results_download_ready.csv"
            if not download_csv.exists():
                logger.warning(f"Download-ready CSV not found: {download_csv}")
                logger.info("Creating download-ready CSV from search results...")
                
                if results_csv.exists():
                    results_df = pd.read_csv(results_csv)
                    download_df = results_df[results_df['md5_hash'].notna()].copy()
                    if not download_df.empty:
                        download_df[['work_id', 'original_title', 'original_author', 'md5_hash', 'file_formats']].to_csv(download_csv, index=False)
                        logger.info(f"Created download-ready CSV: {download_csv}")
                    else:
                        logger.warning("No books with MD5 hashes found for download")
                        return 1
                else:
                    logger.error("No search results found for download")
                    return 1
            
            download_output_dir = output_dir / "downloaded_books"
            download_output_dir.mkdir(exist_ok=True)
            
            if not run_batch_download(str(download_csv), str(download_output_dir)):
                logger.error("Batch download failed. Exiting.")
                return 1
        else:
            logger.info("Skipping batch download (--skip-download)")
        
        # Step 4: Generate comprehensive report
        logger.info("Step 4: Generating automation report...")
        generate_automation_report(str(results_csv), str(output_dir))
        
        logger.info("=" * 60)
        logger.info("Fully automated workflow completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Show summary
        if results_csv.exists():
            results_df = pd.read_csv(results_csv)
            found_count = len(results_df[results_df['md5_hash'].notna()])
            logger.info(f"Summary: {len(results_df)} books searched, {found_count} books found")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during automated workflow: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
