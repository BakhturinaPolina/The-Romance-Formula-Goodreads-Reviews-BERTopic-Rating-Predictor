#!/usr/bin/env python3
"""
Integrated Workflow for Anna's Archive Book Matching and Download
Complete automation from romance dataset to downloaded books
"""

import sys
import logging
import subprocess
from pathlib import Path
import argparse
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_workflow.log'),
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


def check_prerequisites() -> bool:
    """
    Check if all prerequisites are met
    
    Returns:
        True if all prerequisites met, False otherwise
    """
    logger.info("Checking prerequisites...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.warning("Virtual environment not detected. Please activate your virtual environment.")
        return False
    
    # Check if Anna Archive data directories exist
    data_dir = Path("data")
    required_dirs = ["elasticsearch", "aac", "mariadb"]
    
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            logger.error("Please download Anna Archive datasets first.")
            return False
        
        # Check if directory has files
        files = list(dir_path.glob("*"))
        if not files:
            logger.error(f"Directory {dir_path} is empty. Please add Anna Archive data files.")
            return False
    
    logger.info("✓ All prerequisites met")
    return True


def process_anna_archive_data() -> bool:
    """
    Process Anna Archive raw data files
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Processing Anna Archive data files...")
    
    command = "python run_matcher.py --process-data --data-dir data"
    return run_command(command, "Anna Archive data processing")


def run_book_matching(romance_csv: str, sample_size: int = None) -> bool:
    """
    Run book matching process
    
    Args:
        romance_csv: Path to romance books CSV
        sample_size: Optional sample size for testing
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running book matching...")
    
    command = f"python run_matcher.py --romance-csv {romance_csv} --data-dir data --output-dir outputs --similarity-threshold 0.8"
    
    if sample_size:
        command += f" --sample-size {sample_size}"
    
    return run_command(command, "Book matching process")


def run_batch_download() -> bool:
    """
    Run batch download of matched books
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running batch download...")
    
    # Check if download-ready CSV exists
    download_csv = Path("outputs/download_ready_books.csv")
    if not download_csv.exists():
        logger.error(f"Download CSV not found: {download_csv}")
        logger.error("Please run book matching first.")
        return False
    
    # Run the download
    command = f"python ../../standalone_downloader.py --csv {download_csv} --output-dir ../../organized_outputs/epub_downloads"
    return run_command(command, "Batch download of matched books")


def generate_final_report() -> None:
    """
    Generate a final summary report
    """
    logger.info("Generating final report...")
    
    report_path = Path("outputs/final_workflow_report.txt")
    
    try:
        with open(report_path, 'w') as f:
            f.write("Anna Archive Book Matching - Final Workflow Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Check outputs
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                f.write("Generated Files:\n")
                for file_path in outputs_dir.glob("*"):
                    f.write(f"  - {file_path.name}\n")
                f.write("\n")
            
            # Check downloads
            downloads_dir = Path("../../organized_outputs/epub_downloads")
            if downloads_dir.exists():
                epub_files = list(downloads_dir.glob("*.epub"))
                f.write(f"Downloaded Books: {len(epub_files)}\n")
                f.write("Sample downloads:\n")
                for epub_file in epub_files[:10]:  # Show first 10
                    f.write(f"  - {epub_file.name}\n")
                if len(epub_files) > 10:
                    f.write(f"  ... and {len(epub_files) - 10} more\n")
        
        logger.info(f"Final report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating final report: {e}")


def main():
    """
    Main workflow function
    """
    parser = argparse.ArgumentParser(description='Integrated Anna Archive Workflow')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Process only a sample of books (for testing)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip Anna Archive data processing')
    parser.add_argument('--skip-matching', action='store_true',
                       help='Skip book matching (use existing results)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip batch download')
    
    args = parser.parse_args()
    
    logger.info("Starting Anna Archive Integrated Workflow")
    logger.info("=" * 50)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        return 1
    
    # Step 2: Process Anna Archive data
    if not args.skip_processing:
        if not process_anna_archive_data():
            logger.error("Data processing failed. Exiting.")
            return 1
    else:
        logger.info("Skipping data processing (--skip-processing)")
    
    # Step 3: Run book matching
    if not args.skip_matching:
        if not run_book_matching(args.romance_csv, args.sample_size):
            logger.error("Book matching failed. Exiting.")
            return 1
    else:
        logger.info("Skipping book matching (--skip-matching)")
    
    # Step 4: Run batch download
    if not args.skip_download:
        if not run_batch_download():
            logger.error("Batch download failed. Exiting.")
            return 1
    else:
        logger.info("Skipping batch download (--skip-download)")
    
    # Step 5: Generate final report
    generate_final_report()
    
    logger.info("=" * 50)
    logger.info("Integrated workflow completed successfully!")
    logger.info("Check the outputs/ directory for results and reports.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
