#!/usr/bin/env python3
"""
Setup script for Anna's Archive datasets
Downloads and processes the datasets for book matching
"""

import os
import sys
import logging
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        chunk_size: Chunk size for download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def setup_anna_archive_datasets(data_dir: str = "data", 
                               download_datasets: bool = True) -> None:
    """
    Set up Anna's Archive datasets
    
    Args:
        data_dir: Path to data directory
        download_datasets: Whether to download datasets (requires manual download)
    """
    data_path = Path(data_dir)
    
    # Create directory structure
    directories = [
        "elasticsearch", "elasticsearchF",
        "aac", "aacF", 
        "mariadb", "mariadbF"
    ]
    
    for directory in directories:
        (data_path / directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {data_path}")
    
    if download_datasets:
        logger.info("Dataset download information:")
        logger.info("=" * 50)
        logger.info("Anna's Archive datasets are available at:")
        logger.info("https://annas-archive.li/datasets")
        logger.info("")
        logger.info("Required datasets:")
        logger.info("1. Elasticsearch dataset (aarecord_elasticsearch)")
        logger.info("2. AAC dataset (aarecord_aac)")
        logger.info("3. MariaDB dataset (aarecord_mariadb)")
        logger.info("")
        logger.info("Please download the datasets manually and place them in:")
        logger.info(f"  - {data_path / 'elasticsearch'}/ (for .gz files)")
        logger.info(f"  - {data_path / 'aac'}/ (for .zst files)")
        logger.info(f"  - {data_path / 'mariadb'}/ (for .gz files)")
        logger.info("")
        logger.info("Then run: python run_matcher.py --process-data")


def create_sample_config() -> None:
    """
    Create a sample configuration file
    """
    config_content = """# Anna Archive Book Matcher Configuration

# Data paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# Matching parameters
SIMILARITY_THRESHOLD = 0.8
SAMPLE_SIZE = None  # Set to integer for testing with sample

# DuckDB configuration
DUCKDB_MEMORY_LIMIT = "28GB"
DUCKDB_THREADS = 4

# Processing parameters
CHUNK_SIZE = 10485760  # 10MB
BATCH_SIZE = 10000

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "anna_archive_matcher.log"
"""
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    logger.info("Created sample configuration file: config.py")


def main():
    """
    Main setup function
    """
    parser = argparse.ArgumentParser(description='Setup Anna Archive datasets')
    parser.add_argument('--data-dir', default='data',
                       help='Path to data directory')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip download instructions')
    
    args = parser.parse_args()
    
    logger.info("Setting up Anna Archive Book Matcher...")
    
    # Set up directories
    setup_anna_archive_datasets(args.data_dir, not args.no_download)
    
    # Create sample config
    create_sample_config()
    
    logger.info("Setup completed!")
    logger.info("Next steps:")
    logger.info("1. Download Anna Archive datasets (see instructions above)")
    logger.info("2. Run: python run_matcher.py --process-data --romance-csv <your_csv>")
    logger.info("3. Use the generated download_ready_books.csv with your download system")


if __name__ == "__main__":
    main()
