#!/usr/bin/env python3
"""
Test Script for Free Romance Novel Corpus Creation Pipeline

This script tests the free corpus creation pipeline using Anna's Archive
torrent datasets and the annas-mcp tool, eliminating the need for API keys.

Usage:
    python test_free_pipeline.py --storage-dir ./free_corpus
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from .free_pipeline import FreeCorpusCreationPipeline, FreePipelineConfig
from .logging_config import configure_logging

# Try to import optional clients (archived modules)
try:
    from .annas_torrent_client import AnnasTorrentClient
    TORRENT_AVAILABLE = True
except ImportError:
    AnnasTorrentClient = None
    TORRENT_AVAILABLE = False

try:
    from .annas_mcp_client import AnnasMCPClient
    MCP_AVAILABLE = True
except ImportError:
    AnnasMCPClient = None
    MCP_AVAILABLE = False

def setup_logging(level=logging.INFO):
    """Setup logging configuration using the centralized config"""
    configure_logging(
        log_level="DEBUG",
        log_filename="free_corpus_creation_test.log",
        console_level=level
    )

def load_test_books(test_books_data: List[Dict]) -> pd.DataFrame:
    """Load test books from the provided data"""
    return pd.DataFrame(test_books_data)

def main():
    parser = argparse.ArgumentParser(description='Test Free Romance Novel Corpus Creation Pipeline')
    parser.add_argument('--storage-dir', default='./free_corpus_storage',
                       help='Directory to store downloaded books')
    parser.add_argument('--torrent-dir', default='./torrent_data',
                       help='Directory for torrent datasets')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--use-torrents', action='store_true', default=True,
                       help='Use torrent datasets for matching')
    parser.add_argument('--use-mcp', action='store_true', default=True,
                       help='Use annas-mcp tool for matching')

    args = parser.parse_args()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    logger = logging.getLogger(__name__)
    logger.info("Starting free corpus creation pipeline test")

    # Test books data (the 6 diverse books we selected)
    test_books_data = [
        {
            'work_id': '846763',
            'title': 'The Duke and I',
            'author_name': 'Julia Quinn',
            'publication_year': 2000
        },
        {
            'work_id': '859012',
            'title': 'Marriage Most Scandalous',
            'author_name': 'Johanna Lindsey',
            'publication_year': 2005
        },
        {
            'work_id': '6995719',
            'title': 'Poor Little Bitch Girl',
            'author_name': 'Jackie Collins',
            'publication_year': 2010
        },
        {
            'work_id': '43343675',
            'title': 'The Vigilante\'s Lover',
            'author_name': 'Annie Winters',
            'publication_year': 2015
        },
        {
            'work_id': '18968802',
            'title': 'Motorcycle Man',
            'author_name': 'Kristen Ashley',
            'publication_year': 2012
        },
        {
            'work_id': '25223025',
            'title': 'True',
            'author_name': 'Laurann Dohner',
            'publication_year': 2013
        }
    ]

    try:
        # Load test books
        books_df = load_test_books(test_books_data)
        logger.info(f"Loaded {len(books_df)} test books")

        # Create storage directories
        storage_dir = Path(args.storage_dir)
        torrent_dir = Path(args.torrent_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        torrent_dir.mkdir(parents=True, exist_ok=True)

        # Initialize free pipeline
        config = FreePipelineConfig(
            storage_base=storage_dir,
            torrent_base=torrent_dir,
            min_confidence=0.7,
            batch_size=10,
            use_torrents=args.use_torrents,
            use_mcp=args.use_mcp,
            preferred_datasets=['libgen_li', 'z_library', 'internet_archive']
        )

        pipeline = FreeCorpusCreationPipeline(config)

        # Validate setup
        if not pipeline.validate_setup():
            logger.error("Free pipeline setup validation failed")
            return 1

        # Test individual components
        logger.info("=== Testing Individual Components ===")

        # Test torrent client
        if args.use_torrents and TORRENT_AVAILABLE:
            logger.info("Testing torrent client...")
            if pipeline.torrent_client:
                datasets = pipeline.torrent_client.get_available_datasets()
                logger.info(f"Available datasets: {list(datasets.keys())}")

                # Test search in one dataset
                test_matches = pipeline.torrent_client.search_books_in_dataset(
                    'libgen_li', 'The Duke and I', 'Julia Quinn', 2000
                )
                logger.info(f"Test search found {len(test_matches)} matches")
            else:
                logger.warning("Torrent client not available")
        elif args.use_torrents:
            logger.warning("Torrent client module not available (archived)")

        # Test MCP client
        if args.use_mcp and MCP_AVAILABLE:
            logger.info("Testing MCP client...")
            if pipeline.mcp_client and pipeline.mcp_client.validate_installation():
                logger.info("MCP client validated successfully")
                
                # Test search
                test_results = pipeline.mcp_client.search_books('romance novel', limit=3)
                logger.info(f"MCP test search found {len(test_results)} results")
            else:
                logger.warning("MCP client not available - install annas-mcp tool")
        elif args.use_mcp:
            logger.warning("MCP client module not available (archived)")

        # Run full free pipeline
        logger.info("=== Running Full Free Pipeline ===")
        result = pipeline.run_free_pipeline(books_df)

        # Save comprehensive report
        pipeline.save_free_pipeline_report(result)

        # Print summary
        print("\\n" + "=" * 60)
        print("FREE CORPUS CREATION PIPELINE TEST RESULTS")
        print("=" * 60)
        print(f"Books processed: {result.total_books_processed}")
        print(f"Matches found: {result.matches_found}")
        print(f"Downloads successful: {result.downloads_successful}")

        print(f"\\nSource Breakdown:")
        print(f"- Torrent matches: {len(result.torrent_matches)}")
        print(f"- MCP matches: {len(result.mcp_matches)}")

        if result.errors:
            print(f"\\nErrors encountered: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")

        print(f"\\nDetailed results saved to: {storage_dir}/metadata/")
        print("Check free_corpus_creation_test.log for detailed execution logs")

        # Show next steps
        print("\\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Install annas-mcp tool for better results:")
        print("   - Download from: https://github.com/iosifache/annas-mcp/releases")
        print("   - Set ANNAS_DOWNLOAD_PATH environment variable")
        print()
        print("2. Download torrent datasets for bulk access:")
        print("   - Libgen.li: 188TB of fiction books")
        print("   - Z-Library: 75TB of academic and fiction books")
        print("   - Internet Archive: 304TB of digitized books")
        print()
        print("3. Scale to full 6000-book dataset:")
        print("   - Use the same pipeline with your full romance_subdataset_6000.csv")
        print("   - Monitor storage space and download progress")

        return 0 if result.downloads_successful > 0 else 1

    except Exception as e:
        logger.error(f"Free pipeline test failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
