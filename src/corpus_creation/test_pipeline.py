#!/usr/bin/env python3
"""
Test Script for Romance Novel Corpus Creation Pipeline

This script tests the corpus creation pipeline with a small set of books
to validate functionality before scaling to larger datasets.

Usage:
    python test_pipeline.py --api-key YOUR_API_KEY
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from .pipeline import CorpusCreationPipeline, PipelineConfig
from .book_matcher import BookMatcher
from .downloader import BookDownloader
from .logging_config import configure_logging

def setup_logging(level=logging.INFO):
    """Setup logging configuration using the centralized config"""
    configure_logging(
        log_level="DEBUG",
        log_filename="corpus_creation_test.log",
        console_level=level
    )

def load_test_books(test_books_data: List[Dict]) -> pd.DataFrame:
    """Load test books from the provided data"""
    return pd.DataFrame(test_books_data)

def main():
    parser = argparse.ArgumentParser(description='Test Romance Novel Corpus Creation Pipeline')
    parser.add_argument('--api-key', required=True, help='Anna\'s Archive API key')
    parser.add_argument('--storage-dir', default='./corpus_storage',
                       help='Directory to store downloaded books')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    logger = logging.getLogger(__name__)
    logger.info("Starting corpus creation pipeline test")

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

        # Create storage directory
        storage_dir = Path(args.storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline
        config = PipelineConfig(
            api_key=args.api_key,
            storage_base=storage_dir,
            min_confidence=0.7,
            batch_size=10
        )

        pipeline = CorpusCreationPipeline(config)

        # Validate setup
        if not pipeline.validate_setup():
            logger.error("Pipeline setup validation failed")
            return 1

        # Run full pipeline
        logger.info("Running full corpus creation pipeline...")
        result = pipeline.run_full_pipeline(books_df)

        # Save comprehensive report
        pipeline.save_pipeline_report(result)

        # Print summary
        print("\\n" + "=" * 60)
        print("CORPUS CREATION PIPELINE TEST RESULTS")
        print("=" * 60)
        print(f"Books processed: {result.total_books_processed}")
        print(f"Matches found: {result.matches_found}")
        print(f"Downloads successful: {result.downloads_successful}")

        if result.match_results:
            match_stats = pipeline.book_matcher.get_match_statistics(result.match_results)
            print(f"Match success rate: {match_stats['success_rate']".1%"}")
            print(f"Average confidence: {match_stats['avg_confidence']".3f"}")

        if result.download_results:
            download_success_rate = result.downloads_successful / len(result.download_results)
            print(f"Download success rate: {download_success_rate".1%"}")

        storage_stats = pipeline.downloader.get_storage_stats()
        print(f"Files downloaded: {storage_stats['total_files']}")
        print(f"Total size: {storage_stats['total_size_mb']".1f"} MB")

        if result.errors:
            print(f"\\nErrors encountered: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")

        print(f"\\nDetailed results saved to: {storage_dir}/metadata/")
        print("Check corpus_creation_test.log for detailed execution logs")

        return 0 if result.downloads_successful > 0 else 1

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
