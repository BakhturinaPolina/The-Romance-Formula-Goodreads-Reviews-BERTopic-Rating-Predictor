"""
Main Corpus Creation Pipeline

This module orchestrates the entire corpus creation process from Goodreads
metadata to downloaded books from Anna's Archive.

### Coding Agent Pattern
**Intent**: Enable autonomous corpus creation workflow
**Problem**: Complex multi-step process requiring coordination
**Solution**: Centralized pipeline with proper error handling and logging
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime

from .book_matcher import BookMatcher, MatchResult
from .annas_client import AnnasArchiveClient
from .downloader import BookDownloader, DownloadResult
from .logging_config import configure_logging

@dataclass
class PipelineConfig:
    """Configuration for the corpus creation pipeline"""
    api_key: str
    storage_base: Path
    min_confidence: float = 0.7
    batch_size: int = 50
    max_retries: int = 3

@dataclass
class PipelineResult:
    """Results from running the corpus creation pipeline"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_books_processed: int = 0
    matches_found: int = 0
    downloads_successful: int = 0
    match_results: List[MatchResult] = None
    download_results: List[DownloadResult] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.match_results is None:
            self.match_results = []
        if self.download_results is None:
            self.download_results = []
        if self.errors is None:
            self.errors = []

class CorpusCreationPipeline:
    """
    Main pipeline for creating romance novel corpus from Anna's Archive.

    Coordinates the entire process:
    1. Book matching between Goodreads and Anna's Archive
    2. Downloading matched books
    3. Organizing and validating downloads
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the corpus creation pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            configure_logging()

        # Initialize components
        self.annas_client = AnnasArchiveClient(config.api_key)
        self.book_matcher = BookMatcher(self.annas_client)
        self.downloader = BookDownloader(config.storage_base, self.annas_client)

        # Set matching confidence threshold
        self.book_matcher.min_confidence = config.min_confidence

    def validate_setup(self) -> bool:
        """Validate that the pipeline is properly configured"""
        self.logger.info("Validating pipeline setup...")

        # Check API key
        if not self.annas_client.validate_api_key():
            self.logger.error("Invalid API key")
            return False

        # Check storage directory
        if not self.config.storage_base.exists():
            self.logger.error(f"Storage directory does not exist: {self.config.storage_base}")
            return False

        # Check write permissions
        try:
            test_file = self.config.storage_base / 'test_write.txt'
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            self.logger.error(f"Cannot write to storage directory: {e}")
            return False

        self.logger.info("Pipeline setup validated successfully")
        return True

    def run_matching(self, books_df: pd.DataFrame) -> List[MatchResult]:
        """
        Run the book matching phase.

        Args:
            books_df: DataFrame with Goodreads book data

        Returns:
            List of MatchResult objects
        """
        self.logger.info(f"Starting book matching for {len(books_df)} books...")

        # Validate required columns
        required_columns = ['work_id', 'title', 'author_name', 'publication_year']
        missing_columns = [col for col in required_columns if col not in books_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Run matching in batches for memory efficiency
        all_results = []

        for i in range(0, len(books_df), self.config.batch_size):
            batch = books_df.iloc[i:i + self.config.batch_size]
            self.logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(books_df)-1)//self.config.batch_size + 1}")

            batch_results = self.book_matcher.match_books_batch(batch)
            all_results.extend(batch_results)

        self.logger.info(f"Matching completed. Found {sum(1 for r in all_results if r.match_found)}/{len(all_results)} matches")
        return all_results

    def run_downloads(self, match_results: List[MatchResult]) -> List[DownloadResult]:
        """
        Run the download phase for successfully matched books.

        Args:
            match_results: Results from the matching phase

        Returns:
            List of DownloadResult objects
        """
        # Filter to successful matches only
        successful_matches = [
            result for result in match_results
            if result.match_found and result.best_format
        ]

        if not successful_matches:
            self.logger.warning("No successful matches to download")
            return []

        self.logger.info(f"Starting downloads for {len(successful_matches)} matched books...")

        download_results = self.downloader.download_books_batch(successful_matches)

        successful_downloads = sum(1 for result in download_results if result.success)
        self.logger.info(f"Downloads completed. {successful_downloads}/{len(download_results)} successful")

        return download_results

    def run_full_pipeline(self, books_df: pd.DataFrame) -> PipelineResult:
        """
        Run the complete corpus creation pipeline.

        Args:
            books_df: DataFrame with Goodreads book data

        Returns:
            PipelineResult with complete results
        """
        result = PipelineResult(
            start_time=datetime.now(),
            total_books_processed=len(books_df)
        )

        try:
            # Phase 1: Matching
            self.logger.info("=== Phase 1: Book Matching ===")
            match_results = self.run_matching(books_df)
            result.match_results = match_results
            result.matches_found = sum(1 for r in match_results if r.match_found)

            # Save match results
            match_output = self.config.storage_base / 'metadata' / 'match_results.csv'
            self.book_matcher.save_match_results(match_results, match_output)

            # Phase 2: Downloading
            self.logger.info("=== Phase 2: Book Downloads ===")
            download_results = self.run_downloads(match_results)
            result.download_results = download_results
            result.downloads_successful = sum(1 for r in download_results if r.success)

            # Save download results
            download_output = self.config.storage_base / 'metadata' / 'download_results.csv'
            self.downloader.save_download_results(download_results, download_output)

            # Phase 3: Statistics
            self.logger.info("=== Phase 3: Statistics ===")
            storage_stats = self.downloader.get_storage_stats()
            match_stats = self.book_matcher.get_match_statistics(match_results)

            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Books processed: {result.total_books_processed}")
            match_rate = result.matches_found / result.total_books_processed if result.total_books_processed > 0 else 0
            self.logger.info(f"Matches found: {result.matches_found} ({match_rate:.1%})")

            download_rate = result.downloads_successful / max(len(download_results), 1) if download_results else 0
            self.logger.info(f"Downloads successful: {result.downloads_successful} ({download_rate:.1%})")
            self.logger.info(f"Storage: {storage_stats['total_files']} files ({storage_stats['total_size_mb']:.1f} MB)")

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        finally:
            result.end_time = datetime.now()

        return result

    def save_pipeline_report(self, result: PipelineResult, output_path: Optional[Path] = None):
        """Save a comprehensive pipeline report"""
        if output_path is None:
            output_path = self.config.storage_base / 'metadata' / 'pipeline_report.txt'

        with open(output_path, 'w') as f:
            f.write("Romance Novel Corpus Creation Pipeline Report\\n")
            f.write("=" * 50 + "\\n\\n")

            f.write(f"Start Time: {result.start_time}\\n")
            f.write(f"End Time: {result.end_time}\\n")
            f.write(f"Duration: {result.end_time - result.start_time if result.end_time else 'N/A'}\\n\\n")

            f.write("Results Summary:\\n")
            f.write(f"- Books processed: {result.total_books_processed}\\n")
            f.write(f"- Matches found: {result.matches_found}\\n")
            f.write(f"- Downloads successful: {result.downloads_successful}\\n\\n")

            if result.match_results:
                match_stats = self.book_matcher.get_match_statistics(result.match_results)
                f.write("Matching Statistics:\\n")
                f.write(f"- Success rate: {match_stats['success_rate']:.1%}\\n")
                f.write(f"- Average confidence: {match_stats['avg_confidence']:.3f}\\n")
                f.write(f"- Format distribution: {match_stats['format_distribution']}\\n\\n")

            if result.errors:
                f.write("Errors:\\n")
                for error in result.errors:
                    f.write(f"- {error}\\n")

        self.logger.info(f"Pipeline report saved to {output_path}")
