"""
Free Corpus Creation Pipeline

This module provides a free alternative to the API-based pipeline by using
Anna's Archive torrent datasets and the annas-mcp tool.

Based on:
- Anna's Archive datasets: https://annas-archive.li/datasets
- annas-mcp tool: https://github.com/iosifache/annas-mcp
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime

from .logging_config import configure_logging

# Try to import optional clients (archived modules)
try:
    from .annas_torrent_client import AnnasTorrentClient, BookMatch
    TORRENT_AVAILABLE = True
except ImportError:
    AnnasTorrentClient = None
    BookMatch = None
    TORRENT_AVAILABLE = False

try:
    from .annas_mcp_client import AnnasMCPClient, MCPBookResult
    MCP_AVAILABLE = True
except ImportError:
    AnnasMCPClient = None
    MCPBookResult = None
    MCP_AVAILABLE = False

from .book_matcher import BookMatcher
from .downloader import BookDownloader

@dataclass
class FreePipelineConfig:
    """Configuration for the free corpus creation pipeline"""
    storage_base: Path
    torrent_base: Path
    min_confidence: float = 0.7
    batch_size: int = 50
    use_torrents: bool = True
    use_mcp: bool = True
    preferred_datasets: List[str] = None

    def __post_init__(self):
        if self.preferred_datasets is None:
            self.preferred_datasets = ['libgen_li', 'z_library', 'internet_archive']

@dataclass
class FreePipelineResult:
    """Results from running the free corpus creation pipeline"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_books_processed: int = 0
    matches_found: int = 0
    downloads_successful: int = 0
    torrent_matches: List[BookMatch] = None
    mcp_matches: List[MCPBookResult] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.torrent_matches is None:
            self.torrent_matches = []
        if self.mcp_matches is None:
            self.mcp_matches = []
        if self.errors is None:
            self.errors = []

class FreeCorpusCreationPipeline:
    """
    Free corpus creation pipeline using Anna's Archive torrent datasets
    and the annas-mcp tool.

    This approach eliminates the need for API keys while providing access
    to the massive collection of books available through free methods.
    """

    def __init__(self, config: FreePipelineConfig):
        """
        Initialize the free corpus creation pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            configure_logging()

        # Initialize components (with fallback for missing modules)
        if TORRENT_AVAILABLE and config.use_torrents:
            self.torrent_client = AnnasTorrentClient(config.torrent_base)
        else:
            self.torrent_client = None
            if config.use_torrents:
                self.logger.warning("Torrent client not available - torrent features disabled")

        if MCP_AVAILABLE and config.use_mcp:
            self.mcp_client = AnnasMCPClient()
        else:
            self.mcp_client = None
            if config.use_mcp:
                self.logger.warning("MCP client not available - MCP features disabled")

        self.downloader = BookDownloader(config.storage_base, None)

        # Create storage structure
        self._create_storage_structure()

    def _create_storage_structure(self):
        """Create the organized storage directory structure"""
        dirs_to_create = [
            self.config.storage_base,
            self.config.storage_base / 'books',
            self.config.storage_base / 'torrents',
            self.config.storage_base / 'mcp_downloads',
            self.config.storage_base / 'metadata',
            self.config.storage_base / 'temp'
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created storage structure at {self.config.storage_base}")

    def validate_setup(self) -> bool:
        """Validate that the pipeline is properly configured"""
        self.logger.info("Validating free pipeline setup...")

        # Check storage directory
        if not self.config.storage_base.exists():
            self.logger.error(f"Storage directory does not exist: {self.config.storage_base}")
            return False

        # Check torrent client
        if self.config.use_torrents and self.torrent_client:
            datasets = self.torrent_client.get_available_datasets()
            if not datasets:
                self.logger.warning("No torrent datasets available")
            else:
                self.logger.info(f"Available torrent datasets: {list(datasets.keys())}")
        elif self.config.use_torrents:
            self.logger.warning("Torrent client not available")

        # Check MCP client
        if self.config.use_mcp and self.mcp_client:
            if not self.mcp_client.validate_installation():
                self.logger.warning("annas-mcp tool not available")
            else:
                self.logger.info("annas-mcp tool validated")
        elif self.config.use_mcp:
            self.logger.warning("MCP client not available")

        self.logger.info("Free pipeline setup validated")
        return True

    def search_torrent_datasets(self, title: str, author: str, year: Optional[int] = None) -> List:
        """
        Search for books in torrent datasets.

        Args:
            title: Book title
            author: Book author
            year: Publication year (optional)

        Returns:
            List of matching BookMatch objects (empty if torrent client unavailable)
        """
        if not self.torrent_client:
            self.logger.debug("Torrent client not available, returning empty results")
            return []

        all_matches = []

        for dataset_name in self.config.preferred_datasets:
            try:
                matches = self.torrent_client.search_books_in_dataset(
                    dataset_name, title, author, year
                )
                all_matches.extend(matches)
                self.logger.info(f"Found {len(matches)} matches in {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error searching {dataset_name}: {e}")

        return all_matches

    def search_mcp_tool(self, title: str, author: str, year: Optional[int] = None) -> List:
        """
        Search for books using the annas-mcp tool.

        Args:
            title: Book title
            author: Book author
            year: Publication year (optional)

        Returns:
            List of matching MCPBookResult objects (empty if MCP client unavailable)
        """
        if not self.config.use_mcp or not self.mcp_client:
            self.logger.debug("MCP client not available, returning empty results")
            return []

        # Create search query
        query = f'"{title}" "{author}"'
        if year:
            query += f' {year}'

        try:
            results = self.mcp_client.search_books(query, limit=10)
            self.logger.info(f"Found {len(results)} matches via annas-mcp")
            return results
        except Exception as e:
            self.logger.error(f"Error searching with annas-mcp: {e}")
            return []

    def match_books_free(self, books_df: pd.DataFrame) -> FreePipelineResult:
        """
        Match books using free methods (torrents + MCP).

        Args:
            books_df: DataFrame with Goodreads book data

        Returns:
            FreePipelineResult with matching information
        """
        result = FreePipelineResult(
            start_time=datetime.now(),
            total_books_processed=len(books_df)
        )

        self.logger.info(f"Starting free book matching for {len(books_df)} books...")

        # Validate required columns
        required_columns = ['work_id', 'title', 'author_name', 'publication_year']
        missing_columns = [col for col in required_columns if col not in books_df.columns]

        if missing_columns:
            result.errors.append(f"Missing required columns: {missing_columns}")
            return result

        # Process books in batches
        for i in range(0, len(books_df), self.config.batch_size):
            batch = books_df.iloc[i:i + self.config.batch_size]
            self.logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(books_df)-1)//self.config.batch_size + 1}")

            for _, book in batch.iterrows():
                try:
                    # Search torrent datasets
                    if self.config.use_torrents:
                        torrent_matches = self.search_torrent_datasets(
                            book['title'], book['author_name'], book['publication_year']
                        )
                        result.torrent_matches.extend(torrent_matches)

                    # Search MCP tool
                    if self.config.use_mcp:
                        mcp_matches = self.search_mcp_tool(
                            book['title'], book['author_name'], book['publication_year']
                        )
                        result.mcp_matches.extend(mcp_matches)

                    result.matches_found += len(torrent_matches) + len(mcp_matches)

                except Exception as e:
                    error_msg = f"Error processing book {book['work_id']}: {e}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)

        self.logger.info(f"Free matching completed. Found {result.matches_found} total matches")
        return result

    def download_matched_books(self, result: FreePipelineResult) -> FreePipelineResult:
        """
        Download matched books from both sources.

        Args:
            result: FreePipelineResult with matching information

        Returns:
            Updated FreePipelineResult with download information
        """
        self.logger.info("Starting downloads from matched sources...")

        # Download from torrent matches
        if self.torrent_client:
            for match in result.torrent_matches:
                try:
                    success = self.torrent_client.download_book(
                        match, self.config.storage_base / 'books'
                    )
                    if success:
                        result.downloads_successful += 1
                except Exception as e:
                    error_msg = f"Error downloading torrent book {match.title}: {e}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)
        else:
            self.logger.debug("Skipping torrent downloads - client not available")

        # Download from MCP matches
        if self.mcp_client:
            for match in result.mcp_matches:
                try:
                    downloaded_file = self.mcp_client.download_book(match.id, 'epub')
                    if downloaded_file:
                        # Move to organized location
                        final_path = self.config.storage_base / 'books' / downloaded_file.name
                        downloaded_file.rename(final_path)
                        result.downloads_successful += 1
                except Exception as e:
                    error_msg = f"Error downloading MCP book {match.title}: {e}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)
        else:
            self.logger.debug("Skipping MCP downloads - client not available")

        self.logger.info(f"Downloads completed. {result.downloads_successful} successful")
        return result

    def run_free_pipeline(self, books_df: pd.DataFrame) -> FreePipelineResult:
        """
        Run the complete free corpus creation pipeline.

        Args:
            books_df: DataFrame with Goodreads book data

        Returns:
            FreePipelineResult with complete results
        """
        result = FreePipelineResult(
            start_time=datetime.now(),
            total_books_processed=len(books_df)
        )

        try:
            # Phase 1: Matching
            self.logger.info("=== Phase 1: Free Book Matching ===")
            result = self.match_books_free(books_df)

            # Phase 2: Downloading
            self.logger.info("=== Phase 2: Free Book Downloads ===")
            result = self.download_matched_books(result)

            # Phase 3: Statistics
            self.logger.info("=== Phase 3: Statistics ===")
            storage_stats = self.downloader.get_storage_stats()

            self.logger.info("Free pipeline completed successfully!")
            self.logger.info(f"Books processed: {result.total_books_processed}")
            self.logger.info(f"Matches found: {result.matches_found}")
            self.logger.info(f"Downloads successful: {result.downloads_successful}")
            self.logger.info(f"Storage: {storage_stats['total_files']} files ({storage_stats['total_size_mb']:.1f} MB)")

        except Exception as e:
            error_msg = f"Free pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        finally:
            result.end_time = datetime.now()

        return result

    def save_free_pipeline_report(self, result: FreePipelineResult, output_path: Optional[Path] = None):
        """Save a comprehensive free pipeline report"""
        if output_path is None:
            output_path = self.config.storage_base / 'metadata' / 'free_pipeline_report.txt'

        with open(output_path, 'w') as f:
            f.write("Free Romance Novel Corpus Creation Pipeline Report\\n")
            f.write("=" * 60 + "\\n\\n")

            f.write(f"Start Time: {result.start_time}\\n")
            f.write(f"End Time: {result.end_time}\\n")
            f.write(f"Duration: {result.end_time - result.start_time if result.end_time else 'N/A'}\\n\\n")

            f.write("Results Summary:\\n")
            f.write(f"- Books processed: {result.total_books_processed}\\n")
            f.write(f"- Matches found: {result.matches_found}\\n")
            f.write(f"- Downloads successful: {result.downloads_successful}\\n\\n")

            f.write("Source Breakdown:\\n")
            f.write(f"- Torrent matches: {len(result.torrent_matches)}\\n")
            f.write(f"- MCP matches: {len(result.mcp_matches)}\\n\\n")

            if result.errors:
                f.write("Errors:\\n")
                for error in result.errors:
                    f.write(f"- {error}\\n")

        self.logger.info(f"Free pipeline report saved to {output_path}")
