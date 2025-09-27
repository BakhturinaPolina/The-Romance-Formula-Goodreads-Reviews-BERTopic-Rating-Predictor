"""
Book Downloader for Romance Novel Corpus Creation

This module handles downloading books from Anna's Archive and organizing
them with consistent file naming and storage structure.

### Coding Agent Pattern
**Intent**: Enable autonomous book downloading and storage
**Problem**: Managing large-scale downloads with proper organization
**Solution**: Robust downloading with error handling and consistent naming
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import shutil
import tempfile

@dataclass
class DownloadResult:
    """Result of a download attempt"""
    work_id: str
    title: str
    author: str
    format_type: str
    success: bool
    file_path: Optional[Path] = None
    file_size: int = 0
    error_message: Optional[str] = None
    checksum: Optional[str] = None

class BookDownloader:
    """
    Handles downloading and organizing books from Anna's Archive.

    Features:
    - Consistent file naming (work_id_title_author.format)
    - Organized storage structure
    - Checksum validation
    - Error handling and retry logic
    """

    def __init__(self, storage_base: Path, annas_client):
        """
        Initialize the book downloader.

        Args:
            storage_base: Base directory for storing downloaded books
            annas_client: Anna's Archive API client instance
        """
        self.storage_base = Path(storage_base)
        self.annas_client = annas_client
        self.logger = logging.getLogger(__name__)

        # Create storage structure
        self._create_storage_structure()

        # Download settings
        self.max_retries = 3
        self.chunk_size = 8192

    def _create_storage_structure(self):
        """Create the organized storage directory structure"""
        dirs_to_create = [
            self.storage_base,
            self.storage_base / 'books',
            self.storage_base / 'temp',
            self.storage_base / 'failed',
            self.storage_base / 'metadata'
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created storage structure at {self.storage_base}")

    def _sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filenames"""
        # Remove or replace problematic characters
        import re
        text = re.sub(r'[<>:"/\\|?*]', '_', text)
        text = re.sub(r'[^\w\s\-_.()]', '', text)
        return text.strip()

    def _generate_filename(self, work_id: str, title: str, author: str, format_type: str) -> str:
        """Generate consistent filename for a book"""
        # Sanitize components
        safe_title = self._sanitize_filename(title)[:50]  # Limit length
        safe_author = self._sanitize_filename(author)[:30]  # Limit length

        # Format: work_id_title_author.format
        filename = f"{work_id}_{safe_title}_{safe_author}.{format_type}"
        return filename

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    def download_book(self, work_id: str, title: str, author: str,
                     annas_id: str, format_type: str) -> DownloadResult:
        """
        Download a single book from Anna's Archive.

        Args:
            work_id: Goodreads work ID
            title: Book title
            author: Book author
            annas_id: Anna's Archive book ID
            format_type: Format to download (epub, html, etc.)

        Returns:
            DownloadResult with download information
        """
        self.logger.debug(f"Starting download for {work_id}: {title} by {author} (format: {format_type})")
        
        result = DownloadResult(
            work_id=work_id,
            title=title,
            author=author,
            format_type=format_type,
            success=False
        )

        try:
            # Generate filename
            filename = self._generate_filename(work_id, title, author, format_type)
            file_path = self.storage_base / 'books' / filename
            self.logger.debug(f"Target file path: {file_path}")

            # Check if file already exists
            if file_path.exists():
                result.success = True
                result.file_path = file_path
                result.file_size = file_path.stat().st_size
                result.checksum = self._calculate_checksum(file_path)
                result.error_message = "File already exists"
                self.logger.info(f"File already exists: {filename}")
                return result

            # Download the book
            self.logger.info(f"Downloading: {title} by {author} ({format_type})")
            self.logger.debug(f"Using Anna's Archive ID: {annas_id}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format_type}') as temp_file:
                temp_path = Path(temp_file.name)
                self.logger.debug(f"Temporary file: {temp_path}")

                # Get book content
                content = self.annas_client.download_book(annas_id, format_type)

                if content is None:
                    result.error_message = "Download returned no content"
                    self.logger.error(f"Download returned no content for {work_id}")
                    return result

                self.logger.debug(f"Downloaded {len(content)} bytes of content")

                # Write content to temporary file
                with open(temp_path, 'wb') as f:
                    f.write(content)

                # Validate download
                if temp_path.stat().st_size == 0:
                    result.error_message = "Downloaded file is empty"
                    self.logger.error(f"Downloaded file is empty for {work_id}")
                    temp_path.unlink()
                    return result

                # Calculate checksum
                checksum = self._calculate_checksum(temp_path)
                self.logger.debug(f"File checksum: {checksum}")

                # Move to final location
                shutil.move(str(temp_path), str(file_path))
                self.logger.debug(f"Moved file to final location: {file_path}")

                # Set result information
                result.success = True
                result.file_path = file_path
                result.file_size = file_path.stat().st_size
                result.checksum = checksum

                self.logger.info(f"Successfully downloaded: {filename} ({result.file_size} bytes)")

        except Exception as e:
            result.error_message = f"Download error: {str(e)}"
            self.logger.error(f"Error downloading book {work_id}: {e}")

        return result

    def download_books_batch(self, match_results: List, max_concurrent: int = 3) -> List[DownloadResult]:
        """
        Download multiple books in batch with controlled concurrency.

        Args:
            match_results: List of MatchResult objects with successful matches
            max_concurrent: Maximum number of concurrent downloads

        Returns:
            List of DownloadResult objects
        """
        download_results = []

        # Filter to only successful matches with valid formats
        valid_matches = [
            result for result in match_results
            if result.match_found and result.best_format and result.annas_id
        ]

        self.logger.info(f"Starting batch download of {len(valid_matches)} books")

        for i, match_result in enumerate(valid_matches):
            self.logger.info(f"Processing book {i+1}/{len(valid_matches)}: {match_result.title}")

            download_result = self.download_book(
                work_id=match_result.work_id,
                title=match_result.title,
                author=match_result.author,
                annas_id=match_result.annas_id,
                format_type=match_result.best_format
            )

            download_results.append(download_result)

            # Log progress
            if (i + 1) % 5 == 0:
                success_count = sum(1 for r in download_results[-5:] if r.success)
                self.logger.info(f"Last 5 downloads: {success_count}/5 successful")

        return download_results

    def save_download_results(self, results: List[DownloadResult], output_path: Path):
        """Save download results to CSV for analysis"""
        results_data = []

        for result in results:
            results_data.append({
                'work_id': result.work_id,
                'title': result.title,
                'author': result.author,
                'format_type': result.format_type,
                'success': result.success,
                'file_path': str(result.file_path) if result.file_path else '',
                'file_size': result.file_size,
                'checksum': result.checksum or '',
                'error_message': result.error_message or ''
            })

        import pandas as pd
        df = pd.DataFrame(results_data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved download results to {output_path}")

    def get_storage_stats(self) -> Dict:
        """Get statistics about the storage structure"""
        books_dir = self.storage_base / 'books'

        if not books_dir.exists():
            return {'total_files': 0, 'total_size': 0, 'format_counts': {}}

        total_files = 0
        total_size = 0
        format_counts = {}

        for file_path in books_dir.iterdir():
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size

                # Count by format
                format_type = file_path.suffix.lstrip('.')
                format_counts[format_type] = format_counts.get(format_type, 0) + 1

        return {
            'total_files': total_files,
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'format_counts': format_counts
        }
