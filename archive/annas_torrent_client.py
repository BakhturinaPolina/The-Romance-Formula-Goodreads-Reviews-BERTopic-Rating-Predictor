"""
Anna's Archive Torrent Client for Free Dataset Access

This module provides access to Anna's Archive free datasets via torrents,
eliminating the need for API keys while still providing comprehensive
book access for corpus creation.

Based on Anna's Archive datasets page: https://annas-archive.li/datasets
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import sqlite3
import hashlib

@dataclass
class TorrentDataset:
    """Represents a torrent dataset from Anna's Archive"""
    name: str
    size: str
    description: str
    torrent_url: Optional[str] = None
    metadata_url: Optional[str] = None
    local_path: Optional[Path] = None

@dataclass
class BookMatch:
    """Represents a book found in torrent datasets"""
    title: str
    author: str
    year: Optional[int]
    format: str
    file_path: Path
    size: int
    source: str
    md5_hash: Optional[str] = None

class AnnasTorrentClient:
    """
    Client for accessing Anna's Archive free datasets via torrents.

    This approach eliminates the need for API keys while providing access
    to the massive collection of books available through torrent downloads.
    """

    def __init__(self, torrent_base_dir: Path, temp_dir: Optional[Path] = None):
        """
        Initialize the torrent client.

        Args:
            torrent_base_dir: Base directory for storing torrent downloads
            temp_dir: Temporary directory for processing (optional)
        """
        self.torrent_base_dir = Path(torrent_base_dir)
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / 'annas_torrent'
        self.logger = logging.getLogger(__name__)

        # Create directories
        self.torrent_base_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Available datasets (from Anna's Archive datasets page)
        self.datasets = {
            'libgen_li': TorrentDataset(
                name='Libgen.li',
                size='188TB',
                description='Collaboration with Anna\'s Archive - fiction and non-fiction books'
            ),
            'libgen_rs': TorrentDataset(
                name='Libgen.rs',
                size='82TB',
                description='Mirrored by Anna\'s Archive - scientific and technical books'
            ),
            'z_library': TorrentDataset(
                name='Z-Library',
                size='75TB',
                description='Collaboration with Anna\'s Archive - academic and fiction books'
            ),
            'internet_archive': TorrentDataset(
                name='Internet Archive',
                size='304TB',
                description='Scraped by Anna\'s Archive - public domain and digitized books'
            ),
            'hathitrust': TorrentDataset(
                name='HathiTrust',
                size='9TB',
                description='Scraped by Anna\'s Archive - academic and research books'
            )
        }

        # Check for torrent client
        self.torrent_client = self._find_torrent_client()

    def _find_torrent_client(self) -> Optional[str]:
        """Find available torrent client on the system"""
        clients = ['transmission-cli', 'aria2c', 'wget', 'curl']
        
        for client in clients:
            try:
                subprocess.run([client, '--version'], 
                             capture_output=True, check=True)
                self.logger.info(f"Found torrent client: {client}")
                return client
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        self.logger.warning("No torrent client found. Install transmission-cli or aria2c")
        return None

    def get_available_datasets(self) -> Dict[str, TorrentDataset]:
        """Get information about available datasets"""
        return self.datasets.copy()

    def download_dataset_metadata(self, dataset_name: str) -> bool:
        """
        Download metadata for a specific dataset.

        Args:
            dataset_name: Name of the dataset to download metadata for

        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return False

        dataset = self.datasets[dataset_name]
        self.logger.info(f"Downloading metadata for {dataset.name}")

        # For now, we'll create a placeholder metadata structure
        # In a real implementation, you would download the actual metadata
        metadata = {
            'dataset': dataset.name,
            'size': dataset.size,
            'description': dataset.description,
            'books': []  # This would be populated with actual book metadata
        }

        metadata_file = self.torrent_base_dir / f'{dataset_name}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to {metadata_file}")
        return True

    def search_books_in_dataset(self, dataset_name: str, title: str, 
                               author: str, year: Optional[int] = None) -> List[BookMatch]:
        """
        Search for books in a specific dataset.

        Args:
            dataset_name: Name of the dataset to search
            title: Book title to search for
            author: Book author to search for
            year: Publication year (optional)

        Returns:
            List of matching BookMatch objects
        """
        if dataset_name not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return []

        # For demonstration, we'll create mock results
        # In a real implementation, this would search the actual dataset
        matches = []

        # Mock search results based on the dataset
        if dataset_name == 'libgen_li':
            # Libgen.li typically has fiction books
            if 'romance' in title.lower() or 'love' in title.lower():
                matches.append(BookMatch(
                    title=title,
                    author=author,
                    year=year,
                    format='epub',
                    file_path=Path(f'/mock/path/{title.replace(" ", "_")}.epub'),
                    size=1024000,  # 1MB
                    source='libgen_li',
                    md5_hash='mock_hash_123'
                ))

        elif dataset_name == 'z_library':
            # Z-Library has academic and fiction books
            matches.append(BookMatch(
                title=title,
                author=author,
                year=year,
                format='pdf',
                file_path=Path(f'/mock/path/{title.replace(" ", "_")}.pdf'),
                size=2048000,  # 2MB
                source='z_library',
                md5_hash='mock_hash_456'
            ))

        return matches

    def download_book(self, book_match: BookMatch, output_dir: Path) -> bool:
        """
        Download a specific book from the dataset.

        Args:
            book_match: BookMatch object with book information
            output_dir: Directory to save the downloaded book

        Returns:
            True if successful, False otherwise
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # For demonstration, we'll create a mock download
        # In a real implementation, this would download the actual file
        output_file = output_dir / f"{book_match.title.replace(' ', '_')}.{book_match.format}"

        # Create a mock file
        with open(output_file, 'w') as f:
            f.write(f"Mock content for {book_match.title} by {book_match.author}")

        self.logger.info(f"Mock download completed: {output_file}")
        return True

    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """Get statistics about a specific dataset"""
        if dataset_name not in self.datasets:
            return {}

        dataset = self.datasets[dataset_name]
        
        # Mock statistics - in real implementation, these would be calculated
        return {
            'name': dataset.name,
            'size': dataset.size,
            'estimated_books': 1000000,  # Mock number
            'formats': ['epub', 'pdf', 'mobi', 'txt'],
            'last_updated': '2024-01-01',
            'download_status': 'available'
        }

    def create_local_index(self, dataset_name: str) -> bool:
        """
        Create a local searchable index for a dataset.

        Args:
            dataset_name: Name of the dataset to index

        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.datasets:
            return False

        # Create SQLite database for local indexing
        db_path = self.torrent_base_dir / f'{dataset_name}_index.db'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create books table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                year INTEGER,
                format TEXT,
                file_path TEXT,
                size INTEGER,
                md5_hash TEXT,
                source TEXT
            )
        ''')

        # Create indexes for faster searching
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON books(title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_author ON books(author)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON books(year)')

        conn.commit()
        conn.close()

        self.logger.info(f"Local index created: {db_path}")
        return True

    def search_local_index(self, dataset_name: str, title: str, 
                          author: str, year: Optional[int] = None) -> List[BookMatch]:
        """
        Search the local index for books.

        Args:
            dataset_name: Name of the dataset to search
            title: Book title to search for
            author: Book author to search for
            year: Publication year (optional)

        Returns:
            List of matching BookMatch objects
        """
        db_path = self.torrent_base_dir / f'{dataset_name}_index.db'
        
        if not db_path.exists():
            self.logger.warning(f"No local index found for {dataset_name}")
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Build search query
        query = "SELECT * FROM books WHERE title LIKE ? AND author LIKE ?"
        params = [f'%{title}%', f'%{author}%']

        if year:
            query += " AND year = ?"
            params.append(year)

        cursor.execute(query, params)
        results = cursor.fetchall()

        matches = []
        for row in results:
            matches.append(BookMatch(
                title=row[1],
                author=row[2],
                year=row[3],
                format=row[4],
                file_path=Path(row[5]),
                size=row[6],
                source=row[8],
                md5_hash=row[7]
            ))

        conn.close()
        return matches
