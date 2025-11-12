"""
Anna's Archive Book Matcher
Automated book matching system using DuckDB queries on Anna's Archive datasets
"""

import duckdb
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookMatcher:
    """
    Main class for matching romance books with Anna's Archive datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the BookMatcher
        
        Args:
            data_dir: Path to Anna's Archive data directory
        """
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()
        
        # Set up DuckDB configuration for large datasets
        self.conn.execute("SET memory_limit='28GB'")
        self.conn.execute("SET threads=4")
        
        logger.info("BookMatcher initialized with DuckDB connection")
    
    def load_romance_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load the romance books dataset
        
        Args:
            csv_path: Path to the romance books CSV file
            
        Returns:
            DataFrame with romance books data
        """
        logger.info(f"Loading romance dataset from {csv_path}")
        
        # Load the canonicalized dataset
        df = pd.read_csv(csv_path)
        
        # Clean and prepare the data for matching
        df['title_clean'] = df['title'].apply(self._clean_title)
        df['author_clean'] = df['author_name'].apply(self._clean_author)
        
        logger.info(f"Loaded {len(df)} romance books")
        return df
    
    def setup_anna_archive_tables(self) -> None:
        """
        Set up Anna's Archive dataset tables in DuckDB
        """
        logger.info("Setting up Anna's Archive tables...")
        
        # Elasticsearch dataset
        elasticsearch_parquet = self.data_dir / "elasticsearchF" / "*.parquet"
        if elasticsearch_parquet.parent.exists():
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE elasticsearch AS 
                SELECT * FROM read_parquet('{elasticsearch_parquet}')
            """)
            logger.info("Elasticsearch table created")
        
        # AAC dataset
        aac_parquet = self.data_dir / "aacF" / "*.parquet"
        if aac_parquet.parent.exists():
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE aac AS 
                SELECT * FROM read_parquet('{aac_parquet}')
            """)
            logger.info("AAC table created")
        
        # MariaDB dataset
        mariadb_parquet = self.data_dir / "mariadbF" / "*.parquet"
        if mariadb_parquet.parent.exists():
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE mariadb AS 
                SELECT * FROM read_parquet('{mariadb_parquet}')
            """)
            logger.info("MariaDB table created")
    
    def find_matches(self, romance_df: pd.DataFrame, 
                    similarity_threshold: float = 0.8) -> pd.DataFrame:
        """
        Find matches between romance books and Anna's Archive datasets
        
        Args:
            romance_df: DataFrame with romance books
            similarity_threshold: Minimum similarity score for matches
            
        Returns:
            DataFrame with matches and MD5 hashes
        """
        logger.info(f"Finding matches for {len(romance_df)} romance books...")
        
        matches = []
        
        for idx, book in romance_df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing book {idx}/{len(romance_df)}")
            
            # Try different matching strategies
            match = self._find_single_match(book, similarity_threshold)
            if match:
                matches.append(match)
        
        matches_df = pd.DataFrame(matches)
        logger.info(f"Found {len(matches_df)} matches")
        
        return matches_df
    
    def _find_single_match(self, book: pd.Series, 
                          similarity_threshold: float) -> Optional[Dict]:
        """
        Find a single book match using multiple strategies
        
        Args:
            book: Single book record
            similarity_threshold: Minimum similarity score
            
        Returns:
            Match dictionary or None
        """
        title = book['title_clean']
        author = book['author_clean']
        year = book.get('publication_year', None)
        
        # Strategy 1: Exact title and author match
        match = self._exact_match(title, author, year)
        if match:
            return self._create_match_record(book, match, "exact", 1.0)
        
        # Strategy 2: Fuzzy title match with author
        match = self._fuzzy_title_match(title, author, year, similarity_threshold)
        if match:
            return self._create_match_record(book, match, "fuzzy_title", match.get('similarity', 0.8))
        
        # Strategy 3: Author match with similar title
        match = self._author_title_match(title, author, year, similarity_threshold)
        if match:
            return self._create_match_record(book, match, "author_title", match.get('similarity', 0.8))
        
        return None
    
    def _exact_match(self, title: str, author: str, year: Optional[int]) -> Optional[Dict]:
        """
        Try exact matching across all datasets
        """
        # Elasticsearch exact match
        query = """
            SELECT *, 'elasticsearch' as source
            FROM elasticsearch 
            WHERE LOWER(title) = LOWER(?) 
            AND LOWER(author) = LOWER(?)
        """
        
        if year:
            query += f" AND publication_year = {year}"
        
        result = self.conn.execute(query, [title, author]).fetchone()
        if result:
            return dict(zip([col[0] for col in self.conn.description], result))
        
        # AAC exact match
        query = """
            SELECT *, 'aac' as source
            FROM aac 
            WHERE LOWER(title) = LOWER(?) 
            AND LOWER(author) = LOWER(?)
        """
        
        if year:
            query += f" AND publication_year = {year}"
        
        result = self.conn.execute(query, [title, author]).fetchone()
        if result:
            return dict(zip([col[0] for col in self.conn.description], result))
        
        return None
    
    def _fuzzy_title_match(self, title: str, author: str, year: Optional[int], 
                          threshold: float) -> Optional[Dict]:
        """
        Try fuzzy title matching
        """
        # This would require more complex SQL with similarity functions
        # For now, we'll implement a simpler approach
        return None
    
    def _author_title_match(self, title: str, author: str, year: Optional[int], 
                           threshold: float) -> Optional[Dict]:
        """
        Try author matching with title similarity
        """
        # This would require more complex SQL with similarity functions
        # For now, we'll implement a simpler approach
        return None
    
    def _create_match_record(self, book: pd.Series, match: Dict, 
                           match_type: str, similarity: float) -> Dict:
        """
        Create a standardized match record
        """
        return {
            'work_id': book['work_id'],
            'title': book['title'],
            'author_name': book['author_name'],
            'publication_year': book.get('publication_year'),
            'aa_title': match.get('title', ''),
            'aa_author': match.get('author', ''),
            'aa_publication_year': match.get('publication_year'),
            'md5_hash': match.get('md5', ''),
            'file_size': match.get('file_size'),
            'file_extension': match.get('file_extension', ''),
            'source': match.get('source', ''),
            'match_type': match_type,
            'similarity_score': similarity,
            'download_url': f"https://annas-archive.org/md5/{match.get('md5', '')}" if match.get('md5') else ''
        }
    
    def _clean_title(self, title: str) -> str:
        """
        Clean title for matching
        """
        if pd.isna(title):
            return ""
        
        # Remove common punctuation and normalize
        title = re.sub(r'[^\w\s]', ' ', str(title))
        title = re.sub(r'\s+', ' ', title)
        return title.strip().lower()
    
    def _clean_author(self, author: str) -> str:
        """
        Clean author name for matching
        """
        if pd.isna(author):
            return ""
        
        # Remove common punctuation and normalize
        author = re.sub(r'[^\w\s]', ' ', str(author))
        author = re.sub(r'\s+', ' ', author)
        return author.strip().lower()
    
    def extract_md5_hashes(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract MD5 hashes from matches for download
        
        Args:
            matches_df: DataFrame with matches
            
        Returns:
            DataFrame with MD5 hashes ready for download
        """
        logger.info(f"Extracting MD5 hashes from {len(matches_df)} matches...")
        
        # Filter matches with valid MD5 hashes
        valid_matches = matches_df[matches_df['md5_hash'].notna() & (matches_df['md5_hash'] != '')]
        
        # Create download-ready format
        download_df = valid_matches[['work_id', 'title', 'author_name', 'md5_hash', 'file_extension']].copy()
        download_df['filename'] = download_df.apply(
            lambda row: f"{row['title']}_{row['author_name']}_{row['md5_hash'][:8]}.{row['file_extension']}", 
            axis=1
        )
        
        logger.info(f"Extracted {len(download_df)} valid MD5 hashes for download")
        return download_df
    
    def save_matches(self, matches_df: pd.DataFrame, output_path: str) -> None:
        """
        Save matches to CSV file
        
        Args:
            matches_df: DataFrame with matches
            output_path: Output file path
        """
        matches_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(matches_df)} matches to {output_path}")
    
    def close(self) -> None:
        """
        Close DuckDB connection
        """
        self.conn.close()
        logger.info("DuckDB connection closed")
