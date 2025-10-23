#!/usr/bin/env python3
"""
Query Engine for Anna's Archive Local Data

Provides SQL-based search interface using DuckDB to query Parquet files
containing Anna's Archive book metadata. Supports fuzzy matching for
title and author searches.

Usage:
    from query_engine import BookSearchEngine
    
    engine = BookSearchEngine('../../data/anna_archive/parquet/sample_10k/')
    results = engine.search_by_title_author("Fifty Shades", "E.L. James")
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookSearchEngine:
    """DuckDB-based search engine for Anna's Archive book data."""
    
    def __init__(self, parquet_dir: str):
        """
        Initialize search engine with Parquet data directory.
        
        Args:
            parquet_dir: Directory containing Parquet files
        """
        self.parquet_dir = Path(parquet_dir)
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect()
        
        # Load Parquet files
        self._load_data()
        
        # Detect schema
        self.schema_fields = self._detect_schema()
        logger.info(f"Detected {len(self.schema_fields)} fields in schema")
    
    def _load_data(self) -> None:
        """Load Parquet files into DuckDB."""
        parquet_files = list(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {self.parquet_dir}")
        
        logger.info(f"Loading {len(parquet_files)} Parquet files...")
        
        # Create view from all Parquet files
        parquet_pattern = str(self.parquet_dir / "*.parquet")
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW books AS 
            SELECT * FROM read_parquet('{parquet_pattern}')
        """)
        
        # Get record count
        result = self.conn.execute("SELECT COUNT(*) FROM books").fetchone()
        self.total_records = result[0]
        logger.info(f"Loaded {self.total_records:,} records")
    
    def _detect_schema(self) -> Dict[str, str]:
        """Detect available fields in the schema."""
        # Get column information
        columns = self.conn.execute("DESCRIBE books").fetchall()
        
        schema_fields = {}
        for col_name, col_type, _, _, _, _ in columns:
            schema_fields[col_name] = col_type
        
        # Map common field patterns to standard names
        field_mapping = self._create_field_mapping(schema_fields)
        
        return field_mapping
    
    def _create_field_mapping(self, schema_fields: Dict[str, str]) -> Dict[str, str]:
        """Create mapping from standard names to actual field names."""
        mapping = {}
        
        # Common field patterns in Anna's Archive data
        patterns = {
            'title': [
                r'.*title.*best.*',
                r'.*title.*',
                r'.*_source.*file_unified_data.*title.*best.*'
            ],
            'author': [
                r'.*author.*best.*',
                r'.*author.*',
                r'.*_source.*file_unified_data.*author.*best.*'
            ],
            'publisher': [
                r'.*publisher.*best.*',
                r'.*publisher.*',
                r'.*_source.*file_unified_data.*publisher.*best.*'
            ],
            'year': [
                r'.*year.*best.*',
                r'.*year.*',
                r'.*_source.*file_unified_data.*year.*best.*'
            ],
            'md5': [
                r'.*md5.*',
                r'.*identifiers.*md5.*',
                r'.*_source.*file_unified_data.*identifiers.*md5.*'
            ],
            'extension': [
                r'.*extension.*best.*',
                r'.*extension.*',
                r'.*_source.*file_unified_data.*extension.*best.*'
            ],
            'language': [
                r'.*language.*best.*',
                r'.*language.*',
                r'.*_source.*file_unified_data.*language.*best.*'
            ],
            'file_size': [
                r'.*size.*',
                r'.*file_size.*',
                r'.*_source.*file_unified_data.*size.*'
            ]
        }
        
        for standard_name, pattern_list in patterns.items():
            for field_name in schema_fields.keys():
                for pattern in pattern_list:
                    if re.match(pattern, field_name, re.IGNORECASE):
                        mapping[standard_name] = field_name
                        break
                if standard_name in mapping:
                    break
        
        # Log detected mappings
        logger.info("Detected field mappings:")
        for standard_name, actual_field in mapping.items():
            logger.info(f"  {standard_name} -> {actual_field}")
        
        return mapping
    
    def search_by_title_author(
        self,
        title: str,
        author: str,
        fuzzy: bool = True,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for books by title and author.
        
        Args:
            title: Book title to search for
            author: Author name to search for
            fuzzy: Whether to use fuzzy matching (default: True)
            limit: Maximum number of results (default: 10)
            
        Returns:
            List of matching book records
        """
        if not self.schema_fields:
            raise ValueError("No schema fields detected")
        
        # Build query
        query_parts = []
        params = {}
        
        # Title search
        if title and 'title' in self.schema_fields:
            title_field = self.schema_fields['title']
            if fuzzy:
                query_parts.append(f"LOWER({title_field}) LIKE LOWER('%' || $title || '%')")
            else:
                query_parts.append(f"LOWER({title_field}) = LOWER($title)")
            params['title'] = title
        
        # Author search
        if author and 'author' in self.schema_fields:
            author_field = self.schema_fields['author']
            if fuzzy:
                query_parts.append(f"LOWER({author_field}) LIKE LOWER('%' || $author || '%')")
            else:
                query_parts.append(f"LOWER({author_field}) = LOWER($author)")
            params['author'] = author
        
        if not query_parts:
            raise ValueError("Must provide at least title or author")
        
        # Build SELECT clause
        select_fields = []
        for standard_name, actual_field in self.schema_fields.items():
            select_fields.append(f"{actual_field} AS {standard_name}")
        
        # Build WHERE clause
        where_clause = " AND ".join(query_parts)
        
        # Execute query
        query = f"""
            SELECT {', '.join(select_fields)}
            FROM books
            WHERE {where_clause}
            LIMIT {limit}
        """
        
        logger.debug(f"Executing query: {query}")
        logger.debug(f"Parameters: {params}")
        
        try:
            result = self.conn.execute(query, params).fetchdf()
            return result.to_dict('records')
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {params}")
            raise
    
    def search_by_title(self, title: str, fuzzy: bool = True, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for books by title only."""
        return self.search_by_title_author(title, "", fuzzy, limit)
    
    def search_by_author(self, author: str, fuzzy: bool = True, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for books by author only."""
        return self.search_by_title_author("", author, fuzzy, limit)
    
    def search_by_md5(self, md5: str) -> List[Dict[str, Any]]:
        """Search for a book by its MD5 hash."""
        if 'md5' not in self.schema_fields:
            raise ValueError("MD5 field not found in schema")
        
        md5_field = self.schema_fields['md5']
        query = f"""
            SELECT *
            FROM books
            WHERE LOWER({md5_field}) = LOWER($md5)
        """
        
        try:
            result = self.conn.execute(query, {'md5': md5}).fetchdf()
            return result.to_dict('records')
        except Exception as e:
            logger.error(f"MD5 search failed: {e}")
            raise
    
    def get_random_books(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get random books from the dataset."""
        query = f"""
            SELECT *
            FROM books
            ORDER BY RANDOM()
            LIMIT {count}
        """
        
        try:
            result = self.conn.execute(query).fetchdf()
            return result.to_dict('records')
        except Exception as e:
            logger.error(f"Random book query failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        stats = {
            'total_records': self.total_records,
            'schema_fields': len(self.schema_fields),
            'available_fields': list(self.schema_fields.keys())
        }
        
        # Get counts for key fields
        for field_name in ['title', 'author', 'md5', 'extension']:
            if field_name in self.schema_fields:
                actual_field = self.schema_fields[field_name]
                try:
                    result = self.conn.execute(f"""
                        SELECT COUNT(*) as count
                        FROM books
                        WHERE {actual_field} IS NOT NULL AND {actual_field} != ''
                    """).fetchone()
                    stats[f'{field_name}_count'] = result[0]
                except Exception as e:
                    logger.warning(f"Could not get count for {field_name}: {e}")
                    stats[f'{field_name}_count'] = 0
        
        return stats
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Main function for command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query Anna's Archive book data using DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search by title and author
  python query_engine.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --title "Fifty Shades" \\
    --author "E.L. James"

  # Search by title only
  python query_engine.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --title "Romeo and Juliet"

  # Get dataset statistics
  python query_engine.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --stats
        """
    )
    
    parser.add_argument(
        '--parquet-dir',
        required=True,
        help='Directory containing Parquet files'
    )
    
    parser.add_argument(
        '--title',
        help='Book title to search for'
    )
    
    parser.add_argument(
        '--author',
        help='Author name to search for'
    )
    
    parser.add_argument(
        '--md5',
        help='MD5 hash to search for'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum number of results (default: 10)'
    )
    
    parser.add_argument(
        '--exact',
        action='store_true',
        help='Use exact matching instead of fuzzy matching'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show dataset statistics'
    )
    
    parser.add_argument(
        '--random',
        type=int,
        help='Show N random books'
    )
    
    args = parser.parse_args()
    
    try:
        with BookSearchEngine(args.parquet_dir) as engine:
            if args.stats:
                stats = engine.get_stats()
                print("Dataset Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            elif args.random:
                books = engine.get_random_books(args.random)
                print(f"Random {args.random} books:")
                for i, book in enumerate(books, 1):
                    print(f"{i}. {book.get('title', 'No title')} by {book.get('author', 'Unknown author')}")
            
            elif args.md5:
                books = engine.search_by_md5(args.md5)
                print(f"Books with MD5 {args.md5}:")
                for book in books:
                    print(f"  {book.get('title', 'No title')} by {book.get('author', 'Unknown author')}")
            
            elif args.title or args.author:
                books = engine.search_by_title_author(
                    args.title or "",
                    args.author or "",
                    fuzzy=not args.exact,
                    limit=args.limit
                )
                
                print(f"Found {len(books)} books:")
                for i, book in enumerate(books, 1):
                    print(f"{i}. {book.get('title', 'No title')} by {book.get('author', 'Unknown author')}")
                    if book.get('md5'):
                        print(f"   MD5: {book['md5']}")
                    if book.get('year'):
                        print(f"   Year: {book['year']}")
                    if book.get('extension'):
                        print(f"   Format: {book['extension']}")
                    print()
            
            else:
                print("No search criteria provided. Use --title, --author, --md5, --stats, or --random")
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
