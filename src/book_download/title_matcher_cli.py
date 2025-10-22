#!/usr/bin/env python3
"""
Title Matcher CLI for Anna's Archive
Maps CSV titles to MD5 hashes using MariaDB or Elasticsearch backends
Integrates with existing BookDownloadManager for automatic downloads
"""

import os
import sys
import csv
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_manager import BookDownloadManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TitleMatcher:
    """Matches book titles to Anna's Archive MD5 hashes using fuzzy matching"""
    
    def __init__(self, backend: str, **backend_config):
        """
        Initialize title matcher
        
        Args:
            backend: 'mariadb' or 'elasticsearch'
            **backend_config: Backend-specific configuration
        """
        self.backend = backend
        self.config = backend_config
        
        if backend == 'mariadb':
            self._init_mariadb()
        elif backend == 'elasticsearch':
            self._init_elasticsearch()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        logger.info(f"Title matcher initialized with {backend} backend")
    
    def _init_mariadb(self):
        """Initialize MariaDB connection"""
        try:
            import pymysql
            self.pymysql = pymysql
        except ImportError:
            raise ImportError("pymysql required for MariaDB backend. Install with: pip install pymysql")
        
        # Connection parameters
        self.db_config = {
            'host': self.config.get('db_host', 'localhost'),
            'user': self.config.get('db_user', 'annas_user'),
            'password': self.config.get('db_pass', 'annas_pass'),
            'database': self.config.get('db_name', 'annas_archive'),
            'charset': 'utf8mb4'
        }
        
        # Test connection
        try:
            conn = pymysql.connect(**self.db_config)
            conn.close()
            logger.info("MariaDB connection successful")
        except Exception as e:
            logger.error(f"MariaDB connection failed: {e}")
            raise
    
    def _init_elasticsearch(self):
        """Initialize Elasticsearch connection"""
        try:
            from elasticsearch import Elasticsearch
            self.elasticsearch = Elasticsearch
        except ImportError:
            raise ImportError("elasticsearch required for ES backend. Install with: pip install elasticsearch")
        
        # Connection parameters
        es_host = self.config.get('es_host', 'http://localhost:9200')
        self.es_client = Elasticsearch([es_host])
        self.index_name = self.config.get('index', 'aa_records')
        
        # Test connection
        try:
            info = self.es_client.info()
            logger.info(f"Elasticsearch connection successful: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            raise
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy matching"""
        if not text:
            return ""
        
        try:
            from unidecode import unidecode
            return unidecode(text.lower().strip())
        except ImportError:
            # Fallback without unidecode
            return text.lower().strip()
    
    def fuzzy_score(self, text1: str, text2: str) -> float:
        """Calculate fuzzy matching score"""
        try:
            from rapidfuzz import fuzz
            return fuzz.token_set_ratio(text1, text2)
        except ImportError:
            # Fallback to simple similarity
            if not text1 or not text2:
                return 0.0
            return 100.0 if text1 == text2 else 0.0
    
    def search_mariadb(self, title: str, author: str = None, year: int = None) -> List[Dict]:
        """Search MariaDB for matching books"""
        conn = self.pymysql.connect(**self.db_config)
        cursor = conn.cursor(self.pymysql.cursors.DictCursor)
        
        try:
            # Build query - adjust table/column names based on your schema
            query = """
                SELECT md5, title, author, year, language, extension, filesize, isbn10, isbn13
                FROM aa_records
                WHERE title LIKE %s
                LIMIT 200
            """
            
            # Use first part of title for LIKE search
            title_like = f"%{title[:50]}%"
            cursor.execute(query, (title_like,))
            
            results = cursor.fetchall()
            logger.debug(f"MariaDB returned {len(results)} candidates for '{title}'")
            return results
            
        finally:
            cursor.close()
            conn.close()
    
    def search_elasticsearch(self, title: str, author: str = None, year: int = None) -> List[Dict]:
        """Search Elasticsearch for matching books"""
        try:
            # Build search query
            must_clauses = [
                {"match": {"title": {"query": title, "fuzziness": "AUTO"}}}
            ]
            
            if author:
                must_clauses.append({"match": {"author": {"query": author, "fuzziness": "AUTO"}}})
            
            if year:
                must_clauses.append({"term": {"year": int(year)}})
            
            query = {
                "query": {"bool": {"must": must_clauses}},
                "_source": ["md5", "title", "author", "year", "language", "extension", "filesize", "isbn10", "isbn13"],
                "size": 200
            }
            
            response = self.es_client.search(index=self.index_name, body=query)
            results = [hit["_source"] for hit in response["hits"]["hits"]]
            
            logger.debug(f"Elasticsearch returned {len(results)} candidates for '{title}'")
            return results
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def find_best_match(self, title: str, author: str = None, year: int = None) -> Tuple[Optional[Dict], float, str]:
        """
        Find the best matching book for given title/author/year
        
        Returns:
            Tuple of (best_match_dict, confidence_score, match_explanation)
        """
        # Normalize input
        norm_title = self.normalize_text(title)
        norm_author = self.normalize_text(author) if author else None
        
        # Search backend
        if self.backend == 'mariadb':
            candidates = self.search_mariadb(title, author, year)
        else:
            candidates = self.search_elasticsearch(title, author, year)
        
        if not candidates:
            return None, 0.0, "No candidates found"
        
        best_match = None
        best_score = -1
        best_explanation = ""
        
        for candidate in candidates:
            # Normalize candidate data
            cand_title = self.normalize_text(candidate.get('title', ''))
            cand_author = self.normalize_text(candidate.get('author', ''))
            cand_year = candidate.get('year')
            
            # Calculate scores
            title_score = self.fuzzy_score(norm_title, cand_title)
            author_score = self.fuzzy_score(norm_author, cand_author) if norm_author else 0
            year_score = 100 if (year and cand_year and str(year) == str(cand_year)) else 0
            
            # Weighted total score
            total_score = title_score * 0.75 + author_score * 0.2 + year_score * 0.05
            
            # Determine match quality
            if total_score > best_score:
                best_score = total_score
                best_match = candidate
                
                # Generate explanation
                if title_score >= 95 and author_score >= 80:
                    best_explanation = "exact_match"
                elif title_score >= 90 and author_score >= 70:
                    best_explanation = "high_confidence"
                elif title_score >= 80:
                    best_explanation = "medium_confidence"
                else:
                    best_explanation = "low_confidence"
        
        return best_match, best_score, best_explanation
    
    def process_csv(self, input_csv: str, output_csv: str) -> Dict[str, Any]:
        """
        Process CSV file and generate title-to-MD5 mapping
        
        Args:
            input_csv: Path to input CSV with title, author_name, publication_year
            output_csv: Path to output CSV with MD5 mappings
            
        Returns:
            Summary statistics
        """
        logger.info(f"Processing CSV: {input_csv}")
        
        # Read input CSV
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"Loaded {len(df)} rows from {input_csv}")
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise
        
        # Validate required columns
        required_cols = ['title', 'author_name', 'publication_year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process each row
        results = []
        stats = {
            'total_processed': 0,
            'exact_matches': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'no_matches': 0,
            'No candidates found': 0
        }
        
        for idx, row in df.iterrows():
            title = str(row['title']).strip()
            author = str(row['author_name']).strip() if pd.notna(row['author_name']) else None
            year = int(row['publication_year']) if pd.notna(row['publication_year']) else None
            
            logger.info(f"Processing {idx+1}/{len(df)}: '{title}' by {author}")
            
            # Find best match
            match, score, explanation = self.find_best_match(title, author, year)
            
            # Prepare result row
            result_row = {
                'work_id': row.get('work_id', idx),
                'input_title': title,
                'input_author': author,
                'input_year': year,
                'md5': match.get('md5') if match else None,
                'match_score': round(score, 1),
                'match_confidence': explanation,
                'matched_title': match.get('title') if match else None,
                'matched_author': match.get('author') if match else None,
                'matched_year': match.get('year') if match else None,
                'language': match.get('language') if match else None,
                'extension': match.get('extension') if match else None,
                'filesize': match.get('filesize') if match else None,
                'isbn10': match.get('isbn10') if match else None,
                'isbn13': match.get('isbn13') if match else None,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result_row)
            stats['total_processed'] += 1
            stats[explanation] += 1
            
            logger.info(f"  → {explanation}: {score:.1f}% - {match.get('title', 'No match') if match else 'No match'}")
        
        # Write results
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        logger.info(f"Results written to: {output_csv}")
        
        # Print summary
        logger.info("=== MATCHING SUMMARY ===")
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Exact matches: {stats['exact_matches']}")
        logger.info(f"High confidence: {stats['high_confidence']}")
        logger.info(f"Medium confidence: {stats['medium_confidence']}")
        logger.info(f"Low confidence: {stats['low_confidence']}")
        logger.info(f"No matches: {stats['no_matches']}")
        
        return stats

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Match book titles to Anna's Archive MD5 hashes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MariaDB backend
  python title_matcher_cli.py --backend mariadb --in data/titles.csv --out results.csv \\
    --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass

  # Elasticsearch backend  
  python title_matcher_cli.py --backend es --in data/titles.csv --out results.csv \\
    --es-host http://localhost:9200 --index aa_records

  # With automatic download
  python title_matcher_cli.py --backend mariadb --in data/titles.csv --download \\
    --daily-limit 10
        """
    )
    
    # Input/Output
    parser.add_argument("--in", "--input", dest="input_csv", required=True,
                       help="Input CSV file with title, author_name, publication_year columns")
    parser.add_argument("--out", "--output", dest="output_csv",
                       help="Output CSV file (default: title_to_md5_<timestamp>.csv)")
    
    # Backend selection
    parser.add_argument("--backend", choices=['mariadb', 'es'], required=True,
                       help="Backend to use: mariadb or elasticsearch")
    
    # MariaDB options
    parser.add_argument("--db-host", default="localhost", help="MariaDB host")
    parser.add_argument("--db-name", default="annas_archive", help="MariaDB database name")
    parser.add_argument("--db-user", default="annas_user", help="MariaDB username")
    parser.add_argument("--db-pass", default="annas_pass", help="MariaDB password")
    parser.add_argument("--table", default="aa_records", help="MariaDB table name")
    
    # Elasticsearch options
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--index", default="aa_records", help="Elasticsearch index name")
    
    # Download integration
    parser.add_argument("--download", action="store_true",
                       help="Automatically download matched books using BookDownloadManager")
    parser.add_argument("--daily-limit", type=int, default=25,
                       help="Daily download limit (default: 25)")
    parser.add_argument("--download-dir", 
                       default="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download",
                       help="Download directory")
    
    # Other options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set output file if not provided
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"title_to_md5_{timestamp}.csv"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize title matcher
        if args.backend == 'mariadb':
            matcher = TitleMatcher(
                backend='mariadb',
                db_host=args.db_host,
                db_name=args.db_name,
                db_user=args.db_user,
                db_pass=args.db_pass,
                table=args.table
            )
        else:
            matcher = TitleMatcher(
                backend='elasticsearch',
                es_host=args.es_host,
                index=args.index
            )
        
        # Process CSV
        stats = matcher.process_csv(args.input_csv, args.output_csv)
        
        # Download if requested
        if args.download:
            logger.info("=== STARTING AUTOMATIC DOWNLOADS ===")
            
            # Initialize download manager
            download_manager = BookDownloadManager(
                csv_path=args.output_csv,
                download_dir=args.download_dir,
                daily_limit=args.daily_limit
            )
            
            # Run MD5-based download batch
            download_summary = download_manager.run_md5_download_batch()
            
            logger.info("=== DOWNLOAD SUMMARY ===")
            logger.info(f"Processed: {download_summary.get('processed', 0)}")
            logger.info(f"Downloaded: {download_summary.get('downloaded', 0)}")
            logger.info(f"Failed: {download_summary.get('failed', 0)}")
        
        logger.info("Title matching completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Title matching failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
