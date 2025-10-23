#!/usr/bin/env python3
"""
Demo Script: Query 50 Sample Books

Tests the Anna's Archive local data pipeline by searching for the 50 books
in sample_50_books.csv against the local Parquet data. Demonstrates the
complete workflow from data querying to download preparation.

Usage:
    python demo_query_50_books.py --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
                                 --books-csv ../../data/processed/sample_50_books.csv \
                                 --output-dir ../../data/anna_archive/demo_results/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

import pandas as pd
from tqdm import tqdm

from query_engine import BookSearchEngine
from api_downloader import AnnaArchiveDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BookSearchDemo:
    """Demo class for testing book search pipeline."""
    
    def __init__(
        self,
        parquet_dir: str,
        books_csv: str,
        output_dir: str,
        api_key: str = None
    ):
        """
        Initialize the demo.
        
        Args:
            parquet_dir: Directory containing Parquet files
            books_csv: Path to CSV file with books to search
            output_dir: Output directory for results
            api_key: Optional API key for download testing
        """
        self.parquet_dir = Path(parquet_dir)
        self.books_csv = Path(books_csv)
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.search_engine = None
        self.downloader = None
        
        logger.info(f"Initialized demo:")
        logger.info(f"  Parquet dir: {self.parquet_dir}")
        logger.info(f"  Books CSV: {self.books_csv}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def load_test_books(self) -> pd.DataFrame:
        """Load books from CSV file."""
        if not self.books_csv.exists():
            raise FileNotFoundError(f"Books CSV not found: {self.books_csv}")
        
        logger.info(f"Loading test books from {self.books_csv}")
        books_df = pd.read_csv(self.books_csv)
        
        logger.info(f"Loaded {len(books_df)} books")
        logger.info(f"Columns: {list(books_df.columns)}")
        
        return books_df
    
    def search_books(self, books_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Search for books in the local data."""
        logger.info("Initializing search engine...")
        self.search_engine = BookSearchEngine(self.parquet_dir)
        
        # Get dataset stats
        stats = self.search_engine.get_stats()
        logger.info(f"Dataset stats: {stats['total_records']:,} records, {stats['schema_fields']} fields")
        
        results = []
        
        logger.info("Searching for books...")
        for idx, book in tqdm(books_df.iterrows(), total=len(books_df), desc="Searching"):
            title = book.get('title', '')
            author = book.get('author_name', '')
            
            try:
                # Search for the book
                matches = self.search_engine.search_by_title_author(
                    title=title,
                    author=author,
                    fuzzy=True,
                    limit=5
                )
                
                result = {
                    'original_title': title,
                    'original_author': author,
                    'found': len(matches) > 0,
                    'match_count': len(matches),
                    'matches': matches
                }
                
                # Add best match details
                if matches:
                    best_match = matches[0]
                    result.update({
                        'matched_title': best_match.get('title', ''),
                        'matched_author': best_match.get('author', ''),
                        'md5': best_match.get('md5', ''),
                        'year': best_match.get('year', ''),
                        'extension': best_match.get('extension', ''),
                        'publisher': best_match.get('publisher', '')
                    })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error searching for '{title}' by '{author}': {e}")
                results.append({
                    'original_title': title,
                    'original_author': author,
                    'found': False,
                    'match_count': 0,
                    'error': str(e)
                })
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search results and generate statistics."""
        logger.info("Analyzing results...")
        
        total_books = len(results)
        found_books = sum(1 for r in results if r['found'])
        success_rate = found_books / total_books if total_books > 0 else 0
        
        # Count by file format
        formats = {}
        for result in results:
            if result['found'] and result.get('extension'):
                ext = result['extension'].lower()
                formats[ext] = formats.get(ext, 0) + 1
        
        # Count by year range
        years = {}
        for result in results:
            if result['found'] and result.get('year'):
                try:
                    year = int(result['year'])
                    decade = f"{(year // 10) * 10}s"
                    years[decade] = years.get(decade, 0) + 1
                except (ValueError, TypeError):
                    pass
        
        # Count by publisher
        publishers = {}
        for result in results:
            if result['found'] and result.get('publisher'):
                pub = result['publisher']
                publishers[pub] = publishers.get(pub, 0) + 1
        
        analysis = {
            'total_books': total_books,
            'found_books': found_books,
            'success_rate': success_rate,
            'formats': formats,
            'years': years,
            'publishers': dict(sorted(publishers.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        return analysis
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> None:
        """Save results to files."""
        logger.info("Saving results...")
        
        # Save detailed results
        results_file = self.output_dir / "search_results.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary statistics
        summary_file = self.output_dir / "search_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        # Save download-ready CSV (books with MD5 hashes)
        download_ready = []
        for result in results:
            if result['found'] and result.get('md5'):
                download_ready.append({
                    'title': result['matched_title'],
                    'author': result['matched_author'],
                    'md5': result['md5'],
                    'year': result.get('year', ''),
                    'extension': result.get('extension', ''),
                    'publisher': result.get('publisher', ''),
                    'original_title': result['original_title'],
                    'original_author': result['original_author']
                })
        
        if download_ready:
            download_file = self.output_dir / "download_ready.csv"
            download_df = pd.DataFrame(download_ready)
            download_df.to_csv(download_file, index=False)
            logger.info(f"Saved {len(download_ready)} books ready for download to {download_file}")
        
        # Save report
        self._save_report(analysis, results)
    
    def _save_report(self, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """Save a human-readable report."""
        report_file = self.output_dir / "search_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Anna's Archive Local Search Results\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total books searched**: {analysis['total_books']}\n")
            f.write(f"- **Books found**: {analysis['found_books']}\n")
            f.write(f"- **Success rate**: {analysis['success_rate']:.1%}\n\n")
            
            # File formats
            if analysis['formats']:
                f.write("## File Formats Found\n\n")
                for format_name, count in sorted(analysis['formats'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{format_name.upper()}**: {count} books\n")
                f.write("\n")
            
            # Years
            if analysis['years']:
                f.write("## Publication Years\n\n")
                for decade, count in sorted(analysis['years'].items()):
                    f.write(f"- **{decade}**: {count} books\n")
                f.write("\n")
            
            # Top publishers
            if analysis['publishers']:
                f.write("## Top Publishers\n\n")
                for publisher, count in list(analysis['publishers'].items())[:5]:
                    f.write(f"- **{publisher}**: {count} books\n")
                f.write("\n")
            
            # Found books
            f.write("## Found Books\n\n")
            found_books = [r for r in results if r['found']]
            for i, result in enumerate(found_books, 1):
                f.write(f"{i}. **{result['matched_title']}** by {result['matched_author']}\n")
                f.write(f"   - Original: {result['original_title']} by {result['original_author']}\n")
                f.write(f"   - MD5: `{result.get('md5', 'N/A')}`\n")
                f.write(f"   - Format: {result.get('extension', 'N/A')}\n")
                f.write(f"   - Year: {result.get('year', 'N/A')}\n")
                f.write(f"   - Publisher: {result.get('publisher', 'N/A')}\n\n")
            
            # Not found books
            not_found = [r for r in results if not r['found']]
            if not_found:
                f.write("## Books Not Found\n\n")
                for i, result in enumerate(not_found, 1):
                    f.write(f"{i}. **{result['original_title']}** by {result['original_author']}\n")
                f.write("\n")
        
        logger.info(f"Saved report to {report_file}")
    
    def test_downloads(self, results: List[Dict[str, Any]]) -> None:
        """Test downloading a few books if API key is provided."""
        if not self.api_key:
            logger.info("No API key provided, skipping download test")
            return
        
        logger.info("Testing downloads...")
        self.downloader = AnnaArchiveDownloader(self.api_key)
        
        # Find books with MD5 hashes
        books_with_md5 = [r for r in results if r['found'] and r.get('md5')]
        
        if not books_with_md5:
            logger.info("No books with MD5 hashes found for download test")
            return
        
        # Test download first 3 books
        test_books = books_with_md5[:3]
        download_dir = self.output_dir / "test_downloads"
        download_dir.mkdir(exist_ok=True)
        
        logger.info(f"Testing download of {len(test_books)} books...")
        
        for i, result in enumerate(test_books, 1):
            md5 = result['md5']
            title = result['matched_title']
            
            logger.info(f"Testing download {i}/{len(test_books)}: {title}")
            
            try:
                download_path = self.downloader.download_book(md5, str(download_dir))
                if download_path:
                    logger.info(f"✅ Download successful: {download_path}")
                else:
                    logger.warning(f"❌ Download failed: {title}")
            except Exception as e:
                logger.error(f"❌ Download error for {title}: {e}")
    
    def run_demo(self) -> None:
        """Run the complete demo."""
        logger.info("🚀 Starting Anna's Archive Local Search Demo")
        
        try:
            # Load test books
            books_df = self.load_test_books()
            
            # Search for books
            results = self.search_books(books_df)
            
            # Analyze results
            analysis = self.analyze_results(results)
            
            # Save results
            self.save_results(results, analysis)
            
            # Test downloads if API key provided
            self.test_downloads(results)
            
            # Print summary
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETE")
            print("="*60)
            print(f"Total books searched: {analysis['total_books']}")
            print(f"Books found: {analysis['found_books']}")
            print(f"Success rate: {analysis['success_rate']:.1%}")
            print(f"Results saved to: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.search_engine:
                self.search_engine.close()
            if self.downloader:
                self.downloader.close()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Demo script for Anna's Archive local search pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with sample data
  python demo_query_50_books.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --books-csv ../../data/processed/sample_50_books.csv \\
    --output-dir ../../data/anna_archive/demo_results/

  # Run demo with download testing
  python demo_query_50_books.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --books-csv ../../data/processed/sample_50_books.csv \\
    --output-dir ../../data/anna_archive/demo_results/ \\
    --api-key "your_api_key"
        """
    )
    
    parser.add_argument(
        '--parquet-dir',
        required=True,
        help='Directory containing Parquet files'
    )
    
    parser.add_argument(
        '--books-csv',
        required=True,
        help='CSV file with books to search for'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--api-key',
        help='API key for download testing (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        demo = BookSearchDemo(
            parquet_dir=args.parquet_dir,
            books_csv=args.books_csv,
            output_dir=args.output_dir,
            api_key=args.api_key
        )
        
        demo.run_demo()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
