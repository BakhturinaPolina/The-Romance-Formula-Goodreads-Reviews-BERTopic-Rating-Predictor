#!/usr/bin/env python3
"""
Fallback Download System
Implement a fallback system for when Anna's Archive is unavailable
"""

import pandas as pd
import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FallbackDownloadSystem:
    """Fallback system for book downloads when Anna's Archive is unavailable"""
    
    def __init__(self, download_dir: str):
        """
        Initialize fallback download system
        
        Args:
            download_dir: Directory containing downloaded books
        """
        self.download_dir = Path(download_dir)
        self.existing_books = self._scan_existing_books()
        
        logger.info(f"Fallback system initialized with {len(self.existing_books)} existing books")
    
    def _scan_existing_books(self) -> List[Dict]:
        """Scan existing downloaded books"""
        books = []
        
        if not self.download_dir.exists():
            return books
        
        for file_path in self.download_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.epub':
                # Extract book info from filename
                filename = file_path.stem
                
                # Try to parse work_id from filename
                work_id = None
                if '_' in filename:
                    try:
                        work_id = int(filename.split('_')[0])
                    except ValueError:
                        pass
                
                books.append({
                    'work_id': work_id,
                    'title': filename,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'download_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return books
    
    def create_curated_book_list(self, csv_path: str, output_path: str = None) -> str:
        """
        Create a curated book list from existing downloads and CSV data
        
        Args:
            csv_path: Path to the original CSV file
            output_path: Path to save curated list (optional)
            
        Returns:
            Path to the curated book list
        """
        logger.info("Creating curated book list from existing downloads")
        
        # Load original CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} books from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
        
        # Create curated list
        curated_books = []
        
        for _, row in df.iterrows():
            work_id = row['work_id']
            title = row['title']
            author = row['author_name']
            
            # Check if we have this book already
            existing_book = next((b for b in self.existing_books if b['work_id'] == work_id), None)
            
            if existing_book:
                # Book already downloaded
                curated_books.append({
                    'work_id': work_id,
                    'title': title,
                    'author_name': author,
                    'publication_year': row['publication_year'],
                    'status': 'already_downloaded',
                    'file_path': existing_book['file_path'],
                    'file_size': existing_book['file_size'],
                    'download_date': existing_book['download_date']
                })
            else:
                # Book not downloaded - mark for future download
                curated_books.append({
                    'work_id': work_id,
                    'title': title,
                    'author_name': author,
                    'publication_year': row['publication_year'],
                    'status': 'pending_download',
                    'file_path': None,
                    'file_size': None,
                    'download_date': None
                })
        
        # Save curated list
        if output_path is None:
            output_path = self.download_dir / "curated_book_list.json"
        
        with open(output_path, 'w') as f:
            json.dump(curated_books, f, indent=2)
        
        logger.info(f"Curated book list saved to: {output_path}")
        return str(output_path)
    
    def generate_download_statistics(self) -> Dict:
        """Generate statistics about existing downloads"""
        if not self.existing_books:
            return {'total_books': 0}
        
        total_size = sum(book['file_size'] for book in self.existing_books)
        
        # Group by year if we can extract it
        books_by_year = {}
        for book in self.existing_books:
            # Try to extract year from filename or use current year
            year = datetime.now().year
            books_by_year[year] = books_by_year.get(year, 0) + 1
        
        return {
            'total_books': len(self.existing_books),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'average_size_mb': round(total_size / (1024 * 1024) / len(self.existing_books), 2),
            'books_by_year': books_by_year,
            'existing_books': self.existing_books
        }
    
    def create_test_dataset(self, num_books: int = 10) -> str:
        """
        Create a test dataset from existing books for system testing
        
        Args:
            num_books: Number of books to include in test dataset
            
        Returns:
            Path to test dataset CSV
        """
        logger.info(f"Creating test dataset with {num_books} books")
        
        if not self.existing_books:
            logger.warning("No existing books found for test dataset")
            return None
        
        # Take first N books
        test_books = self.existing_books[:num_books]
        
        # Create test dataset
        test_data = []
        for book in test_books:
            test_data.append({
                'work_id': book['work_id'] or 999999,
                'title': book['title'],
                'author_name': 'Test Author',
                'publication_year': 2020,
                'file_path': book['file_path'],
                'status': 'test_book'
            })
        
        # Save test dataset
        test_df = pd.DataFrame(test_data)
        test_path = self.download_dir / "test_dataset.csv"
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Test dataset saved to: {test_path}")
        return str(test_path)
    
    def simulate_download_workflow(self, csv_path: str, max_books: int = 5) -> Dict:
        """
        Simulate the download workflow using existing books
        
        Args:
            csv_path: Path to CSV file with book metadata
            max_books: Maximum number of books to process
            
        Returns:
            Simulation results
        """
        logger.info(f"Simulating download workflow with {max_books} books")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} books from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return {'error': str(e)}
        
        # Process first N books
        test_df = df.head(max_books)
        results = []
        
        for idx, row in test_df.iterrows():
            work_id = row['work_id']
            title = row['title']
            author = row['author_name']
            
            logger.info(f"Processing: {title} by {author}")
            
            # Check if we have this book
            existing_book = next((b for b in self.existing_books if b['work_id'] == work_id), None)
            
            if existing_book:
                result = {
                    'work_id': work_id,
                    'title': title,
                    'author_name': author,
                    'publication_year': row['publication_year'],
                    'status': 'downloaded',
                    'file_path': existing_book['file_path'],
                    'file_size': existing_book['file_size'],
                    'download_date': existing_book['download_date'],
                    'simulation': True
                }
                logger.info(f"✓ Found existing book: {title}")
            else:
                result = {
                    'work_id': work_id,
                    'title': title,
                    'author_name': author,
                    'publication_year': row['publication_year'],
                    'status': 'not_found',
                    'file_path': None,
                    'error': 'Book not in existing collection',
                    'simulation': True
                }
                logger.info(f"✗ Book not found: {title}")
            
            results.append(result)
            time.sleep(0.5)  # Simulate processing time
        
        # Calculate statistics
        successful = sum(1 for r in results if r['status'] == 'downloaded')
        failed = sum(1 for r in results if r['status'] == 'not_found')
        success_rate = (successful / len(results)) * 100 if results else 0
        
        summary = {
            'total_processed': len(results),
            'successful_downloads': successful,
            'failed_downloads': failed,
            'success_rate': round(success_rate, 1),
            'results': results,
            'simulation': True
        }
        
        logger.info(f"Simulation completed: {successful}/{len(results)} successful ({success_rate:.1f}%)")
        return summary

def main():
    """Main function for testing fallback system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fallback download system')
    parser.add_argument('--download-dir',
                       default='organized_outputs/anna_archive_download',
                       help='Directory containing downloaded books')
    parser.add_argument('--csv-path',
                       default='data/processed/test_subdataset.csv',
                       help='Path to CSV file with book metadata')
    parser.add_argument('--action',
                       choices=['scan', 'curate', 'test', 'simulate'],
                       default='scan',
                       help='Action to perform')
    parser.add_argument('--max-books', type=int, default=5,
                       help='Maximum books for test/simulation')
    
    args = parser.parse_args()
    
    # Initialize fallback system
    fallback = FallbackDownloadSystem(args.download_dir)
    
    if args.action == 'scan':
        # Scan existing books
        stats = fallback.generate_download_statistics()
        print(f"Found {stats['total_books']} existing books")
        print(f"Total size: {stats['total_size_mb']} MB")
        print(f"Average size: {stats['average_size_mb']} MB")
        
    elif args.action == 'curate':
        # Create curated book list
        curated_path = fallback.create_curated_book_list(args.csv_path)
        if curated_path:
            print(f"Curated book list created: {curated_path}")
        
    elif args.action == 'test':
        # Create test dataset
        test_path = fallback.create_test_dataset(args.max_books)
        if test_path:
            print(f"Test dataset created: {test_path}")
        
    elif args.action == 'simulate':
        # Simulate download workflow
        results = fallback.simulate_download_workflow(args.csv_path, args.max_books)
        print(f"Simulation results: {results['successful_downloads']}/{results['total_processed']} successful")
        print(f"Success rate: {results['success_rate']}%")

if __name__ == "__main__":
    main()
