#!/usr/bin/env python3
"""
Manual Search Helper for Anna's Archive
Provides search URLs and tracking for manual book searches
"""

import pandas as pd
import webbrowser
import logging
from pathlib import Path
from typing import List, Dict
import time

logger = logging.getLogger(__name__)


class ManualSearchHelper:
    """
    Helper for manual searching on Anna's Archive
    """
    
    def __init__(self, template_csv: str):
        """
        Initialize with manual search template
        
        Args:
            template_csv: Path to manual search template CSV
        """
        self.template_df = pd.read_csv(template_csv)
        self.current_index = 0
        self.base_url = "https://annas-archive.org/search"
        
        logger.info(f"Loaded {len(self.template_df)} books for manual search")
    
    def get_search_url(self, book_row: pd.Series) -> str:
        """
        Generate Anna's Archive search URL for a book
        
        Args:
            book_row: Book data row
            
        Returns:
            Search URL
        """
        title = book_row['title']
        author = book_row['author_name']
        
        # Create search query
        query = f'"{title}" "{author}" romance'
        
        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        return f"{self.base_url}?q={encoded_query}"
    
    def open_next_book(self) -> bool:
        """
        Open the next book's search page in browser
        
        Returns:
            True if more books available, False if done
        """
        if self.current_index >= len(self.template_df):
            print("All books processed!")
            return False
        
        book = self.template_df.iloc[self.current_index]
        search_url = self.get_search_url(book)
        
        print(f"\nBook {self.current_index + 1}/{len(self.template_df)}")
        print(f"Title: {book['title']}")
        print(f"Author: {book['author_name']}")
        print(f"Year: {book['publication_year']}")
        print(f"Rating: {book['average_rating_weighted_mean']}")
        print(f"Reviews: {book['ratings_count_sum']}")
        print(f"Search URL: {search_url}")
        
        # Open in browser
        webbrowser.open(search_url)
        
        return True
    
    def record_finding(self, found: bool, md5_hash: str = "", 
                      download_url: str = "", file_format: str = "",
                      notes: str = "") -> None:
        """
        Record the result of a manual search
        
        Args:
            found: Whether the book was found
            md5_hash: MD5 hash if found
            download_url: Download URL if found
            file_format: File format if found
            notes: Additional notes
        """
        if self.current_index < len(self.template_df):
            self.template_df.loc[self.current_index, 'found_on_anna_archive'] = 'Yes' if found else 'No'
            self.template_df.loc[self.current_index, 'md5_hash'] = md5_hash
            self.template_df.loc[self.current_index, 'download_url'] = download_url
            self.template_df.loc[self.current_index, 'file_format'] = file_format
            self.template_df.loc[self.current_index, 'search_notes'] = notes
            
            print(f"Recorded: {'Found' if found else 'Not found'}")
    
    def save_progress(self, output_file: str) -> None:
        """
        Save current progress to CSV
        
        Args:
            output_file: Path to save progress
        """
        self.template_df.to_csv(output_file, index=False)
        logger.info(f"Progress saved to {output_file}")
    
    def show_statistics(self) -> None:
        """
        Show current search statistics
        """
        total_processed = self.current_index
        if total_processed == 0:
            print("No books processed yet.")
            return
        
        found_count = len(self.template_df[
            (self.template_df['found_on_anna_archive'] == 'Yes') & 
            (self.template_df.index < self.current_index)
        ])
        
        print(f"\nSearch Statistics:")
        print(f"Books processed: {total_processed}")
        print(f"Books found: {found_count}")
        print(f"Success rate: {found_count/total_processed*100:.1f}%")
        print(f"Remaining: {len(self.template_df) - self.current_index}")
    
    def interactive_search(self) -> None:
        """
        Interactive search session
        """
        print("Starting interactive manual search session...")
        print("Commands:")
        print("  'n' or 'next' - Open next book")
        print("  'f' or 'found' - Mark current book as found")
        print("  'nf' or 'not found' - Mark current book as not found")
        print("  's' or 'stats' - Show statistics")
        print("  'q' or 'quit' - Quit and save progress")
        print("  'h' or 'help' - Show this help")
        
        while True:
            if not self.open_next_book():
                break
            
            command = input("\nEnter command (n/f/nf/s/q/h): ").strip().lower()
            
            if command in ['n', 'next']:
                self.current_index += 1
            elif command in ['f', 'found']:
                md5 = input("MD5 hash (optional): ").strip()
                url = input("Download URL (optional): ").strip()
                fmt = input("File format (epub/pdf, optional): ").strip()
                notes = input("Notes (optional): ").strip()
                
                self.record_finding(True, md5, url, fmt, notes)
                self.current_index += 1
            elif command in ['nf', 'not found']:
                notes = input("Notes (optional): ").strip()
                self.record_finding(False, notes=notes)
                self.current_index += 1
            elif command in ['s', 'stats']:
                self.show_statistics()
            elif command in ['q', 'quit']:
                break
            elif command in ['h', 'help']:
                print("Commands:")
                print("  'n' or 'next' - Open next book")
                print("  'f' or 'found' - Mark current book as found")
                print("  'nf' or 'not found' - Mark current book as not found")
                print("  's' or 'stats' - Show statistics")
                print("  'q' or 'quit' - Quit and save progress")
                print("  'h' or 'help' - Show this help")
            else:
                print("Unknown command. Type 'h' for help.")
        
        # Save progress
        output_file = f"manual_search_progress_{int(time.time())}.csv"
        self.save_progress(output_file)
        print(f"Progress saved to {output_file}")


def main():
    """
    Main function for manual search helper
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Search Helper for Anna Archive')
    parser.add_argument('--template', required=True,
                       help='Path to manual search template CSV')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search session')
    parser.add_argument('--book-index', type=int, default=0,
                       help='Start from specific book index')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize helper
    helper = ManualSearchHelper(args.template)
    helper.current_index = args.book_index
    
    if args.interactive:
        helper.interactive_search()
    else:
        # Show first book
        helper.open_next_book()
        print("\nUse --interactive flag for full interactive session")


if __name__ == "__main__":
    main()
