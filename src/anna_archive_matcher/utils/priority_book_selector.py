#!/usr/bin/env python3
"""
Priority Book Selector for Manual Search
Creates prioritized lists of romance books for manual searching on Anna's Archive
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class PriorityBookSelector:
    """
    Select priority books for manual search and download
    """
    
    def __init__(self, romance_csv_path: str):
        """
        Initialize with romance books dataset
        
        Args:
            romance_csv_path: Path to romance books CSV file
        """
        self.romance_csv_path = romance_csv_path
        self.df = pd.read_csv(romance_csv_path)
        logger.info(f"Loaded {len(self.df)} romance books")
    
    def create_priority_lists(self, output_dir: str = "priority_lists") -> Dict[str, str]:
        """
        Create multiple priority lists for different search strategies
        
        Args:
            output_dir: Directory to save priority lists
            
        Returns:
            Dictionary mapping list names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        priority_lists = {}
        
        # List 1: Top-rated popular books
        top_rated = self.df[
            (self.df['average_rating_weighted_mean'] > 4.0) & 
            (self.df['ratings_count_sum'] > 50000)
        ].sort_values('average_rating_weighted_mean', ascending=False)
        
        top_rated_path = output_path / "top_rated_popular_books.csv"
        top_rated.to_csv(top_rated_path, index=False)
        priority_lists['top_rated'] = str(top_rated_path)
        logger.info(f"Created top-rated list: {len(top_rated)} books")
        
        # List 2: Most reviewed books
        most_reviewed = self.df[
            self.df['ratings_count_sum'] > 100000
        ].sort_values('ratings_count_sum', ascending=False)
        
        most_reviewed_path = output_path / "most_reviewed_books.csv"
        most_reviewed.to_csv(most_reviewed_path, index=False)
        priority_lists['most_reviewed'] = str(most_reviewed_path)
        logger.info(f"Created most-reviewed list: {len(most_reviewed)} books")
        
        # List 3: Recent popular books (2010+)
        recent_popular = self.df[
            (self.df['publication_year'] >= 2010) & 
            (self.df['ratings_count_sum'] > 20000)
        ].sort_values('publication_year', ascending=False)
        
        recent_popular_path = output_path / "recent_popular_books.csv"
        recent_popular.to_csv(recent_popular_path, index=False)
        priority_lists['recent_popular'] = str(recent_popular_path)
        logger.info(f"Created recent popular list: {len(recent_popular)} books")
        
        # List 4: Popular authors (multiple books)
        author_counts = self.df['author_name'].value_counts()
        popular_authors = author_counts[author_counts >= 3].index
        
        popular_author_books = self.df[
            self.df['author_name'].isin(popular_authors)
        ].sort_values(['author_name', 'ratings_count_sum'], ascending=[True, False])
        
        popular_author_path = output_path / "popular_author_books.csv"
        popular_author_books.to_csv(popular_author_path, index=False)
        priority_lists['popular_authors'] = str(popular_author_path)
        logger.info(f"Created popular authors list: {len(popular_author_books)} books")
        
        # List 5: Series books
        series_books = self.df[
            (self.df['series_title'] != 'stand_alone') & 
            (self.df['series_title'].notna())
        ].sort_values(['series_title', 'publication_year'])
        
        series_path = output_path / "series_books.csv"
        series_books.to_csv(series_path, index=False)
        priority_lists['series'] = str(series_path)
        logger.info(f"Created series list: {len(series_books)} books")
        
        # List 6: Small sample for testing (50 books)
        test_sample = self.df.sample(n=min(50, len(self.df)), random_state=42)
        test_sample_path = output_path / "test_sample_50_books.csv"
        test_sample.to_csv(test_sample_path, index=False)
        priority_lists['test_sample'] = str(test_sample_path)
        logger.info(f"Created test sample: {len(test_sample)} books")
        
        return priority_lists
    
    def create_search_queries(self, priority_list_path: str, output_file: str) -> None:
        """
        Create search queries for Anna's Archive
        
        Args:
            priority_list_path: Path to priority list CSV
            output_file: Path to save search queries
        """
        df = pd.read_csv(priority_list_path)
        
        queries = []
        for idx, book in df.iterrows():
            # Create multiple search query variations
            title = book['title']
            author = book['author_name']
            
            # Query 1: Exact title and author
            query1 = f'"{title}" "{author}"'
            
            # Query 2: Title with romance keyword
            query2 = f'"{title}" romance'
            
            # Query 3: Author with romance keyword
            query3 = f'"{author}" romance'
            
            # Query 4: Simplified title
            simplified_title = title.split(':')[0].split('(')[0].strip()
            query4 = f'"{simplified_title}" "{author}"'
            
            queries.extend([
                {
                    'work_id': book['work_id'],
                    'title': title,
                    'author': author,
                    'query': query1,
                    'type': 'exact'
                },
                {
                    'work_id': book['work_id'],
                    'title': title,
                    'author': author,
                    'query': query2,
                    'type': 'title_romance'
                },
                {
                    'work_id': book['work_id'],
                    'title': title,
                    'author': author,
                    'query': query3,
                    'type': 'author_romance'
                },
                {
                    'work_id': book['work_id'],
                    'title': title,
                    'author': author,
                    'query': query4,
                    'type': 'simplified'
                }
            ])
        
        queries_df = pd.DataFrame(queries)
        queries_df.to_csv(output_file, index=False)
        logger.info(f"Created {len(queries)} search queries in {output_file}")
    
    def create_manual_search_template(self, priority_list_path: str, output_file: str) -> None:
        """
        Create a template for manual search tracking
        
        Args:
            priority_list_path: Path to priority list CSV
            output_file: Path to save search template
        """
        df = pd.read_csv(priority_list_path)
        
        # Add columns for manual search tracking
        template_df = df.copy()
        template_df['found_on_anna_archive'] = ''
        template_df['anna_archive_title'] = ''
        template_df['anna_archive_author'] = ''
        template_df['md5_hash'] = ''
        template_df['download_url'] = ''
        template_df['file_format'] = ''
        template_df['file_size'] = ''
        template_df['search_notes'] = ''
        template_df['download_status'] = ''
        
        template_df.to_csv(output_file, index=False)
        logger.info(f"Created manual search template: {output_file}")


def main():
    """
    Main function to create priority lists
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Priority Book Lists')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--output-dir', default='priority_lists',
                       help='Output directory for priority lists')
    parser.add_argument('--create-queries', action='store_true',
                       help='Create search queries for Anna Archive')
    parser.add_argument('--create-template', action='store_true',
                       help='Create manual search template')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize selector
    selector = PriorityBookSelector(args.romance_csv)
    
    # Create priority lists
    priority_lists = selector.create_priority_lists(args.output_dir)
    
    print("\nCreated priority lists:")
    for name, path in priority_lists.items():
        print(f"  {name}: {path}")
    
    # Create search queries if requested
    if args.create_queries:
        print("\nCreating search queries...")
        for name, path in priority_lists.items():
            if name == 'test_sample':  # Start with test sample
                queries_file = Path(args.output_dir) / f"{name}_search_queries.csv"
                selector.create_search_queries(path, str(queries_file))
                print(f"  Created queries: {queries_file}")
    
    # Create manual search template if requested
    if args.create_template:
        print("\nCreating manual search template...")
        test_sample_path = priority_lists['test_sample']
        template_file = Path(args.output_dir) / "manual_search_template.csv"
        selector.create_manual_search_template(test_sample_path, str(template_file))
        print(f"  Created template: {template_file}")
    
    print(f"\nNext steps:")
    print(f"1. Start with: {priority_lists['test_sample']}")
    print(f"2. Search these books manually on Anna's Archive")
    print(f"3. Use the template to track your findings")


if __name__ == "__main__":
    main()
