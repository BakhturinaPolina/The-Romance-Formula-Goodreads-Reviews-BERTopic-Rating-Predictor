#!/usr/bin/env python3
"""
Book Search CLI for Anna's Archive Local Data

Simple command-line interface for searching books in Anna's Archive
Parquet data using the query engine.

Usage:
    python book_search_cli.py --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
                             --title "Fifty Shades" --author "E.L. James"
"""

import argparse
import sys
from pathlib import Path

from query_engine import BookSearchEngine


def format_book_result(book: dict, index: int = None) -> str:
    """Format a book result for display."""
    lines = []
    
    if index is not None:
        lines.append(f"{index}. {book.get('title', 'No title')} by {book.get('author', 'Unknown author')}")
    else:
        lines.append(f"{book.get('title', 'No title')} by {book.get('author', 'Unknown author')}")
    
    # Add additional details
    details = []
    if book.get('year'):
        details.append(f"Year: {book['year']}")
    if book.get('publisher'):
        details.append(f"Publisher: {book['publisher']}")
    if book.get('extension'):
        details.append(f"Format: {book['extension']}")
    if book.get('language'):
        details.append(f"Language: {book['language']}")
    if book.get('file_size'):
        details.append(f"Size: {book['file_size']} bytes")
    
    if details:
        lines.append("   " + " | ".join(details))
    
    # Add MD5 hash
    if book.get('md5'):
        lines.append(f"   MD5: {book['md5']}")
    
    return "\n".join(lines)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Search Anna's Archive book data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search by title and author
  python book_search_cli.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --title "Fifty Shades" \\
    --author "E.L. James"

  # Search by title only
  python book_search_cli.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --title "Romeo and Juliet"

  # Search by author only
  python book_search_cli.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --author "Shakespeare"

  # Get dataset statistics
  python book_search_cli.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --stats

  # Show random books
  python book_search_cli.py \\
    --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --random 5
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
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate parquet directory
    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        print(f"Error: Parquet directory not found: {args.parquet_dir}")
        return 1
    
    try:
        with BookSearchEngine(args.parquet_dir) as engine:
            if args.stats:
                print("📊 Dataset Statistics")
                print("=" * 50)
                stats = engine.get_stats()
                
                print(f"Total records: {stats['total_records']:,}")
                print(f"Schema fields: {stats['schema_fields']}")
                print()
                
                print("Field coverage:")
                for field in ['title', 'author', 'md5', 'extension']:
                    count = stats.get(f'{field}_count', 0)
                    percentage = (count / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
                    print(f"  {field.capitalize()}: {count:,} ({percentage:.1f}%)")
                
                print()
                print("Available fields:")
                for field in stats['available_fields']:
                    print(f"  - {field}")
            
            elif args.random:
                print(f"🎲 Random {args.random} Books")
                print("=" * 50)
                books = engine.get_random_books(args.random)
                
                for i, book in enumerate(books, 1):
                    print(format_book_result(book, i))
                    print()
            
            elif args.md5:
                print(f"🔍 Search by MD5: {args.md5}")
                print("=" * 50)
                books = engine.search_by_md5(args.md5)
                
                if books:
                    for book in books:
                        print(format_book_result(book))
                        print()
                else:
                    print("No books found with that MD5 hash.")
            
            elif args.title or args.author:
                # Build search description
                search_parts = []
                if args.title:
                    search_parts.append(f"title: '{args.title}'")
                if args.author:
                    search_parts.append(f"author: '{args.author}'")
                
                search_desc = " and ".join(search_parts)
                match_type = "exact" if args.exact else "fuzzy"
                
                print(f"🔍 Search Results ({match_type} matching)")
                print(f"Query: {search_desc}")
                print("=" * 50)
                
                books = engine.search_by_title_author(
                    args.title or "",
                    args.author or "",
                    fuzzy=not args.exact,
                    limit=args.limit
                )
                
                if books:
                    print(f"Found {len(books)} books:")
                    print()
                    
                    for i, book in enumerate(books, 1):
                        print(format_book_result(book, i))
                        print()
                else:
                    print("No books found matching your criteria.")
                    print()
                    print("💡 Tips:")
                    print("  - Try using --exact for precise matching")
                    print("  - Use partial titles/authors for broader results")
                    print("  - Check spelling and try variations")
            
            else:
                print("❌ No search criteria provided.")
                print()
                print("Available options:")
                print("  --title 'Book Title'     Search by title")
                print("  --author 'Author Name'   Search by author")
                print("  --md5 'hash'            Search by MD5 hash")
                print("  --stats                 Show dataset statistics")
                print("  --random N              Show N random books")
                print()
                print("Use --help for more information.")
                return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
