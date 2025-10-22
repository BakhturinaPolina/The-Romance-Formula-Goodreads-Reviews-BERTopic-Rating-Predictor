#!/usr/bin/env python3
"""
Create download instructions and alternative methods for accessing books
"""

import csv
import json
from pathlib import Path

def create_download_instructions():
    """Create comprehensive download instructions"""
    
    print("📖 Creating Download Instructions and Alternative Methods")
    print("=" * 60)
    
    # Read the MD5 hashes CSV
    books = []
    with open('book_md5_hashes.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            books.append(row)
    
    print(f"📚 Processing {len(books)} books with MD5 hashes")
    
    # Create detailed instructions
    instructions = """
# 📚 Book Download Instructions

## 🎯 Overview
You have **229 books** with MD5 hashes ready for download from multiple sources.

## 🔍 What You Have
- **Z-Library**: 216 books (including romance novels)
- **LibGen Fiction**: 13 books
- **Total**: 229 books with verified MD5 hashes

## 📥 Download Methods

### Method 1: Direct MD5 Search
Use the MD5 hashes to search on these sites:

#### Z-Library Sites:
- https://b-ok.cc/
- https://z-lib.org/
- https://1lib.limited/
- https://booksc.eu/

#### LibGen Sites:
- https://libgen.li/
- https://libgen.rs/
- https://libgen.is/

### Method 2: MD5 Hash Search
1. Go to any of the above sites
2. Use the search function
3. Search by MD5 hash (32-character string)
4. Download the file

### Method 3: Alternative Download Tools
- **JDownloader2**: Can handle many of these sites
- **wget/curl with proper headers**: May work with some sites
- **Browser extensions**: Some sites have browser extensions

## 📋 Book Lists by Category

"""
    
    # Categorize books
    romance_books = []
    fiction_books = []
    other_books = []
    
    for book in books:
        title_lower = book['title'].lower()
        if any(keyword in title_lower for keyword in ['romance', 'love', 'heart', 'kiss', 'wedding', 'bride', 'groom']):
            romance_books.append(book)
        elif book['source'] == 'LibGen Fiction':
            fiction_books.append(book)
        else:
            other_books.append(book)
    
    # Add romance books section
    if romance_books:
        instructions += f"\n### 💕 Romance Novels ({len(romance_books)} books)\n\n"
        for book in romance_books:
            instructions += f"- **{book['title']}** by {book['author']} ({book['year']})\n"
            instructions += f"  - MD5: `{book['md5']}`\n"
            instructions += f"  - Format: {book['extension']}\n"
            instructions += f"  - Size: {book['filesize']} bytes\n\n"
    
    # Add fiction books section
    if fiction_books:
        instructions += f"\n### 📖 Fiction Books ({len(fiction_books)} books)\n\n"
        for book in fiction_books:
            instructions += f"- **{book['title']}** by {book['author']} ({book['year']})\n"
            instructions += f"  - MD5: `{book['md5']}`\n"
            instructions += f"  - Format: {book['extension']}\n"
            instructions += f"  - Size: {book['filesize']} bytes\n\n"
    
    # Add other books section
    if other_books:
        instructions += f"\n### 📚 Other Books ({len(other_books)} books)\n\n"
        for book in other_books[:10]:  # Show first 10
            instructions += f"- **{book['title']}** by {book['author']} ({book['year']})\n"
            instructions += f"  - MD5: `{book['md5']}`\n"
            instructions += f"  - Format: {book['extension']}\n"
            instructions += f"  - Size: {book['filesize']} bytes\n\n"
        
        if len(other_books) > 10:
            instructions += f"... and {len(other_books) - 10} more books (see CSV file for complete list)\n\n"
    
    # Add technical details
    instructions += """
## 🔧 Technical Details

### MD5 Hash Verification
After downloading, verify the file integrity:
```bash
md5sum downloaded_file.epub
# Should match the MD5 hash from the CSV
```

### File Formats Available
- **EPUB**: Most common e-book format
- **PDF**: Portable Document Format
- **FB2**: FictionBook format
- **LRF**: Sony Reader format
- **LIT**: Microsoft Reader format

### Download Tips
1. **Use different mirrors**: If one site is down, try another
2. **Check file size**: Should match the expected size in the CSV
3. **Verify MD5**: Always verify the downloaded file's MD5 hash
4. **Use VPN if needed**: Some sites may be geo-blocked

## 📁 Files Provided
- `book_md5_hashes.csv`: Complete list with all details
- `download_instructions.md`: This file
- `romance_books_only.csv`: Romance novels only
- `fiction_books_only.csv`: Fiction books only

## 🚀 Quick Start
1. Open `book_md5_hashes.csv` in a spreadsheet program
2. Find a book you want
3. Copy the MD5 hash
4. Go to https://b-ok.cc/ or https://libgen.li/
5. Search for the MD5 hash
6. Download the file
7. Verify with MD5 hash

Happy reading! 📚✨
"""
    
    # Save instructions
    with open('download_instructions.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Created: download_instructions.md")
    
    # Create category-specific CSV files
    if romance_books:
        with open('romance_books_only.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['source', 'md5', 'title', 'author', 'year', 'extension', 'filesize', 'download_url', 'additional_info']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for book in romance_books:
                writer.writerow(book)
        print(f"✅ Created: romance_books_only.csv ({len(romance_books)} books)")
    
    if fiction_books:
        with open('fiction_books_only.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['source', 'md5', 'title', 'author', 'year', 'extension', 'filesize', 'download_url', 'additional_info']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for book in fiction_books:
                writer.writerow(book)
        print(f"✅ Created: fiction_books_only.csv ({len(fiction_books)} books)")
    
    # Create a simple text list for easy copying
    with open('md5_hashes_list.txt', 'w') as f:
        f.write("# MD5 Hashes for Book Downloads\n\n")
        for book in books:
            f.write(f"{book['md5']} - {book['title']} by {book['author']} ({book['year']})\n")
    
    print("✅ Created: md5_hashes_list.txt")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  💕 Romance novels: {len(romance_books)}")
    print(f"  📖 Fiction books: {len(fiction_books)}")
    print(f"  📚 Other books: {len(other_books)}")
    print(f"  📁 Total files created: 4")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Read download_instructions.md for detailed instructions")
    print(f"  2. Use the MD5 hashes to search on Z-Library or LibGen sites")
    print(f"  3. Download books manually using the provided hashes")
    print(f"  4. Verify downloads using MD5 hash verification")

def main():
    create_download_instructions()

if __name__ == "__main__":
    main()
