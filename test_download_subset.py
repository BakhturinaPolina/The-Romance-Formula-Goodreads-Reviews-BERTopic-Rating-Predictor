#!/usr/bin/env python3
"""
Test download functionality with a small subset of books
"""

import csv
import subprocess
import os
from pathlib import Path

def test_download_subset():
    """Test downloading a small subset of books"""
    
    print("🧪 Testing Download Functionality with Subset")
    print("=" * 50)
    
    # Read the MD5 hashes CSV
    books = []
    with open('book_md5_hashes.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            books.append(row)
    
    print(f"📚 Total books available: {len(books)}")
    
    # Select a small subset for testing (first 5 books)
    test_books = books[:5]
    
    print(f"🎯 Testing with {len(test_books)} books:")
    for i, book in enumerate(test_books, 1):
        print(f"  {i}. \"{book['title']}\" by {book['author']} ({book['year']}) - {book['extension']}")
    
    # Create test directory
    test_dir = Path("test_downloads")
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Created test directory: {test_dir}")
    
    # Test downloads
    successful_downloads = 0
    failed_downloads = 0
    
    for i, book in enumerate(test_books, 1):
        print(f"\n📥 Downloading {i}/{len(test_books)}: {book['title']}")
        
        # Create filename
        title_clean = "".join(c for c in book['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        author_clean = "".join(c for c in book['author'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{author_clean} - {title_clean}.{book['extension']}"
        filepath = test_dir / filename
        
        # Download using curl
        try:
            cmd = [
                'curl', '-L', 
                '--connect-timeout', '30',
                '--max-time', '300',  # 5 minute timeout
                '--retry', '2',
                '--retry-delay', '5',
                '--user-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                '-o', str(filepath),
                book['download_url']
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and filepath.exists() and filepath.stat().st_size > 0:
                file_size = filepath.stat().st_size
                print(f"  ✅ Success! Downloaded {file_size:,} bytes")
                successful_downloads += 1
            else:
                print(f"  ❌ Failed! Return code: {result.returncode}")
                if result.stderr:
                    print(f"     Error: {result.stderr.strip()}")
                failed_downloads += 1
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout! Download took too long")
            failed_downloads += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_downloads += 1
    
    # Summary
    print(f"\n📊 Download Test Results:")
    print(f"  ✅ Successful: {successful_downloads}")
    print(f"  ❌ Failed: {failed_downloads}")
    print(f"  📁 Files saved to: {test_dir}")
    
    if successful_downloads > 0:
        print(f"\n🎉 Test successful! {successful_downloads} books downloaded.")
        print("   You can now run the full download script with confidence.")
        
        # Show downloaded files
        print(f"\n📁 Downloaded files:")
        for file in test_dir.iterdir():
            if file.is_file():
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
    else:
        print(f"\n⚠️  Test failed! No books were downloaded successfully.")
        print("   Check your internet connection and try again.")
    
    return successful_downloads > 0

def create_romance_subset_script():
    """Create a script to download only romance novels"""
    
    print(f"\n💕 Creating Romance Novel Download Script...")
    
    # Read the MD5 hashes CSV
    romance_books = []
    with open('book_md5_hashes.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Look for romance-related keywords
            title_lower = row['title'].lower()
            if any(keyword in title_lower for keyword in ['romance', 'love', 'heart', 'kiss', 'wedding', 'bride', 'groom']):
                romance_books.append(row)
    
    if not romance_books:
        print("  ❌ No romance novels found in the dataset")
        return
    
    print(f"  📚 Found {len(romance_books)} romance novels")
    
    # Create romance download script
    with open('download_romance_books.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Download romance novels using MD5 hashes\n")
        f.write("# Usage: ./download_romance_books.sh\n\n")
        
        f.write("mkdir -p romance_downloads\n")
        f.write("cd romance_downloads\n\n")
        
        for book in romance_books:
            title_clean = "".join(c for c in book['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            author_clean = "".join(c for c in book['author'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename_clean = f"{author_clean} - {title_clean}.{book['extension']}"
            
            f.write(f"# {book['title']} by {book['author']} ({book['year']})\n")
            f.write(f"echo \"Downloading: {book['title']} by {book['author']}\"\n")
            f.write(f"curl -L -o \"{filename_clean}\" \"{book['download_url']}\"\n\n")
    
    # Make executable
    os.chmod('download_romance_books.sh', 0o755)
    
    print(f"  ✅ Created: download_romance_books.sh")
    print(f"  📚 Contains {len(romance_books)} romance novels")
    
    # Show romance books found
    print(f"\n💕 Romance novels found:")
    for book in romance_books:
        print(f"  - \"{book['title']}\" by {book['author']} ({book['year']})")

def main():
    print("🎯 Book Download Test Suite")
    print("=" * 50)
    
    # Test with small subset
    success = test_download_subset()
    
    if success:
        # Create romance-specific script
        create_romance_subset_script()
        
        print(f"\n🚀 Ready for full downloads!")
        print(f"   Available scripts:")
        print(f"   - ./download_books.sh (all 229 books)")
        print(f"   - ./download_romance_books.sh (romance novels only)")
    else:
        print(f"\n⚠️  Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
