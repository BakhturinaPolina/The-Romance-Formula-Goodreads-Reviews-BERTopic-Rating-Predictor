#!/usr/bin/env python3
"""
Book Download Research Component - Repair Existing Files
One-shot repair script to fix mislabeled files in the download directory
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aa_epub_guard import sniff_format, ensure_valid_epub

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def repair_existing_files(download_dir: str = None):
    """
    Repair mislabeled files in the download directory
    
    Args:
        download_dir: Path to download directory (defaults to organized_outputs/anna_archive_download)
    """
    if download_dir is None:
        download_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download"
    
    root = Path(download_dir)
    
    if not root.exists():
        logger.error(f"Download directory not found: {download_dir}")
        return False
    
    logger.info(f"Scanning download directory: {download_dir}")
    
    # Check for calibre availability
    calibre_bin = os.environ.get("EBOOK_CONVERT_BIN", "ebook-convert")
    calibre_available = check_calibre_availability(calibre_bin)
    
    if not calibre_available:
        logger.warning("Calibre not available - will only rename files, not convert")
        convert_mobi = False
    else:
        logger.info("Calibre available - will convert MOBI files to EPUB")
        convert_mobi = True
    
    repaired_count = 0
    error_count = 0
    
    # Process all .epub files
    for epub_file in root.glob("*.epub"):
        try:
            logger.info(f"Checking {epub_file.name}")
            
            # Sniff the actual format
            actual_format = sniff_format(epub_file)
            logger.info(f"  Detected format: {actual_format}")
            
            if actual_format == "epub":
                logger.info(f"  ✓ Already valid EPUB")
                continue
            elif actual_format == "mobi":
                logger.info(f"  🔄 MOBI file mislabeled as EPUB - fixing...")
                try:
                    final_path = ensure_valid_epub(epub_file, convert_mobi=convert_mobi, calibre_bin=calibre_bin)
                    if str(final_path) != str(epub_file):
                        logger.info(f"  ✓ Fixed: {epub_file.name} -> {final_path.name}")
                        repaired_count += 1
                    else:
                        logger.info(f"  ✓ Renamed to .mobi: {epub_file.name}")
                        repaired_count += 1
                except Exception as e:
                    logger.error(f"  ✗ Failed to fix {epub_file.name}: {e}")
                    error_count += 1
            elif actual_format == "zip":
                logger.error(f"  ✗ Invalid EPUB structure (ZIP but not EPUB): {epub_file.name}")
                error_count += 1
            elif actual_format == "html":
                logger.error(f"  ✗ HTML file (likely error page): {epub_file.name}")
                error_count += 1
            else:
                logger.warning(f"  ? Unknown format: {epub_file.name}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"  ✗ Error processing {epub_file.name}: {e}")
            error_count += 1
    
    # Process any .mobi files
    mobi_files = list(root.glob("*.mobi"))
    if mobi_files:
        logger.info(f"Found {len(mobi_files)} MOBI files")
        for mobi_file in mobi_files:
            logger.info(f"  📱 {mobi_file.name} (already correct format)")
    
    # Summary
    logger.info(f"\n=== REPAIR SUMMARY ===")
    logger.info(f"Files repaired: {repaired_count}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info(f"Calibre available: {calibre_available}")
    
    if repaired_count > 0:
        logger.info("✓ Some files were successfully repaired")
    if error_count > 0:
        logger.warning("⚠️  Some files had issues - check logs above")
    
    return error_count == 0

def check_calibre_availability(calibre_bin: str) -> bool:
    """Check if calibre is available for conversion"""
    import subprocess
    
    try:
        result = subprocess.run([calibre_bin, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✓ Calibre available: {result.stdout.strip()}")
            return True
        else:
            logger.warning(f"✗ Calibre not working: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning("✗ Calibre not found in PATH")
        logger.info("To install calibre:")
        logger.info("  Ubuntu/Debian: sudo apt-get install calibre")
        logger.info("  macOS: brew install --cask calibre")
        logger.info("  Windows: Download from https://calibre-ebook.com/download")
        return False
    except Exception as e:
        logger.error(f"✗ Calibre test error: {e}")
        return False

def main():
    """Main function"""
    logger.info("=== REPAIRING EXISTING DOWNLOAD FILES ===")
    
    success = repair_existing_files()
    
    if success:
        logger.info("\n🎉 File repair completed successfully!")
    else:
        logger.info("\n⚠️  File repair completed with some issues")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
