#!/usr/bin/env python3
"""
Shared log parsing utilities for review extraction monitoring.

This module provides common functions for parsing progress information
from review extraction log files.
"""

import re
from pathlib import Path
from typing import Optional, Dict


def parse_log_progress(log_file: Path) -> Optional[Dict]:
    """
    Parse the latest progress from log file.
    
    Returns a dictionary with progress information, or None if no progress found.
    Returns {'completed': True} if extraction is complete.
    Returns {'error': str} if an error occurred.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary with progress information or None/error dict
    """
    if not log_file.exists():
        return None
    
    try:
        # Read last 100 lines to find latest progress
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Search backwards for progress line
            for line in reversed(lines[-100:]):
                if 'Progress:' in line:
                    # Parse: "Progress: 2,305,000 processed | Rate: 144.8 reviews/sec | Matched: 1,322,565 (57.38%) | English: 1,232,141 (93.16% of matched) | Written: 1,232,141 (100.00% of English) | Elapsed: 265.3 min"
                    match = re.search(r'Progress: ([\d,]+) processed', line)
                    if match:
                        processed = int(match.group(1).replace(',', ''))
                        
                        # Extract other stats
                        rate_match = re.search(r'Rate: ([\d.]+) reviews/sec', line)
                        matched_match = re.search(r'Matched: ([\d,]+) \(([\d.]+)%\)', line)
                        english_match = re.search(r'English: ([\d,]+) \(([\d.]+)% of matched\)', line)
                        written_match = re.search(r'Written: ([\d,]+) \(([\d.]+)% of English\)', line)
                        elapsed_match = re.search(r'Elapsed: ([\d.]+) min', line)
                        
                        return {
                            'processed': processed,
                            'rate': float(rate_match.group(1)) if rate_match else 0.0,
                            'matched': int(matched_match.group(1).replace(',', '')) if matched_match else 0,
                            'matched_pct': float(matched_match.group(2)) if matched_match else 0.0,
                            'english': int(english_match.group(1).replace(',', '')) if english_match else 0,
                            'english_pct': float(english_match.group(2)) if english_match else 0.0,
                            'written': int(written_match.group(1).replace(',', '')) if written_match else 0,
                            'written_pct': float(written_match.group(2)) if written_match else 0.0,
                            'elapsed_min': float(elapsed_match.group(1)) if elapsed_match else 0.0,
                            'timestamp': line.split(' - ')[0] if ' - ' in line else ''
                        }
        
        # If no progress found, check for completion message
        for line in reversed(lines[-50:]):
            if 'Extraction complete!' in line or 'Script completed successfully!' in line:
                return {'completed': True, 'timestamp': line.split(' - ')[0] if ' - ' in line else ''}
        
        return None
    except Exception as e:
        return {'error': str(e)}


def find_latest_log_file(log_dir: Path) -> Optional[Path]:
    """
    Find the latest extract_reviews log file in the given directory.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Path to latest log file or None if not found
    """
    if not log_dir.exists():
        return None
    
    log_files = sorted(
        log_dir.glob('extract_reviews_*.log'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    return log_files[0] if log_files else None

