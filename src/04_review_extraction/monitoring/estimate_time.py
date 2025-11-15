#!/usr/bin/env python3
"""
Estimate remaining time for review extraction.

Usage:
    python3 estimate_time.py [--log-file PATH]
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

from .log_parser import parse_log_progress, find_latest_log_file

# Total reviews in the input file (counted once)
TOTAL_REVIEWS = 3_565_378


def calculate_estimate(processed: int, rate: float) -> dict:
    """Calculate time estimates."""
    remaining = TOTAL_REVIEWS - processed
    
    # Time estimates (in hours)
    time_remaining_hours = remaining / (rate * 3600)
    
    # Conservative (10% slower)
    conservative_rate = rate * 0.9
    time_remaining_conservative = remaining / (conservative_rate * 3600)
    
    # Optimistic (10% faster)
    optimistic_rate = rate * 1.1
    time_remaining_optimistic = remaining / (optimistic_rate * 3600)
    
    # Completion times
    now = datetime.now()
    completion_current = now + timedelta(hours=time_remaining_hours)
    completion_conservative = now + timedelta(hours=time_remaining_conservative)
    completion_optimistic = now + timedelta(hours=time_remaining_optimistic)
    
    progress_pct = (processed / TOTAL_REVIEWS) * 100
    
    return {
        'remaining': remaining,
        'progress_pct': progress_pct,
        'time_remaining_hours': time_remaining_hours,
        'time_remaining_conservative': time_remaining_conservative,
        'time_remaining_optimistic': time_remaining_optimistic,
        'completion_current': completion_current,
        'completion_conservative': completion_conservative,
        'completion_optimistic': completion_optimistic
    }


def main():
    parser = argparse.ArgumentParser(
        description='Estimate remaining time for review extraction'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Path to log file (auto-detects latest if not specified)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect log file if not specified
    if not args.log_file:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent.parent
        log_dir = project_root / 'logs'
        latest_log = find_latest_log_file(log_dir)
        if latest_log:
            args.log_file = latest_log
    
    if not args.log_file or not args.log_file.exists():
        print("Error: Could not find log file")
        return 1
    
    # Parse progress
    progress = parse_log_progress(args.log_file)
    
    if not progress:
        print("Error: Could not parse progress from log file")
        return 1
    
    if progress.get('completed'):
        print("=" * 70)
        print("EXTRACTION COMPLETED!")
        print("=" * 70)
        if progress.get('timestamp'):
            print(f"Completed at: {progress['timestamp']}")
        return 0
    
    if progress.get('error'):
        print(f"Error: {progress['error']}")
        return 1
    
    processed = progress['processed']
    rate = progress['rate']
    elapsed = progress['elapsed_min']
    
    # Calculate estimates
    estimates = calculate_estimate(processed, rate)
    
    # Print results
    print("=" * 70)
    print("REVIEW EXTRACTION TIME ESTIMATE")
    print("=" * 70)
    print(f"\nTotal reviews in file: {TOTAL_REVIEWS:,}")
    print(f"Reviews processed: {processed:,}")
    print(f"Remaining reviews: {estimates['remaining']:,}")
    print(f"\nCurrent processing rate: {rate:.1f} reviews/sec")
    print(f"Elapsed time: {elapsed:.1f} minutes")
    print("\n" + "-" * 70)
    print("TIME REMAINING ESTIMATES:")
    print("-" * 70)
    print(f"  Optimistic (10% faster):  {estimates['time_remaining_optimistic']:.2f} hours ({estimates['time_remaining_optimistic']*60:.1f} minutes)")
    print(f"  Current rate:              {estimates['time_remaining_hours']:.2f} hours ({estimates['time_remaining_hours']*60:.1f} minutes)")
    print(f"  Conservative (10% slower): {estimates['time_remaining_conservative']:.2f} hours ({estimates['time_remaining_conservative']*60:.1f} minutes)")
    print("\n" + "-" * 70)
    print("ESTIMATED COMPLETION TIME:")
    print("-" * 70)
    print(f"  Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Optimistic:   {estimates['completion_optimistic'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Current rate: {estimates['completion_current'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Conservative: {estimates['completion_conservative'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-" * 70)
    print("PROGRESS:")
    print("-" * 70)
    print(f"  Completed: {estimates['progress_pct']:.2f}%")
    print(f"  Remaining: {100 - estimates['progress_pct']:.2f}%")
    print("\n" + "=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())

