#!/usr/bin/env python3
"""
Real-time monitoring script for review extraction process.

Usage:
    python3 monitor_extraction.py [--pid PID] [--log-file PATH] [--output-file PATH] [--interval SECONDS]
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple
import argparse
from datetime import datetime

from .log_parser import parse_log_progress, find_latest_log_file

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"

def check_process(pid: int) -> Optional[Dict]:
    """Check if process exists and get its stats."""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'pid,state,%cpu,%mem,etime,cmd'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode != 0:
            return None
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Parse header and data
        header = lines[0].split()
        data = lines[1].split(None, len(header) - 1)
        
        # Map column names (ps uses abbreviated names)
        column_map = {
            'PID': 'pid',
            'S': 'state',  # State column is abbreviated as 'S'
            '%CPU': 'cpu',
            '%MEM': 'mem',
            'ETIME': 'etime',
            'CMD': 'cmd'
        }
        
        # Map to dict
        process_info = {}
        for i, key in enumerate(header):
            if i < len(data):
                # Use mapped name or lowercase original
                mapped_key = column_map.get(key, key.lower())
                process_info[mapped_key] = data[i]
        
        # Get command (everything after the header columns)
        cmd = ' '.join(data[len(header)-1:]) if len(data) > len(header) - 1 else ''
        
        return {
            'pid': process_info.get('pid', str(pid)),
            'state': process_info.get('state', '?'),
            'cpu': process_info.get('cpu', '0.0'),
            'mem': process_info.get('mem', '0.0'),
            'etime': process_info.get('etime', '00:00'),
            'cmd': cmd
        }
    except Exception:
        return None

# parse_log_progress is now imported from .log_parser

def get_output_file_info(output_file: Path) -> Dict:
    """Get information about the output file."""
    if not output_file.exists():
        return {'exists': False, 'size': 0, 'lines': 0}
    
    try:
        size = output_file.stat().st_size
        
        # Count lines (quick estimate for large files)
        # For very large files, we'll use wc -l
        try:
            result = subprocess.run(
                ['wc', '-l', str(output_file)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = int(result.stdout.split()[0])
            else:
                lines = 0
        except Exception:
            lines = 0
        
        return {
            'exists': True,
            'size': size,
            'lines': lines
        }
    except Exception:
        return {'exists': False, 'size': 0, 'lines': 0}

def get_recent_log_entries(log_file: Path, num_lines: int = 5) -> list:
    """Get recent log entries."""
    if not log_file.exists():
        return []
    
    try:
        result = subprocess.run(
            ['tail', '-n', str(num_lines), str(log_file)],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except Exception:
        return []

def monitor_extraction(pid: int, log_file: Path, output_file: Path, interval: int = 10):
    """Monitor review extraction progress in real-time."""
    
    print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.BOLD}  Review Extraction Monitor{Colors.RESET}")
    print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"Process ID: {pid}")
    print(f"Log file: {log_file}")
    print(f"Output file: {output_file}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to stop")
    print()
    
    last_processed = 0
    last_written = 0
    
    try:
        while True:
            # Clear screen (optional - comment out if you want to keep history)
            os.system('clear' if os.name != 'nt' else 'cls')
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{Colors.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print(f"{Colors.BOLD}Status at {timestamp}{Colors.RESET}")
            print(f"{Colors.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print()
            
            # Check process status
            print(f"{Colors.BOLD}🔄 Process Status:{Colors.RESET}")
            process_info = check_process(pid)
            if process_info:
                state = process_info['state']
                # Process states: R=Running, S=Sleeping, T=Stopped, Z=Zombie, etc.
                if state == 'R':
                    state_color = Colors.GREEN
                    state_text = 'Running'
                elif state == 'T' or state == 't':
                    state_color = Colors.YELLOW
                    state_text = 'Stopped/Paused'
                elif state == 'S':
                    state_color = Colors.CYAN
                    state_text = 'Sleeping (waiting)'
                else:
                    state_color = Colors.YELLOW
                    state_text = f'State: {state}'
                print(f"  PID: {process_info['pid']}")
                print(f"  State: {state_color}{state_text} ({state}){Colors.RESET}")
                print(f"  CPU: {process_info['cpu']}%")
                print(f"  Memory: {process_info['mem']}%")
                print(f"  Runtime: {process_info['etime']}")
            else:
                print(f"  {Colors.RED}✗ Process not found (may have completed or crashed){Colors.RESET}")
            print()
            
            # Parse log progress
            print(f"{Colors.BOLD}📊 Progress (from log):{Colors.RESET}")
            progress = parse_log_progress(log_file)
            if progress:
                if progress.get('completed'):
                    print(f"  {Colors.GREEN}✓ Extraction completed!{Colors.RESET}")
                    if progress.get('timestamp'):
                        print(f"  Completed at: {progress['timestamp']}")
                elif progress.get('error'):
                    print(f"  {Colors.YELLOW}⚠ Error reading log: {progress['error']}{Colors.RESET}")
                else:
                    processed = progress.get('processed', 0)
                    rate = progress.get('rate', 0.0)
                    matched = progress.get('matched', 0)
                    matched_pct = progress.get('matched_pct', 0.0)
                    english = progress.get('english', 0)
                    english_pct = progress.get('english_pct', 0.0)
                    written = progress.get('written', 0)
                    written_pct = progress.get('written_pct', 0.0)
                    elapsed = progress.get('elapsed_min', 0.0)
                    
                    # Calculate change since last check
                    processed_delta = processed - last_processed if last_processed > 0 else 0
                    written_delta = written - last_written if last_written > 0 else 0
                    
                    print(f"  Reviews processed: {processed:,}")
                    if processed_delta > 0:
                        print(f"    {Colors.CYAN}(+{processed_delta:,} since last check){Colors.RESET}")
                    
                    print(f"  Processing rate: {rate:.1f} reviews/sec")
                    print(f"  Matched: {matched:,} ({matched_pct:.2f}%)")
                    print(f"  English: {english:,} ({english_pct:.2f}% of matched)")
                    print(f"  Written: {written:,} ({written_pct:.2f}% of English)")
                    if written_delta > 0:
                        print(f"    {Colors.CYAN}(+{written_delta:,} since last check){Colors.RESET}")
                    
                    print(f"  Elapsed time: {format_time(elapsed * 60)}")
                    
                    # Estimate time remaining (rough)
                    if rate > 0 and processed > 0:
                        # Very rough estimate - we don't know total reviews
                        # But we can estimate based on current rate
                        remaining_estimate = "Unknown (total reviews unknown)"
                        print(f"  Time remaining: {remaining_estimate}")
                    
                    last_processed = processed
                    last_written = written
            else:
                print(f"  {Colors.YELLOW}⚠ No progress information found in log{Colors.RESET}")
            print()
            
            # Output file info
            print(f"{Colors.BOLD}📁 Output File:{Colors.RESET}")
            file_info = get_output_file_info(output_file)
            if file_info['exists']:
                size = file_info['size']
                lines = file_info['lines']
                print(f"  Size: {format_size(size)}")
                print(f"  Lines: {lines:,} (including header)")
                if lines > 1:
                    print(f"  Reviews in file: {lines - 1:,}")
            else:
                print(f"  {Colors.YELLOW}⚠ File not found yet{Colors.RESET}")
            print()
            
            # Recent log entries
            print(f"{Colors.BOLD}📝 Recent Log Entries:{Colors.RESET}")
            recent_entries = get_recent_log_entries(log_file, 5)
            if recent_entries:
                for entry in recent_entries[-5:]:
                    # Color code by log level
                    if 'ERROR' in entry:
                        color = Colors.RED
                    elif 'WARNING' in entry:
                        color = Colors.YELLOW
                    elif 'INFO' in entry:
                        color = Colors.CYAN
                    else:
                        color = Colors.RESET
                    print(f"  {color}{entry}{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}No log entries available{Colors.RESET}")
            print()
            
            # Wait for next update
            print(f"{Colors.CYAN}Refreshing in {interval} seconds... (Ctrl+C to stop){Colors.RESET}")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print()
        print(f"{Colors.YELLOW}Monitoring stopped by user.{Colors.RESET}")
        sys.exit(0)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monitor review extraction process in real-time'
    )
    parser.add_argument(
        '--pid',
        type=int,
        default=155101,
        help='Process ID to monitor (default: 155101)'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / 'logs' / 'extract_reviews_20251111_190005.log',
        help='Path to log file'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'romance_reviews_english.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Update interval in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect log file if default doesn't exist
    if not args.log_file.exists():
        log_dir = args.log_file.parent
        if log_dir.exists():
            latest_log = find_latest_log_file(log_dir)
            if latest_log:
                args.log_file = latest_log
                print(f"{Colors.YELLOW}Auto-detected log file: {args.log_file}{Colors.RESET}")
    
    monitor_extraction(args.pid, args.log_file, args.output_file, args.interval)

if __name__ == '__main__':
    main()

