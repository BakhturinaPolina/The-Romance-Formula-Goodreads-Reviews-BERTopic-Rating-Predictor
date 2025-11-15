#!/usr/bin/env python3
"""
Monitoring script for prepare_bertopic_input.py.

Monitors progress and displays updates in real-time.
"""

import sys
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

from venv_utils import get_project_root, verify_venv

# Verify we're using the correct venv
verify_venv()

project_root = get_project_root()
log_file = Path("/tmp/bertopic_prep_monitor.log")
output_file = project_root / "data" / "processed" / "review_sentences_for_bertopic.parquet"
monitor_interval = 30  # Check every 30 seconds

# Change to project root
import os
os.chdir(project_root)

print("=" * 50)
print("BERTopic Preparation Monitor")
print("=" * 50)
print(f"Monitoring log: {log_file}")
print(f"Output file: {output_file}")
print(f"Check interval: {monitor_interval}s")
print("Press Ctrl+C to stop monitoring")
print("=" * 50)
print()

def get_process_info():
    """Get information about the prepare_bertopic_input process."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        lines = result.stdout.split('\n')
        for line in lines:
            if 'prepare_bertopic_input' in line and 'grep' not in line and 'monitor' not in line and 'bash' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'runtime': parts[9]
                    }
    except Exception:
        pass
    return None

def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

try:
    while True:
        # Clear screen
        subprocess.run(['clear'], check=False)
        
        print("=" * 50)
        print(f"BERTopic Preparation Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        print()
        
        # Check process
        process_info = get_process_info()
        if not process_info:
            print("⚠️  Process Status: NOT RUNNING")
            print()
            print("Checking if output file exists...")
            if output_file.exists():
                print("✓ Output file exists!")
                size = format_size(output_file.stat().st_size)
                print(f"   {output_file.name}: {size}")
                print()
                print("Script may have completed. Check log for details.")
            else:
                print("✗ Output file not found. Script may have failed.")
        else:
            print(f"✓ Process Status: RUNNING (PID: {process_info['pid']})")
            print(f"  CPU: {process_info['cpu']}% | Memory: {process_info['mem']}% | Runtime: {process_info['runtime']}")
            print()
            
            if output_file.exists():
                size = format_size(output_file.stat().st_size)
                print(f"✓ Output file exists: {size}")
            else:
                print("⏳ Output file not yet created...")
        
        print()
        print("--- Latest Progress (last 15 lines) ---")
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    relevant = [l for l in lines[-15:] if re.search(r'(Chunk|Progress|Processed|ETA|complete|✓|ERROR|WARNING)', l)]
                    for line in relevant[-10:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"Error reading log: {e}")
        else:
            print(f"Log file not found: {log_file}")
        
        print()
        print("--- Key Statistics ---")
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Extract chunk progress
                    chunk_match = re.search(r'\[Chunk (\d+)/20\]', content)
                    if chunk_match:
                        print(f"Chunk: {chunk_match.group(1)}/20")
                    
                    # Extract processing rate
                    rate_match = re.search(r'(\d+\.?\d*)\s*reviews/sec', content)
                    if rate_match:
                        print(f"Rate: {rate_match.group(1)} reviews/sec")
                    
                    # Extract ETA
                    eta_match = re.search(r'ETA:\s*([^\n]+)', content)
                    if eta_match:
                        print(f"ETA: {eta_match.group(1)}")
                    
                    # Extract total processed
                    processed_match = re.search(r'Progress:\s*([\d,]+)/[\d,]+ reviews', content)
                    if processed_match:
                        print(f"Processed: {processed_match.group(1)} reviews")
            except Exception as e:
                print(f"Error parsing log: {e}")
        
        print()
        print(f"Next update in {monitor_interval}s... (Press Ctrl+C to stop)")
        time.sleep(monitor_interval)
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user.")
    sys.exit(0)

