#!/usr/bin/env python3
"""
Quick status check for BERTopic preparation.

Displays current process status, progress, and file information.
"""

import sys
import subprocess
import re
from pathlib import Path

from venv_utils import get_project_root, verify_venv

# Verify we're using the correct venv
verify_venv()

project_root = get_project_root()
log_file = Path("/tmp/bertopic_prep_monitor.log")
output_file = project_root / "data" / "processed" / "review_sentences_for_bertopic.parquet"
temp_dir = project_root / "data" / "processed" / "review_sentences_temp"

# Change to project root
import os
os.chdir(project_root)

print("=== BERTopic Preparation Status ===")
print()

# Check process
try:
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True,
        timeout=2
    )
    
    process_info = None
    for line in result.stdout.split('\n'):
        if 'prepare_bertopic_input' in line and 'grep' not in line and 'bash' not in line and 'check' not in line:
            parts = line.split()
            if len(parts) >= 11:
                process_info = {
                    'pid': parts[1],
                    'cpu': parts[2],
                    'mem': parts[3],
                    'runtime': parts[9]
                }
                break
    
    if not process_info:
        print("❌ Process: NOT RUNNING")
    else:
        print(f"✅ Process: RUNNING (PID: {process_info['pid']})")
        print(f"   CPU: {process_info['cpu']}% | Memory: {process_info['mem']}% | Runtime: {process_info['runtime']}")
except Exception:
    print("❌ Process: NOT RUNNING (or error checking)")

print()
print("--- Latest Progress ---")
if log_file.exists():
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            relevant = [l for l in lines[-20:] if re.search(r'(Chunk|Progress|Processed|ETA|complete|✓|ERROR)', l)]
            for line in relevant[-5:]:
                print(line.rstrip())
    except Exception:
        print("Error reading log file")
else:
    print("Log file not found")

print()
print("--- Output File ---")
if output_file.exists():
    size_bytes = output_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"✓ {output_file.name}: {size_mb:.2f} MB")
else:
    print("⏳ Final file not yet created")

print()
print("--- Incremental Progress (Chunk Files) ---")
if temp_dir.exists():
    chunk_files = list(temp_dir.glob("chunk_*.parquet"))
    if chunk_files:
        total_size = sum(f.stat().st_size for f in chunk_files)
        total_mb = total_size / (1024 * 1024)
        print(f"✅ Chunks saved: {len(chunk_files)} files")
        print(f"   Total size: {total_mb:.2f} MB")
        print("   Latest chunks:")
        for f in sorted(chunk_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")
    else:
        print("⏳ No chunk files yet")
else:
    print("⏳ Temp directory not yet created")

