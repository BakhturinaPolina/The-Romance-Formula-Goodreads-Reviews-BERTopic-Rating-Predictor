#!/usr/bin/env python3
"""
Wrapper script to run prepare_bertopic_input.py with proper logging and venv.

This script ensures the preparation script runs with the correct virtual environment
and logs output to a file.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Import venv_utils with fallback for direct script execution
try:
    from .venv_utils import get_project_root, get_venv_python, verify_venv
except ImportError:
    # If running as script, add current directory to path and import
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from venv_utils import get_project_root, get_venv_python, verify_venv

# Verify we're using the correct venv
verify_venv()

# Get paths
project_root = get_project_root()
venv_python = get_venv_python()
script_path = Path(__file__).parent.parent / "core" / "prepare_bertopic_input.py"
log_file = Path("/tmp/bertopic_prep_monitor.log")

# Change to project root
import os
os.chdir(project_root)

print("=" * 50)
print("Starting BERTopic Preparation Script")
print("=" * 50)
print(f"Python: {venv_python}")
print(f"Script: {script_path}")
print(f"Log file: {log_file}")
print(f"Project root: {project_root}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)
print()

# Run the script and capture output
try:
    with open(log_file, 'w', encoding='utf-8') as log_f:
        process = subprocess.run(
            [str(venv_python), str(script_path)] + sys.argv[1:],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Write to both log file and stdout
        output = process.stdout
        log_f.write(output)
        print(output, end='')
    
    exit_code = process.returncode
    
    print()
    print("=" * 50)
    print(f"Script finished with exit code: {exit_code}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    sys.exit(exit_code)
    
except KeyboardInterrupt:
    print("\n\nScript interrupted by user")
    sys.exit(130)
except Exception as e:
    print(f"\n\nError running script: {e}", file=sys.stderr)
    sys.exit(1)

