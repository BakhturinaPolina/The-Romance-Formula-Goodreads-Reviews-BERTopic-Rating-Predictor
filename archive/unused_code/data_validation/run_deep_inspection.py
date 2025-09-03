#!/usr/bin/env python3
"""
Run deep data inspection for invisible issues.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from data_validation.deep_data_inspector import DeepDataInspector

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run deep inspection
    inspector = DeepDataInspector()
    results = inspector.run_deep_inspection()
    
    # Print summary
    inspector.print_summary()
    
    # Return appropriate exit code
    if results['summary']['status'] == 'FAIL':
        sys.exit(1)
    elif results['summary']['status'] == 'WARNING':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
