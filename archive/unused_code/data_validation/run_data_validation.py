#!/usr/bin/env python3
"""
Run comprehensive data validation on processed CSV files.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from data_validation.data_sanity_checker import DataSanityChecker

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    checker = DataSanityChecker()
    results = checker.run_comprehensive_validation()
    
    # Print summary
    checker.print_validation_summary()
    
    # Return appropriate exit code
    if results['summary']['overall_status'] == 'FAIL':
        sys.exit(1)
    elif results['summary']['overall_status'] == 'WARNING':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
