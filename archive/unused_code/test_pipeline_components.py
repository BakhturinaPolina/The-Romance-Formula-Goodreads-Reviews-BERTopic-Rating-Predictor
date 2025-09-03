#!/usr/bin/env python3
"""
Test Pipeline Components
========================

Simple script to test existing pipeline components before running the full pipeline.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.config_loader import ConfigLoader
from src.data_processing.data_loader import DataLoader
from src.data_processing.quality_filters import QualityFilters
from src.data_processing.data_integrator import DataIntegrator
from src.data_processing.pipeline_validator import PipelineValidator

def test_config_loader():
    """Test ConfigLoader component."""
    print("🔧 Testing ConfigLoader...")
    try:
        config_loader = ConfigLoader("config")
        
        # Test loading configurations
        variable_selection = config_loader.get_variable_selection()
        sampling_policy = config_loader.get_sampling_policy()
        fields_required = config_loader.get_fields_required()
        
        print(f"✅ ConfigLoader: Loaded {len(variable_selection)} variable selections")
        print(f"✅ ConfigLoader: Loaded sampling policy with {len(sampling_policy)} sections")
        print(f"✅ ConfigLoader: Loaded {len(fields_required)} required fields")
        
        return True
    except Exception as e:
        print(f"❌ ConfigLoader test failed: {e}")
        return False

def test_data_loader():
    """Test DataLoader component."""
    print("📚 Testing DataLoader...")
    try:
        data_loader = DataLoader()
        config_loader = ConfigLoader("config")
        variable_selection = config_loader.get_variable_selection()
        
        # Test loading a small sample of books using batch loading
        print("   Loading 10 books for testing...")
        sample_books = []
        for batch in data_loader.load_books_data_batch(variable_selection, batch_size=10, max_batches=1):
            sample_books.extend(batch)
            break  # Only take the first batch
        
        if sample_books and len(sample_books) > 0:
            print(f"✅ DataLoader: Successfully loaded {len(sample_books)} books")
            print(f"✅ DataLoader: Sample book has {len(sample_books[0])} fields")
            return True
        else:
            print("❌ DataLoader: No books loaded")
            return False
            
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False

def test_data_files():
    """Test data file availability."""
    print("📁 Testing data file availability...")
    
    data_raw_dir = project_root / "data" / "raw"
    required_files = [
        "goodreads_books_romance.json.gz",
        "goodreads_reviews_romance.json.gz",
        "goodreads_book_authors.json.gz",
        "goodreads_book_works.json.gz"
    ]
    
    all_files_exist = True
    for file_name in required_files:
        file_path = data_raw_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Data file: {file_name} ({file_size:.2f} MB)")
        else:
            print(f"❌ Data file missing: {file_name}")
            all_files_exist = False
    
    return all_files_exist

def main():
    """Run all component tests."""
    print("🧪 Testing Pipeline Components")
    print("=" * 50)
    
    tests = [
        ("Data Files", test_data_files),
        ("ConfigLoader", test_config_loader),
        ("DataLoader", test_data_loader),
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    if all_passed:
        print("\n🎉 All tests passed! Pipeline components are ready.")
        print("💡 You can now run the pipeline with:")
        print("   python src/run_pipeline_5000_sample.py")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running the pipeline.")
        return 1

if __name__ == "__main__":
    exit(main())
