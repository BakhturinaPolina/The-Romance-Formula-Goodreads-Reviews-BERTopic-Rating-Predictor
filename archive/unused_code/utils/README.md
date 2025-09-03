# Utilities Module

## Overview

The Utilities module provides shared utility functions, file handlers, and common operations used throughout the Romance Novel NLP Research project. These utilities are designed to be reusable, well-tested, and provide comprehensive error handling.

## Components

### 📁 File Handlers (`file_handlers.py`)

**Purpose**: Comprehensive file handling utilities for JSON and CSV operations with error handling.

**Key Features**:
- Gzipped JSON file reading with progress tracking
- Chunked file processing for memory efficiency
- CSV file operations with error handling
- File validation and integrity checking
- Comprehensive logging and error reporting

**Main Classes**:

#### JSONFileHandler
Handles reading and processing of JSON files, especially gzipped JSON files.

**Key Methods**:
- `read_json_gz()`: Read gzipped JSON file line by line
- `read_json_gz_chunked()`: Read gzipped JSON file in chunks
- `count_records()`: Count records in a JSON file
- `validate_json_file()`: Validate JSON file integrity

**Usage**:
```python
from src.utils.file_handlers import JSONFileHandler
from pathlib import Path

# Initialize handler
handler = JSONFileHandler(chunk_size=1000)

# Read JSON file with progress tracking
file_path = Path("data/raw/goodreads_books_romance.json.gz")
for record in handler.read_json_gz(file_path, max_records=1000):
    # Process each record
    process_record(record)

# Read in chunks for large files
for chunk in handler.read_json_gz_chunked(file_path, max_records=10000):
    # Process chunk of records
    process_chunk(chunk)

# Count records
record_count = handler.count_records(file_path)
print(f"File contains {record_count} records")
```

#### CSVFileHandler
Handles CSV file operations with comprehensive error handling.

**Key Methods**:
- `read_csv_safe()`: Safely read CSV files with error handling
- `write_csv_safe()`: Safely write CSV files with error handling
- `validate_csv_file()`: Validate CSV file structure and content
- `get_csv_info()`: Get information about CSV file structure

**Usage**:
```python
from src.utils.file_handlers import CSVFileHandler

# Initialize handler
csv_handler = CSVFileHandler()

# Read CSV file safely
file_path = Path("data/processed/romance_books_cleaned.csv")
try:
    df = csv_handler.read_csv_safe(file_path)
    print(f"Successfully read {len(df)} records")
except Exception as e:
    print(f"Error reading CSV: {e}")

# Write CSV file safely
output_path = Path("data/processed/output.csv")
csv_handler.write_csv_safe(df, output_path, index=False)
```

### 🛠️ Lightweight Handlers (`lightweight_handlers.py`)

**Purpose**: Core Python library utilities and helper functions for common operations.

**Key Features**:
- Data type conversion utilities
- String processing and cleaning
- Date and time handling
- List and dictionary operations
- Validation and checking functions

**Main Functions**:

#### Data Type Conversion
```python
from src.utils.lightweight_handlers import (
    safe_int, safe_float, safe_str, safe_bool,
    convert_to_int, convert_to_float, convert_to_str
)

# Safe type conversion with error handling
value = safe_int("123")  # Returns 123
value = safe_int("abc")  # Returns None or default value

# Batch conversion
numbers = ["1", "2", "3", "invalid"]
converted = [safe_int(x) for x in numbers]
```

#### String Processing
```python
from src.utils.lightweight_handlers import (
    clean_string, normalize_text, extract_year,
    truncate_text, remove_special_chars
)

# Clean and normalize text
text = "  Hello, World!  "
cleaned = clean_string(text)  # "Hello, World!"

# Extract year from text
year = extract_year("Published in 2020")  # 2020

# Truncate long text
truncated = truncate_text("Very long text...", max_length=50)
```

#### Date and Time Handling
```python
from src.utils.lightweight_handlers import (
    parse_date, format_date, is_valid_date,
    get_year_from_date, calculate_age
)

# Parse various date formats
date = parse_date("2020-01-15")  # datetime object
date = parse_date("15/01/2020")  # datetime object

# Check date validity
is_valid = is_valid_date("2020-13-45")  # False

# Extract year
year = get_year_from_date("2020-01-15")  # 2020
```

#### List and Dictionary Operations
```python
from src.utils.lightweight_handlers import (
    flatten_list, deduplicate_list, merge_dicts,
    filter_dict, sort_dict_by_value
)

# Flatten nested lists
nested = [[1, 2], [3, 4], [5]]
flat = flatten_list(nested)  # [1, 2, 3, 4, 5]

# Remove duplicates while preserving order
duplicates = [1, 2, 2, 3, 3, 4]
unique = deduplicate_list(duplicates)  # [1, 2, 3, 4]

# Merge dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = merge_dicts(dict1, dict2)  # {"a": 1, "b": 3, "c": 4}
```

## Dependencies

### Internal Dependencies
- `config/logging_config.py`: Logging configuration
- Project-specific configuration and settings

### External Dependencies
- `pandas`: Data manipulation and CSV operations
- `numpy`: Numerical operations
- `gzip`: Compressed file handling
- `json`: JSON processing
- `pathlib`: File path operations
- `tqdm`: Progress bars for long operations
- `logging`: Logging framework

## Usage Patterns

### File Processing Pattern
```python
from src.utils.file_handlers import JSONFileHandler, CSVFileHandler
from pathlib import Path

# Process large JSON files efficiently
json_handler = JSONFileHandler(chunk_size=1000)
csv_handler = CSVFileHandler()

input_file = Path("data/raw/large_file.json.gz")
output_file = Path("data/processed/output.csv")

# Process in chunks to manage memory
for chunk in json_handler.read_json_gz_chunked(input_file):
    # Process chunk
    processed_chunk = process_data(chunk)
    
    # Write to CSV
    csv_handler.write_csv_safe(processed_chunk, output_file, mode='a')
```

### Data Validation Pattern
```python
from src.utils.lightweight_handlers import safe_int, clean_string
from src.utils.file_handlers import JSONFileHandler

# Validate and clean data during processing
handler = JSONFileHandler()

for record in handler.read_json_gz(file_path):
    # Clean and validate fields
    record['title'] = clean_string(record.get('title', ''))
    record['publication_year'] = safe_int(record.get('publication_year'))
    
    # Skip invalid records
    if not record['title'] or record['publication_year'] is None:
        continue
    
    # Process valid record
    process_record(record)
```

### Error Handling Pattern
```python
from src.utils.file_handlers import JSONFileHandler
import logging

logger = logging.getLogger(__name__)
handler = JSONFileHandler()

try:
    for record in handler.read_json_gz(file_path):
        process_record(record)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"JSON decode error: {e}")
    # Continue processing or handle gracefully
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Performance Characteristics

### File Processing Performance
- **Memory Efficiency**: Chunked processing for large files
- **Progress Tracking**: Real-time progress monitoring
- **Error Recovery**: Graceful handling of file errors
- **Parallel Processing**: Support for parallel file operations

### Optimization Features
- **Lazy Loading**: Load data only when needed
- **Caching**: Cache frequently accessed data
- **Streaming**: Process files without loading entire content
- **Compression**: Efficient handling of compressed files

## Error Handling

### Error Types
- **File Access Errors**: Missing, inaccessible, or corrupted files
- **Data Format Errors**: Malformed JSON or CSV data
- **Memory Errors**: Insufficient memory for large operations
- **Type Conversion Errors**: Invalid data type conversions
- **Validation Errors**: Data validation failures

### Recovery Strategies
- **Graceful Degradation**: Continue processing with available data
- **Error Logging**: Comprehensive error logging and reporting
- **Retry Mechanisms**: Automatic retry for transient errors
- **Fallback Values**: Use default values for missing or invalid data

## Testing

### Test Coverage
- **Unit Tests**: Individual function and method tests
- **Integration Tests**: End-to-end file processing tests
- **Error Handling Tests**: Error scenario testing
- **Performance Tests**: Performance benchmarks

### Test Files
- `tests/utils/test_file_handlers.py`
- `tests/utils/test_lightweight_handlers.py`
- `tests/utils/test_integration.py`

## Best Practices

### File Handling
1. **Use Chunked Processing**: Process large files in chunks to manage memory
2. **Validate Files**: Always validate file integrity before processing
3. **Handle Errors Gracefully**: Implement proper error handling and recovery
4. **Use Progress Tracking**: Show progress for long-running operations

### Data Processing
1. **Validate Input Data**: Always validate data before processing
2. **Use Safe Conversions**: Use safe type conversion functions
3. **Handle Missing Data**: Implement proper handling of missing values
4. **Log Operations**: Log important operations for debugging

### Performance
1. **Use Appropriate Chunk Sizes**: Balance memory usage and performance
2. **Implement Caching**: Cache frequently accessed data
3. **Use Streaming**: Process data without loading entire files
4. **Monitor Memory Usage**: Track memory usage for large operations

## Future Enhancements

### Planned Features
1. **Parallel Processing**: Multi-threaded file processing
2. **Advanced Caching**: Intelligent caching strategies
3. **Compression Support**: Additional compression formats
4. **Streaming APIs**: Streaming data processing interfaces
5. **Performance Monitoring**: Real-time performance monitoring

### Potential Improvements
1. **Distributed Processing**: Support for distributed file processing
2. **Cloud Storage**: Integration with cloud storage services
3. **Data Validation**: Advanced data validation capabilities
4. **Format Conversion**: Additional file format support
5. **Automated Optimization**: Self-optimizing processing parameters
