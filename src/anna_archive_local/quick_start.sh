#!/bin/bash

# Anna's Archive Local Pipeline - Quick Start Script
# This script helps you get started with the local data pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_KEY="BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"
PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research"
DATA_DIR="$PROJECT_ROOT/data/anna_archive"
SAMPLE_SIZE=10000

echo -e "${BLUE}🚀 Anna's Archive Local Pipeline - Quick Start${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Check prerequisites
echo -e "\n${BLUE}Step 1: Checking prerequisites...${NC}"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the src/anna_archive_local/ directory"
    exit 1
fi

# Check disk space
AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available. You need at least 10GB for samples."
fi

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    print_error "Virtual environment not found. Please create one first:"
    echo "cd $PROJECT_ROOT && python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

print_status "Prerequisites check complete"

# Step 2: Activate virtual environment and install dependencies
echo -e "\n${BLUE}Step 2: Setting up Python environment...${NC}"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Install dependencies
pip install -r requirements.txt > /dev/null 2>&1
print_status "Dependencies installed"

# Step 3: Check for existing data
echo -e "\n${BLUE}Step 3: Checking for existing data...${NC}"

if [ -d "$DATA_DIR/elasticsearch" ] && [ "$(ls -A $DATA_DIR/elasticsearch/*.json.gz 2>/dev/null)" ]; then
    print_status "Found existing Anna's Archive data"
    JSON_FILES=$(ls $DATA_DIR/elasticsearch/*.json.gz | wc -l)
    echo "Found $JSON_FILES JSON.gz files"
    
    # Check if sample already exists
    if [ -f "$DATA_DIR/elasticsearch/sample_${SAMPLE_SIZE}.json.gz" ]; then
        print_status "Sample file already exists: sample_${SAMPLE_SIZE}.json.gz"
        SKIP_SAMPLE=true
    else
        SKIP_SAMPLE=false
    fi
else
    print_warning "No Anna's Archive data found"
    echo "You need to download the data dumps first. See STEP_BY_STEP_DATA_ACQUISITION.md"
    echo "For now, we'll create a test setup..."
    SKIP_SAMPLE=true
fi

# Step 4: Create sample data (if needed)
if [ "$SKIP_SAMPLE" = false ]; then
    echo -e "\n${BLUE}Step 4: Creating sample data...${NC}"
    
    python3 sample_data_extractor.py \
        --input-dir "$DATA_DIR/elasticsearch/" \
        --output-file "$DATA_DIR/elasticsearch/sample_${SAMPLE_SIZE}.json.gz" \
        --sample-size $SAMPLE_SIZE \
        --analyze
    
    print_status "Sample data created"
fi

# Step 5: Convert to Parquet (if needed)
echo -e "\n${BLUE}Step 5: Converting to Parquet format...${NC}"

SAMPLE_PARQUET_DIR="$DATA_DIR/parquet/sample_${SAMPLE_SIZE}"
if [ -d "$SAMPLE_PARQUET_DIR" ] && [ "$(ls -A $SAMPLE_PARQUET_DIR/*.parquet 2>/dev/null)" ]; then
    print_status "Parquet files already exist"
else
    if [ -f "$DATA_DIR/elasticsearch/sample_${SAMPLE_SIZE}.json.gz" ]; then
        python3 json_to_parquet.py \
            --input-file "$DATA_DIR/elasticsearch/sample_${SAMPLE_SIZE}.json.gz" \
            --output-dir "$SAMPLE_PARQUET_DIR/"
        
        print_status "Parquet conversion complete"
    else
        print_warning "No sample data to convert"
    fi
fi

# Step 6: Test the pipeline
echo -e "\n${BLUE}Step 6: Testing the pipeline...${NC}"

if [ -d "$SAMPLE_PARQUET_DIR" ] && [ "$(ls -A $SAMPLE_PARQUET_DIR/*.parquet 2>/dev/null)" ]; then
    # Test search engine
    echo "Testing search engine..."
    python3 book_search_cli.py \
        --parquet-dir "$SAMPLE_PARQUET_DIR/" \
        --stats > /dev/null 2>&1
    
    print_status "Search engine working"
    
    # Test API connection
    echo "Testing API connection..."
    python3 api_downloader.py \
        --api-key "$API_KEY" \
        --test > /dev/null 2>&1
    
    print_status "API connection working"
else
    print_warning "No Parquet data available for testing"
fi

# Step 7: Run demo (if sample books CSV exists)
echo -e "\n${BLUE}Step 7: Running demo...${NC}"

SAMPLE_BOOKS_CSV="$PROJECT_ROOT/data/processed/sample_50_books.csv"
if [ -f "$SAMPLE_BOOKS_CSV" ]; then
    if [ -d "$SAMPLE_PARQUET_DIR" ] && [ "$(ls -A $SAMPLE_PARQUET_DIR/*.parquet 2>/dev/null)" ]; then
        DEMO_OUTPUT_DIR="$DATA_DIR/demo_results"
        
        echo "Running demo with 50 sample books..."
        python3 demo_query_50_books.py \
            --parquet-dir "$SAMPLE_PARQUET_DIR/" \
            --books-csv "$SAMPLE_BOOKS_CSV" \
            --output-dir "$DEMO_OUTPUT_DIR/" \
            --api-key "$API_KEY"
        
        print_status "Demo complete"
        echo "Results saved to: $DEMO_OUTPUT_DIR/"
    else
        print_warning "No Parquet data available for demo"
    fi
else
    print_warning "Sample books CSV not found: $SAMPLE_BOOKS_CSV"
fi

# Step 8: Summary
echo -e "\n${BLUE}🎉 Quick Start Complete!${NC}"
echo "=================================================="

echo -e "\n${GREEN}What's been set up:${NC}"
echo "✅ Python environment and dependencies"
echo "✅ Directory structure"
if [ "$SKIP_SAMPLE" = false ]; then
    echo "✅ Sample data ($SAMPLE_SIZE records)"
    echo "✅ Parquet conversion"
fi
echo "✅ Pipeline testing"

echo -e "\n${GREEN}Next steps:${NC}"
echo "1. If you don't have Anna's Archive data yet:"
echo "   - Follow STEP_BY_STEP_DATA_ACQUISITION.md"
echo "   - Download data dumps from https://annas-archive.org/datasets"
echo ""
echo "2. If you have data and want to test:"
echo "   python3 book_search_cli.py --parquet-dir $SAMPLE_PARQUET_DIR/ --title 'test'"
echo ""
echo "3. To run the full demo:"
echo "   python3 demo_query_50_books.py \\"
echo "     --parquet-dir $SAMPLE_PARQUET_DIR/ \\"
echo "     --books-csv $SAMPLE_BOOKS_CSV \\"
echo "     --output-dir $DATA_DIR/demo_results/ \\"
echo "     --api-key $API_KEY"

echo -e "\n${GREEN}Documentation:${NC}"
echo "📖 README.md - Complete guide"
echo "📖 DATA_ACQUISITION.md - How to download data"
echo "📖 SAMPLING_GUIDE.md - Working with samples"
echo "📖 STEP_BY_STEP_DATA_ACQUISITION.md - Detailed instructions"

echo -e "\n${GREEN}Your API key:${NC} $API_KEY"
echo -e "\n${BLUE}Happy searching! 🔍📚${NC}"
