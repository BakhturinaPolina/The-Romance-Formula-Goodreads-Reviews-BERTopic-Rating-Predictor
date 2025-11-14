#!/bin/bash
# Monitoring script for prepare_bertopic_input.py
# Monitors progress and displays updates

PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research"
LOG_FILE="/tmp/bertopic_prep_monitor.log"
OUTPUT_FILE="$PROJECT_ROOT/data/processed/review_sentences_for_bertopic.parquet"
MONITOR_INTERVAL=30  # Check every 30 seconds

# Change to project root
cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "BERTopic Preparation Monitor"
echo "=========================================="
echo "Monitoring log: $LOG_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Check interval: ${MONITOR_INTERVAL}s"
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "BERTopic Preparation Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if process is running (get the actual Python process, not the shell)
    PID=$(ps aux | grep "prepare_bertopic_input" | grep -v grep | grep -v "bash" | grep -v "monitor" | awk '{print $2}' | head -1)
    if [ -z "$PID" ]; then
        echo "⚠️  Process Status: NOT RUNNING"
        echo ""
        echo "Checking if output file exists..."
        if [ -f "$OUTPUT_FILE" ]; then
            echo "✓ Output file exists!"
            ls -lh "$OUTPUT_FILE"
            echo ""
            echo "Script may have completed. Check log for details."
        else
            echo "✗ Output file not found. Script may have failed."
        fi
    else
        # Get process stats
        STATS=$(ps aux | grep "$PID" | grep -v grep)
        CPU=$(echo "$STATS" | awk '{print $3}')
        MEM=$(echo "$STATS" | awk '{print $4}')
        RUNTIME=$(echo "$STATS" | awk '{print $10}')
        
        echo "✓ Process Status: RUNNING (PID: $PID)"
        echo "  CPU: ${CPU}% | Memory: ${MEM}% | Runtime: ${RUNTIME}"
        echo ""
        
        # Check output file size if it exists
        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
            echo "✓ Output file exists: $FILE_SIZE"
        else
            echo "⏳ Output file not yet created..."
        fi
    fi
    
    echo ""
    echo "--- Latest Progress (last 15 lines) ---"
    if [ -f "$LOG_FILE" ]; then
        tail -15 "$LOG_FILE" | grep -E "(Chunk|Progress|Processed|ETA|complete|✓|ERROR|WARNING)" | tail -10
    else
        echo "Log file not found: $LOG_FILE"
    fi
    
    echo ""
    echo "--- Key Statistics ---"
    if [ -f "$LOG_FILE" ]; then
        # Extract chunk progress
        CHUNK_INFO=$(grep -E "\[Chunk [0-9]+/20\]" "$LOG_FILE" | tail -1)
        if [ ! -z "$CHUNK_INFO" ]; then
            echo "$CHUNK_INFO"
        fi
        
        # Extract processing rate
        RATE=$(grep -E "reviews/sec" "$LOG_FILE" | tail -1)
        if [ ! -z "$RATE" ]; then
            echo "$RATE"
        fi
        
        # Extract ETA
        ETA=$(grep -E "ETA:" "$LOG_FILE" | tail -1)
        if [ ! -z "$ETA" ]; then
            echo "$ETA"
        fi
        
        # Extract total processed
        PROCESSED=$(grep -E "Progress: [0-9,]+/[0-9,]+ reviews" "$LOG_FILE" | tail -1)
        if [ ! -z "$PROCESSED" ]; then
            echo "$PROCESSED"
        fi
    fi
    
    echo ""
    echo "Next update in ${MONITOR_INTERVAL}s... (Press Ctrl+C to stop)"
    sleep "$MONITOR_INTERVAL"
done

