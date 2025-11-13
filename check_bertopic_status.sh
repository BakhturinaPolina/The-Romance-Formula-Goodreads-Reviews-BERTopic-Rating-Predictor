#!/bin/bash
# Quick status check for BERTopic preparation

echo "=== BERTopic Preparation Status ==="
echo ""

# Check process
PID=$(ps aux | grep "python3.*prepare_bertopic_input" | grep -v grep | grep -v "bash" | awk '{print $2}' | head -1)
if [ -z "$PID" ]; then
    echo "❌ Process: NOT RUNNING"
else
    STATS=$(ps aux | grep "$PID" | grep -v grep)
    CPU=$(echo "$STATS" | awk '{print $3}')
    MEM=$(echo "$STATS" | awk '{print $4}')
    RUNTIME=$(echo "$STATS" | awk '{print $10}')
    echo "✅ Process: RUNNING (PID: $PID)"
    echo "   CPU: ${CPU}% | Memory: ${MEM}% | Runtime: ${RUNTIME}"
fi

echo ""
echo "--- Latest Progress ---"
tail -20 /tmp/bertopic_prep_monitor.log 2>/dev/null | grep -E "(Chunk|Progress|Processed|ETA|complete|✓|ERROR)" | tail -5

echo ""
echo "--- Output File ---"
if [ -f "data/processed/review_sentences_for_bertopic.parquet" ]; then
    ls -lh data/processed/review_sentences_for_bertopic.parquet
else
    echo "⏳ Final file not yet created"
fi

echo ""
echo "--- Incremental Progress (Chunk Files) ---"
TEMP_DIR="data/processed/review_sentences_temp"
if [ -d "$TEMP_DIR" ]; then
    CHUNK_COUNT=$(ls -1 "$TEMP_DIR"/chunk_*.parquet 2>/dev/null | wc -l)
    if [ "$CHUNK_COUNT" -gt 0 ]; then
        TOTAL_SIZE=$(du -sh "$TEMP_DIR" 2>/dev/null | awk '{print $1}')
        echo "✅ Chunks saved: $CHUNK_COUNT files"
        echo "   Total size: $TOTAL_SIZE"
        echo "   Latest chunks:"
        ls -lht "$TEMP_DIR"/chunk_*.parquet 2>/dev/null | head -5 | awk '{print "   - " $9 " (" $5 ")"}'
    else
        echo "⏳ No chunk files yet"
    fi
else
    echo "⏳ Temp directory not yet created"
fi
