# Review Extraction - Pause Information

**Date Paused:** 2025-11-11 23:25  
**Process ID:** 155101  
**Status:** PAUSED (SIGSTOP - can be resumed)

## Current Progress

- **Reviews Processed:** 2,295,000
- **Reviews Written:** 1,227,428
- **Output File:** `romance_novel_nlp_research/data/processed/romance_reviews_english.csv`
- **Output File Size:** ~1.0 GB (check with `ls -lh`)
- **Log File:** `romance_novel_nlp_research/logs/extract_reviews_20251111_190005.log`

## How to Resume

### Option 1: Resume Paused Process (Recommended)

The process is paused with SIGSTOP and can be resumed immediately:

```bash
# Resume the paused process
kill -CONT 155101

# Monitor progress
tail -f romance_novel_nlp_research/logs/extract_reviews_20251111_190005.log
```

**Advantages:**
- ✅ Continues from exact position (no data loss)
- ✅ No need to modify script
- ✅ Fastest resume method

### Option 2: Check if Process Still Exists

If the process was killed or doesn't exist:

```bash
# Check if process exists
ps -p 155101

# If process doesn't exist, you'll need to restart (see Option 3)
```

### Option 3: Restart with Resume Support (If Process Died)

If the process was killed (not just paused), you'll need to modify the script to skip already processed lines. The current script doesn't support this natively.

**Workaround:** The script processes line-by-line from a gzipped file. To resume:
1. Count lines in current output file
2. Modify script to skip that many lines from input
3. Append to existing output file

**Note:** This is complex because:
- Gzipped files can't easily seek to a specific line
- Would need to decompress and count, or use a different approach

## Current Status Check

```bash
# Check if process is paused
ps -p 155101 -o pid,state,cmd

# State 'T' = Stopped (paused)
# State 'R' = Running
# If process doesn't exist, it was killed
```

## System Load Impact

- **Before pause:** Process using ~95.7% CPU
- **After pause:** CPU usage should drop significantly
- **System load should decrease** from ~23 to ~15-18

## Next Steps

1. ✅ Process is paused
2. ✅ Can proceed with database operations
3. ⏳ Resume when ready: `kill -CONT 155101`
4. ⏳ Monitor: `tail -f romance_novel_nlp_research/logs/extract_reviews_20251111_190005.log`

## Important Notes

⚠️ **If you kill the process (SIGKILL) instead of pausing:**
- The process will be lost
- Output file will have partial data
- Would need to restart from beginning or implement resume logic

✅ **Current approach (SIGSTOP):**
- Process is safely paused
- Can resume immediately with SIGCONT
- No data loss
- Continues from exact position

