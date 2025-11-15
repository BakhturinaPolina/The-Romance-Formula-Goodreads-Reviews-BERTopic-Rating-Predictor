# Memory Optimization Guide for Laptops

## Overview

The BERTopic+OCTIS pipeline has been optimized for laptops with limited memory. Several command-line options have been added to reduce memory usage and processing time.

## Key Optimizations Added

### 1. Limit Number of Embedding Models
**Problem**: Processing 4 embedding models simultaneously loads 4× memory for embeddings.

**Solution**: Use `--max-embedding-models` to limit how many models are processed.

```bash
# Process only 1 model instead of all 4
python bertopic_plus_octis.py --max-embedding-models 1
```

### 2. Reduce Embedding Batch Size
**Problem**: Default batch size of 32 can be too large for laptops.

**Solution**: Use `--embedding-batch-size` to reduce memory during embedding calculation.

```bash
# Use smaller batches (8 = very memory-efficient, 16 = balanced)
python bertopic_plus_octis.py --embedding-batch-size 8
```

### 3. Reduce Optimization Runs
**Problem**: Default is 15 × number of parameters (e.g., 11 params = 165 runs), which is very time-consuming.

**Solution**: Use `--optimization-multiplier` to reduce optimization runs.

```bash
# Use 5× multiplier instead of 15× (faster, less thorough)
python bertopic_plus_octis.py --optimization-multiplier 5
```

### 4. Limit Dataset Size
**Problem**: Loading full dataset (8.6M sentences) is too heavy.

**Solution**: Use `--max-sentences` to limit dataset size.

```bash
# Test with 10K sentences first
python bertopic_plus_octis.py --max-sentences 10000
```

### 5. Process One Model at a Time
**Problem**: Keeping all embeddings in memory simultaneously uses too much RAM.

**Solution**: Use `--process-one-model` to process models sequentially, clearing memory between models.

```bash
# Process one model at a time (most memory-efficient)
python bertopic_plus_octis.py --process-one-model
```

## Recommended Settings for Laptops

### Lightweight Testing (Quick Test)
```bash
python bertopic_plus_octis.py \
    --max-embedding-models 1 \
    --max-sentences 10000 \
    --embedding-batch-size 8 \
    --optimization-multiplier 5 \
    --process-one-model
```

**Memory Impact**: ~2-4 GB RAM
**Time**: ~30-60 minutes

### Balanced (Good Quality, Manageable)
```bash
python bertopic_plus_octis.py \
    --max-embedding-models 2 \
    --max-sentences 50000 \
    --embedding-batch-size 16 \
    --optimization-multiplier 10 \
    --process-one-model
```

**Memory Impact**: ~4-8 GB RAM
**Time**: ~2-4 hours

### Production (Full Dataset, All Models)
```bash
python bertopic_plus_octis.py \
    --embedding-batch-size 16 \
    --optimization-multiplier 15 \
    --process-one-model
```

**Memory Impact**: ~8-16 GB RAM (one model at a time)
**Time**: ~8-12 hours

## Memory Usage Breakdown

### Without Optimizations
- **4 embedding models**: ~4 × 2-4 GB = 8-16 GB (all in memory)
- **Full dataset (8.6M sentences)**: ~2-3 GB
- **BERTopic optimization**: ~2-4 GB
- **Total**: ~12-23 GB RAM

### With Optimizations (Recommended)
- **1 embedding model at a time**: ~2-4 GB
- **Limited dataset (10K-50K sentences)**: ~0.1-0.5 GB
- **BERTopic optimization**: ~1-2 GB
- **Total**: ~3-7 GB RAM

## Step-by-Step Optimization Strategy

### Step 1: Start Small
```bash
# Test with minimal settings
python bertopic_plus_octis.py \
    --max-embedding-models 1 \
    --max-sentences 10000 \
    --embedding-batch-size 8 \
    --optimization-multiplier 5
```

### Step 2: If Successful, Scale Up
```bash
# Increase dataset size
python bertopic_plus_octis.py \
    --max-embedding-models 1 \
    --max-sentences 50000 \
    --embedding-batch-size 16 \
    --optimization-multiplier 10
```

### Step 3: Add More Models (One at a Time)
```bash
# Process 2 models, one at a time
python bertopic_plus_octis.py \
    --max-embedding-models 2 \
    --max-sentences 50000 \
    --embedding-batch-size 16 \
    --optimization-multiplier 10 \
    --process-one-model
```

## Command-Line Arguments Summary

| Argument | Default | Description | Memory Impact |
|----------|---------|-------------|---------------|
| `--max-embedding-models` | None (all) | Limit number of models | High |
| `--embedding-batch-size` | 16 | Batch size for embeddings | Medium |
| `--optimization-multiplier` | 15 | Optimization runs multiplier | Low (time) |
| `--max-sentences` | None (all) | Limit dataset size | High |
| `--process-one-model` | False | Process models sequentially | High |
| `--config` | None | Path to config file | None |

## Tips

1. **Use Test Dataset First**: Create a small test dataset using `sample_test_dataset.py` before running full optimization.

2. **Monitor Memory**: Watch system memory usage during first run to determine optimal settings.

3. **Embeddings are Cached**: Once embeddings are calculated, they're saved to disk. Subsequent runs will be faster.

4. **One Model at a Time**: The `--process-one-model` flag is the most effective memory optimization for multiple models.

5. **Reduce Optimization Runs**: For initial testing, use `--optimization-multiplier 5` instead of 15. You can always resume with more runs later.

## Example: Complete Lightweight Run

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research
source romance-novel-nlp-research/.venv/bin/activate

# Run with all optimizations
python src/reviews_analysis/BERTopic_OCTIS/bertopic_plus_octis.py \
    --max-embedding-models 1 \
    --max-sentences 10000 \
    --embedding-batch-size 8 \
    --optimization-multiplier 5 \
    --process-one-model
```

This will:
- Process only 1 embedding model
- Use 10K sentences (quick test)
- Use small batch size (8) for embeddings
- Run 5× parameters optimization (faster)
- Process one model at a time (memory-efficient)

## Troubleshooting

### Out of Memory Errors
1. Reduce `--max-sentences` to 5000-10000
2. Reduce `--embedding-batch-size` to 4-8
3. Use `--process-one-model` flag
4. Limit to 1 model with `--max-embedding-models 1`

### Too Slow
1. Reduce `--optimization-multiplier` to 5-10
2. Use smaller dataset with `--max-sentences`
3. Process fewer models with `--max-embedding-models`

### Want Better Results
1. Increase `--optimization-multiplier` back to 15
2. Use larger dataset (remove `--max-sentences` or increase it)
3. Process more models (increase `--max-embedding-models`)

