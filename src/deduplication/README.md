# Enhanced Dedupe Pipeline

A robust token deduplication pipeline with quality diagnostics and advanced filtering techniques.

## 🚨 Key Fix: Edge Sample Validation

**Problem**: Edge samples contained similarities below the threshold (e.g., 0.319 when threshold ≥0.6), indicating sampling from unfiltered data or weak clustering chains.

**Solution**: 
- ✅ Fixed `build_clusters()` to sample only from filtered edges
- ✅ Added validation to ensure all samples meet threshold requirements
- ✅ Implemented cluster quality metrics and weak edge pruning

## 🛠️ Features

### Core Pipeline
- **Filtering**: DuckDB-based filtering with configurable thresholds
- **Clustering**: Union-Find clustering with proper edge sampling
- **Quality Metrics**: Min/mean/max similarity, triangle rate analysis
- **Diagnostics**: Comprehensive cluster quality assessment

### Advanced Filtering
- **Mutual Nearest Neighbors**: Prevent hub chaining by requiring bidirectional top-k relationships
- **Triangle Constraints**: Ensure local consistency by requiring shared neighbors
- **Weak Edge Pruning**: Remove edges below quality thresholds

## 📋 Commands

### Basic Commands
```bash
# Run full pipeline with quality metrics
python dedupe_pipeline.py all input_pairs.parquet output_dir --threshold 0.6

# Filter only
python dedupe_pipeline.py filter input_pairs.parquet output_dir --threshold 0.6

# Cluster only (on existing filtered data)
python dedupe_pipeline.py cluster filtered_pairs.parquet output_dir

# Statistics only
python dedupe_pipeline.py stats input_pairs.parquet --threshold 0.6
```

### Diagnostic Commands
```bash
# Diagnose existing results
python dedupe_pipeline.py diagnose output_dir --threshold 0.6

# Prune weak edges with mutual NN
python dedupe_pipeline.py prune filtered_pairs.parquet output_dir --k 5
```

## 📊 Output Files

### Standard Outputs
- `pairs_filtered.parquet` - Filtered edge pairs
- `clusters_token_map.parquet` - Token to cluster ID mapping
- `clusters_summary.parquet` - Basic cluster statistics
- `clusters_edges_samples.parquet` - Sample edges per cluster
- `meta.json` - Configuration and metadata

### Quality Metrics (New)
- `cluster_cohesion_metrics.parquet` - Min/mean/max sim, triangle rates
- `clusters_flagged_low_quality.parquet` - Problematic clusters
- `clusters_edges_samples_fixed.parquet` - Validated edge samples

### Pruned Outputs
- `pairs_mutual_nn.parquet` - Mutual nearest neighbor filtered edges
- `clusters_*_pruned.parquet` - Re-clustered results on pruned data

## 🔍 Quality Metrics

### Cluster Cohesion
- **min_sim**: Minimum similarity within cluster
- **mean_sim**: Average similarity within cluster  
- **max_sim**: Maximum similarity within cluster
- **triangle_rate**: Fraction of edges with shared neighbors
- **edges**: Number of intra-cluster edges

### Quality Thresholds
- **min_sim_threshold**: Default 0.62 (clusters with min sim below this are flagged)
- **min_triangle_rate**: Default 0.10 (clusters with triangle rate below this are flagged)

## 🎯 Usage Examples

### 1. Basic Pipeline
```bash
python dedupe_pipeline.py all data/token_pairs.parquet outputs/deduped --threshold 0.6
```

### 2. Quality Assessment
```bash
python dedupe_pipeline.py diagnose outputs/deduped --threshold 0.6
```

### 3. High-Quality Clustering
```bash
# Apply mutual NN filtering for cleaner clusters
python dedupe_pipeline.py prune outputs/deduped/pairs_filtered.parquet outputs/pruned --k 5
```

### 4. Custom Thresholds
```bash
# More aggressive filtering
python dedupe_pipeline.py all data/token_pairs.parquet outputs/strict --threshold 0.7 --min-len 4
```

## ⚙️ Configuration

### Key Parameters
- `threshold`: Minimum similarity for edge inclusion (default: 0.50)
- `min_len`: Minimum token length (default: 3)
- `max_rank`: Maximum rank for edge inclusion (default: None)
- `block_numeric`: Block numeric tokens (default: True)
- `sample_edges_per_cluster`: Sample size per cluster (default: 5)

### Quality Parameters
- `min_sim_threshold`: Flag clusters below this min similarity (default: 0.62)
- `min_triangle_rate`: Flag clusters below this triangle rate (default: 0.10)
- `k`: Top-k for mutual nearest neighbors (default: 5)

## 🔧 Troubleshooting

### Common Issues

1. **Edge samples below threshold**
   ```bash
   # Run diagnostics to identify the issue
   python dedupe_pipeline.py diagnose output_dir --threshold 0.6
   ```

2. **Low-quality clusters**
   ```bash
   # Check flagged clusters
   python -c "import pandas as pd; print(pd.read_parquet('output_dir/clusters_flagged_low_quality.parquet'))"
   ```

3. **Too many weak connections**
   ```bash
   # Apply mutual NN filtering
   python dedupe_pipeline.py prune filtered_pairs.parquet output_dir --k 3
   ```

### Performance Tips
- Use higher thresholds (0.6-0.7) for better quality
- Apply mutual NN filtering to reduce hub chaining
- Monitor triangle rates for local consistency
- Use `max_rank` to limit edge density

## 📈 Quality Assessment

### Good Clusters
- High min_sim (>0.7)
- High triangle_rate (>0.3)
- Reasonable size (not too large/small)

### Problematic Clusters
- Low min_sim (<0.6) - weak connections
- Low triangle_rate (<0.1) - chain-like structure
- Very large size - potential hub chaining

### Recommended Workflow
1. Run with moderate threshold (0.6)
2. Diagnose quality metrics
3. Apply mutual NN filtering if needed
4. Re-cluster on pruned data
5. Validate final results

## 🚀 Advanced Usage

### Custom Quality Thresholds
```python
from dedupe_pipeline import cluster_cohesion_metrics, flag_low_quality_clusters

# Load your data
cohesion = cluster_cohesion_metrics("filtered_pairs.parquet", token_cluster)

# Custom thresholds
low_quality = flag_low_quality_clusters(
    cohesion, 
    min_sim_threshold=0.7,  # Stricter
    min_triangle_rate=0.2   # Require more triangles
)
```

### Mutual NN Analysis
```python
from dedupe_pipeline import mutual_nearest_neighbors

# Apply mutual NN filtering
mutual_edges = mutual_nearest_neighbors("filtered_pairs.parquet", k=3)
print(f"Reduced from {len(original)} to {len(mutual_edges)} edges")
```

## 📝 Notes

- The pipeline now ensures all edge samples meet the specified threshold
- Quality metrics help identify problematic clusters
- Mutual NN filtering prevents hub chaining
- Triangle constraints ensure local consistency
- All outputs include comprehensive metadata
