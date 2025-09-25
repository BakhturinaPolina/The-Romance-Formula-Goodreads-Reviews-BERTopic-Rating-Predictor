# Complete Deduplication Pipeline

A comprehensive token deduplication pipeline with graph refinement, triangle-based growth, and frequency-based canonical label normalization.

## 🎯 **Pipeline Overview**

The complete pipeline consists of three main components:

1. **Graph Refinement** - Build clean, dense graphs from raw candidates
2. **Triangle Growth** - Expand clusters via triangle-based attachment  
3. **Canonical Normalization** - Export frequency-based medoids and normalize labels

## 🛠️ **Core Components**

### 1. Graph Refinement (`dedupe_pipeline.py` + `graph_refine.py`)

**Purpose**: Build clean, dense graphs from raw token similarity candidates.

**Key Features**:
- **Top-k per token** filtering (k=5-8) with mutual nearest neighbors
- **Hub capping** (max degree = 50-100) to prevent hub chaining
- **Adaptive thresholds** (short tokens <5 chars need ≥0.8 similarity)
- **Size-aware quality assessment** (different criteria for pairs vs multi-token clusters)

**Commands**:
```bash
# Basic pipeline
python src/deduplication/dedupe_pipeline.py all input_pairs.parquet output_dir --threshold 0.6

# Refined pipeline with graph building
python src/deduplication/dedupe_pipeline.py refined-all input_pairs.parquet output_dir --base-threshold 0.6 --k 5

# Just build clean graph
python src/deduplication/dedupe_pipeline.py graph-build input_pairs.parquet output_dir --k 5 --max-degree 50
```

### 2. Triangle-Based Growth (`graph_grow.py` + `grow_runner.py`)

**Purpose**: Expand seed clusters by adding nodes that create triangles with existing cluster members.

**Key Features**:
- **Triangle requirement** - Only add nodes that create triangles with cluster
- **Support threshold** - Require edges to ≥2 cluster tokens
- **Growth limits** - Max new nodes per cluster to prevent overgrowth
- **Iterative growth** - Multiple passes for larger clusters

**Commands**:
```bash
# Complete growth pipeline
python src/deduplication/grow_runner.py input_pairs.parquet output_dir --k 8 --grow-min-sim 0.75

# With custom growth parameters
python src/deduplication/grow_runner.py input_pairs.parquet output_dir \
  --base-threshold 0.60 --k 8 --max-degree 100 \
  --grow-min-sim 0.75 --grow-min-support 2 --grow-iterations 1
```

### 3. Canonical Normalization (`mapping_export.py`)

**Purpose**: Export frequency-based medoids and normalize token labels to canonical forms.

**Key Features**:
- **Frequency-based medoid selection** - Most frequent token per cluster
- **Streaming CSV processing** - Handle large datasets (250k chunks)
- **Canonical column generation** - `shelves_str_canonical`, `genres_str_canonical`
- **Flexible token separation** - Configurable separators and normalization

**Commands**:
```bash
# Export medoids from corpus
python src/deduplication/mapping_export.py export-medoids \
  clusters_token_map.parquet output_dir \
  --csv data/processed/books.csv --col shelves_str --col genres_str

# Normalize CSV columns
python src/deduplication/mapping_export.py normalize-columns \
  input.csv token_to_medoid.parquet output.csv \
  --col shelves_str --col genres_str

# One-shot normalization
python src/deduplication/mapping_export.py normalize-shelves-genres \
  clusters_token_map.parquet books.csv output_dir
```

## 📊 **Quality Assessment**

### Size-Aware Quality Criteria

The pipeline uses different quality criteria based on cluster size:

- **Pairs (size=2)**: Accept if `min_sim ≥ 0.85` (no triangle requirement)
- **Small clusters (3-5)**: Require `min_sim ≥ 0.75` AND `triangle_rate ≥ 0.33`
- **Big clusters (>5)**: Require `min_sim ≥ 0.70`, `mean_sim ≥ 0.78`, `triangle_rate ≥ 0.20`

### Quality Metrics

- **min_sim**: Minimum similarity within cluster
- **mean_sim**: Average similarity within cluster
- **max_sim**: Maximum similarity within cluster
- **triangle_rate**: Fraction of edges with shared neighbors
- **size**: Number of tokens in cluster

## 🎯 **Complete Workflow Example**

```bash
# 1. Build clean graph and cluster
python src/deduplication/dedupe_pipeline.py refined-all \
  data/intermediate/token_pairs.parquet outputs/clean \
  --base-threshold 0.6 --k 5 --mutual-nn --max-degree 50

# 2. Grow clusters via triangles
python src/deduplication/grow_runner.py \
  data/intermediate/token_pairs.parquet outputs/grown \
  --base-threshold 0.6 --k 8 --grow-min-sim 0.75

# 3. Export frequency-based medoids
python src/deduplication/mapping_export.py export-medoids \
  outputs/grown/clusters_token_map.parquet outputs/mapping \
  --csv data/processed/books.csv --col shelves_str --col genres_str

# 4. Normalize corpus with canonical labels
python src/deduplication/mapping_export.py normalize-columns \
  data/processed/books.csv outputs/mapping/token_to_medoid.parquet \
  data/processed/books.canonical.csv --col shelves_str --col genres_str
```

## 📈 **Results on Sample Data**

**Graph Refinement**:
- **339.9x fewer edges**: 8,157 → 24 edges with top-k + mutual-NN
- **562.8x fewer clusters**: 6,753 → 12 clusters
- **25% good clusters**: 3/12 meet size-aware criteria

**Triangle Growth**:
- **No additional growth** in sample (already optimal)
- **Maintains quality** while allowing expansion
- **Configurable growth limits** prevent overgrowth

**Canonical Normalization**:
- **Frequency-based medoids** from corpus usage
- **Canonical columns** preserve original + normalized
- **Streaming processing** handles large datasets

## 🔧 **Configuration Options**

### Graph Building
- `base_threshold`: Minimum similarity floor (default: 0.60)
- `k`: Top-k neighbors per token (default: 5)
- `max_degree`: Hub capping limit (default: 50)
- `adaptive_short_thr`: Threshold for short tokens (default: 0.80)

### Triangle Growth
- `grow_min_sim`: Minimum similarity for growth (default: 0.75)
- `grow_min_support`: Minimum cluster support (default: 2)
- `grow_require_triangle`: Require triangle formation (default: True)
- `grow_max_new_per_cluster`: Growth limit per cluster (default: 10)

### Quality Assessment
- `pair_min_sim`: Minimum similarity for pairs (default: 0.85)
- `small_min_sim`: Minimum similarity for small clusters (default: 0.75)
- `big_min_sim`: Minimum similarity for big clusters (default: 0.70)

### Normalization
- `sep`: Token separator (default: ",")
- `lowercase`: Lowercase tokens (default: False)
- `chunksize`: CSV processing chunk size (default: 250,000)

## 📁 **Output Files**

### Standard Outputs
- `pairs_clean_graph.parquet` - Clean graph edges
- `pairs_grown_graph.parquet` - Grown graph edges
- `clusters_token_map.parquet` - Token to cluster mapping
- `cluster_cohesion_metrics.parquet` - Quality metrics
- `clusters_flagged_size_aware.parquet` - Quality flags

### Mapping Outputs
- `cluster_medoids.parquet` - Frequency-based medoids
- `token_to_medoid.parquet` - Token to medoid mapping
- `*_canonical.csv` - Normalized CSV with canonical columns

## 🎯 **Key Insights**

1. **Size-2 components are normal** - triangle_rate≈0 is expected for pairs
2. **Graph density matters** - top-k + mutual-NN creates much cleaner graphs
3. **Triangle growth is conservative** - maintains quality while allowing expansion
4. **Frequency-based medoids work** - corpus usage determines canonical forms
5. **Size-aware criteria are essential** - different rules for different cluster sizes

## 🚀 **Performance Tips**

- Use **higher k values** (8-10) for denser graphs
- Apply **mutual NN filtering** to prevent hub chaining
- Monitor **triangle rates** for local consistency
- Use **streaming processing** for large datasets
- **Raise thresholds** for better quality over quantity

## 🔍 **Troubleshooting**

### Common Issues

1. **Low triangle rates**: Expected for size-2 components, use size-aware criteria
2. **Too many small clusters**: Increase k or lower thresholds
3. **Hub chaining**: Use mutual NN filtering and hub capping
4. **Memory issues**: Use streaming processing and smaller chunks
5. **Poor growth**: Lower growth thresholds or increase iterations

### Quality Assessment

```bash
# Check cluster quality
python -c "
import pandas as pd
coh = pd.read_parquet('outputs/cluster_cohesion_metrics.parquet')
print(f'Total clusters: {len(coh)}')
print(f'Good clusters: {len(coh[coh[\"min_sim\"] >= 0.85])}')
print(f'Average triangle rate: {coh[\"triangle_rate\"].mean():.3f}')
"
```

This complete pipeline provides a robust, scalable solution for token deduplication with proper quality assessment and canonical label normalization.
