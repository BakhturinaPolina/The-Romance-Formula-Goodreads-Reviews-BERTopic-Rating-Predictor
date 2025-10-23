# Anna's Archive Data Acquisition Guide

## Overview

This guide explains how to obtain Anna's Archive data dumps for offline book searching. Anna's Archive provides public database dumps (Elasticsearch JSON and MariaDB SQL) that contain metadata for all books in their archive.

## Why Use Data Dumps?

- **No rate limiting**: Search locally without web scraping restrictions
- **Faster queries**: SQL-based search on indexed data
- **Offline capability**: Work without internet connection after setup
- **Bulk processing**: Handle thousands of books efficiently
- **Reliable**: No dependency on website availability

## Data Sources

Anna's Archive provides several data dumps:

### 1. Elasticsearch Data (Primary)
- **Content**: Main book metadata index
- **Format**: JSON.gz files
- **Size**: ~200-300GB compressed
- **Contains**: Title, author, publisher, year, MD5 hash, file info

### 2. ElasticsearchAux Data (Auxiliary)
- **Content**: Additional metadata and relationships
- **Format**: JSON.gz files  
- **Size**: ~50-100GB compressed
- **Contains**: Series info, enhanced metadata, cross-references

### 3. AAC Data (Anna's Archive Combined)
- **Content**: Combined metadata from multiple sources
- **Format**: .zst compressed files
- **Size**: ~100-200GB compressed
- **Contains**: Internet Archive metadata, enhanced book info

### 4. MariaDB Data (Relational)
- **Content**: Relational database dumps
- **Format**: SQL .gz files
- **Size**: ~50-100GB compressed
- **Contains**: Structured relationships, user data, advanced queries

## Download Methods

### Method 1: Torrent Download (Recommended)

1. **Visit Anna's Archive Datasets Page**
   - Go to: https://annas-archive.org/datasets
   - Look for "Database Dumps" section

2. **Download Torrent Files**
   - `elasticsearch.torrent` - Main book metadata
   - `elasticsearchAux.torrent` - Auxiliary data
   - `aac.torrent` - Combined data (optional)
   - `mariadb.torrent` - Relational data (optional)

3. **Use Torrent Client**
   ```bash
   # Install qBittorrent or similar
   sudo apt install qbittorrent
   
   # Download torrents to your data directory
   qbittorrent elasticsearch.torrent
   ```

### Method 2: Direct Download (If Available)

Some dumps may be available for direct download:
- Check Anna's Archive datasets page for direct links
- Use `wget` or `curl` for large files
- Ensure sufficient disk space and stable connection

## Storage Requirements

### Full Dataset
- **Disk Space**: 500GB+ (compressed + processed)
- **RAM**: 30GB+ recommended for processing
- **Processing Time**: 4-8 hours for conversion

### Sample Dataset (Recommended for Testing)
- **Disk Space**: 5-10GB
- **RAM**: 8GB+ 
- **Processing Time**: 30-60 minutes

## Directory Structure

Place downloaded files in this structure:

```
data/anna_archive/
├── elasticsearch/           # Main JSON.gz files
│   ├── part-00000.json.gz
│   ├── part-00001.json.gz
│   └── ...
├── elasticsearchAux/        # Auxiliary JSON.gz files
│   ├── aux-part-00000.json.gz
│   └── ...
├── aac/                     # AAC .zst files (optional)
│   ├── aac-part-00000.zst
│   └── ...
├── mariadb/                 # SQL dump files (optional)
│   ├── mariadb-dump.sql.gz
│   └── ...
└── parquet/                 # Converted Parquet files (created by processing)
    ├── elasticsearchF/
    ├── elasticsearchAuxF/
    └── ...
```

## Verification

After download, verify your files:

```bash
# Check file sizes (should be substantial)
ls -lh data/anna_archive/elasticsearch/

# Verify file integrity (if checksums provided)
sha256sum data/anna_archive/elasticsearch/*.json.gz

# Test decompression
gunzip -t data/anna_archive/elasticsearch/part-00000.json.gz
```

## Legal and Ethical Considerations

- **Personal Use**: Downloads should be for personal research use only
- **Terms of Service**: Ensure compliance with Anna's Archive terms
- **Copyright Respect**: Only access books you have rights to
- **Rate Limiting**: Be respectful of their servers during download

## Troubleshooting

### Common Issues

1. **Insufficient Disk Space**
   - Free up space or use external storage
   - Consider sampling approach for testing

2. **Slow Download**
   - Use torrent client with multiple connections
   - Download during off-peak hours
   - Consider partial downloads for testing

3. **Corrupted Files**
   - Re-download affected files
   - Verify checksums if available
   - Test decompression before processing

### Getting Help

- Check Anna's Archive FAQ: https://annas-archive.org/faq
- Reddit community: r/Annas_Archive
- GitHub issues in related projects

## Next Steps

After acquiring the data:

1. **Start with Sampling**: Use `SAMPLING_GUIDE.md` to create test datasets
2. **Process Data**: Run `json_to_parquet.py` to convert to queryable format
3. **Test Queries**: Use `query_engine.py` to search for books
4. **Scale Up**: Process full dataset once testing is successful

## References

- [Anna's Archive Datasets](https://annas-archive.org/datasets)
- [Data Science Starter Kit](https://github.com/RArtutos/Data-science-starter-kit-Enhance)
- [Anna's Archive FAQ](https://annas-archive.org/faq)
