# Proxy Setup Guide for Anna's Archive Access

## Overview

Anna's Archive requires VPN or Tor to access. This guide shows how to configure proxy support for the automated search system.

## Supported Proxy Types

### 1. Tor (Recommended)
- **Type**: SOCKS5 proxy
- **Default Port**: 9050
- **Setup**: Install and run Tor browser or Tor service

### 2. SOCKS5 Proxy
- **Type**: SOCKS5 proxy
- **Port**: Custom (usually 1080)
- **Use Case**: VPN services that provide SOCKS5

### 3. HTTP Proxy
- **Type**: HTTP proxy
- **Port**: Custom (usually 8080)
- **Use Case**: HTTP proxy services

## Quick Setup

### Option 1: Tor (Easiest)

1. **Install Tor**:
   ```bash
   # Ubuntu/Debian
   sudo apt install tor
   
   # Start Tor service
   sudo systemctl start tor
   sudo systemctl enable tor
   ```

2. **Run with Tor**:
   ```bash
   cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
   source .venv/bin/activate
   cd src/anna_archive_matcher
   
   # Test proxy connection
   python utils/proxy_automated_search.py \
     --romance-csv utils/priority_lists/test_sample_50_books.csv \
     --max-books 5 \
     --proxy-type tor \
     --test-proxy
   
   # Run full search with Tor
   python utils/proxy_automated_search.py \
     --romance-csv utils/priority_lists/top_rated_popular_books.csv \
     --max-books 20 \
     --proxy-type tor
   ```

### Option 2: Custom SOCKS5 Proxy

```bash
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/test_sample_50_books.csv \
  --max-books 10 \
  --proxy-type socks5 \
  --proxy-host 127.0.0.1 \
  --proxy-port 1080 \
  --test-proxy
```

### Option 3: HTTP Proxy

```bash
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/test_sample_50_books.csv \
  --max-books 10 \
  --proxy-type http \
  --proxy-host 127.0.0.1 \
  --proxy-port 8080 \
  --proxy-user username \
  --proxy-pass password \
  --test-proxy
```

## Tor Installation and Setup

### Ubuntu/Debian
```bash
# Install Tor
sudo apt update
sudo apt install tor

# Start Tor service
sudo systemctl start tor
sudo systemctl enable tor

# Check if Tor is running
sudo systemctl status tor

# Check Tor port
sudo netstat -tlnp | grep 9050
```

### macOS
```bash
# Install Tor with Homebrew
brew install tor

# Start Tor service
brew services start tor

# Or run Tor manually
tor
```

### Windows
1. Download Tor Browser from https://www.torproject.org/
2. Install and run Tor Browser
3. Tor will run on port 9050 by default

## VPN Setup

If you're using a VPN service that provides proxy access:

### 1. Get Proxy Details
- **Host**: Usually your VPN server IP or localhost
- **Port**: SOCKS5 port (usually 1080) or HTTP port (usually 8080)
- **Credentials**: Username/password if required

### 2. Configure the Search
```bash
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/test_sample_50_books.csv \
  --max-books 10 \
  --proxy-type socks5 \
  --proxy-host YOUR_VPN_HOST \
  --proxy-port YOUR_VPN_PORT \
  --proxy-user YOUR_USERNAME \
  --proxy-pass YOUR_PASSWORD \
  --test-proxy
```

## Testing Proxy Connection

Always test your proxy connection before running the full search:

```bash
# Test Tor connection
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/test_sample_50_books.csv \
  --max-books 1 \
  --proxy-type tor \
  --test-proxy

# Test custom proxy
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/test_sample_50_books.csv \
  --max-books 1 \
  --proxy-type socks5 \
  --proxy-host 127.0.0.1 \
  --proxy-port 1080 \
  --test-proxy
```

## Troubleshooting

### Common Issues

1. **"Connection refused"**
   - Check if Tor/VPN is running
   - Verify port number
   - Check firewall settings

2. **"Proxy authentication failed"**
   - Verify username/password
   - Check proxy type (HTTP vs SOCKS5)

3. **"SSL certificate verify failed"**
   - This is normal - the system handles SSL issues automatically

4. **"Timeout errors"**
   - Increase delay between requests
   - Check proxy performance
   - Try different proxy server

### Debug Commands

```bash
# Check if Tor is running
sudo systemctl status tor

# Check Tor port
sudo netstat -tlnp | grep 9050

# Test Tor connection manually
curl --socks5 127.0.0.1:9050 https://httpbin.org/ip

# Check proxy logs
tail -f proxy_automated_search.log
```

## Performance Tips

### 1. Optimize Delays
```bash
# Faster search (higher risk of blocking)
python utils/proxy_automated_search.py \
  --delay-min 1.0 \
  --delay-max 2.0

# Slower search (more reliable)
python utils/proxy_automated_search.py \
  --delay-min 3.0 \
  --delay-max 6.0
```

### 2. Batch Size
```bash
# Small batches (more reliable)
--max-books 10

# Larger batches (faster but riskier)
--max-books 100
```

### 3. Proxy Selection
- **Tor**: Most reliable, slower
- **VPN SOCKS5**: Faster, depends on VPN quality
- **HTTP Proxy**: Fastest, least reliable

## Security Notes

1. **Use trusted proxies only**
2. **Don't share proxy credentials**
3. **Monitor for unusual activity**
4. **Respect rate limits**
5. **Follow terms of service**

## Example Workflows

### Quick Test with Tor
```bash
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/test_sample_50_books.csv \
  --max-books 5 \
  --proxy-type tor \
  --test-proxy
```

### Production Run with Tor
```bash
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/top_rated_popular_books.csv \
  --max-books 50 \
  --proxy-type tor \
  --delay-min 2.0 \
  --delay-max 4.0
```

### VPN SOCKS5 Proxy
```bash
python utils/proxy_automated_search.py \
  --romance-csv utils/priority_lists/most_reviewed_books.csv \
  --max-books 30 \
  --proxy-type socks5 \
  --proxy-host YOUR_VPN_HOST \
  --proxy-port 1080 \
  --proxy-user YOUR_USERNAME \
  --proxy-pass YOUR_PASSWORD
```

The proxy-enabled system will automatically handle connection issues and provide detailed logging for troubleshooting.
