# Quick Start Guide: Romance Books Without Torrents

## 🎯 What You Have Now

I've created a complete system to help you find romance books on Anna's Archive without downloading full datasets or using torrents. Here's what's ready for you:

### ✅ **Priority Lists Created**
- **Top-rated books**: 129 books (rating > 4.0, reviews > 50k)
- **Most-reviewed books**: 85 books (reviews > 100k)
- **Recent popular books**: 443 books (2010+, reviews > 20k)
- **Test sample**: 50 books for getting started
- **Search queries**: Ready-to-use search terms
- **Manual search template**: For tracking your findings

### ✅ **Tools Ready**
- **Priority book selector**: Creates focused lists
- **Manual search helper**: Interactive browser-based search
- **Alternative sources**: Open Library, Internet Archive integration

## 🚀 **Quick Start (5 minutes)**

### Step 1: Start with Test Sample
```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
source .venv/bin/activate
cd src/anna_archive_matcher/utils

# View your test sample
head -5 priority_lists/test_sample_50_books.csv
```

### Step 2: Interactive Manual Search
```bash
# Start interactive search session
python manual_search_helper.py \
  --template priority_lists/manual_search_template.csv \
  --interactive
```

This will:
- Open Anna's Archive search pages in your browser
- Guide you through each book
- Let you record findings (found/not found, MD5 hashes, etc.)
- Track your progress automatically

### Step 3: Use Your Existing Download System
Once you have MD5 hashes from manual searches:
```bash
# Use your existing download system
cd ../../../
python standalone_downloader.py \
  --csv src/anna_archive_matcher/utils/manual_search_progress_*.csv \
  --output-dir organized_outputs/epub_downloads
```

## 📊 **Expected Results**

### **Realistic Expectations**
- **Manual search success rate**: 15-30% of books found
- **Time per book**: 1-2 minutes of searching
- **Test sample (50 books)**: 7-15 books found in 1-2 hours
- **Top-rated list (129 books)**: 20-40 books found in 3-4 hours

### **Quality Matches**
- **Exact matches**: Same title and author
- **Good matches**: Same book, different edition
- **Acceptable matches**: Same book, slightly different title formatting

## 🎯 **Recommended Strategy**

### **Phase 1: Test with 50 Books (1-2 hours)**
1. Use the test sample to learn the process
2. Get familiar with Anna's Archive search interface
3. Test the manual search helper
4. Download a few books to verify the workflow

### **Phase 2: Focus on Top Books (3-4 hours)**
1. Process the top-rated popular books (129 books)
2. Focus on books with high ratings and many reviews
3. These are most likely to be found and downloaded

### **Phase 3: Expand Gradually (as needed)**
1. Process most-reviewed books (85 books)
2. Add recent popular books (443 books)
3. Scale up based on your success rate

## 🛠️ **Manual Search Process**

### **For Each Book:**
1. **Search Anna's Archive** using the provided search URL
2. **Look for exact matches** in title and author
3. **Check file formats** (EPUB preferred, PDF acceptable)
4. **Record findings**:
   - Found: Yes/No
   - MD5 hash (if found)
   - Download URL
   - File format
   - Any notes

### **Search Tips:**
- **Try different search terms**: Exact title, simplified title, author + romance
- **Check multiple results**: Sometimes books appear with slightly different titles
- **Look for series**: Books in series are often easier to find
- **Check file sizes**: Larger files are usually better quality

## 📁 **Files Created for You**

```
src/anna_archive_matcher/utils/priority_lists/
├── test_sample_50_books.csv              # Start here (50 books)
├── top_rated_popular_books.csv           # High-rated books (129 books)
├── most_reviewed_books.csv               # Popular books (85 books)
├── recent_popular_books.csv              # Recent books (443 books)
├── test_sample_search_queries.csv        # Ready search terms
└── manual_search_template.csv            # Tracking template
```

## 🔧 **Alternative Approaches**

### **If Manual Search is Too Slow:**
1. **Use alternative sources**: Open Library, Internet Archive
2. **Focus on public domain**: Project Gutenberg for older romance classics
3. **Check library resources**: Many libraries have digital romance collections

### **If You Want More Automation:**
1. **Web scraping**: Create scripts to search Anna's Archive programmatically
2. **API integration**: Use Anna's Archive API if available
3. **Batch processing**: Process multiple books simultaneously

## 📈 **Success Tracking**

The system automatically tracks:
- **Books processed**: How many you've searched
- **Books found**: How many were located
- **Success rate**: Percentage of books found
- **MD5 hashes**: Ready for batch downloading
- **Progress**: Resume where you left off

## 🎉 **Next Steps**

1. **Start with test sample**: `python manual_search_helper.py --template priority_lists/manual_search_template.csv --interactive`
2. **Search 10-20 books** to get familiar with the process
3. **Download found books** using your existing system
4. **Scale up** to larger lists based on your success rate
5. **Track progress** and adjust strategy as needed

## 💡 **Pro Tips**

- **Start small**: Test with 10-20 books first
- **Focus on quality**: Better to find fewer high-quality matches
- **Use multiple search terms**: Try different variations
- **Check series**: Books in popular series are easier to find
- **Save progress regularly**: Don't lose your work
- **Be patient**: Manual search takes time but gives good results

This approach will give you a substantial collection of romance books without requiring torrent downloads or full dataset processing. You can start immediately and scale up based on your success rate and available time.
