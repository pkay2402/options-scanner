# RSS Feed Setup Guide

## How to Configure Your Google Alert RSS Feeds

### Step 1: Get Your RSS URLs from Google Alerts

1. Go to https://www.google.com/alerts
2. For each alert you created:
   - Click on the **pencil icon** (edit) next to the alert
   - Click **"Show options"**
   - Under **"Deliver to"**, select **"RSS feed"**
   - Click **"Update Alert"**
3. Click the **RSS icon** (🔔) next to each alert to see the feed URL
4. Copy both URLs - they will look like:
   ```
   https://www.google.com/alerts/feeds/12345678901234567890/1234567890123456789
   ```

### Step 2: Add URLs to Your Code

Open `Main_Dashboard.py` and find this section (around line 1375):

```python
# Replace these with your actual Google Alert RSS URLs
rss_feeds = {
    'Feed 1': 'YOUR_FIRST_RSS_URL_HERE',
    'Feed 2': 'YOUR_SECOND_RSS_URL_HERE'
}
```

Replace with your actual URLs and custom names:

```python
rss_feeds = {
    'Market News': 'https://www.google.com/alerts/feeds/YOUR_FEED_1_URL',
    'Stock Alerts': 'https://www.google.com/alerts/feeds/YOUR_FEED_2_URL'
}
```

Do the same in `pages/13_Trading_Hub.py` (same location).

### Step 3: Install feedparser

```bash
pip install feedparser
```

Or if already in requirements.txt (which I added):
```bash
pip install -r requirements.txt
```

### Step 4: Restart Streamlit

```bash
streamlit run Main_Dashboard.py
```

## Features

- **Auto-refresh**: Cached for 5 minutes, updates automatically
- **Collapsible**: Expander is collapsed by default to save space
- **Two feeds side-by-side**: Each feed shows latest 5 articles
- **Clickable links**: Click article titles to open in new tab
- **Published dates**: Shows when each alert was published

## Customization Options

### Show More/Less Articles
Change `[:5]` to show different number of articles:
```python
for entry in feed.entries[:10]:  # Show 10 instead of 5
```

### Expand by Default
Set `expanded=True`:
```python
with st.expander("📰 Market News & Alerts", expanded=True):
```

### Change Cache Duration
Adjust TTL (time-to-live):
```python
@st.cache_data(ttl=600)  # 10 minutes instead of 5
```

### Add More Feeds
Add a third column:
```python
news_col1, news_col2, news_col3 = st.columns(3)
# Add third feed in news_col3
```

## Troubleshooting

**No articles showing?**
- Verify RSS URL is correct
- Check if Google Alerts has any new items
- Clear Streamlit cache with 🔄 button

**Error loading feeds?**
- Check internet connection
- Verify feedparser is installed
- Check logs for specific errors

**Slow loading?**
- Increase cache TTL to reduce API calls
- Consider showing fewer articles per feed
