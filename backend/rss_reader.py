import re
import feedparser

# Function to fetch and parse RSS feed
def get_rss_feed(url):
    """
    Parses the given RSS feed URL and returns a list of dictionaries, each representing a news item.
    """
    feed = feedparser.parse(url)
    news_items = []
    
    for entry in feed.entries:
        news_item = {
            'title': entry.get('title', 'No Title'),
            'link': entry.get('link', ''),
            'published': entry.get('published', 'No publish date provided'),
            'summary': entry.get('summary', entry.get('description', ''))
        }
        news_items.append(news_item)
    
    return news_items

# Function to detect RSS source and category
def rss_detector(message):
    """
    Extracts the news source and category from the given message using regex.
    """
    pattern = r".*open\s+rss\s+([\w\s-]+)\s+([\w\s-]+).*"
    match = re.search(pattern, message, re.IGNORECASE)

    if match:
        rss_source = match.group(1).strip().lower()  # Normalize to lowercase and remove spaces
        category = match.group(2).strip().lower()
        return rss_source, category
    
    return None, None

# Function to get the appropriate RSS feed URL
def get_rss_url(rss_source, category):
    """
    Returns the corresponding RSS feed URL for a given source and category.
    """

    rss_feeds = {
        "skynews": {
            "home": "https://feeds.skynews.com/feeds/rss/home.xml",
            "world": "https://feeds.skynews.com/feeds/rss/world.xml",
            "business": "https://feeds.skynews.com/feeds/rss/business.xml",
            "technology": "https://feeds.skynews.com/feeds/rss/technology.xml",
            "politics": "https://feeds.skynews.com/feeds/rss/politics.xml"
        },
        "nasa": {
            "news": "https://www.nasa.gov/news-release/feed/",
            "recent content": "https://www.nasa.gov/feed/",
            "image of the day": "https://www.nasa.gov/feeds/iotd-feed/"
        },
        "the west australia": {
            "recent": "https://thewest.com.au/rss",
            "wa": "https://thewest.com.au/news/wa/rss",
            "national": "https://thewest.com.au/news/australia/rss",
            "international": "https://thewest.com.au/news/world/rss",
            "business": "https://thewest.com.au/business/rss",
            "politics": "https://thewest.com.au/politics/rss"
        },
        "rba": {
            "daily exchange rate": "https://www.rba.gov.au/rss/rss-cb-exchange-rates.xml",
            "media releases": "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
            "bulletin": "https://www.rba.gov.au/rss/rss-cb-bulletin.xml",
            "monetary policy statement": "https://www.rba.gov.au/rss/rss-cb-smp.xml"
        },
        "google news": {
            "general": "https://news.google.com/rss",
            "technology": "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
            "australia": "https://news.google.com/rss?hl=en-AU&gl=AU&ceid=AU:en",
            "world": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVLQUFQAQ?hl=en-AU&gl=AU&ceid=AU:en",
            "business": "https://news.google.com/rss/search?q=business&hl=en-AU&gl=AU&ceid=AU:en",
            "science": "https://news.google.com/rss/search?q=science&hl=en-AU&gl=AU&ceid=AU:en",
            "health": "https://news.google.com/rss/search?q=health&hl=en-AU&gl=AU&ceid=AU:en"
        },
        "us federal reserve": {
            "press release": "https://www.federalreserve.gov/feeds/press_all.xml",
            "regulatory policies": "https://www.federalreserve.gov/feeds/press_bcreg.xml"
        }
    }

    return rss_feeds.get(rss_source, {}).get(category, None)

# Function to fetch and display RSS feed
def display_rss_feed(message):
    """
    Extracts RSS details from the message, fetches the feed, and prints the news articles.
    """
    rss_source, category = rss_detector(message)
    
    if rss_source and category:
        news_url = get_rss_url(rss_source, category)
        
        if news_url:
            print(f"Fetching news from {rss_source.title()} - {category.title()}...\n")
            items = get_rss_feed(news_url)
            
            if not items:
                print("No news found in this category.")
                return
            
            for item in items[:5]:  # Limit to 5 news items for display
                print(f"Title: {item['title']}")
                print(f"Link: {item['link']}")
                print(f"Published: {item['published']}")
                print("-" * 40)
        else:
            print("Invalid news category or source.")
    else:
        print("Could not understand your RSS request.")

# Example usage
display_rss_feed("open rss skynews world")
display_rss_feed("open rss google news technology")
display_rss_feed("open rss nasa image of the day")
