import feedparser


#Feedparsing function - Returns a list of dictionaries, each representing a news item.
def get_rss_feed(url):

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

# Functionality Testing
if __name__ == '__main__':
    test_url = "https://feeds.skynews.com/feeds/rss/home.xml"
    items = get_rss_feed(test_url)
    for item in items:
        print("Title:", item['title'])
        print("Link:", item['link'])
        print("Published:", item['published'])
        print("-" * 40)