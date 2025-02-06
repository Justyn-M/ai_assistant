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
    print('Which news provider would you like to access? (SkyNews, 9news.com)')
    usrinput = input().strip().lower() #convert to lowercase
    
    if usrinput == 'skynews':
        print('What news section would you like to read? (Home, World, Business, Technology, Politics)')
        newsoption = input().strip().lower()

        if newsoption == 'home':
            test_url = "https://feeds.skynews.com/feeds/rss/home.xml"

        elif newsoption == 'world':
            test_url = "https://feeds.skynews.com/feeds/rss/world.xml"

        elif newsoption == 'business':
            test_url = "https://feeds.skynews.com/feeds/rss/business.xml"
            
        elif newsoption == 'technology':
            test_url = "https://feeds.skynews.com/feeds/rss/technology.xml"
            
        elif newsoption == 'politics':
            test_url = "https://feeds.skynews.com/feeds/rss/politics.xml"
        else:
            print("Invalid Input")
        
        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)


    elif usrinput == '9news.com':
        test_url = "https://www.9news.com/feeds/syndication/rss/news"
        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)
    else:
        print("Invalid Input")