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
    print('Which news provider would you like to access? (SkyNews, Nasa, West Australian, RBA, Google News, US Federal Reserve)')
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

    elif usrinput == 'nasa':
        print('What news section would you like to read? (News, Recent Content, Image of the Day)')
        newsoption = input().strip().lower()

        if newsoption == 'news':
            test_url = "https://www.nasa.gov/news-release/feed/"

        elif newsoption == 'recent content':
            test_url = "https://www.nasa.gov/feed/"

        elif newsoption == 'image of the day':
            test_url = "https://www.nasa.gov/feeds/iotd-feed/"
            
        else:
            print("Invalid Input")
        
        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)

    elif usrinput == 'the west australia':
        print('What news section would you like to read? (Recent, WA, national, international, business, politics)')
        newsoption = input().strip().lower()

        if newsoption == 'recent':
            test_url = "https://thewest.com.au/rss"

        elif newsoption == 'wa':
            test_url = "https://thewest.com.au/news/wa/rss"

        elif newsoption == 'national':
            test_url = "https://thewest.com.au/news/australia/rss"
            
        elif newsoption == 'international':
            test_url = "https://thewest.com.au/news/world/rss"

        elif newsoption == 'business':
            test_url = "https://thewest.com.au/business/rss"
            
        elif newsoption == 'politics':
            test_url = "https://thewest.com.au/politics/rss"
        else:
            print("Invalid Input")

        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)

    elif usrinput == 'rba':
        print('What news section would you like to read? (Daily Exchange Rates, Media Releases, Bulletin, Stability Review, Monetary Policy Statement, politics)')
        newsoption = input().strip().lower()

        if newsoption == 'daily exchange rates':
            test_url = "https://www.rba.gov.au/rss/rss-cb-exchange-rates.xml"

        elif newsoption == 'media releases':
            test_url = "https://www.rba.gov.au/rss/rss-cb-media-releases.xml"

        elif newsoption == 'bulletin':
            test_url = "https://www.rba.gov.au/rss/rss-cb-bulletin.xml"
            
        elif newsoption == 'Monetary Policy Statement':
            test_url = "https://www.rba.gov.au/rss/rss-cb-smp.xml"

        elif newsoption == 'business':
            test_url = "https://thewest.com.au/business/rss"
            
        elif newsoption == 'politics':
            test_url = "https://thewest.com.au/politics/rss"
        else:
            print("Invalid Input")

        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)

    elif usrinput == 'google news':
        print('What news section would you like to read? (General, australia, world, business, science, health)')
        newsoption = input().strip().lower()

        if newsoption == 'general':
            test_url = "https://news.google.com/rss"

        elif newsoption == 'technology':
            test_url = "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"

        elif newsoption == 'australia':
            test_url = "https://news.google.com/rss?hl=en-AU&gl=AU&ceid=AU:en"
            
        elif newsoption == 'world':
            test_url = "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVLQUFQAQ?hl=en-AU&gl=AU&ceid=AU:en"

        elif newsoption == 'business':
            test_url = "https://news.google.com/rss/search?q=business&hl=en-AU&gl=AU&ceid=AU:en"
            
        elif newsoption == 'science':
            test_url = "https://news.google.com/rss/search?q=science&hl=en-AU&gl=AU&ceid=AU:en"
        
        elif newsoption == 'health':
            test_url = "https://news.google.com/rss/search?q=health&hl=en-AU&gl=AU&ceid=AU:en"

        else:
            print("Invalid Input")

        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)

    elif usrinput == 'us federal reserve':
        print('What news section would you like to read? (Press Release, Regulatory Policies, Bulletin, Stability Review, Monetary Policy Statement, politics)')
        newsoption = input().strip().lower()

        if newsoption == 'press release':
            test_url = "https://www.federalreserve.gov/feeds/press_all.xml"

        elif newsoption == 'regulatory policies':
            test_url = "https://www.federalreserve.gov/feeds/press_bcreg.xml"

        elif newsoption == 'bulletin':
            test_url = "https://www.rba.gov.au/rss/rss-cb-bulletin.xml"
            
        elif newsoption == 'Monetary Policy Statement':
            test_url = "https://www.rba.gov.au/rss/rss-cb-smp.xml"

        elif newsoption == 'business':
            test_url = "https://thewest.com.au/business/rss"
            
        elif newsoption == 'politics':
            test_url = "https://thewest.com.au/politics/rss"
        else:
            print("Invalid Input")

        items = get_rss_feed(test_url)
        for item in items:
            print("Title:", item['title'])
            print("Link:", item['link'])
            print("Published:", item['published'])
            print("-" * 40)

    else:
        print("Invalid Input")