import requests


# This is the scraper that analyzed reddit comments with the keyword (name)
def get_reddit_comments(name):
    limit = 150
    # Public Reddit search API 
    url = "https://api.pullpush.io/reddit/search/comment/"
    params = {
        "q": name,
        "size": limit,
        "fields": ["body"],
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", [])
    except:
        return []

    
    comments = []
    # Creates a list of comments (we only do 150 max so its not too slow)
    for item in data:
        body = item.get("body", "")
        if body and name.lower() in body.lower():
            comments.append(body)

    return comments[:limit]
