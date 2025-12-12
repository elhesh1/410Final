import requests


# This is the scraper that analyzed reddit comments with the keyword (name)
import requests

def get_reddit_comments(name, limit=150):
    url = "https://api.pullpush.io/reddit/search/comment/"
    params = {
        "q": name,
        "size": limit,
        "fields": ["body"],
    }

    # Trys twice, because sometimes the api times outs
    for i in range(2):
        try:

            r = requests.get(url, params=params, timeout=30)
            data = r.json().get("data", [])

            # Returns array of all the comments.
            return [d.get("body", "") for d in data if d.get("body")]
        except Exception as e:
            print(f"scraper failed: {i}  ", e)

    return []
