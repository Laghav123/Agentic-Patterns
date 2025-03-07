import json
import requests

def get_current_temprature(
        location: str,
        unit: str = "celsius"
) -> str:
    if(unit == "celsius"):
       return "30"
    return "45"

def fetch_top_hacker_news_stories (top_n: int):
    """
    Fetch the top stories from Hacker News.
    This function retrieves the top top_n' stories from Hacker News using the Hacker News API.
    Each story contains the title, URL, score, author, and time of submission. The data is fetched
    from the official Firebase Hacker News API, which returns story details in JSON format.
    Args:
    top_n (int): The number of top stories to retrieve.
    """
    top_stories_url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    try:
        response = requests.get(top_stories_url)
        response.raise_for_status() # Check for HTTP errors
        # Get the top story IDs
        top_story_ids = response.json()[:top_n]
        top_stories = []
        # For each story ID, fetch the story details
        for story_id in top_story_ids:
            story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            story_response = requests.get(story_url)
            story_response. raise_for_status() # Check for HTTP errors
            story_data = story_response.json()
            # Append the story title and URL (or other relevant info) to the list
            top_stories.append({
                'title': story_data.get('title', 'No title'),
                'url': story_data.get('url', 'No URL available'),
            })
        return json.dumps (top_stories)
    except requests.exceptions. RequestException as e:
        print(f"An error occurred: {e}")



tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temprature",
                "description": "Use this to retrieve temprature of any particular location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. Hyderabas, Telangana"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature to use. Defaults to celsius."
                        }
                    },
                    "required": ["location"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_top_hacker_news_stories",
                "description": "This function can make an API call to hacker news platform to fetch top n latest news",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "int",
                            "description": "number of latest news to fetch"
                        },
                        
                    },
                    "required": ["top_n"]
                },
            },
        }
]

available_functions = {
    "get_current_temprature" : get_current_temprature,
    "fetch_top_hacker_news_stories": fetch_top_hacker_news_stories,
}