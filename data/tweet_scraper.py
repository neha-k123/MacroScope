import praw
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
import pytesseract
import pandas as pd
from datetime import datetime
import os


reddit = praw.Reddit(
    client_id="ActbsRUTEIPcm0aPMSTqZQ",  # replace with your real ID
    client_secret="69nJvSYIVEcjERqAbwtFW2WTtYZ87w",  # replace with real secret
  user_agent="macOS:macroscope-fetch:v1.0 (by /u/SavingsMundane368)"
)

# === Setup ===
subreddit = reddit.subreddit("TrumpTweets2")
output_dir = "data/trump_images"
os.makedirs(output_dir, exist_ok=True)

# === Download Images ===
for post in subreddit.new(limit=50):
    if post.url.endswith((".jpg", ".jpeg", ".png")):
        timestamp = datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{post.id}.jpg"
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            try:
                img_data = requests.get(post.url).content
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Failed to download {post.url}: {e}")




# # Test connection
# print(reddit.read_only)  # Should print True




# Test that you can scrape r/python
# import praw

# reddit = praw.Reddit(
#     client_id="ActbsRUTEIPcm0aPMSTqZQ",  # replace with your real ID
#     client_secret="69nJvSYIVEcjERqAbwtFW2WTtYZ87w",  # replace with real secret
#   user_agent="macOS:macroscope-fetch:v1.0 (by /u/SavingsMundane368)"
# )

# print("Auth check:", reddit.read_only)  # Should say True

# try:
#   subreddit = reddit.subreddit("python")
#   for post in subreddit.hot(limit=10):
#       print(post.title)
# except Exception as e:
#     print("Failed:", e)
