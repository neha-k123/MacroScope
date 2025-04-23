import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import os

import toml
config = toml.load("config.toml")

# Access keys
fred_api_key = config["FRED"]["api_key"]

reddit_id = config["REDDIT"]["client_id"]
reddit_secret = config["REDDIT"]["client_secret"]
reddit_agent = config["REDDIT"]["user_agent"]

# === Load OCR + sentiment data ===
df = pd.read_csv("data/trump_ocr_texts_tagged.csv")

# === Average Sentiment Score ===
avg_sentiment = df["sentiment"].mean()
normalized_score = (avg_sentiment + 1) / 2  # Scale -1 to 1 → 0 to 1

st.markdown("## President Trump's Sentiment Overview & USD Performance")

st.markdown("""
<div>
  <p style="font-size: 20px;">
    Gathering text from 50 of President Trump's Truth Social Post, I derived an overall sentiment score. <br>
    The Trump Tweet Sentiment Score is a number between 0 and 1 that reflects the emotional tone of his recent posts, where 0 is very negative and 1 is very positive. <br>
    A higher score suggests more optimistic or approving language, while a lower score indicates more critical or combative tone.
  </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div>
  <p style="font-size: 32px;">
    Average Sentiment Score (from recent tweets)
  </p>
</div>
""", unsafe_allow_html=True)

# Add Trump icon (hovering effect)
trump_img_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg"

st.markdown(f"""
<div style="position: relative; width: 100%; height: 80px;">
    <img src="{trump_img_url}" width="60" style="position: absolute; top: -30px; left: calc({normalized_score*100}% - 30px); border-radius: 50%;">
</div>
""", unsafe_allow_html=True)

# Slider (disabled)
st.slider("", 0.0, 1.0, value=normalized_score, disabled=True)

# Divider
st.markdown("---")  # Optional visual break
st.markdown("<br>", unsafe_allow_html=True)

# === Word Cloud ===
st.markdown("### Most Common Words in Trump’s Recent Tweets")

st.markdown("""
<div>
  <p style="font-size: 20px;">
    After filtering some filler words, this word cloud highlights the most commonly used words in the Truth Social posts. <br>
    The larger the term, the more frequently it was used. <br>
  </p>
</div>
""", unsafe_allow_html=True)

# Base stopwords
stopwords = set(STOPWORDS)

# Custom additions
custom_stopwords = {
    "j", "donald", "trump", "real", "realdonaldtrump", "pm", "apr", "retruths", "likes", "rt", "com",
    "q", "am", "ie", "uv", "m", "name", "s", "t", "x", "z", "re", "http", "https",
    "youtube", "youtu", "egg", "logo"  # any others from OCR artifacts
}

stopwords.update(custom_stopwords)

# Generate word cloud
text_blob = " ".join(df["text"].dropna().astype(str)).lower()

wordcloud = WordCloud(
    width=500,
    height=250,
    background_color="white",
    stopwords=stopwords,
    max_words=100,
    collocations=True
).generate(text_blob)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# === Last updated date ===
df["timestamp"] = pd.to_datetime(df["filename"].str.extract(r'(\d{4}-\d{2}-\d{2})')[0], errors='coerce')
latest = df["timestamp"].dropna().max()

if pd.notnull(latest):
    st.markdown(f"**Tweets last pulled on:** {latest.strftime('%B %d, %Y')}")
else:
    st.markdown("**Tweets last pulled on:** Date unknown")

# Divider
st.markdown("---")  # Optional visual break
st.markdown("<br>", unsafe_allow_html=True)

# Part 2 -- Trends for USD 
st.subheader("USD Performance with Major Trump Events")

st.markdown("""
<div>
  <p style="font-size: 20px;">
    As of April 23, 2025, the USD has dropped to its lowest level in nearly 3 years. <br>
    This is highlighted below and is annotated with vertical lines to show influence of some recent events 
    during President Trump's presidency that I found personally significant.
  </p>
</div>
""", unsafe_allow_html=True)

# Construct the correct path to the image
image_path = os.path.join("scripts", "usd_with_trump_events.png")

# Load and display the image
st.image(Image.open(image_path))