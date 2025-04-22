import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# === Load OCR + sentiment data ===
df = pd.read_csv("data/trump_ocr_texts_tagged.csv")

# === Average Sentiment Score ===
avg_sentiment = df["sentiment"].mean()
normalized_score = (avg_sentiment + 1) / 2  # Scale -1 to 1 → 0 to 1

st.markdown("## Trump's Tweet Sentiment Overview")
st.markdown("**Average Sentiment Score (from recent tweets):**")

# Add Trump icon (hovering effect)
trump_img_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg"

st.markdown(f"""
<div style="position: relative; width: 100%; height: 80px;">
    <img src="{trump_img_url}" width="60" style="position: absolute; top: -30px; left: calc({normalized_score*100}% - 30px); border-radius: 50%;">
</div>
""", unsafe_allow_html=True)

# Slider (disabled)
st.slider("", 0.0, 1.0, value=normalized_score, disabled=True)

# === Word Cloud ===
st.markdown("### Most Common Words in Trump’s Recent Tweets")

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


# Part 2 -- Trends for USD and other assets
#st.subheader("💵 USD & Global Asset Performance with Trump Annotations")

# # === Sidebar Toggles (Reduced) ===
# st.sidebar.subheader("📊 Select Trends to Display")
# show_eur = st.sidebar.checkbox("EUR/USD", value=False)
# show_gold = st.sidebar.checkbox("Gold (USD/oz)", value=False)
# show_oil = st.sidebar.checkbox("Oil (WTI)", value=False)
# show_treasury = st.sidebar.checkbox("10Y Treasury Yield", value=False)

# # === Cached data fetch ===
# @st.cache_data
# def load_data():
#     tickers = {
#         "USD_Index": "DX-Y.NYB",
#         "Gold": "GC=F",
#         "Oil": "CL=F"
#     }

#     series_list = []

#     for label, symbol in tickers.items():
#         df = yf.download(symbol, start="2022-01-01", end="2025-04-30", interval="1mo", progress=False)
#         if not df.empty and "Close" in df.columns:
#             series = df["Close"]
#             series.name = label  # ✅ Correct way to name the series
#             series_list.append(series)
#         else:
#             print(f"⚠️ Skipping {label}: No valid data or missing 'Close' column")

#     if series_list:
#         return pd.concat(series_list, axis=1)
#     else:
#         return pd.DataFrame()

#     # # Combine into one DataFrame
#     # df_all = pd.DataFrame(data)

#     # if frames:
#     #     df_all = pd.concat(frames, axis=1).dropna(how="all")
#     #     df_all.index.name = "Date"
#     #     return df_all
#     # else:
#     #     return pd.DataFrame()  # fallback empty df

# df_all = load_data()

# # === Trump Events ===
# trump_events = {
#     "2023-01-15": "Tariff Announcement",
#     "2023-07-04": "Inflation Speech",
#     "2024-03-20": "Attack on Fed Chair"
# }
# trump_events = {pd.to_datetime(k): v for k, v in trump_events.items()}

# # debug
# st.write(df_all.columns)

# # === Build Plot ===
# fig = go.Figure()


# # USD Index always on
# fig.add_trace(go.Scatter(
#     x=df_all.index, y=df_all["USD_Index"],
#     name="USD Index (UUP ETF)",
#     line=dict(color="blue", width=2)
# ))




# if show_eur:
#     fig.add_trace(go.Scatter(x=df_all.index, y=df_all["EUR/USD"], name="EUR/USD", line=dict(color="green")))
# if show_gold:
#     fig.add_trace(go.Scatter(x=df_all.index, y=df_all["Gold"], name="Gold (USD/oz)", line=dict(color="gold")))
# if show_oil:
#     fig.add_trace(go.Scatter(x=df_all.index, y=df_all["Oil"], name="Oil (WTI)", line=dict(color="black")))
# if show_treasury:
#     fig.add_trace(go.Scatter(x=df_all.index, y=df_all["10Y_Yield"], name="10Y Treasury Yield", line=dict(color="red", dash="dot")))

# # === Trump Annotations ===
# # === Add Trump Annotations as vertical shapes ===
# for date, label in trump_events.items():
#     fig.add_shape(
#         type="line",
#         x0=date, x1=date,
#         y0=0, y1=1,
#         xref='x',
#         yref='paper',
#         line=dict(color="red", width=1, dash="dot")
#     )
#     fig.add_annotation(
#         x=date,
#         y=1.01,
#         xref='x',
#         yref='paper',
#         showarrow=False,
#         text=label,
#         font=dict(size=10, color="red"),
#         align="left"
#     )


# fig.update_layout(
#     title="USD and Global Asset Prices with Trump Events",
#     xaxis_title="Date",
#     yaxis_title="Value",
#     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     margin=dict(t=40)
# )

# st.plotly_chart(fig, use_container_width=True)