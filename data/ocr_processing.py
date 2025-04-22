import os
import pytesseract
from PIL import Image
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# === Setup ===
image_dir = "data/trump_images"
output_csv = "data/trump_ocr_texts_tagged.csv"
keywords = ["economy", "dollar", "tariff", "inflation", "budget"]
analyzer = SentimentIntensityAnalyzer()

# === Utility: Clean text ===
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === OCR + Sentiment + Tagging ===
results = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path)
            raw_text = pytesseract.image_to_string(image)
            cleaned = clean_text(raw_text)
            sentiment = analyzer.polarity_scores(cleaned)["compound"]
            is_extreme = sentiment >= 0.7 or sentiment <= -0.7
            contains_keyword = any(k in cleaned.lower() for k in keywords)
            results.append({
                "filename": filename,
                "text": cleaned,
                "sentiment": sentiment,
                "extreme": is_extreme,
                "has_keyword": contains_keyword
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# === Save and display matches ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print("\n📢 First 5 tweets with keywords (dollar, tariff, inflation, recession) sorted by sentiment:\n")

filtered = df[df["has_keyword"]].sort_values(by="sentiment", ascending=False).head(5)

for i, row in filtered.iterrows():
    print(f"--- [{row['filename']}] ---")
    print(f"Sentiment Score: {row['sentiment']:.2f}")
    print(row["text"])
    print("="*80)



# How to interpret the sentiment score:
# As produced by Chat GPT
# The sentiment score ranges from -1 to +1, where negative values indicate negative sentiment, positive values indicate positive sentiment, and values near 0 suggest neutral or mixed tone. A score closer to -1 signals strong criticism or negativity, while a score closer to +1 reflects strong praise or enthusiasm.
