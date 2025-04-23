import streamlit as st
import pandas as pd
import numpy as np
from model_recession import get_data, train_model
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LogisticRegression




# Import the functions from model_recession script
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Because model_recession is in a different folder

from model_recession import get_cached_model

import toml
config = toml.load("config.toml")

# Access keys
fred_api_key = config["FRED"]["api_key"]

reddit_id = config["REDDIT"]["client_id"]
reddit_secret = config["REDDIT"]["client_secret"]
reddit_agent = config["REDDIT"]["user_agent"]


# FRED setup
fred = Fred(api_key=fred_api_key)

st.title("Recession Probability Forecast")

st.markdown("""
<div>
  <p style="font-size: 20px;">
        This page shows a real-time forecast for the probability of a U.S. recession in the next quarter, as computed by an ML model,
        based on the latest macro indicators.
  </p>
</div>
""", unsafe_allow_html=True)



# Show the input values associated date
date_str = pd.Timestamp.today().strftime("%B %d, %Y")
st.markdown(f"**Data as of:** {date_str}")




# === Load model and scaler ===
# From model
model, X_test, y_test, y_pred, coef_df, df = get_cached_model()
latest_cols = X_test.columns
scaler = StandardScaler().fit(df[latest_cols])



# === Pull latest values ===
def get_latest_feature_values():
    # === Fetch raw values ===
    umcsent = fred.get_series("UMCSENT")[-1]
    icsa = fred.get_series("ICSA")[-1]
    pce = fred.get_series("PCE").pct_change().dropna()[-1]
    t10y2y = fred.get_series("T10Y2Y")[-1]
    vix = yf.download("^VIX", period="7d", interval="1d")["Close"].dropna().values[-1]


    # === MA3 of Unemployment ===
    unemp = fred.get_series("UNRATE")
    ma3 = unemp.rolling(3).mean().dropna()[-1]

    # === 3M SP500 return ===
    spy = yf.download("SPY", start="2022-01-01", interval="1d")["Close"]
    spy.index = pd.to_datetime(spy.index)

    spy_monthly = spy.resample("M").last()
    spy_return = spy_monthly.pct_change(3).dropna().values[-1]

    # === Combine into input DataFrame ===
    return pd.DataFrame([{
        "T10Y2Y": t10y2y,
        "UMCSENT": umcsent,
        "ICSA": icsa,
        "PCE_Change": pce,
        "MA3": ma3,
        "SP500_3M_Return": spy_return,
        "VIX": vix
    }])


# Live data
live_input = get_latest_feature_values()
live_input = live_input[latest_cols]  # Ensure exact same column order
live_scaled = scaler.transform(live_input)


# Input values

# Transpose and format for vertical display
formatted = live_input.copy()
formatted = formatted.T  # Transpose
formatted.columns = ["Value"]  # Rename the column for clean display

# Format specific values
formatted.loc["SP500_3M_Return", "Value"] = f"{float(formatted.loc['SP500_3M_Return', 'Value']):.2%}"
formatted.loc["VIX", "Value"] = f"{float(formatted.loc['VIX', 'Value']):.2f}"


st.subheader("Current Macro Indicators")
st.dataframe(formatted.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

# Legend for the indicators with hyperlinks to FRED site
st.markdown("### Indicator Definitions")
st.markdown("""
- **[T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y)**: 10-Year minus 2-Year Treasury Yield Spread  
- **[UMCSENT](https://fred.stlouisfed.org/series/UMCSENT)**: University of Michigan Consumer Sentiment Index  
- **[MA3](https://fred.stlouisfed.org/series/UNRATE)**: 3-Month Moving Average of the U.S. Unemployment Rate  
- **[ICSA](https://fred.stlouisfed.org/series/ICSA)**: Initial Weekly Jobless Claims (Seasonally Adjusted)  
- **[PCE_Change](https://fred.stlouisfed.org/series/PCE)**: Monthly Percent Change in Personal Consumption Expenditures  
- **[SP500_3M_Return](https://finance.yahoo.com/quote/SPY)**: S&P 500 Return over the Past 3 Months  
- **[VIX](https://finance.yahoo.com/quote/%5EVIX)**: CBOE Volatility Index (Market "Fear Gauge")  
""")

# === Predict ===
prob = model.predict_proba(live_scaled)[0][1]

# Divider
st.markdown("---")  # Optional visual break
st.markdown("<br>", unsafe_allow_html=True)

# Probability
st.subheader("Model Forecast")

# Color by severity
if prob > 0.6:
    box_color = "#441111"  # deep red
elif prob > 0.3:
    box_color = "#332600"  # dark orange
else:
    box_color = "#113311"  # dark green

# Display the original recession probability
with st.container():
    st.markdown(f"""
    <div style="padding: 20px; border: 1px solid #888; border-radius: 8px;
                background-color: {box_color};">
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: white;">
            Recession Probability: {prob:.2%}
        </div>
    </div>
    """, unsafe_allow_html=True)


# Visual slider of probability
st.slider("Recession Severity", 0.0, 1.0, prob, format="%.2f", disabled=True)



# Optional: traffic light indicator
if prob >= 0.7:
    st.error("High likelihood of recession")
elif prob >= 0.4:
    st.warning("Moderate likelihood")
else:
    st.success("Low likelihood of recession")

# Divider
st.markdown("---")  # Optional visual break
st.markdown("<br>", unsafe_allow_html=True)


# Fine tune the model weights
st.subheader("Fine-Tune Model Weights")


st.markdown("""
<div>
  <p style="font-size: 20px;">
        Please use the sliders below for the model features to explore how changing the weight of each 
        feature impacts the recession forecast. <br>   
  </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div>
  <p style="font-size: 20px;">
    Interpreting Feature Weights: <br>
    A <b> negative weight</b> means that as the feature increases, the model is <b>less likely</b> to predict a recession.  
    A <b>positive weight</b> means the feature increases the <b>likelihood</b> of a recession prediction.
  </p>
</div>
""", unsafe_allow_html=True)

# Step 1: Create initial weights
initial_weights = dict(zip(coef_df['Feature'], coef_df['Coefficient']))

# st.write("📊 Live scaled input", live_scaled)


# Step 2: Let user adjust sliders
tuning_weights = {}
for col in latest_cols:
    default = float(initial_weights.get(col, 1.0))
    tuning_weights[col] = st.slider(
        f"{col} weight",
        min_value=-5.0,
        max_value=5.0,
        value=round(default, 2),
        step=0.1
    )


# Recreate the model structure
tuned_model = LogisticRegression()
tuned_model.classes_ = np.array([0, 1])
tuned_model.intercept_ = np.array([model.intercept_[0]])  # keep the same for now

# Update coefficients with slider values
weights_array = np.array([tuning_weights[col] for col in latest_cols])
tuned_model.coef_ = np.array([weights_array])

tuned_prob = tuned_model.predict_proba(live_scaled)[0][1]
#print("Tuned prob: " , tuned_prob)


# Color by severity
if tuned_prob > 0.6:
    box_color = "#441111"  # deep red
elif tuned_prob > 0.3:
    box_color = "#332600"  # dark orange
else:
    box_color = "#113311"  # dark green


# Display tuned prob. in styled container
with st.container():
    st.markdown(f"""
    <div style="padding: 20px; border: 1px solid #888; border-radius: 8px;
                background-color: {box_color};">
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: white; margin-top: 10px;">
            Tuned Recession Probability: {tuned_prob:.2%}
        </div>
    </div>
    """, unsafe_allow_html=True)



if st.button("🔄 Retrain Model"):
    st.cache_data.clear()
    st.experimental_rerun()