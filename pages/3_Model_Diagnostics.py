import streamlit as st
from model_recession import get_data, train_model, plot_predictions
import seaborn as sns
import matplotlib.pyplot as plt

import toml
config = toml.load("config.toml")

# Access keys
fred_api_key = config["FRED"]["api_key"]

reddit_id = config["REDDIT"]["client_id"]
reddit_secret = config["REDDIT"]["client_secret"]
reddit_agent = config["REDDIT"]["user_agent"]

st.title("Model Diagnostics")

st.markdown("""
<div>
  <p style="font-size: 20px;">
    I trained a Logistic Regression model using economic indicators as model features as seen on the previous page:
      3-month moving average of unemployment, volatility, yield curve, etc. <br>
    As you can see below, the model doesn't do great, it often overpredicts. <br>
    I spent a while trying to correct this before realizing that this can <b> never be perfect. </b> <br>
    There are so many other factors that may lead to a recession that can't be predicted e.g. geopolitical turmoil. <br>
    That inspired me to create the Trends subpage, where you can take a look at some major political events. <br>
  </p>
</div>
""", unsafe_allow_html=True)

df = get_data()
model, X_test, y_test, y_pred, coef_df = train_model(df)

# --- Prediction Plot ---
st.subheader("Recession Predictions vs. Actual")
fig_pred = plot_predictions(X_test, y_test, y_pred)
st.pyplot(fig_pred)


# Divider
st.markdown("---")  # Optional visual break
st.markdown("<br>", unsafe_allow_html=True)



# --- Correlation Heatmap ---
st.subheader("Feature Correlation Heatmap")

st.markdown("""
<div>
  <p style="font-size: 20px;">
    In future developments, if I were to improve my model pipeline, I would use the following heatmap to derive correlations in features. <br>
    Ideally, should be uncorrelated as to provide distinct signals and reduce redundancy in the model. <br>
    As seen below, MA3 and UMCSENT have a correlation of almost 70%, which is not ideal, yet it is highly intuitive. <br>
  </p>
</div>
""", unsafe_allow_html=True)


corr = df.corr()
fig_corr, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig_corr)

