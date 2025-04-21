import streamlit as st
from model_recession import get_data, train_model, plot_predictions
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Model Diagnostics")

df = get_data()
model, X_test, y_test, y_pred, coef_df = train_model(df)

# --- Prediction Plot ---
st.subheader("📈 Recession Predictions vs. Actual")
fig_pred = plot_predictions(X_test, y_test, y_pred)
st.pyplot(fig_pred)

# --- Correlation Heatmap ---
st.subheader("🔍 Feature Correlation Heatmap")
corr = df.corr()
fig_corr, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig_corr)

