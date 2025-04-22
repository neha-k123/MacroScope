# recession_modeling.py

import pandas as pd
import yfinance as yf
import seaborn as sns

import numpy as np
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import streamlit as st


# === SETUP ===
fred = Fred(api_key='b80e439e493aa86fbb9f080c3c2600f9')


# 4.22.2025: Caching the model, otherwise the page takes too long to open up
@st.cache_data(show_spinner="Training model...")
def get_cached_model():
    df = get_data()
    model, X_test, y_test, y_pred, coef_df = train_model(df)
    return model, X_test, y_test, y_pred, coef_df, df

# === FETCH DATA ===
def get_data():
    # Yield Curve
    yield_curve = fred.get_series('T10Y2Y').to_frame('T10Y2Y')
    yield_curve.index = pd.to_datetime(yield_curve.index)

    # Unemployment
    unemp = fred.get_series('UNRATE').to_frame('Unemployment')
    unemp.index = pd.to_datetime(unemp.index)
    unemp['MA3'] = unemp['Unemployment'].rolling(3).mean()
    unemp = unemp.drop(columns=["Unemployment"])


    # Consumer Sentiment
    umcsent = fred.get_series('UMCSENT').to_frame('UMCSENT')
    umcsent.index = pd.to_datetime(umcsent.index)

    # Jobless Claims
    icsa = fred.get_series('ICSA').to_frame('ICSA')
    icsa.index = pd.to_datetime(icsa.index)

    # PCE (Personal Consumption Expenditures)
    pce = fred.get_series('PCE').to_frame('PCE')
    pce.index = pd.to_datetime(pce.index)
    pce['PCE_Change'] = pce['PCE'].pct_change()

    df = yield_curve.join(umcsent, how='outer')
    df = df.join(unemp, how='outer')
    df = df.join(icsa, how='outer')
    df = df.join(pce['PCE_Change'], how='outer')
    df = df.resample('ME').last().ffill()

    # Download SPY daily data
    spy = yf.download("SPY", start="1980-01-01", interval="1d")["Close"]
    print("Downloaded SPY shape:", spy.shape)
    print("First few SPY dates:", spy.index[:5])
    spy.index = pd.to_datetime(spy.index)



    # Resample to month-end to match df (this will now match df.index properly)
    spy_monthly = spy.resample("ME").last()
    spy_returns = spy_monthly.pct_change(3)
    spy_returns.name = "SP500_3M_Return"


    df['SP500_3M_Return'] = spy_returns

    # === VIX (Volatility Index) ===
    vix = yf.download("^VIX", start="1990-01-01", interval="1d")["Close"]
    vix.index = pd.to_datetime(vix.index)
    vix_monthly = vix.resample("M").last()
    vix_monthly.name = "VIX"

    df['VIX']= vix_monthly

    # Recession Target
    recession = fred.get_series("USREC").astype(int)
    recession.index = pd.to_datetime(recession.index)
    recession = recession.resample("M").mean()  # Match df's monthly frequency
    df['Recession'] = recession.shift(-3).rolling(3).max()

    #drop the last 6 months since we have no prediction
    df = df.iloc[:-6] 

    # Remove outliers (clip at 1st and 99th percentile)
    for col in df.columns:
        if col != 'Recession':
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    print("Features are ready")
    important_cols = ['T10Y2Y', 'MA3', 'UMCSENT', 'ICSA', 'PCE_Change', 'SP500_3M_Return', 'Recession', 'VIX']

    missing_report = df[important_cols].isna().mean().sort_values(ascending=False)
    print("\nMissing percentage per column:")
    print(missing_report)

    # Drop rows with any NaNs in key columns
    df_clean = df.dropna(subset=important_cols)
    print(f"\nFinal shape after dropna: {df_clean.shape}")
    df = df['1980':'2025']


    # Compute correlation matrix on cleaned df
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr = df[numeric_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


    return df_clean

# === TRAIN MODEL ===
def train_model(df):
    print("Final data shape:", df.shape)

    X = df.drop(columns=['Recession'])
    y = df['Recession']

    print(f"X shape before scaling: {X.shape}")
    print(f"Any NaNs? {X.isna().sum().sum()}")

    # Normalize features
    scaler = StandardScaler() 

    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    logits = model.decision_function(X_test)
    logits_noisy = logits + np.random.normal(0, 0.1, size=logits.shape)

    # Convert to probability
    probs = 1 / (1 + np.exp(-logits_noisy))
    y_pred_noisy = (probs >= 0.5).astype(int)


    #y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred_noisy)
    f1 = f1_score(y_test, y_pred_noisy)

    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Display model coefficients
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))

    return model, X_test, y_test, y_pred_noisy, coef_df

# === PLOT ===
def plot_predictions(X_test, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(X_test.index, y_pred, label='Predicted Recession', color='red')
    ax.plot(X_test.index, y_test.values, label='Actual Recession', color='black', linewidth=3, alpha=0.5)
    ax.set_title("Recession Prediction vs. Actual")
    ax.legend()
    plt.tight_layout()
    return fig




# === MAIN ===
if __name__ == '__main__':
    df = get_data()
    model, X_test, y_test, y_pred = train_model(df)
    plot_predictions(X_test, y_test, y_pred)
