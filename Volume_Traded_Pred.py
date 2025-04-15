import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
import time

st.set_page_config(page_title="Stock Volume Prediction App", layout="wide")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", [
    "Top Traded Stocks",
    "User Input",
    "Prediction Output",
    "Model Explainability"
])

st.title("📊 Stock Volume Prediction App")

# -------------------- Tab 1: Top 3 Most Traded Stocks (Last Month, yfinance) --------------------
st.header("📊 1. Top 3 Most Traded Stocks Over the Past Month")

# Choose a list of popular stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']

# Function to get 1-month average volume using yfinance
@st.cache_data(ttl=3600)
def get_avg_volume_yf(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        return df['Volume'].mean()
    except Exception as e:
        st.warning(f"Error processing {ticker}: {e}")
        return None

# Get average volumes
avg_volumes = []
for tkr in tickers:
    vol = get_avg_volume_yf(tkr)
    if vol is not None:
        avg_volumes.append((tkr, vol))

# Get top 3 by average volume
top3_tickers = [t[0] for t in sorted(avg_volumes, key=lambda x: x[1], reverse=True)[:3]]

# Load volume data for top 3
volume_data = []
for ticker in top3_tickers:
    df = yf.download(ticker, period="1mo", interval="1d", progress=False)
    if not df.empty:
        df = df[['Volume']].copy()
        df['Ticker'] = ticker
        df['Date'] = df.index
        volume_data.append(df)

# Plot
if volume_data:
    combined_df = pd.concat(volume_data, ignore_index=True)
    combined_df = combined_df.pivot(index='Date', columns='Ticker', values='Volume')

    st.subheader("Top 3 Stocks by Average Daily Volume (Last Month)")
    st.line_chart(combined_df)
else:
    st.warning("No volume data available.")

# -------------------- Tab 2 --------------------
elif page == "User Input":
    st.header("Select Stock and Time Range")
    ticker = st.text_input("Enter stock ticker:", value="TSLA")
    start = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("today"))
    interval = st.selectbox("Interval", options=['1d', '1wk', '1mo'])

    if st.button("Load Stock Data"):
        user_data = yf.download(ticker, start=start, end=end, interval=interval)
        if not user_data.empty:
            st.success("Data loaded successfully!")
            st.dataframe(user_data.tail())

            st.subheader("📈 Volume Chart")
            st.line_chart(user_data['Volume'])
        else:
            st.error("No data found for the selected inputs.")

# -------------------- Tab 3 --------------------
elif page == "Prediction Output":
    st.header("Prediction Model Output")

    if 'user_data' in locals() and not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])
        random_factors = pd.Series(np.random.uniform(0.95, 1.05, len(user_data)), index=user_data.index)
        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Load data to see prediction output.")

# -------------------- Tab 4 --------------------
elif page == "Model Explainability":
    st.header("Feature Importance (Model Explainability)")

    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances_raw = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances_raw})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)

    st.caption("_Note: Replace simulated predictions and importances with real model outputs when ready._")
