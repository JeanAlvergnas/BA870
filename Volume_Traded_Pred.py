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

# -------------------- Tab 1 --------------------
if page == "Top Traded Stocks":
    st.header("Top 3 Most Traded Stocks Over the Past Month")

    API_KEY = "CXW22KLIBXMMW6KU"
    ALPHA_URL = "https://www.alphavantage.co/query"
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']

    def get_avg_volume(ticker):
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": API_KEY
        }
        response = requests.get(ALPHA_URL, params=params)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            st.warning(f"Error processing {ticker}: {data.get('Note') or data.get('Information') or 'Unexpected format'}")
            return None

        try:
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            last_30 = df[df.index >= pd.to_datetime("today") - pd.Timedelta(days=30)]
            return last_30['6. volume'].mean()
        except Exception as e:
            st.warning(f"Error parsing data for {ticker}: {e}")
            return None

    avg_volumes = []
    for i, tkr in enumerate(tickers):
        vol = get_avg_volume(tkr)
        if vol is not None:
            avg_volumes.append((tkr, vol))
        time.sleep(12)

    top3_tickers = [t[0] for t in sorted(avg_volumes, key=lambda x: x[1], reverse=True)[:3]]

    volume_data = []
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    for ticker in top3_tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty and 'Volume' in df.columns:
            df = df[['Volume']].copy()
            df['Ticker'] = ticker
            df['Date'] = df.index
            volume_data.append(df)

    if volume_data:
        combined_df = pd.concat(volume_data, ignore_index=True)
        combined_df = combined_df.pivot(index='Date', columns='Ticker', values='Volume')

        st.subheader("Top 3 Stocks by Average Daily Volume (Last Month)")
        st.line_chart(combined_df)
    else:
        st.warning("No volume data available to display.")

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
