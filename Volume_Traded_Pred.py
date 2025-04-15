# Volume_Traded_Pred.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 Stock Volume Prediction App", layout="wide")

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Top 3 Traded", "User Input", "Prediction", "Feature Importance"])

# Common config
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']
end_date = pd.to_datetime("today")
start_date = end_date - pd.Timedelta(days=30)

# -------------------- TAB 1 --------------------
if page == "Top 3 Traded":
    st.title("📈 Top 3 Most Traded Stocks (Past Month)")
    avg_volumes = []
    volume_data = []

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty and "Volume" in df.columns:
            df = df.dropna(subset=["Volume"])
            if not df.empty:
                try:
                    avg_volume = float(df["Volume"].mean())
                    avg_volumes.append((ticker, avg_volume))
                    df["Ticker"] = ticker
                    df["Date"] = df.index
                    volume_data.append(df[["Date", "Ticker", "Volume"]])
                except Exception as e:
                    st.warning(f"Skipping {ticker}: {e}")

    if avg_volumes:
        avg_df = pd.DataFrame(avg_volumes, columns=["Ticker", "AvgVolume"]).dropna()
        top3 = avg_df.sort_values(by="AvgVolume", ascending=False).head(3)
        top3_data = [df for df in volume_data if df["Ticker"].iloc[0] in top3["Ticker"].values]

        if top3_data:
            combined_df = pd.concat(top3_data, ignore_index=True)
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            combined_df['Ticker'] = combined_df['Ticker'].astype(str)
            combined_df['Volume'] = pd.to_numeric(combined_df['Volume'], errors='coerce')
            combined_df.dropna(subset=['Volume'], inplace=True)

            if {'Date', 'Ticker', 'Volume'}.issubset(combined_df.columns):
                pivot_df = combined_df.pivot_table(index='Date', columns='Ticker', values='Volume')
                st.line_chart(pivot_df)
            else:
                st.error("Pivot failed due to missing columns.")
        else:
            st.warning("No top 3 data available.")
    else:
        st.warning("Could not retrieve average volumes.")

# -------------------- TAB 2 --------------------
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
            st.subheader("📉 Volume Chart")
            st.line_chart(user_data['Volume'])
        else:
            st.error("No data found.")

# -------------------- TAB 3 --------------------
elif page == "Prediction":
    st.header("Prediction Model Output")

    if "user_data" in locals() and not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])
        random_factors = pd.Series(np.random.uniform(0.95, 1.05, size=user_data.shape[0]), index=user_data.index)
        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("📊 Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Please load stock data from the 'User Input' tab.")

# -------------------- TAB 4 --------------------
elif page == "Feature Importance":
    st.header("Feature Importance (Simulated Model Output)")
    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)

    st.markdown("_Note: This is placeholder data. Replace with actual model feature importances._")
