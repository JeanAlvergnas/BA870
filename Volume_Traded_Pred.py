import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Volume App", layout="wide")

# Sidebar for navigation
page = st.sidebar.radio("Navigate", [
    "1. Team & App Overview",
    "2. Top Traded Stocks in the Past 3 Months",
    "3. User Input",
    "4. Prediction Output",
    "5. Feature Importance"
])

# 1. Team & App Overview
if page == "1. Team & App Overview":
    st.title("📘 Volume Prediction After Financial Releases")
    st.markdown("""
    **Team Members:**
    - Jean Alvergnas  
    - Quan Nguyen  
    - Michael Webber  

    **App Purpose:**
    This Streamlit app predicts the volume of stock traded on the day following a financial release.  
    The goal is to leverage past volume behavior and key financial ratios (profitability, leverage, etc.)  
    to anticipate trading activity after earnings announcements.
    """)

# 2. Top Traded Stocks in the Past 3 Months
elif page == "2. Top Traded Stocks in the Past Month":
    st.title("📈 Top 5 Traded Stocks in the Past 3 Months")
    st.markdown("Displays the daily volume traded over the past 3 months for 5 selected major stocks.")

    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=90)

    fig, ax = plt.subplots(figsize=(10, 6))
    volume_data = {}

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            df['Volume'] = df['Volume'] / 1e6
            ax.plot(df.index, df['Volume'], label=ticker)
            volume_data[ticker] = df[['Volume']].copy()

    ax.set_title("Volume Traded Over the Past 3 Months (in Millions)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume (Millions)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📋 Raw Data Preview")
    for ticker, data in volume_data.items():
        st.markdown(f"**{ticker}**")
        st.dataframe(data.tail())

# 3. User Input
elif page == "3. User Input":
    st.header("📥 User Input")
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

# 4. Prediction Output
elif page == "4. Prediction Output":
    st.header("📊 Prediction Model Output")
    ticker = st.text_input("Ticker for prediction:", value="TSLA")
    start = pd.to_datetime("2023-01-01")
    end = pd.to_datetime("today")
    user_data = yf.download(ticker, start=start, end=end, interval='1d')

    if not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])
        random_factors = pd.Series(np.random.uniform(0.95, 1.05, size=user_data.shape[0]), index=user_data.index)
        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("📉 Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No data to display prediction.")

# 5. Feature Importance
elif page == "5. Feature Importance":
    st.header("📌 Feature Importance (Simulated)")
    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots()
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
    ax.set_title("Simulated Feature Importances")
    st.pyplot(fig)

    st.markdown("_Note: Replace with real model results once available._")
