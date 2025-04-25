import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Volume App", layout="wide")

# Sidebar for navigation
page = st.sidebar.radio("Navigate", [
    "1. Team & App Overview",
    "2. Volume Prediction After Financials",
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

# 2. Volume Prediction After Financials
elif page == "Volume Prediction After Financials":
    st.title("📊 Volume Prediction After Financials")
    st.markdown("This section displays volume trends for top traded stocks to illustrate the app’s intended future predictive use after financial releases.")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    avg_volumes = []
    volume_data = []

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            if 'Volume' in df.columns:
                avg_volume = float(df['Volume'].mean())
                avg_volumes.append((ticker, avg_volume))
                df = df[['Volume']].copy()
                df['Date'] = df.index
                df['Ticker'] = ticker
                volume_data.append(df)
            else:
                st.warning(f"Volume column missing for {ticker}")
        else:
            st.warning(f"No data available for {ticker}")

    if avg_volumes:
        avg_df = pd.DataFrame(avg_volumes, columns=['Ticker', 'AvgVolume'])
        top3 = avg_df.sort_values(by='AvgVolume', ascending=False).head(3)['Ticker'].tolist()

        top3_data = [df for df in volume_data if df['Ticker'].iloc[0] in top3]

        if top3_data:
            combined_df = pd.concat(top3_data)
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            combined_df['Ticker'] = combined_df['Ticker'].astype(str)

            if 'Volume' in combined_df.columns:
                combined_df['Volume'] = pd.to_numeric(combined_df['Volume'], errors='coerce')
                combined_df.dropna(subset=['Volume'], inplace=True)

                if {'Date', 'Ticker', 'Volume'}.issubset(combined_df.columns):
                    pivot_df = combined_df.pivot(index='Date', columns='Ticker', values='Volume')
                    st.subheader("Daily Volume for Top 3 Stocks (Past Month)")
                    st.line_chart(pivot_df)
                else:
                    st.warning("Missing necessary columns after cleaning.")
            else:
                st.warning("Volume column missing after processing.")
        else:
            st.warning("No valid top 3 stock data available.")
    else:
        st.warning("No average volume data could be calculated.")

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
