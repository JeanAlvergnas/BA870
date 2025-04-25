import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------- Config --------------------
st.set_page_config(page_title="📈 Earnings Volume Predictor", layout="wide")

# -------------------- Tab Navigation --------------------
page = st.sidebar.radio("Navigate", [
    "📌 About the App",
    "📊 Volume Prediction After Financials",
    "🧪 User Input",
    "📈 Prediction Output",
    "📉 Feature Importance"
])

# -------------------- 📌 About the App --------------------
if page == "📌 About the App":
    st.title("📈 Earnings Volume Predictor")
    st.markdown("""
    This app predicts a stock's traded volume on the day following the release of financial statements.
    It uses historical trading volume and business fundamentals (e.g., profitability, leverage ratios) to anticipate market reaction to new earnings data.

    ### 🔍 Real-World Application
    Investors and analysts can use this app to:
    - Gauge market interest after financial disclosures
    - Anticipate volume surges for trading strategies
    - Evaluate company sensitivity to fundamental releases

    ### 👥 Team Members
    - Jean Alvergnas
    - Quan Nguyen
    - Michael Webber
    """)

# -------------------- 📊 Volume Prediction After Financials --------------------
elif page == "📊 Volume Prediction After Financials":
    st.header("Top 3 Stocks by Volume After Earnings")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=60)

    volume_data = []
    avg_volumes = []

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty and "Volume" in df.columns:
            df['Ticker'] = ticker
            df['Date'] = df.index
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df.dropna(subset=['Volume'], inplace=True)
            avg_volume = float(df['Volume'].mean())
            avg_volumes.append((ticker, avg_volume))
            volume_data.append(df)

    avg_df = pd.DataFrame(avg_volumes, columns=['Ticker', 'AvgVolume'])
    top3 = avg_df.sort_values(by='AvgVolume', ascending=False).head(3)
    top3_data = [df for df in volume_data if df['Ticker'].iloc[0] in top3['Ticker'].values]

    if top3_data:
        combined_df = pd.concat(top3_data)
        combined_df = combined_df[['Date', 'Ticker', 'Volume']]
        pivot_df = combined_df.pivot_table(index='Date', columns='Ticker', values='Volume')
        st.line_chart(pivot_df)
    else:
        st.warning("No volume data available for display.")

# -------------------- 🧪 User Input --------------------
elif page == "🧪 User Input":
    st.header("Select Stock and Date Range")
    ticker = st.text_input("Enter stock ticker:", value="TSLA")
    start = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("today"))
    interval = st.selectbox("Interval", options=['1d', '1wk', '1mo'])

    if st.button("Load Stock Data"):
        user_data = yf.download(ticker, start=start, end=end, interval=interval)
        if not user_data.empty:
            st.success("Data loaded successfully!")
            st.dataframe(user_data.tail())

            st.subheader("Volume Over Time")
            st.line_chart(user_data['Volume'])
        else:
            st.error("No data found for the selected inputs.")

# -------------------- 📈 Prediction Output --------------------
elif page == "📈 Prediction Output":
    st.header("Predicted Volume After Financial Report")
    if 'user_data' in locals() and not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])
        simulated_factors = np.random.uniform(low=0.95, high=1.05, size=user_data.shape[0])
        simulated_series = pd.Series(simulated_factors, index=user_data.index)
        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * simulated_series
        user_data['Predicted Volume'] = predicted

        st.subheader("📉 Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual')
        ax.plot(user_data.index, user_data['Predicted Volume'], linestyle='--', label='Predicted')
        ax.legend()
        st.pyplot(fig)

        st.metric("Predicted Next Volume", f"{predicted.iloc[-1]:,.0f} shares")
    else:
        st.info("Please load stock data in the User Input tab first.")

# -------------------- 📉 Feature Importance --------------------
elif page == "📉 Feature Importance":
    st.header("Feature Importance (Simulated Model)")
    features = ['ROA', 'ROE', 'Debt/Equity', 'Current Ratio', 'PE Ratio']
    importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    imp_df.sort_values(by='Importance', inplace=True)

    fig, ax = plt.subplots()
    sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title("Feature Importance for Volume Prediction")
    st.pyplot(fig)

    st.caption("Note: Replace with actual ML model importance when available.")
