import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Config --------------------
st.set_page_config(page_title="Stock Volume Prediction App", layout="wide")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("📂 Navigate")
page = st.sidebar.radio("Go to", ["Top Traded Stocks", "User Input", "Prediction Output", "Feature Importance"])

# -------------------- Tab 1: Top Traded Stocks --------------------
if page == "Top Traded Stocks":
    st.title("📊 Stock Volume Prediction App")
    st.header("Top 3 Most Traded Stocks Over the Past Month")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    avg_volumes = []

    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            avg_volume = df['Volume'].mean()
            avg_volumes.append((ticker, avg_volume))

    avg_df = pd.DataFrame(avg_volumes, columns=["Ticker", "AvgVolume"])
    top3_tickers = avg_df.sort_values("AvgVolume", ascending=False).head(3)["Ticker"].tolist()

    volume_data = []
    for ticker in top3_tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            df = df[['Volume']].copy()
            df['Ticker'] = ticker
            df['Date'] = df.index
            volume_data.append(df)

    if volume_data:
        combined_df = pd.concat(volume_data, ignore_index=True)
        combined_df = combined_df.pivot(index='Date', columns='Ticker', values='Volume')
        st.line_chart(combined_df)
    else:
        st.warning("No volume data available.")

# -------------------- Tab 2: User Input --------------------
elif page == "User Input":
    st.header("Select Stock and Time Range")
    ticker = st.text_input("Enter stock ticker:", value="TSLA")
    start = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("today"))
    interval = st.selectbox("Interval", options=['1d', '1wk', '1mo'])

    if st.button("Load Stock Data"):
        user_data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if not user_data.empty:
            st.success("Data loaded successfully!")
            st.dataframe(user_data.tail())
            st.subheader("📈 Volume Chart")
            st.line_chart(user_data['Volume'])
            st.session_state['user_data'] = user_data
        else:
            st.error("No data found.")

# -------------------- Tab 3: Prediction Output --------------------
elif page == "Prediction Output":
    st.header("Prediction Model Output")
    if 'user_data' in st.session_state:
        user_data = st.session_state['user_data'].copy()
        user_data = user_data.dropna(subset=["Volume"])

        random_factors = np.random.uniform(low=0.95, high=1.05, size=user_data.shape[0])
        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        st.metric("📊 Predicted Volume for Next Period", f"{predicted.iloc[-1]:,.0f} shares")
    else:
        st.info("Please load a stock in 'User Input' tab.")

# -------------------- Tab 4: Feature Importance --------------------
elif page == "Feature Importance":
    st.header("Feature Importance (Simulated)")
    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)
    st.caption("Note: Replace with actual model-based feature importances when available.")
