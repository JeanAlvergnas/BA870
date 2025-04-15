import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stock Volume Prediction App", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", [
    "Top Traded Stocks", 
    "User Input", 
    "Prediction Output", 
    "Feature Importance"
])

# -------------------- Tab 1: Top Traded Stocks --------------------
if page == "Top Traded Stocks":
    st.title("📊 Top 3 Most Traded Stocks (Past Month)")

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "NFLX", "AMD"]

    avg_volumes = []
    for ticker in tickers:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if not df.empty:
            avg_volume = df['Volume'].mean()
            avg_volumes.append((ticker, avg_volume))

    avg_df = pd.DataFrame(avg_volumes, columns=["Ticker", "AvgVolume"])
    avg_df["AvgVolume"] = pd.to_numeric(avg_df["AvgVolume"], errors="coerce")
    avg_df.dropna(subset=["AvgVolume"], inplace=True)
    top3_tickers = avg_df.sort_values("AvgVolume", ascending=False).head(3)["Ticker"].tolist()

    volume_data = []
    for ticker in top3_tickers:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if not df.empty and "Volume" in df.columns:
            df = df[["Volume"]].copy()
            df["Ticker"] = ticker
            df["Date"] = df.index
            volume_data.append(df)

    if volume_data:
        combined_df = pd.concat(volume_data, ignore_index=True)
        combined_df["Date"] = pd.to_datetime(combined_df["Date"])
        combined_df = combined_df.pivot(index="Date", columns="Ticker", values="Volume")

        st.line_chart(combined_df)
    else:
        st.warning("No volume data available to display.")

# -------------------- Tab 2: User Input --------------------
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

# -------------------- Tab 3: Prediction Output --------------------
elif page == "Prediction Output":
    st.header("Prediction Model Output")

    if 'user_data' in locals() and not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])

        random_factors_raw = np.random.uniform(low=0.95, high=1.05, size=user_data.shape[0])
        random_factors = pd.Series(random_factors_raw, index=user_data.index)

        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        st.metric(
            label="📊 Predicted Volume for Next Period",
            value=f"{user_data['Predicted Volume'].iloc[-1]:,.0f} shares"
        )
        st.caption("Note: This is a placeholder until your ML model is connected.")
    else:
        st.warning("Please load a stock in 'User Input' tab to enable prediction.")

# -------------------- Tab 4: Feature Importance --------------------
elif page == "Feature Importance":
    st.header("Feature Importance (Model Explainability)")

    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances_raw = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances_raw})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)

    st.markdown("_Note: Replace simulated predictions and importances with real model outputs when ready._")


