import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------- App Config --------------------
st.set_page_config(page_title="Stock Volume App", layout="wide")

# -------------------- Sidebar Navigation --------------------
page = st.sidebar.radio("Select a tab", [
    "Top Traded Stocks",
    "User Input",
    "Prediction Output",
    "Feature Importance"
])

# -------------------- Top Traded Stocks --------------------
if page == "Top Traded Stocks":
    st.title("📊 Top 3 Most Traded Stocks - Last 30 Days")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    avg_volumes = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty and 'Volume' in df.columns:
            avg_volume = df['Volume'].mean()
            avg_volumes.append((ticker, avg_volume))

    avg_df = pd.DataFrame(avg_volumes, columns=["Ticker", "AvgVolume"])
    avg_df = avg_df.dropna(subset=["AvgVolume"])
    avg_df["AvgVolume"] = pd.to_numeric(avg_df["AvgVolume"], errors="coerce")
    top3 = avg_df.sort_values("AvgVolume", ascending=False).head(3)["Ticker"].tolist()

    volume_data = []
    for ticker in top3:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            df = df[['Volume']].copy()
            df['Date'] = df.index
            df['Ticker'] = ticker
            volume_data.append(df)

    if volume_data:
        combined = pd.concat(volume_data)
        pivoted = combined.pivot(index='Date', columns='Ticker', values='Volume')
        st.line_chart(pivoted)
    else:
        st.warning("No data available to display.")

# -------------------- User Input --------------------
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

# -------------------- Prediction Output --------------------
elif page == "Prediction Output":
    st.header("Prediction Model Output")

    if 'user_data' in locals() and not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])
        random_factors = pd.Series(np.random.uniform(0.95, 1.05, size=len(user_data)), index=user_data.index)
        predicted = user_data['Volume'].shift(1).bfill() * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Please load stock data in the User Input tab.")

# -------------------- Feature Importance --------------------
elif page == "Feature Importance":
    st.header("Feature Importance (Simulated)")
    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_imp = df_imp.sort_values("Importance")

    fig, ax = plt.subplots()
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)
    st.caption("Note: Replace with your trained model's feature importances.")
