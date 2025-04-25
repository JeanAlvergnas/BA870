import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- App Configuration --------------------
st.set_page_config(page_title="📈 Volume Predictor After Financials", layout="wide")

# -------------------- Navigation Tabs --------------------
page = st.sidebar.radio("Navigate", [
    "📘 About This App",
    "📊 Top Traded Stocks (1M)",
    "🧠 User Input",
    "🔮 Volume Prediction",
    "📈 Feature Importance"
])

# -------------------- Tab 1: About --------------------
if page == "📘 About This App":
    st.title("📈 Predicting Stock Volume After Financials")
    st.subheader("Team Members")
    st.markdown("""
    - Jean Alvergnas  
    - Quan Nguyen  
    - Michael Webber
    """)
    
    st.subheader("Purpose")
    st.markdown("""
    This app is designed to **predict the volume traded the day following a company's financial disclosures**, 
    using historical volume and financial ratios (such as profitability, leverage, and liquidity).

    It can assist investors and analysts by:
    - Providing insights into expected market reactions to earnings reports
    - Supporting pre-trade analysis for high-liquidity opportunities
    - Enabling comparison across firms and sectors
    """)

# -------------------- Tab 2: Top Traded Stocks (1 Month) --------------------
elif page == "📊 Top Traded Stocks (1M)":
    st.title("📊 Top 3 Traded Stocks in the Past Month")
    st.caption("This section displays the daily volume traded over the past 30 days for the top 3 stocks by volume.")

    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "JPM", "NFLX", "AMD"]

    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    avg_volumes = []
    volume_data = []

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if not df.empty and "Volume" in df.columns:
            df["Ticker"] = ticker
            df["Date"] = df.index
            avg_volumes.append((ticker, float(df["Volume"].mean())))
            volume_data.append(df[["Date", "Ticker", "Volume"]])

    avg_df = pd.DataFrame(avg_volumes, columns=["Ticker", "AvgVolume"])
    top3 = avg_df.sort_values(by="AvgVolume", ascending=False).head(3)

    top3_data = [df for df in volume_data if df["Ticker"].iloc[0] in top3["Ticker"].values]

    if top3_data:
        combined_df = pd.concat(top3_data)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df['Ticker'] = combined_df['Ticker'].astype(str)
        combined_df['Volume'] = pd.to_numeric(combined_df['Volume'], errors='coerce')
        combined_df.dropna(subset=['Volume'], inplace=True)

        if {'Date', 'Ticker', 'Volume'}.issubset(combined_df.columns):
            pivot_df = combined_df.pivot_table(index='Date', columns='Ticker', values='Volume')
            st.line_chart(pivot_df)
        else:
            st.warning("Missing columns in volume data.")
    else:
        st.warning("No volume data available.")

# -------------------- Tab 3: User Input --------------------
elif page == "🧠 User Input":
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

# -------------------- Tab 4: Volume Prediction --------------------
elif page == "🔮 Volume Prediction":
    st.header("Predicted Volume After Financials")
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
    else:
        st.info("Load data to see prediction output.")

# -------------------- Tab 5: Feature Importance --------------------
elif page == "📈 Feature Importance":
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
