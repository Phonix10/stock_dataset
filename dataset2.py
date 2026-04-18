import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from textblob import TextBlob

# ---------------------------------
# CONFIG
# ---------------------------------

YEARS = 15
NEWS_PER_STOCK = 100 # number of news articles

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * YEARS)

# NIFTY 50
nifty50 = [
"RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","HINDUNILVR.NS",
"ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
"ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS",
"NESTLEIND.NS","BAJFINANCE.NS","BAJAJFINSV.NS","HCLTECH.NS","WIPRO.NS",
"ONGC.NS","NTPC.NS","POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS",
"COALINDIA.NS","JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","GRASIM.NS",
"TECHM.NS","INDUSINDBK.NS","DRREDDY.NS","CIPLA.NS","BRITANNIA.NS",
"EICHERMOT.NS","DIVISLAB.NS","HEROMOTOCO.NS","APOLLOHOSP.NS",
"TATACONSUM.NS","BAJAJ-AUTO.NS","BPCL.NS","UPL.NS","HDFCLIFE.NS",
"SBILIFE.NS","ICICIPRULI.NS","SHREECEM.NS","TATAMOTORS.NS"
]

os.makedirs("dataset", exist_ok=True)

# ---------------------------------
# NEWS SCRAPER (GOOGLE NEWS)
# ---------------------------------

def get_news(company):
    try:
        url = f"https://news.google.com/search?q={company}%20stock&hl=en-IN&gl=IN&ceid=IN:en"
        res = requests.get(url, timeout=10)

        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.select("article")[:NEWS_PER_STOCK]

        sentiments = []

        for art in articles:
            try:
                title = art.find("a").text
                score = TextBlob(title).sentiment.polarity
                sentiments.append(score)
            except:
                continue

        if len(sentiments) == 0:
            return 0, 0

        return np.mean(sentiments), len(sentiments)

    except:
        return 0, 0


# ---------------------------------
# MACRO DATA
# ---------------------------------

print("Downloading macro data...")

oil = yf.download("CL=F", start=start_date, end=end_date)
usd = yf.download("INR=X", start=start_date, end=end_date)
sp500 = yf.download("^GSPC", start=start_date, end=end_date)

for df_macro in [oil, usd, sp500]:
    df_macro.columns = df_macro.columns.get_level_values(0)

oil = oil[["Close"]].rename(columns={"Close": "OilPrice"})
usd = usd[["Close"]].rename(columns={"Close": "USDINR"})
sp500 = sp500[["Close"]].rename(columns={"Close": "SP500"})

oil.reset_index(inplace=True)
usd.reset_index(inplace=True)
sp500.reset_index(inplace=True)

# ---------------------------------
# MAIN LOOP
# ---------------------------------

for ticker in nifty50:

    print("\nProcessing:", ticker)

    try:
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            print("No data:", ticker)
            continue

        df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)

        # -------------------------
        # INDICATORS
        # -------------------------

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])

        df["OBV"] = obv

        # Fibonacci
        max_price = df["High"].max()
        min_price = df["Low"].min()
        diff = max_price - min_price

        df["Fib23"] = max_price - 0.236 * diff
        df["Fib38"] = max_price - 0.382 * diff
        df["Fib50"] = max_price - 0.5 * diff
        df["Fib61"] = max_price - 0.618 * diff

        # -------------------------
        # MERGE MACRO
        # -------------------------

        df = df.merge(oil, on="Date", how="left")
        df = df.merge(usd, on="Date", how="left")
        df = df.merge(sp500, on="Date", how="left")

        # -------------------------
        # NEWS (ONE FEATURE)
        # -------------------------

        company = ticker.replace(".NS", "")
        sentiment, news_count = get_news(company)

        df["NewsSentiment"] = sentiment
        df["NewsCount"] = news_count

        # -------------------------
        # FINAL CLEAN
        # -------------------------

        df.dropna(inplace=True)

        stock_name = ticker.replace(".NS", "")
        file_path = f"dataset/{stock_name}.csv"

        df.to_csv(file_path, index=False)

        print("Saved:", file_path)
        print("News Sentiment:", sentiment, "| Articles:", news_count)

    except Exception as e:
        print("Error:", ticker, e)

print("\n✅ DATASET READY")