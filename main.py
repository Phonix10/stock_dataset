import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

TICKER = "HDFCBANK.NS"
YEARS = 5

end_date = datetime.today()
start_date = end_date - timedelta(days=365*YEARS)

print("Downloading stock data...")

df = yf.download(TICKER, start=start_date, end=end_date)
df.columns = df.columns.get_level_values(0)
df.reset_index(inplace=True)

# -------------------
# TECHNICAL INDICATORS
# -------------------

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

# -------------------
# FIBONACCI
# -------------------

max_price = df["High"].max()
min_price = df["Low"].min()

diff = max_price - min_price

df["Fib23"] = max_price - 0.236 * diff
df["Fib38"] = max_price - 0.382 * diff
df["Fib50"] = max_price - 0.5 * diff
df["Fib61"] = max_price - 0.618 * diff

# -------------------
# MACRO DATA
# -------------------

print("Downloading macro indicators...")

oil = yf.download("CL=F", start=start_date, end=end_date)
usd = yf.download("INR=X", start=start_date, end=end_date)
sp500 = yf.download("^GSPC", start=start_date, end=end_date)

oil.columns = oil.columns.get_level_values(0)
usd.columns = usd.columns.get_level_values(0)
sp500.columns = sp500.columns.get_level_values(0)

oil = oil[["Close"]].rename(columns={"Close":"OilPrice"})
usd = usd[["Close"]].rename(columns={"Close":"USDINR"})
sp500 = sp500[["Close"]].rename(columns={"Close":"SP500"})

oil.reset_index(inplace=True)
usd.reset_index(inplace=True)
sp500.reset_index(inplace=True)

df = df.merge(oil,on="Date",how="left")
df = df.merge(usd,on="Date",how="left")
df = df.merge(sp500,on="Date",how="left")

# -------------------
# GDELT MONTHLY TONE
# -------------------

print("Fetching monthly GDELT tone...")

def get_gdelt_tone(start,end):

    try:

        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?"
            f"query=India%20economy&mode=ArtList&maxrecords=50&format=json"
            f"&startdatetime={start}000000"
            f"&enddatetime={end}235959"
        )

        r = requests.get(url,timeout=10)
        data = r.json()

        articles = data.get("articles",[])

        tones = []

        for a in articles:
            tone = a.get("tone")
            if tone:
                tones.append(float(tone))

        if len(tones)==0:
            return 0

        return np.mean(tones)

    except:
        return 0


df["YearMonth"] = df["Date"].dt.to_period("M")

tone_map = {}

for month in df["YearMonth"].unique():

    start = str(month.start_time.strftime("%Y%m%d"))
    end = str(month.end_time.strftime("%Y%m%d"))

    tone = get_gdelt_tone(start,end)

    tone_map[month] = tone

df["GlobalNewsTone"] = df["YearMonth"].map(tone_map)

df.drop(columns=["YearMonth"],inplace=True)

# -------------------
# SAVE DATASET
# -------------------

file = "HDFC_5year_research_dataset.xlsx"

df.to_excel(file,index=False)

print("Dataset created")
print("Rows:",len(df))
print("Saved:",file)