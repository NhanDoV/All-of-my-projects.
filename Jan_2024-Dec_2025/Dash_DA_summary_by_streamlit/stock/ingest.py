import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Config
days = 60  # 2 months ~60 days
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    "vs_currency": "usd",
    "days": days,
    "interval": "daily"  # daily prices
}

# Fetch data
res = requests.get(url, params=params)
data = res.json()

# Extract dates & prices
prices = data["prices"]  # [[timestamp_ms, price], ...]
df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
df = df[["date", "price"]]  # Chỉ 2 cột cần

print(df.head())
df.to_csv("btc_2months_daily.csv", index=False)
print(f"Downloaded {len(df)} days of BTC data")