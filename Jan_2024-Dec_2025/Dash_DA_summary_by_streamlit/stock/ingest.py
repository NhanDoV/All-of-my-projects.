import pandas as pd
import requests
from io import StringIO

# 1. Download từ URL
url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
response = requests.get(url)
data = response.text

# 2. Split lines & bỏ dòng 1 (CryptoDataDownload header)
lines = data.strip().split('\n')
lines = lines[1:]  # Bỏ dòng đầu

# 3. Join lại & đọc bằng pandas
csv_data = '\n'.join(lines)
df = pd.read_csv(StringIO(csv_data))

# 4. Clean data hoàn toàn
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# Convert dates
df['Unix'] = pd.to_datetime(df['Unix'], unit='ms', errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Select & rename columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume BTC']].copy()
df.columns = ['date', 'open', 'high', 'low', 'price', 'volume']
df['date'] = df['date'].dt.date

# Clean & sort
df = df.dropna(subset=['date', 'price'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)

# 5. Stats
print(f"\n✅ Clean data: {len(df)} ngày")
print(f"From: {df['date'].min().strftime('%Y-%m-%d')}")
print(f"To: {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print("\nHead 5 rows:")
print(df.head())

# 6. Saved data
df.to_csv("data/all_data.csv", index = False)
print("Crawl all-data successfully!!!")