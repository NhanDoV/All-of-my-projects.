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
newest_data = df.head(1)

# Load historical
historic_db = pd.read_csv("data/all_data.csv")
print("Historic cols:", historic_db.columns.tolist())
start_dt = historic_db['date'].min()
ended_dt = historic_db['date'].max()
print(start_dt, ended_dt)

def merge_btc_data(newest_df, historic_file="data/all_data.csv"):
    """
        Merge new Binance data với historical data.
        Logic: nếu schema giống + new data mới hơn → append, else return historic.
    """
    
    # Load historical data
    try:
        historic_df = pd.read_csv(historic_file)
        print(f"✅ Loaded historic: {len(historic_df)} rows từ {historic_df['date'].min()} → {historic_df['date'].max()}")
    except:
        print("❌ Không tìm thấy historic file, chỉ dùng new data")
        return newest_df
    
    # 1. Check schema compatibility
    required_cols = ['date', 'open', 'high', 'low', 'price', 'volume']
    new_cols = set(newest_df.columns)
    hist_cols = set(historic_df.columns)
    
    if new_cols != hist_cols or new_cols != set(required_cols):
        print("❌ Schema khác nhau!")
        print(f"New cols: {list(new_cols)}")
        print(f"Historic cols: {list(hist_cols)}")
        return historic_df
    
    # 2. Check date overlap & gap
    hist_max_date = pd.to_datetime(historic_df['date'].max()).date()
    new_min_date = newest_df['date'].min()
    new_max_date = newest_df['date'].max()
    
    print(f"Historic end: {hist_max_date}")
    print(f"New data: {new_min_date} → {new_max_date}")
    
    # 3. Merge logic
    if new_min_date > hist_max_date:
        # New data hoàn toàn mới → APPEND
        combined = pd.concat([
            historic_df, 
            newest_df
        ], ignore_index=True)
        combined = combined.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        
        print(f"✅ MERGED: {len(combined)} rows ({len(historic_df)} + {len(newest_df)})")

        return combined
    
    elif new_min_date <= hist_max_date < new_max_date:
        # Overlap một phần → UPDATE từ hist_max_date
        new_after_hist = newest_df[newest_df['date'] > hist_max_date]

        if len(new_after_hist) > 0:
            combined = pd.concat([
                historic_df, 
                new_after_hist
            ], ignore_index=True)
            combined = combined.sort_values('date').drop_duplicates('date').reset_index(drop=True)
            print(f"✅ PARTIAL UPDATE: +{len(new_after_hist)} new days")
            new_after_hist.to_csv("data/newest.csv")

        else:
            combined = historic_df
            print("ℹ️ No new data")

        return combined
    
    else:
        # New data cũ hơn hoặc trùng hết → keep historic
        print("ℹ️ New data cũ hơn historic, giữ nguyên historic")
        return historic_df

# SỬ DỤNG:
final_df = merge_btc_data(newest_data, "data/all_data.csv")

# Save kết quả
final_df.to_csv("data/all_data.csv", index=False)
print(f"💾 Saved {len(final_df)} rows: data/all_data.csv")
print(f"Date range: {final_df['date'].min()} → {final_df['date'].max()}")