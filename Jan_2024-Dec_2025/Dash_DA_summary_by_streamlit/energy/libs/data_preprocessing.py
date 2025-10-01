# Basic libraries
import os
import calendar
from datetime import timedelta
import pandas as pd
import streamlit as st
from pathlib import Path

# For time-series analysis
from statsmodels.tsa.seasonal import seasonal_decompose

# For web-crawler
from app_store_web_scraper import AppStoreEntry
from google_play_scraper import Sort, reviews, reviews_all, app

# Plot & UI
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go

colors = {
    "Consumption（kW）": "#1f77b4",  # blue
    "Production（kW）": "#2ca02c",   # green
    "Purchasing（kW）": "#d62728",   # red
}

app_store_id_BITCH_EVN = 1408655940
android_id_BITCH_EVN = "vn.evnspc.cskh.cskhevnspc.CSKHEVNSPC"

class crawl_EVN_the_BITCH:
    def crawl_all_data_appstore(self):
        df_all_ios = pd.DataFrame({})
        for country in ['vn', 'us', 'de', 'fr']:
            app_ios = AppStoreEntry(app_id = app_store_id_BITCH_EVN, country = country)
            df_ios = pd.DataFrame(app_ios.reviews())
            df_ios['country'] = country
            df_all_ios = pd.concat([df_all_ios, df_ios]).reset_index(drop=True)  

        return df_all_ios

    def crawl_all_data_android(self):
        df_all_adr = pd.DataFrame({})
        for lang, country in [('vi', 'vn'), ('en', 'us')]:
            res = reviews(app_id = android_id_BITCH_EVN, sort=Sort.NEWEST,
                        lang=lang, country=country
            )
            df_android = pd.DataFrame(res[0])
            df_android['location'] = f"{lang}-{country}"
            df_all_adr = pd.concat([df_all_adr, df_android]).reset_index(drop=True)
        return df_all_adr

    def save_sentiment_data(self, df, name: str, folder: str = "sentiment_data"):
        os.makedirs(folder, exist_ok=True)

        # Drop timezone info nếu có
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_localize(None) 
            
        file_path = os.path.join(folder, f"{name}.xlsx")            
        df.to_excel(file_path, index=False, engine="openpyxl")
        return file_path
        
bitch_EVN_as_JSon = {
    "2025" : {
        "May": {
            "0-50 kWh": 1984,
            "51-100 kWh": 2050,
            "101-200 kWh": 2380,
            "201-300 kWh": 2998,
            "301-400 kWh": 3350,
            "> 401 kWh" : 3460
        }
    },
    "2024" : {
        "Jan": {
            "0-50 kWh": 1806,
            "51-100 kWh": 1866,
            "101-200 kWh": 2167,
            "201-300 kWh": 2729,
            "301-400 kWh": 3050,
            "> 401 kWh" : 3151                        
        },
        "May": {
            "0-50 kWh": 1858,
            "51-100 kWh": 1919,
            "101-200 kWh": 2227,
            "201-300 kWh": 2805,
            "301-400 kWh": 3136,
            "> 401 kWh" : 3238
        }
    },  
    "2023" : {
        "May": {
            "0-50 kWh": 1695,
            "51-100 kWh": 1752,
            "101-200 kWh": 2034,
            "201-300 kWh": 2561,
            "301-400 kWh": 2863,
            "> 401 kWh" : 2956
        }
    },                
    "2021" : {
        "Jan": {
            "0-50 kWh": 1549,
            "51-100 kWh": 1600,
            "101-200 kWh": 1858,
            "201-300 kWh": 2340,
            "301-400 kWh": 2615,
            "> 401 kWh" : 2701
        },
        "Jun": {
            "0-50 kWh": 1678,
            "51-100 kWh": 1734,
            "101-200 kWh": 2014,
            "201-300 kWh": 2536,
            "301-400 kWh": 2834,
            "> 401 kWh" : 2927
        }
    },
    "2019": {
        "Jan": {
            "0-50 kWh": 1646,
            "51-100 kWh": 1701,
            "101-200 kWh": 1976,
            "201-300 kWh": 2487,
            "301-400 kWh": 2780,
            "> 401 kWh" : 2871
        }                    
    },
    "2017": {
        "Jan": {
            "0-50 kWh": 1583,
            "51-100 kWh": 1635,
            "101-200 kWh": 1893,
            "201-300 kWh": 2374,
            "301-400 kWh": 2650,
            "> 401 kWh" : 2736
        }                    
    },
    "2015": {
        "Jan": {
            "0-50 kWh": 1484,
            "51-100 kWh": 1533,
            "101-200 kWh": 1786,
            "201-300 kWh": 2242,
            "301-400 kWh": 2503,
            "> 401 kWh" : 2587
        }                    
    }
} 

class datetime_df_processing:

    def get_list_of_compared_hhmm(self, df: pd.DataFrame) -> list:
        last_ts = df["Time"].max()
        # normalize to today's date but keep HH:MM
        reference_ls = pd.date_range(
            start=last_ts.replace(hour=0, minute=0, second=0, microsecond=0),
            end=last_ts,
            freq="5min"   # 5 minutes
        ).strftime("%H:%M").tolist()
        
        reference_ls = reference_ls[::-1]
        return reference_ls

    def get_nday_month_year(self, month: int, year: int) -> int:
        return calendar.monthrange(year, month)[1]
    
    def get_today_values(self):
        return pd.Timestamp.today()

    def get_this_month_df(self, df: pd.DataFrame) -> pd.DataFrame:
        today = self.get_today_values()
        this_month = today.month
        this_year = today.year

        # Filter rows where both year & month match today
        filtered_df = df[
            (df['Time'].dt.month == this_month) &
            (df['Time'].dt.year == this_year)
        ]
        if filtered_df.empty:
            st.warning(f"No records found for {this_month}/{this_year}")
            return pd.DataFrame()  # return empty DataFrame
        else:
            return filtered_df

    def get_firstday_this_month_df(self, df: pd.DataFrame) -> pd.Timestamp | None:
        extract_df = self.get_this_month_df(df)

        if extract_df.empty:
            return None  # Không có dữ liệu tháng này
        else:
            return extract_df['Time'].min().normalize()
    
    def get_last_timestamps_in_df(self, df: pd.DataFrame, input_ts: str) -> pd.DataFrame:
        """
            Compute stats up to a given cutoff time (input_ts = "HH:MM").
            Returns: DataFrame with today total, weekly avg, monthly avg.
        """
        # Ensure datetime
        df["Time"] = pd.to_datetime(df["Time"])
        latest_ts = df["Time"].max()

        # Parse input_ts into a cutoff time
        cutoff_time = pd.to_datetime(input_ts, format="%H:%M").time()
        numeric_cols = ["Consumption（kW）", "Production（kW）", "Purchasing（kW）"]

        # ---------------- Today ----------------
        df_today = df[df["Time"].dt.date == latest_ts.date()]
        df_today = df_today[df_today["Time"].dt.time <= cutoff_time]
        today_sum = df_today[numeric_cols].sum()

        # ---------------- This Week ----------------
        week_start = latest_ts - pd.to_timedelta(latest_ts.weekday(), unit="D")
        df_week = df[(df["Time"].dt.date >= week_start.date()) &
                     (df["Time"].dt.date <= latest_ts.date())]
        df_week = df_week[df_week["Time"].dt.time <= cutoff_time]
        week_avg = df_week.groupby(df_week["Time"].dt.date)[numeric_cols].sum().mean()

        # ---------------- This Month ----------------
        month_start = latest_ts.replace(day=1)
        df_month = df[(df["Time"].dt.date >= month_start.date()) &
                      (df["Time"].dt.date <= latest_ts.date())]
        df_month = df_month[df_month["Time"].dt.time <= cutoff_time]
        month_avg = df_month.groupby(df_month["Time"].dt.date)[numeric_cols].sum().mean()

        # ---------------- All time ----------------
        df_all = df[df["Time"] >= '2025-09-18']
        df_all = df_all[df_all["Time"].dt.time <= cutoff_time]
        all_avg = df_all.groupby(df_all["Time"].dt.date)[numeric_cols].sum().mean()

        # ---------------- Combine ----------------
        result = pd.DataFrame({
            "Today": today_sum,
            "AVG (This week)": week_avg,
            "AVG (This month)": month_avg,
            "AVG (all-time)": all_avg
        })
        return result / 12

    def get_df_which_enough_24hours_per_day(self, df: pd.DataFrame, flag: bool) -> pd.DataFrame:
        """
            Keep only days that have full 24 hours of data (288 rows for 5-min interval).
            If flag=False, return df unchanged.
        """
        # Count number of records per day
        df['Date'] = df['Time'].dt.date
        if not flag:
            return df

        counts = df.groupby('Date').size()

        # Get only valid days (288 records)
        valid_dates = counts[counts == 288].index

        # Filter df
        filtered_df = df[df['Date'].isin(valid_dates)].copy()
        return filtered_df

    def get_ndays(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        first_date = df["Time"].min().normalize()
        last_date = df["Time"].max().normalize()

        # +1 because if first_date == last_date we still count 1 day
        return (last_date - first_date).days + 1

    def get_ndays_this_month(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        # Lấy min & max timestamp trong dataset
        first_ts = df["Time"].min()
        last_ts = df["Time"].max()

        # Chuyển first_ts thành ngày đầu tiên của tháng đó
        first_date = first_ts.replace(day=1).normalize()
        last_date = last_ts.normalize()

        # +1 vì nếu cùng ngày thì vẫn tính là 1
        return (last_date - first_date).days + 1
    
    def get_day_part(self, hour):
        if 0 <= hour < 8:
            return "Off-peak"
        elif 8 <= hour < 16:
            return "Mid-load"
        else:  # 16–24
            return "Peak-load"

    def get_season(self, month: int) -> str:
        """
            Return season based on month (April–October = Rainy, else Sunny).
        """
        if 4 <= month <= 10:
            return "Rainy"
        else:
            return "Sunny"

    def add_time_feature(self, df: pd.DataFrame, flag_spec: str) -> pd.DataFrame:
        """
        Add new column based on flag_spec.
        - "day-part": classify by hour
        - "season": classify by month
        """
        numeric_cols = ["Production（kW）", "Consumption（kW）", "Purchasing（kW）"]
        df["Purchasing（kW）"] = abs(df["Purchasing（kW）"])
        if flag_spec == "day-part":
            df["day-part"] = df["Time"].dt.hour.map(self.get_day_part)
            df = df.groupby('day-part')[numeric_cols].sum()
        elif flag_spec == "season":
            df["season"] = df["Time"].dt.month.map(self.get_season)
            df = df.groupby('season')[numeric_cols].sum()
        elif flag_spec == "weekend": 
            df["Date"] = df["Time"].dt.date
            df = df.groupby('Date')[numeric_cols].sum().reset_index()
            df["weekend"] = pd.to_datetime(df["Date"]).dt.day_name().isin(["Saturday", "Sunday"])
            df["weekend"] = df["weekend"].map({True: "weekend", False: "weekday"})
            df = df.groupby('weekend')[numeric_cols].mean()
        else:
            raise ValueError(f"Unknown flag_spec: {flag_spec}")
        return df 

    def get_valid_weekyear(self, df: pd.DataFrame) -> list:
        df['Week'] = df['Time'].dt.isocalendar().week.apply(lambda x: f"W{x}") # e.g 2025-W20
        df['Year'] = df['Time'].dt.year.apply(lambda x: str(x))
        df['WeekYear'] = df['Year'] + '-' + df['Week']
        valid_weekyear = df['WeekYear'].unique().tolist()
        
        return df, valid_weekyear

    def get_valid_monthyear(self, df: pd.DataFrame) -> list:
        df['Month'] = df['Time'].dt.month_name().apply(lambda x: x[:3])
        df['Year'] = df['Time'].dt.year.apply(lambda x: str(x))
        df['MonthYear'] = df['Month'] + '-' + df['Year']
        valid_monthyear = df['MonthYear'].unique().tolist()
        
        return df, valid_monthyear

    def get_df_by_freq(self, df: pd.DataFrame, col: str, freq: str) -> pd.DataFrame:
        if freq == "Hourly":
            freq_df = df.set_index("Time")[col].resample("h").sum() / 12
        elif freq == "Daily":
            freq_df = df.set_index("Time")[col].resample("D").sum() / 12
        
        return freq_df

    def shift_to_lookback_ts(self, df: pd.DataFrame, lookback_ndays: int) -> pd.DataFrame:
        """
            Keep only records starting from today - lookback_ndays.
            Drops the first incomplete day (hardcoded as 2025-09-18 for now).
        """
        # Ensure datetime
        df["Time"] = pd.to_datetime(df["Time"])

        # Drop first incomplete day
        df = df[df["Time"] >= "2025-09-18"]

        # Compute starting timestamp
        today_values = self.get_today_values().normalize()  # midnight today
        first_time_started = today_values - timedelta(days = lookback_ndays - 1)

        # Filter
        df = df[df["Time"] > first_time_started]

        return df

class EVN_the_BITCH:
    def get_ls_years(self, elec_dict: dict) -> list:
        """Return list of available years in tariff dictionary."""
        return list(elec_dict.keys())

    def get_period_ls(self, elec_dict: dict, year: str) -> list:
        """Return list of available months in a given year."""
        return list(elec_dict[year].keys())

    def get_corresponding_tiers(self, consumption: float, year: str, month: str, tariff_dict: dict) -> list:
        """
            Parse tier strings into [(low, high), price] ranges.
            Example: "0-50 kWh" → (0, 50), "> 401 kWh" → (401, inf).
        """
        tiers = tariff_dict[year][month]
        parsed_tiers = []
        for k, v in tiers.items():
            if ">" in k:  # open-ended
                lower = int(k.split(">")[1].split()[0])
                parsed_tiers.append(((lower, float("inf")), v))
            else:
                low, high = map(int, k.split()[0].split("-"))
                parsed_tiers.append(((low, high), v))
        return sorted(parsed_tiers, key=lambda x: x[0][0])

    def streamlit_get_interval(self, consumption: int, year: str, month: str, tariff_dict: dict) -> tuple:
        """Return the (low, high) interval where given consumption belongs."""
        for (low, high), _ in self.get_corresponding_tiers(consumption, year, month, tariff_dict):
            if low <= consumption <= high or (high == float("inf") and consumption >= low):
                return (low, high)

    def convert_consume(self, consumption: float, year: str, month: str, tariff_dict: dict) -> int:
        """Compute electricity bill based on tiered pricing."""
        parsed_tiers = self.get_corresponding_tiers(consumption, year, month, tariff_dict)
        price = 0
        remaining = consumption
        for (low, high), rate in parsed_tiers:
            if remaining <= 0:
                break
            if high == float("inf"):
                units = remaining
            else:
                units = min(remaining, high - low + 1)  # adjust if half-open
            price += units * rate
            remaining -= units
        return price 

    def get_corresponding_consumptions(self, money: float, year: str, month: str, tariff_dict: dict) -> float:
        """
            Given a budget (money), estimate the maximum electricity consumption
            under the tiered pricing structure.
        """
        tiers = self.get_corresponding_tiers(0, year, month, tariff_dict)
        consumption = 0
        remaining_money = money
        for (low, high), rate in tiers:
            if remaining_money <= 0:
                break
            # units available in this tier
            if high == float("inf"):
                tier_units = float("inf")
            else:
                tier_units = high - low + 1
            # cost for full tier
            tier_cost = tier_units * rate if high != float("inf") else float("inf")
            if remaining_money >= tier_cost and high != float("inf"):
                # can pay the full tier
                consumption += tier_units
                remaining_money -= tier_cost
            else:
                # can only pay partially in this tier
                units_affordable = remaining_money / rate
                consumption += units_affordable
                remaining_money = 0
                break 
        return round(consumption, 2)

def load_all_excels(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root).resolve()

    # collect all xlsx files recursively
    files = sorted(data_root.rglob("*.xlsx"))

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, parse_dates=["Time"])
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}")

    if not dfs:
        raise RuntimeError("No Excel files found in data directory.")

    all_db = pd.concat(dfs, ignore_index=True)

    # Drop duplicated values to avoid copy last-time / first time of 2 consecutive dates
    all_db = all_db.drop_duplicates()

    return all_db

def overview(all_db):
    c1, _, c2, _, c3 = st.columns([9, 1, 7, 1, 5])
    # --- Date range selection (based on Time column) ---
    with c1:
        min_date = all_db["Time"].min().date()
        max_date = all_db["Time"].max().date()
        date_range = st.date_input("Select date range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

    # --- View mode ---
    with c2:
        view_mode = st.selectbox("View mode", ["all rows", "top-N-rows", "bot-N-rows"])

    # --- Number of rows (only if not "all rows") ---
    with c3:
        Nrows = None
        if view_mode != "all rows":
            Nrows = st.number_input("N rows", min_value=1, value=5, step=1)

    # --- Filter by date range ---
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (all_db["Time"].dt.date >= start_date) & (all_db["Time"].dt.date <= end_date)
        filtered_db = all_db.loc[mask]
    else:
        filtered_db = all_db

    # --- Apply view mode ---
    if view_mode == "top-N-rows" and Nrows:
        display_df = filtered_db.head(Nrows)
    elif view_mode == "bot-N-rows" and Nrows:
        display_df = filtered_db.tail(Nrows)
    else:
        display_df = filtered_db

    # --- Show dataframe ---
    st.dataframe(display_df[display_df.columns[:5]], hide_index=True)

def aggregated_summary(df, mode="All-time", specific_timestamp=None):
    # Each row is 5 minutes = 1/12 hour
    time_interval_hours = 5/60 
    df = df.copy()
    df["Year"] = df["Time"].dt.year
    df["Month"] = df["Time"].dt.to_period("M").astype(str)
    df["Week"] = df["Time"].dt.isocalendar().week
    df["Date"] = df["Time"].dt.date

    # ---- filter based on specific timestamp ----
    if mode == "Yearly" and specific_timestamp is not None:
        df = df[df["Year"] == specific_timestamp]
    elif mode == "Monthly" and specific_timestamp is not None:
        df = df[df["Month"] == specific_timestamp]
    elif mode == "Weekly" and specific_timestamp is not None:
        df = df[df["Week"] == specific_timestamp]
    elif mode == "Daily" and specific_timestamp is not None:
        df = df[df["Date"] == specific_timestamp]

    # ---- aggregate (sum) ----
    total_consumption = (df["Consumption（kW）"] * time_interval_hours).sum()
    total_PV = (df["Production（kW）"] * time_interval_hours).sum()
    total_grid_NET = (df["Purchasing（kW）"] * time_interval_hours).sum()

    # ---- display metrics ----
    _, c1, c2, c3, _ = st.columns([0.2, 3, 2, 2, 0.1])
    c1.metric("Total Consumption (kWh)", f"{total_consumption:.2f}")
    c2.metric("Total PV (kWh)", f"{total_PV:.2f}")
    c3.metric("Total NET (kWh)", f"{total_grid_NET:.2f}")

def time_series_show(df):
    st.warning("You can also see the Trend / Seasonality analytic in the next Tab to understand more the usage-behavior")
    y_cols = ['Production（kW）', 'Consumption（kW）', 'Purchasing（kW）']
    c1, _, c = st.columns([5, 1, 11])
    procesor = datetime_df_processing()
    with c1:
        view_mode = st.selectbox("view-mode", 
                                 ["today", "any-date-range", "all-time-daily", "all-time-monthly"]) 
    # Only show c2, c3 when NOT today
    if view_mode in ["today", "any-date-range"]:
        agg_func = None
    if view_mode == "today":
        date_range = None
        with c1:
            reference_ls = procesor.get_list_of_compared_hhmm(df)
            other_timestamps = st.selectbox("Select any time-stamp", reference_ls)
        with c:
            c2, _, c3 = st.columns([7, 1, 3])
            with c2:                 
                sdf = procesor.get_last_timestamps_in_df(df, other_timestamps)
                st.dataframe(sdf, use_container_width = True )
            with c3:
                st.write(f"All the info is computed until `{df['Time'].max().strftime('%H:%M')}`")
    else:
        with c:
            c2, _, c3 = st.columns([5, 1, 5])
            with c3:
                min_date = df["Time"].min().date()
                max_date = df["Time"].max().date()
                date_range = st.date_input("date range", 
                                        value=[min_date, max_date], min_value=min_date, max_value=max_date)
    # generate dataframe
    if view_mode == "today":
        today = pd.Timestamp.today().normalize()  # normalize = 00:00 today
        tomorrow = today + pd.Timedelta(days=1)
        temp_df = df[(df['Time'] >= today) & (df['Time'] < tomorrow)]
        fig = px.line(temp_df, 
                      x = "Time", y = ['Production（kW）', 'Consumption（kW）', 'Purchasing（kW）'], 
                      color_discrete_map = colors)

    elif view_mode == "any-date-range":
        temp_df = df.copy()
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (temp_df["Time"].dt.date >= start_date) & (temp_df["Time"].dt.date <= end_date)
            temp_df = temp_df.loc[mask]

        fig = px.line(temp_df, x = "Time", y = y_cols, color_discrete_map = colors)

    elif view_mode == "all-time-daily":
        temp_df = df.copy()
        temp_df['Date'] = temp_df['Time'].dt.date
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (temp_df["Time"].dt.date >= start_date) & (temp_df["Time"].dt.date <= end_date)
            temp_df = temp_df.loc[mask]
        temp_df = (temp_df.groupby('Date')[y_cols].sum() / 12).reset_index()
        fig = px.line(temp_df, x = "Date", y = y_cols, color_discrete_map = colors)

    else:
        temp_df = df.copy()
        temp_df['Month'] = temp_df['Time'].dt.month
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (temp_df["Time"].dt.date >= start_date) & (temp_df["Time"].dt.date <= end_date)
            temp_df = temp_df.loc[mask]
        temp_df = (temp_df.groupby('Month')[y_cols].sum() / 12).reset_index()
        fig = px.line(temp_df, x = "Month", y = y_cols, color_discrete_map = colors)

    st.plotly_chart(fig, use_container_width=True)
    
def agg_NET_chart(df, mode="All-time"):
    time_interval_hours = 5/60
    y_cols = ['Production（kW）', 'Consumption（kW）', 'Purchasing（kW）']
    labels={"value": "kW", "variable": "Category"}
    colors = {
        "Consumption（kW）": "#1f77b4",  # blue
        "Production（kW）": "#2ca02c",   # green
        "Purchasing（kW）": "#d62728",   # red
    }
    if mode == "All-time":
        total_consumption = (df["Consumption（kW）"] * time_interval_hours).sum()
        total_PV = (df["Production（kW）"] * time_interval_hours).sum()
        total_grid_NET = (df["Purchasing（kW）"] * time_interval_hours).sum()
        values = [total_consumption, total_PV, -total_grid_NET]
        labels = ["Consumption", "PV", "Grid NET"]
        colors = ["#1f77b4", "#2ca02c", "#d62728"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{abs(v):.2f}" for v in values],
            textposition="outside"
        ))
        fig.update_layout(
            title="Energy Balance",
            xaxis=dict(title="kWh", zeroline=True, zerolinewidth=2, zerolinecolor="black"),
            yaxis=dict(title=""),
            bargap=0.4,
            height=400
        )        

    elif mode == "Yearly":
        df["Year"] = df['Time'].dt.year
        df = df.groupby("Year")[y_cols].sum() / 12
        df = df.reset_index()
        fig = px.bar(df, x = 'Year', y = y_cols, color_discrete_map = colors, barmode="group")

    elif mode == "Monthly":
        df["Month"] = df['Time'].dt.month
        df = df.groupby("Month")[y_cols].sum() / 12
        df = df.reset_index()
        fig = px.line(df, x = 'Month', y = y_cols, color_discrete_map = colors)

    elif mode == "Weekly":
        df["Week"] = df["Time"].dt.strftime("%G-W%V")
        df = df.groupby("Week")[y_cols].sum() / 12
        df = df.reset_index()
        fig = px.line(df, x = 'Week', y = y_cols, color_discrete_map = colors)

    elif mode == "Daily":
        df["Date"] = df['Time'].dt.date
        df = df.groupby("Date")[y_cols].sum() / 12
        df = df.reset_index()
        fig = px.line(df, x = 'Date', y = y_cols, color_discrete_map = colors)

    fig.update_layout(
        legend=dict(
            orientation="h",      # horizontal layout
            yanchor="bottom",     # anchor to bottom of legend box
            y=1.2,                # push it above the plot area
            xanchor="center",     # center it horizontally
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def summary(df):
    c1, _, c2, _, c3 = st.columns([5, 1, 5, 1, 5])
    with c1:
        aggregated_mode = st.selectbox("Aggregated mode", ["All-time", "Yearly", "Monthly", "Weekly", "Daily"])
    with c2:
        if aggregated_mode == "Yearly":
            res = st.selectbox("Select year", sorted(df["Time"].dt.year.unique()))
        elif aggregated_mode == "Monthly":
            res = st.selectbox("Select month", sorted(df["Time"].dt.to_period("M").unique().astype(str)))
        elif aggregated_mode == "Weekly":
            res = st.selectbox("Select week", sorted(df["Time"].dt.isocalendar().week.unique()))
        elif aggregated_mode == "Daily":
            res = st.date_input("Select date", value=df["Time"].dt.date.min())
        else:
            res = None

    with c3:
        show_chart = st.checkbox("Show chart", value=True)

    aggregated_summary(df, mode = aggregated_mode, specific_timestamp = res)
    if show_chart:
        agg_NET_chart(df, aggregated_mode) 

def streamlit_spec_flag_solar(df):
    # st.write("Use pie-chart to show distribution for each category in flag [day-part / season /]")
    processor = datetime_df_processing()
    c1, _, c2 = st.columns([1, 0.1, 2])
    with c1:
        flag_spec = st.selectbox("spec-flag", ["day-part", "weekend", "season"])
        metric_col = st.selectbox( "Select metric", 
                                  ["Consumption（kW）", "Production（kW）", "Purchasing（kW）"])
        if flag_spec == "day-part":
            st.json({
                "Off-peak": "00:00 - 07:59",
                "Mid-load": "08:00 - 15:59",
                "Peak-load": "16:00 - 23:59"
            })
            color_mapping = { 
                             "Off-peak": "darkblue",
                             "Mid-load": "violet",
                             "Peak-load": "darkgreen"
                            }
        elif flag_spec == "weekend":
            st.json({
                "week-end": "Sat+Sun",
                "week-date": "Mon-Fri"
            })
            color_mapping = { 
                "weekend": "blue",
                "weekdate": "orange"
                }
        else:
            st.json({
                "rainy-season": "Annually Apr-Oct"
            })
            color_mapping = { 
                "Rainy": "blue",
                "Sunny": "orange"
                }
    with c2:
        df_temp = (processor.add_time_feature(df.copy(), flag_spec) / 12).reset_index()
        fig = px.pie(df_temp, names = flag_spec,  # slice labels
                        values = metric_col,  # slice size
                        title = f"{metric_col} by {flag_spec}",
                        hole = 0.3,  # donut style
                        color = flag_spec,
                        color_discrete_map = color_mapping
                    )
        # adjust font sizes
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(font=dict(size=18)),      # title font size
            legend=dict(font=dict(size=14)),     # legend font size
            font=dict(size=16)                   # general font size (slice labels, etc.)
        )
        st.plotly_chart(fig, use_container_width=True)

def NhanDV_est_and_forecasting(df):
    c1, _, c2 = st.columns([3, 0.5, 6])
    y_cols = ['Production（kW）', 'Consumption（kW）', 'Purchasing（kW）']
    datetime_processor = datetime_df_processing()
    whore_s_price_processor = EVN_the_BITCH()
    with c1:
        your_consideration = st.selectbox("You want?", 
                                          ["Forecast total consumption until the end of this month",
                                           "Convert real-consumption to VND, and vice versa"
                                           ]
                                          )
        Show_basic_info_as_text = st.checkbox("Show basic info as text", True)
        perfect_est = st.checkbox("Included day enough 24h only", True)
        if your_consideration == "Convert real-consumption to VND, and vice versa":
            valid_year = whore_s_price_processor.get_ls_years(bitch_EVN_as_JSon)
            c11, _, c12 = st.columns([6, 1, 6])
            with c11:
                sel_year = st.selectbox("Select year", valid_year)
            with c12:
                valid_periods = whore_s_price_processor.get_period_ls(bitch_EVN_as_JSon, sel_year)
                sel_period = st.selectbox("Select period", valid_periods) 
            c11_, _, c12_ = st.columns([3, 0.1, 1])
            with c11_:
                wanna_vice_versa = st.number_input("Input the money then I will tell how much you used?",
                                               value = 400000, min_value=100000, max_value=100000000)
            with c12_:
                VAT_ = st.slider("VAT (%):", value=8, min_value=8, max_value=20)     
    with c2:
        if your_consideration == "Forecast total consumption until the end of this month":
            today_values = datetime_processor.get_today_values()
            n_days_this_month = datetime_processor.get_nday_month_year(today_values.month, today_values.year)
            if Show_basic_info_as_text:
                st.write(f"""
                            #### Month-year: **{today_values.strftime("%B")}-{today_values.year}**
                            
                            Until now, we have observed `{datetime_processor.get_ndays_this_month(df)}` days [in total `{n_days_this_month} days` of this month], and
                         """)
            else:
                st.write("#### ")

            this_month_df = datetime_processor.get_this_month_df(df)
            this_month_df = datetime_processor.get_df_which_enough_24hours_per_day(df, perfect_est)
            this_month_df = this_month_df.groupby("Date")[y_cols].sum() / 12
            first_observed_day = datetime_processor.get_firstday_this_month_df(df).date()
            nday_remain = n_days_this_month - first_observed_day.day + 1
            avg_consumption = this_month_df['Consumption（kW）'].mean()
            avg_purchasing = -this_month_df['Purchasing（kW）'].mean()

            st.write(f"""
                        - The first days observed in this month data is `{first_observed_day}`; 
                        hence we remained `{ nday_remain }` days in this month,
                        and `{ n_days_this_month - nday_remain }` days WITHOUT SOLAR
                        - AVG Consumption (without Solar) : `{avg_consumption:.3f}`
                        - AVG Purchasing (subtracting Solar) `{avg_purchasing:.3f}`

                        ------------------------
                        Then,
                        - Without Solar; we will cost `{(avg_consumption * n_days_this_month) :.3f} kWh`
                        - Including Solar, we just pay `{(avg_purchasing * nday_remain + avg_consumption * (n_days_this_month - nday_remain )):.3f} kWh`
                        """)

        elif your_consideration == "Convert real-consumption to VND, and vice versa":
            c21, _, c22 = st.columns([1, 0.1, 3])
            with c21:
                st.success("REFERENCE PRICES")
                st.json(bitch_EVN_as_JSon[sel_year][sel_period])
            with c22:
                coef_vat = (100 + VAT_) / 100
                # Refer the cost
                today_values = datetime_processor.get_today_values()
                n_days_this_month = datetime_processor.get_nday_month_year(today_values.month, today_values.year)
                this_month_df = datetime_processor.get_this_month_df(df)
                this_month_df = datetime_processor.get_df_which_enough_24hours_per_day(df, perfect_est)
                this_month_df = this_month_df.groupby("Date")[y_cols].sum() / 12
                first_observed_day = this_month_df.index[0]
                nday_remain = n_days_this_month - first_observed_day.day + 1
                avg_consumption = this_month_df['Consumption（kW）'].mean()
                avg_purchasing = -this_month_df['Purchasing（kW）'].mean()
                prediction_purchase = avg_purchasing * nday_remain + avg_consumption * (n_days_this_month - nday_remain )
                st.write("##### Convert consumption -> VND")
                if Show_basic_info_as_text:
                    st.write(f"""
                             You can see the explaination by re-select `You want: Forecast total ...`; 
                             now, your current consumption's prediction is `{prediction_purchase:.3f} kWh`
                             """)
                itv = whore_s_price_processor.streamlit_get_interval(prediction_purchase, sel_year, sel_period, bitch_EVN_as_JSon)
                pr = whore_s_price_processor.convert_consume(prediction_purchase, sel_year, sel_period, bitch_EVN_as_JSon)
                st.write(f"""
                         The corresponding interval of your prediction is `{itv}`
                         - The total price [without tax is] : `{pr:,.4f} VND`
                         - Including `{coef_vat} % VAT`, we must pay `{( coef_vat * pr):,.4f} VND`
                         """)
                ignore_tax = wanna_vice_versa / coef_vat
                est_consumption = whore_s_price_processor.get_corresponding_consumptions(ignore_tax, sel_year, sel_period, bitch_EVN_as_JSon)
                st.write(f"""
                         ----------------------
                         ##### Vice-versa
                         
                         The money you input is `{wanna_vice_versa:,.4f} VND`; 
                         eliminate `{coef_vat} % VAT` we have costed exactly on `{ignore_tax:,.4f} VND`; coresponding about `{est_consumption} kWh`  
                         """)

def eda_solar_agg(df):
    agg_df = df.copy()
    agg_df = agg_df[agg_df['Time'] >= '2025-09-18'] # just ignore the first day
    numeric_cols = ["Production（kW）", "Consumption（kW）", "Purchasing（kW）"]
    c1, _, c2 = st.columns([9, 1, 9])
    with c1:
        agg_func = st.selectbox("Agg_func", ["AVG - STD", "MAX - MIN"], 
                                help="you can refer SUM in the Homepage")
    with c2:
        group_by = st.selectbox("Groupby", ["Year", "YYYY-MM", "YYYY-WW", "Per-day", "Per-hour"])
        if group_by == "Year":
            agg_df[group_by] = agg_df["Time"].dt.year
        elif group_by == "YYYY-MM":
            # Year-Month in YYYY-MM format
            agg_df[group_by] = agg_df["Time"].dt.to_period("M").astype(str)

        elif group_by == "YYYY-WW":
            # Year-Week in YYYY-WW format
            iso_year = agg_df["Time"].dt.isocalendar().year.astype(str)
            iso_week = agg_df["Time"].dt.isocalendar().week.astype(str).str.zfill(2)
            agg_df[group_by] = iso_year + "-W" + iso_week

        elif group_by == "Per-day":
            # Date only (YYYY-MM-DD)
            agg_df[group_by] = agg_df["Time"].dt.date

        elif group_by == "Per-hour":
            # Hour-level (YYYY-MM-DD HH:00)
            agg_df[group_by] = agg_df["Time"].dt.to_period("H").astype(str)
    
    agg_df = agg_df.groupby(group_by)[numeric_cols].sum() / 12

    if agg_func == "AVG - STD":
        st.write(agg_df.agg(['mean', 'std']))

    elif agg_func == "MAX - MIN":
        a, b, c = st.columns(3)
        with a:
            st.metric("**MAX**(Production) in kWh", 
                      value=f"{agg_df['Production（kW）'].max():.4f}", delta=f"at {agg_df['Production（kW）'].idxmax()}")
            st.metric("**MIN**(Production) in kWh", 
                      value=f"{agg_df.loc[agg_df['Production（kW）'] > 0, 'Production（kW）'].min():.4f}", 
                      delta=f"▼ at {agg_df.loc[agg_df['Production（kW）'] > 0, 'Production（kW）'].idxmin()}",
                      delta_color="inverse"  # red for down
                    )
        with b:
            st.metric("**MAX**(Consumption) in kWh", 
                    value=f"{agg_df['Consumption（kW）'].max():.4f}", 
                    delta=f"▲ at {agg_df['Consumption（kW）'].idxmax()}",
                    delta_color="normal")

            st.metric("**MIN**(Consumption) in kWh", 
                    value=f"{agg_df.loc[agg_df['Consumption（kW）'] > 0.1, 'Consumption（kW）'].min():.4f}", 
                    delta=f"▼ at {agg_df.loc[agg_df['Consumption（kW）'] > 0.1, 'Consumption（kW）'].idxmin()}",
                    delta_color="inverse")
        with c:
            # MAX (least negative, closest to 0)
            max_val = agg_df.loc[agg_df['Purchasing（kW）'] < -0.1, 'Purchasing（kW）'].max()
            max_idx = agg_df.loc[agg_df['Purchasing（kW）'] < -0.1, 'Purchasing（kW）'].idxmax()
            st.metric("**MIN**(Purchasing) in kWh", 
                    value=f"{abs(max_val):.4f}", 
                    delta=f"▲ at {max_idx}",
                    delta_color="normal")

            # MIN (most negative, largest magnitude)
            min_val = agg_df.loc[agg_df['Purchasing（kW）'] < 0, 'Purchasing（kW）'].min()
            min_idx = agg_df.loc[agg_df['Purchasing（kW）'] < 0, 'Purchasing（kW）'].idxmin()
            st.metric("**MAX**(Purchasing) in kWh", 
                    value=f"{abs(min_val):.4f}", 
                    delta=f"▼ at {min_idx}",
                    delta_color="inverse") 

    # with st.expander("See details"): 
    #   st.write(agg_df) 

def heat_map_plot(df, sel_col):
    if sel_col == "Purchasing（kW）":
        df = -df
    black_blue = LinearSegmentedColormap.from_list("black_blue", ["black", "blue"])
    if sel_col == "Production（kW）":
        cmap_colors = black_blue
    else:
        cmap_colors = "RdBu_r"
    n_days_in_week = len(df)    
    fig = plt.figure(figsize = (24, n_days_in_week + 0.5))
    ax = sns.heatmap(df, annot=True, linewidth=.5, fmt=".1f", cmap = cmap_colors,
                        annot_kws={"size": 14},                         # font-size
                        cbar=False                                      # remove color bar (legend on the right)
                        )
    # Move the Hour labels (columns) to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # Increase tick label font size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=-10)
    # Axis labels (the words "Hour" and "Date")
    ax.set_xlabel("Hour", fontsize=16, color="darkgreen", fontweight="bold")
    ax.set_ylabel("Date", fontsize=16, color="darkblue", fontweight="bold")
    st.pyplot(fig)
    
def hourly_analytic_heatmap(df):
    df['Date'] = df['Time'].dt.date
    df['Hour'] = df['Time'].dt.hour.apply(lambda x: f"{x:02d}")
    datetime_processor = datetime_df_processing()
    c1, _, c2 = st.columns([1.2, 0.02, 7.5])
    with c1:
        sel_col = st.selectbox("Observed quantity", ["Consumption（kW）", "Production（kW）", "Purchasing（kW）"])
        level_1 = st.selectbox("Analysis granularity", ["Week-Year", "Month-Year"])
    if level_1 == "Week-Year": 
        with c1:
            df, valid_weekyear = datetime_processor.get_valid_weekyear(df)
            sel_weekyear = st.selectbox("Week-year", valid_weekyear)
        with c2:
            ext_df = df.loc[df['WeekYear'] == sel_weekyear, 
                            ['Date', 'Time', 'WeekYear', 'Hour', sel_col]] # included weekyear easier to double-check             
            # st.dataframe(ext_df)
            ext_df = pd.pivot_table(ext_df, values = sel_col, aggfunc = 'sum',
                                    columns='Hour', index='Date') / 12
            heat_map_plot(ext_df, sel_col)
    else:
        with c1:
            df, valid_monthyear = datetime_processor.get_valid_monthyear(df)
            sel_monthyear = st.selectbox("Month-year", valid_monthyear)
        with c2:
            ext_df = df.loc[df['MonthYear'] == sel_monthyear, 
                            ['Date', 'Time', 'MonthYear', 'Hour', sel_col]]
            ext_df = pd.pivot_table(ext_df, values = sel_col, aggfunc = 'sum',
                                    columns='Hour', index='Date') / 12
            heat_map_plot(ext_df, sel_col)

def trend_and_season_decompo_analytic(df):
    numeric_cols = ["Consumption（kW）", "Production（kW）", "Purchasing（kW）"]
    ts_processpr = datetime_df_processing()
    max_valid_days = ts_processpr.get_ndays(df)
    c1, _, c2 = st.columns([1.5, 0.1, 9])
    with c1:
        freq = st.selectbox("Frequent:", ["Hourly", "Daily"])
        analysis_focus = st.selectbox("Select component to analyze", 
                                      ["Trend", "Seasonality"], help = "Time-series models was assummed additive" )
        lookback_ndays = st.number_input("How many days you looked back", 
                                         value = 10, min_value=5, max_value = max_valid_days,
                                         help = "First date in the data is 2025-09-17; we dont have any observations before this date.")
        shift_df = ts_processpr.shift_to_lookback_ts(df, lookback_ndays)
        analysis_cols = st.selectbox("Analysis column:", numeric_cols)
    with c2:
        ext_df_by_freq = ts_processpr.get_df_by_freq(shift_df, analysis_cols, freq)
        if freq == "Hourly":
            # 24 hours to make a cycle
            processed_series = seasonal_decompose(ext_df_by_freq, model='additive', period=24,
                                                two_sided=True, extrapolate_trend=0)
        elif freq == "Daily":
            # 7 days to make a cycle
            adjusted_period = max(2, max_valid_days / 7)
            processed_series = seasonal_decompose(ext_df_by_freq, model='additive', period=adjusted_period,
                                                two_sided=True, extrapolate_trend=0)

        if analysis_focus == "Trend":
            trend = processed_series.trend.dropna() 
            # simple slope: last value - first value
            slope = trend.iloc[-1] - trend.iloc[0]
            st.write(f"Trend slope over lookback window: {slope:.2f}")

            st.line_chart(trend)

        elif analysis_focus == "Seasonality":
            seasonal = processed_series.seasonal 
            # Plot one representative cycle
            if freq == "Hourly":
                cycle = seasonal.groupby(seasonal.index.hour).mean()
                st.bar_chart(cycle)

            elif freq == "Daily":
                cycle = seasonal.groupby(seasonal.index.dayofweek).mean()
                cycle.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                st.bar_chart(cycle)

def EDA(df):
    c1, _, c2 = st.columns([9, 1, 9])
    with c1:
        with st.expander("Special flag [day-part, dayofweek, sunny-season]"):
            streamlit_spec_flag_solar(df)
    with c2:
        with st.expander("Customer behavior [ AVG | MAX / MIN | STD ]"):
            eda_solar_agg(df)

    with st.expander("Saving & estimated due to?"):
        NhanDV_est_and_forecasting(df)

    with st.expander("Trend & Seasonality analytics"):
        trend_and_season_decompo_analytic(df)

    with st.expander("Hourly behavior-observation w.r.t Week-Year / Month-Year"):
        hourly_analytic_heatmap(df)
        
def sentiment_analytic():
    dit_banh_loz_con_duy_EVN_ = crawl_EVN_the_BITCH()
    with st.expander("Overview: Crawl & show basic info"):
        c1, _, c2, _, c3 = st.columns([2, 0.1, 3, 0.1, 6])

        with c1:
            crawl_all_data = st.selectbox("Crawl again", ["YES", "DEL"])
            wanna_updated = st.checkbox("Wanna updated data", value=False)

        with c2:
            if crawl_all_data == "YES":
                adr_df = dit_banh_loz_con_duy_EVN_.crawl_all_data_android()
                ios_df = dit_banh_loz_con_duy_EVN_.crawl_all_data_appstore()
                
                st.subheader("Android Reviews")
                st.dataframe(adr_df.head(10))  # show more, in a scrollable table
                dit_banh_loz_con_duy_EVN_.save_sentiment_data(adr_df, "android_reviews")
                st.subheader("iOS Reviews")
                st.dataframe(ios_df.head(10))
                dit_banh_loz_con_duy_EVN_.save_sentiment_data(ios_df, "ios_reviews")
            else: 
                st.info("No more data crawled yet. Select 'YES' to fetch new reviews.")

                # now use pandas to read the pre-saved data
                try:
                    adr_df = pd.read_excel("sentiment_data/android_reviews.xlsx")
                    ios_df = pd.read_excel("sentiment_data/ios_reviews.xlsx")

                    st.subheader("Android Reviews (loaded from saved data)")
                    st.dataframe(adr_df.head())

                    st.subheader("iOS Reviews (loaded from saved data)")
                    st.dataframe(ios_df.head())

                except FileNotFoundError:
                    st.warning("⚠️ No saved sentiment data found. Please select 'YES' to crawl new reviews.")                
                
            if wanna_updated:
                st.write("")