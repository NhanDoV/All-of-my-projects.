import os
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

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

    return pd.concat(dfs, ignore_index=True)

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
    if isinstance(date_range, list) and len(date_range) == 2:
        mask = (all_db["Time"].dt.date >= date_range[0]) & (all_db["Time"].dt.date <= date_range[1])
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

def agg_NET_chart(df, mode="All-time"):
    time_interval_hours = 5/60
    colors = {
        "Consumption（kW）": "#1f77b4",  # blue
        "Production（kW）": "#2ca02c",   # green
        "Purchasing（kW）": "#d62728",   # red
    }
    y_cols = ['Production（kW）', 'Consumption（kW）', 'Purchasing（kW）']
    labels={"value": "kW", "variable": "Category"}

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
        fig = px.line(df, x = 'Year', y = y_cols, color_discrete_map = colors)

    elif mode == "Monthly":
        df["Month"] = df['Time'].dt.month
        df = df.groupby("Month")[y_cols].sum() / 12
        df = df.reset_index()
        fig = px.line(df, x = 'Month', y = y_cols, color_discrete_map = colors)

    elif mode == "Weekly":
        df["Week"] = df['Time'].dt.isocalendar().week
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

def streamlit_box_plot_solar(df):
    c1, _, c2 = st.columns([9, 1, 9])

def eda_solar_agg(df):
    c1, _, c2 = st.columns([9, 1, 9])
    with c1:
        agg_func = st.selectbox("Agg_func", ["AVG - STD", "MAX - MIN"], 
                                help="you can refer SUM in the Homepage")
    with c2:
        groupby = st.selectbox("Groupby", ["Year", "Month", "Week", "Date", "Hour"])

def EDA(df):
    c1, _, c2 = st.columns([9, 1, 9])
    with c1:
        with st.expander("Show boxplot"):
            streamlit_box_plot_solar(df)

    with c2:
        with st.expander("Customer behavior [ AVG | MAX / MIN | STD ]"):
            eda_solar_agg(df)