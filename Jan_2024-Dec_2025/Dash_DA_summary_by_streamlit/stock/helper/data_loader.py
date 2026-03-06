import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def get_description_info():
    st.title("🏠 HOME – Bitcoin Market Overview")

    st.markdown("""
                    ### ₿ Bitcoin – Market Time Series Overview
                    High-level view of **BTC price & volume dynamics** over time.  
                    Focus on **trend, volatility & market regime**.
                """)

def get_market_overview(df):
    
    def metric_card(label, value):
        return f"""
        <div class="metric-card metric-center">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    st.subheader("📌 Market Overview")

    # Tạo HTML strings trước khi nhúng
    date_html = metric_card("Date range", f"{df.date.min():%Y-%m-%d} → {df.date.max():%Y-%m-%d}")
    close_html = metric_card("Last close", f"${df.price.iloc[-1]:,.2f}")
    return_html = metric_card("Total return", f"{(df.price.iloc[-1] / df.price.iloc[0] - 1) * 100:.2f}%")
    range_html = metric_card("Max daily range", f"${(df.high - df.low).max():,.0f}")
    volume_html = metric_card("Max volume", f"{df.volume.max():,.0f}")

    st.markdown(f"""
        <div class="market-grid">
            {date_html}
            {close_html}
        </div>
        <div class="market-grid-3">
            {return_html}
            {range_html}
            {volume_html}
        </div>
    """, unsafe_allow_html=True)
