import streamlit as st
from helper.data_loader import *
from helper.viz import *

# ======================================================
# LOAD DATA
# ======================================================
df = load_data("btc_data.csv")

# ======================================================
# LOAD CSS (SAFE TO KEEP – shared style)
# ======================================================
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ======================================================
# PAGE HEADER
# ======================================================
c1, c2 = st.columns([4, 3])
with c1:
    get_description_info()

with c2:
    get_market_overview(df)

# ======================================================
# SECTION NAVIGATION (LEVEL 2 – SUBSECTIONS)
# ======================================================
tab_basic, tab_advanced = st.tabs(
    ["📊 Basic Analytics", "🧠 Advanced Analytics"]
)

# ======================================================
# BASIC ANALYTICS TAB
# ======================================================
with tab_basic:
    st.subheader("📊 Basic Analytics")

    option = st.selectbox(
        "Choose chart",
        ["Price", "Volume"],
        key="basic_chart_select"
    )

    if option == "Price":
        st.plotly_chart(
            plot_price(df),
            use_container_width=True
        )
    else:
        st.plotly_chart(
            plot_volume(df),
            use_container_width=True
        )

# ======================================================
# ADVANCED ANALYTICS TAB
# ======================================================
with tab_advanced:
    st.subheader("🧠 Advanced Analytics")

    analytic = st.selectbox(
        "Choose analytic",
        [
            "Market regime",
            "Volatility",
            "Drawdown",
            "Return distribution",
            "Volume–price relationship",
        ],
        key="advanced_analytic_select"
    )

    if analytic == "Market regime":
        col1, col2, col3 = st.columns(3)
        with col1:
            ma_window = st.selectbox("Trend window (MA)", [50, 100, 200])
        with col2:
            vol_window = st.selectbox("Volatility window", [30, 60, 90])
        with col3:
            sensitivity = st.selectbox("Regime sensitivity", ["Low", "Medium", "High"])

        df_regime = compute_market_regime(
            df,
            ma_window = ma_window,
            vol_window = vol_window,
            sensitivity = sensitivity,
        )

        st.plotly_chart(
            plot_market_regime(df_regime),
            use_container_width=True
        )

        info = interpret_current_regime(df_regime)

        st.success(
            f"""
            **📌 Current Market Regime: {info['regime']}**  
            - Trend: {info['trend']}  
            - Volatility: {info['volatility']}
            """
        )

    elif analytic == "Volatility":
        vol_window = st.selectbox(
            "Volatility window (days)",
            [30, 60, 90],
            index=1
        )

        df_vol = compute_volatility(df, vol_window=vol_window)

        st.plotly_chart(
            plot_volatility(df_vol),
            use_container_width=True
        )

        info = interpret_volatility(df_vol)

        st.warning(
            f"""
            **📌 Current volatility: {info['state']}**  
            - Percentile: {info['percentile']}%  
            - Rolling window: {vol_window} days  
            """
        )
    
    elif analytic == "Drawdown":
        df_dd = compute_drawdown(df)

        st.plotly_chart(
            plot_drawdown(df_dd),
            use_container_width=True
        )

        info = interpret_drawdown(df_dd)

        st.error(
            f"""
            **📌 Current drawdown: {info['current_dd']:.0%}**  
            - Regime: {info['state']}  
            - Max historical drawdown: {df_dd['drawdown'].min():.0%}
            """
        )

    elif analytic == "Return distribution":
        show_normal = st.checkbox(
            "Overlay normal distribution",
            value=True
        )

        returns, stats = compute_return_distribution(df)

        st.plotly_chart(
            plot_return_distribution(
                returns,
                show_normal=show_normal
            ),
            use_container_width=True
        )

        info = interpret_return_distribution(stats)

        st.success(
            f"""
            **📌 Return distribution characteristics**  
            - Skewness: {info['skew']:.2f} → {info['skew_desc']}  
            - Kurtosis: {info['kurtosis']:.2f} → {info['tail']}  
            """
        )
    
    else:
        window = st.slider(
            "Rolling correlation window",
            10, 60, 30
        )

        data, summary = compute_volume_price_relation(
            df,
            window=window
        )
        x1, x2 = st.columns(2)
        with x1:
            st.plotly_chart(
                plot_volume_price_scatter(data),
                use_container_width=True
            )
        with x2:
            st.plotly_chart(
                plot_volume_price_corr(data),
                use_container_width=True
            )

        info = interpret_volume_price(summary)

        st.success(
            f"""
            **📌 Volume–price insight**  
            - Latest rolling correlation: {info['latest_corr']:.2f}  
            - Interpretation: **{info['regime']}**
            """
        )