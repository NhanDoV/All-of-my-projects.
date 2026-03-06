import streamlit as st
import numpy as np
from helper.data_loader import load_data
from helper.FE_and_modeling import *

# ======================================================
# LOAD CSS
# ======================================================
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
df = load_data("btc_data.csv")

# ======================================================
# PAGE HEADER
# ======================================================
st.title("🔮 PREDICT – Direction Forecasting (Learning mode)")

st.markdown("""
This page focuses on **directional forecasting** for Bitcoin.

We proceed step-by-step:
- diagnose stationarity  
- choose explainable models  
- interpret results before moving forward  

⚠️ This is **not auto-trading**.
""")

st.divider()

# ======================================================
# STATIONARITY DIAGNOSTICS (ALWAYS ON)
# ======================================================
st.subheader("📉 Stationarity diagnostics (Returns)")

alpha = st.selectbox(
    "Significance level α",
    [0.01, 0.05, 0.1],
    index=1
)

stationary, score, detail = weighted_stationarity_test(
    df["price"],
    alpha=alpha
)

st.metric("Stationarity score", f"{score:.2f}")

st.caption(f"""
ADF p-value: {detail['adf_p']:.4f}  
KPSS p-value: {detail['kpss_p']:.4f}
""")

if stationary:
    st.success("Return series is stationary")
else:
    st.warning("Return series is NOT stationary")

st.markdown("""
**Why this matters:**  
AR-based models rely on stationarity.  
If returns are non-stationary → AR results are unreliable.
""")

st.divider()

# ======================================================
# TARGET CONFIG
# ======================================================
st.subheader("🎯 Prediction target")

horizon = st.selectbox(
    "Forecast horizon (days ahead)",
    [1, 3, 5, 7],
    index=0
)

data = prepare_direction_data(df, horizon=horizon)

st.markdown(f"""
**Target definition:**  
Direction = sign of **{horizon}-day future log-return**
""")

st.divider()

# ======================================================
# MODEL SELECTION
# ======================================================
st.subheader("🧠 Model selection")

model_blocks = st.multiselect(
    "Choose models to explore",
    [
        "Moving Average (Baseline)",
        "Autoregressive (AR)",
        "Regime-aware Direction",
        "Expected Return",
        "Volatility",
    ],
    default=["Moving Average (Baseline)"]
)

st.divider()

# ======================================================
# MOVING AVERAGE DIRECTION
# ======================================================
if "Moving Average (Baseline)" in model_blocks:
    with st.expander("📈 Moving Average Direction", expanded=True):

        c1, c2 = st.columns(2)
        with c1:
            short = st.slider("Short MA window", 3, 20, 5)
        with c2:
            long = st.slider("Long MA window", 10, 60, 20)

        ma_df = ma_direction_signal(data, short=short, long=long)

        acc_ma = hit_rate(ma_df["target"], ma_df["pred"])
        last_signal = ma_df["pred"].iloc[-1]
        signal_text = "UP 🟢" if last_signal == 1 else "DOWN 🔴"

        st.metric("MA hit-rate", f"{acc_ma*100:.1f}%")
        st.metric("Current signal", signal_text)

        st.info("""
        **Interpretation:**  
        Direction inferred from short-term vs long-term trend.
        Simple, robust, widely used by traders.
        """)

# ======================================================
# AUTOREGRESSIVE DIRECTION
# ======================================================
if "Autoregressive (AR)" in model_blocks:
    with st.expander("🧠 Autoregressive (AR) Direction"):

        if stationary:
            lags = st.slider("Number of AR lags", 1, 10, 3)

            direction, forecast = ar_direction_signal(
                data["return"],
                horizon=horizon,
                lags=lags
            )

            ar_signal = "UP 🟢" if direction == 1 else "DOWN 🔴"
            st.metric("AR predicted direction", ar_signal)

            st.info("""
            **Interpretation:**  
            Direction inferred from autocorrelation in returns.
            Valid only under stationarity assumption.
            """)
        else:
            st.error("""
            ❌ AR model disabled  

            Reason: return series is non-stationary.
            """)

# ======================================================
# REGIME-AWARE DIRECTION
# ======================================================
if "Regime-aware Direction" in model_blocks:
    with st.expander("🌗 Regime-aware Direction"):

        def volatility_regime(returns, window=20, q=0.7):
            vol = returns.rolling(window).std()
            threshold = vol.quantile(q)
            return (vol < threshold).astype(int)

        window = st.slider("Volatility window", 10, 60, 20)
        q = st.slider("Volatility threshold (quantile)", 0.5, 0.9, 0.7)

        tmp = data.copy()
        tmp["ma_pred"] = ma_direction_signal(tmp)["pred"]
        tmp["regime"] = volatility_regime(tmp["return"], window, q)
        tmp["final_pred"] = tmp["ma_pred"] * tmp["regime"]

        acc_regime = hit_rate(tmp["target"], tmp["final_pred"])
        last_regime_signal = tmp["final_pred"].iloc[-1]
        regime_text = "UP 🟢" if last_regime_signal == 1 else "NO-TRADE / DOWN 🔴"

        st.metric("Regime-aware hit-rate", f"{acc_regime*100:.1f}%")
        st.metric("Current signal", regime_text)

        st.info("""
        **Interpretation:**  
        MA signals are trusted only during **low-volatility regimes**.

        High volatility suppresses trend-following signals.
        """)

# =====================================================
# EXPECTED RETURN
# ====================================================
if "Expected Return" in model_blocks:
    with st.expander("📐 Expected Return Forecast"):

        st.markdown("""
        **Definition:**  
        Expected return = average log-return over forecast horizon.
        """)

        # ----------------------------
        # BASELINE: Rolling Mean
        # ----------------------------
        st.markdown("### 📉 Rolling Mean (Baseline)")

        window = st.slider(
            "Rolling window",
            5, 60, 20,
            key="er_roll_window"
        )

        er_roll = rolling_expected_return(
            data["return"],
            window
        ).iloc[-1]

        st.metric(
            "Rolling expected return",
            f"{er_roll*100:.3f}%"
        )

        st.info("""
        **Interpretation:**  
        Assumes future returns resemble recent average behavior.
        """)

        st.divider()

        # ----------------------------
        # AR EXPECTED RETURN
        # ----------------------------
        st.markdown("### 🧠 AR Expected Return")

        if stationary:
            lags = st.slider(
                "AR lags",
                1, 10, 3,
                key="er_ar_lags"
            )

            er_ar, path = ar_expected_return(
                data["return"],
                horizon=horizon,
                lags=lags
            )

            st.metric(
                "AR expected return",
                f"{er_ar*100:.3f}%"
            )

            st.info("""
            **Interpretation:**  
            Expected return inferred from autocorrelation structure
            of historical returns.
            """)

        else:
            st.error("""
            ❌ AR Expected Return disabled  

            Reason: return series is non-stationary.
            """)

# ====================================================

# ===================================================
if "Volatility" in model_blocks:
    with st.expander("🌪 Volatility Forecast"):

        st.markdown("""
        **Definition:**  
        Volatility measures the expected magnitude of price fluctuations.
        """)

        # ----------------------------
        # ROLLING VOLATILITY
        # ----------------------------
        st.markdown("### 📉 Rolling Volatility (Baseline)")

        window = st.slider(
            "Rolling window",
            5, 60, 20,
            key="vol_roll_window"
        )

        vol_roll = rolling_volatility(
            data["return"],
            window
        ).iloc[-1]

        st.metric(
            "Rolling volatility",
            f"{vol_roll*100:.2f}%"
        )

        st.info("""
        **Interpretation:**  
        Assumes future risk resembles recent realized volatility.
        """)

        st.divider()

        # ----------------------------
        # GARCH VOLATILITY
        # ----------------------------
        st.markdown("### 🧠 GARCH(1,1) Volatility")

        vol_garch, garch_path = garch_volatility(
            data["return"],
            horizon=horizon
        )

        st.metric(
            "GARCH expected volatility",
            f"{vol_garch*100:.2f}%"
        )

        st.info("""
        **Interpretation:**  
        GARCH captures volatility clustering:
        high volatility tends to follow high volatility.
        """)

st.divider()

# ======================================================
# FINAL SUMMARY
# ======================================================
st.subheader("📝 Model interpretation summary")

st.markdown(f"""
- **Target:** Direction of {horizon}-day return  
- **Stationarity:** {'Yes' if stationary else 'No'}  
- **Models explored:** {", ".join(model_blocks) if model_blocks else "None"}  

⚠️ All direction forecasts express **probabilistic bias**, not certainty.
""")

st.warning("""
⚠️ Expected return is **not a guarantee**.

Use it together with:
- Direction (sign)
- Volatility (risk)
before making decisions.
""")

st.warning("""
⚠️ Volatility is **risk**, not direction.

High expected volatility:
- increases uncertainty
- reduces reliability of directional signals

Always combine:
Direction + Expected Return + Volatility
""")