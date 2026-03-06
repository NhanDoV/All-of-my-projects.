import plotly.graph_objects as go

def update_layout(fig, title=""):
    fig.update_layout(
        title=dict(  
            text=title,
            font=dict(
                color="#051536",
                family="Inter, Segoe UI, sans-serif",
                size=18
            ),
            x=0.5,
            xanchor="center"
        ),
        
        template="ggplot2",
        yaxis_title="USD",
        xaxis_title="Date",
        
        # Backgrounds
        paper_bgcolor="#F6F8FB",
        plot_bgcolor="#9daed4",
        
        # Font cho toàn bộ
        font=dict(
            color="#051536",
            family="Inter, Segoe UI, sans-serif"
        ),
        
        # LEGEND - Thêm phần này để fix màu đen
        legend=dict(
            font=dict(
                color="#051536",  # Màu legend labels
                family="Inter, Segoe UI, sans-serif",
                size=12
            ),
            bgcolor="rgba(255,255,255,0.8)",  # Background legend (tùy chọn)
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        ),
        
        # Axes
        xaxis=dict(
            title_font=dict(color="#051536"),
            tickfont=dict(color="#051536"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            title_font=dict(color="#051536"),
            tickfont=dict(color="#051536"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            showline=False
        ),
        
        margin=dict(l=20, r=20, t=60, b=40), 
    )
    return fig

def plot_price(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = df.date,
            y = df.price,
            mode = "lines",
            name = "BTC Price"
        )
    )

    return update_layout(fig, title = "📈 Price evolution")

def plot_volume(df):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.date,
            y=df.volume,
            name="Volume"
        )
    )
    fig.update_layout(
        template="simple_white",
        yaxis_title="Volume",
        xaxis_title="Date"
    )

    return update_layout(fig, title = "📊 Volume over time")

# =========================== ADVANCED ANALYTIC =========================================
import numpy as np
import pandas as pd

def compute_market_regime(
    df,
    ma_window=200,
    vol_window=60,
    sensitivity="Medium",
):
    data = df.copy()

    # ---- Trend ----
    data["ma"] = data["price"].rolling(ma_window).mean()
    data["trend"] = np.where(
        data["price"] > data["ma"], "up", "down"
    )

    # ---- Volatility ----
    log_ret = np.log(data["price"]).diff()
    data["vol"] = log_ret.rolling(vol_window).std()

    vol_pct = data["vol"].rank(pct=True)

    if sensitivity == "Low":
        high_th = 0.8
    elif sensitivity == "High":
        high_th = 0.6
    else:
        high_th = 0.7

    data["vol_level"] = np.where(
        vol_pct > high_th, "high", "normal"
    )

    # ---- Regime ----
    def classify(row):
        if pd.isna(row["ma"]):
            return "undefined"
        if row["trend"] == "up" and row["vol_level"] == "normal":
            return "bull"
        if row["trend"] == "up" and row["vol_level"] == "high":
            return "bull_high_vol"
        if row["trend"] == "down" and row["vol_level"] == "normal":
            return "bear"
        if row["trend"] == "down" and row["vol_level"] == "high":
            return "bear_stress"
        return "sideways"

    data["regime"] = data.apply(classify, axis=1)

    return data

REGIME_COLOR = {
    "bull": "rgba(34,197,94,0.25)",
    "bull_high_vol": "rgba(234,179,8,0.25)",
    "bear": "rgba(249,115,22,0.25)",
    "bear_stress": "rgba(239,68,68,0.25)",
}

def plot_market_regime(df_regime):
    fig = go.Figure()

    # Price
    fig.add_trace(
        go.Scatter(
            x=df_regime["date"],
            y=df_regime["price"],
            name="BTC Price",
            line=dict(color="blue", width=2),
        )
    )

    # MA
    fig.add_trace(
        go.Scatter(
            x=df_regime["date"],
            y=df_regime["ma"],
            name="Moving Average",
            line=dict(color="#60A5FA", dash="dash"),
        )
    )

    # Background regime shading
    for regime, color in REGIME_COLOR.items():
        mask = df_regime["regime"] == regime
        fig.add_trace(
            go.Scatter(
                x=df_regime.loc[mask, "date"],
                y=df_regime.loc[mask, "price"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=color,
                showlegend=True,
                name=regime.replace("_", " ").title(),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
    )

    return update_layout(fig, title = "📈 Market Regime Analysis")

def interpret_current_regime(df_regime):
    last = df_regime.iloc[-1]

    regime_map = {
        "bull": "Bullish (Healthy)",
        "bull_high_vol": "Bullish (High Volatility)",
        "bear": "Bearish (Orderly)",
        "bear_stress": "Bearish (Stress)",
    }

    return {
        "regime": regime_map.get(last["regime"], "Undefined"),
        "trend": "Above MA" if last["price"] > last["ma"] else "Below MA",
        "volatility": "Elevated" if last["vol_level"] == "high" else "Normal",
    }

# ================= VOLATILITY ANALYTICS
def compute_volatility(
    df,
    vol_window=60,
):
    data = df.copy()

    log_ret = np.log(data["price"]).diff()
    data["vol"] = log_ret.rolling(vol_window).std()

    # Percentile vs history
    data["vol_pct"] = data["vol"].rank(pct=True)

    def vol_state(p):
        if p < 0.3:
            return "low"
        elif p < 0.7:
            return "normal"
        else:
            return "high"

    data["vol_state"] = data["vol_pct"].apply(vol_state)

    return data

VOL_COLOR = {
    "low": "#22c55e",      # green
    "normal": "#eab308",   # yellow
    "high": "#ef4444",     # red
}

def plot_volatility(df_vol):
    fig = go.Figure()

    # Rolling volatility
    fig.add_trace(
        go.Scatter(
            x=df_vol["date"],
            y=df_vol["vol"],
            name="Rolling Volatility",
            line=dict(color="#1f2937", width=2),
        )
    )

    # Current point
    fig.add_trace(
        go.Scatter(
            x=[df_vol["date"].iloc[-1]],
            y=[df_vol["vol"].iloc[-1]],
            mode="markers",
            marker=dict(
                size=10,
                color=VOL_COLOR.get(df_vol["vol_state"].iloc[-1], "#000")
            ),
            name="Current volatility"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Volatility",
        legend=dict(orientation="h"),
    )

    return update_layout(fig, title = "🌪️ Volatility Analysis")

def interpret_volatility(df_vol):
    last = df_vol.iloc[-1]

    state_map = {
        "low": "Low",
        "normal": "Normal",
        "high": "High",
    }

    return {
        "state": state_map.get(last["vol_state"], "Undefined"),
        "percentile": int(last["vol_pct"] * 100),
        "window": "Rolling volatility",
    }

# ========================= DRAW-DOWN
def compute_drawdown(df):
    data = df.copy()

    data["cum_max"] = data["price"].cummax()
    data["drawdown"] = data["price"] / data["cum_max"] - 1

    return data

def drawdown_stats(df_dd):
    drawdown = df_dd["drawdown"]

    max_dd = drawdown.min()

    # Recovery time
    peak_idx = df_dd["price"].idxmax()
    trough_idx = drawdown.idxmin()

    recovery_time = None
    if trough_idx < len(df_dd) - 1:
        post_trough = df_dd.iloc[trough_idx + 1 :]
        recovered = post_trough[post_trough["price"] >= df_dd.loc[peak_idx, "price"]]
        if not recovered.empty:
            recovery_time = (recovered.index[0] - trough_idx)

    return {
        "max_drawdown": max_dd,
        "recovery_time": recovery_time,
    }

def interpret_drawdown(df_dd):
    current_dd = df_dd["drawdown"].iloc[-1]

    if current_dd > -0.1:
        state = "Shallow pullback"
    elif current_dd > -0.3:
        state = "Moderate correction"
    elif current_dd > -0.5:
        state = "Deep drawdown"
    else:
        state = "Extreme drawdown"

    return {
        "current_dd": current_dd,
        "state": state,
    }

import plotly.graph_objects as go

def plot_drawdown(df_dd):
    fig = go.Figure()

    # Drawdown area
    fig.add_trace(
        go.Scatter(
            x=df_dd["date"],
            y=df_dd["drawdown"],
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#ef4444"),
        )
    )

    # Current point
    fig.add_trace(
        go.Scatter(
            x=[df_dd["date"].iloc[-1]],
            y=[df_dd["drawdown"].iloc[-1]],
            mode="markers",
            marker=dict(size=10, color="#991b1b"),
            name="Current drawdown",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis_tickformat=".0%",
        yaxis_title="Drawdown",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
    )

    return update_layout(fig, title = "📉 Drawdown Analysis")

# =========================== RETURN 
from scipy.stats import skew, kurtosis, norm

def compute_return_distribution(df):
    data = df.copy()

    data["log_return"] = np.log(data["price"]).diff()
    returns = data["log_return"].dropna()

    stats = {
        "mean": returns.mean(),
        "std": returns.std(),
        "skew": skew(returns),
        "kurtosis": kurtosis(returns, fisher=False),  # normal = 3
    }

    return returns, stats

def plot_return_distribution(returns, bins=60, show_normal=True):
    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=bins,
            histnorm="probability density",
            name="BTC returns",
            marker_color="#60A5FA",
            opacity=0.75,
        )
    )

    # Normal distribution overlay
    if show_normal:
        x = np.linspace(returns.min(), returns.max(), 300)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=norm.pdf(x, returns.mean(), returns.std()),
                name="Normal dist (same mean/std)",
                line=dict(color="#ef4444", dash="dash"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Daily log-return",
        yaxis_title="Density",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
    )

    return update_layout(fig, title = "📊 Return Distribution Analysis")

def interpret_return_distribution(stats):
    skew_val = stats["skew"]
    kurt = stats["kurtosis"]

    if kurt > 5:
        tail = "Fat-tailed (extreme moves common)"
    elif kurt > 3:
        tail = "Moderately fat-tailed"
    else:
        tail = "Near-normal"

    if skew_val > 0.5:
        skew_desc = "Positively skewed (strong upside tail)"
    elif skew_val < -0.5:
        skew_desc = "Negatively skewed (dominant downside tail)"
    else:
        skew_desc = "Approximately symmetric"

    return {
        "tail": tail,
        "skew_desc": skew_desc,
        "skew": skew_val,
        "kurtosis": kurt,
    }

# ====================== Volume-Price interpretion
def compute_volume_price_relation(df, window=30):
    data = df.copy()

    data["return"] = data["price"].pct_change()
    data["vol_change"] = data["volume"].pct_change()

    data["rolling_corr"] = (
        data["return"]
        .rolling(window)
        .corr(data["vol_change"])
    )

    clean = data.dropna()

    summary = {
        "mean_corr": clean["rolling_corr"].mean(),
        "latest_corr": clean["rolling_corr"].iloc[-1],
    }

    return clean, summary

def plot_volume_price_scatter(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["vol_change"],
            y=data["return"],
            mode="markers",
            marker=dict(
                size=6,
                color=data["return"],
                colorscale="RdBu",
                showscale=True,
                opacity=0.7,
            ),
            name="Daily moves",
        )
    )

    # Zero lines (quadrants)
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Volume change (%)",
        yaxis_title="Price return (%)",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


def plot_volume_price_corr(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["rolling_corr"],
            mode="lines",
            line=dict(color="#2563EB", width=2),
            name="Rolling corr",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis_title="Correlation",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig

def interpret_volume_price(summary):
    corr = summary["latest_corr"]

    if corr > 0.2:
        regime = "Volume confirms price moves"
    elif corr < -0.2:
        regime = "Volume–price divergence (risk warning)"
    else:
        regime = "Weak volume–price relationship"

    return {
        "latest_corr": corr,
        "regime": regime,
    }
