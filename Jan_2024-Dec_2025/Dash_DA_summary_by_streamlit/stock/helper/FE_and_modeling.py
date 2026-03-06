import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss

def prepare_direction_data(df, horizon=1):
    """
        Create log-return & direction target
        target = 1 if return_{t+h} > 0 else 0
    """
    data = df.copy()
    data["log_price"] = np.log(data["price"])
    data["return"] = data["log_price"].diff()
    data["target"] = (data["return"].shift(-horizon) > 0).astype(int)
    return data.dropna()

def ma_direction_signal(df, short=5, long=20):
    data = df.copy()
    data["ma_short"] = data["price"].rolling(short).mean()
    data["ma_long"] = data["price"].rolling(long).mean()
    data["pred"] = (data["ma_short"] > data["ma_long"]).astype(int)
    return data.dropna()

def ar_direction_signal(returns, horizon=1, lags=3):
    """
    Fit AR on return and predict direction
    """
    model = AutoReg(returns, lags=lags, old_names=False).fit()
    forecast = model.predict(
        start=len(returns),
        end=len(returns) + horizon - 1
    )
    direction = int(forecast.mean() > 0)
    return direction, forecast

def hit_rate(y_true, y_pred):
    return (y_true == y_pred).mean()

def weighted_stationarity_test(data, alpha=0.05, w_adf=0.5, w_kpss=0.5):
    """
        Combine ADF & KPSS using weighted voting.
        Return:
            stationary (bool)
            score (0–1)
            detail dict
    """
    data = data.dropna()

    if len(data) < 20:
        return False, None, {"error": "Data too short"}

    # ADF
    adf_stat, adf_p, *_ = adfuller(data, autolag="AIC")
    adf_stationary = adf_p < alpha

    # KPSS
    kpss_stat, kpss_p, *_ = kpss(data, regression="c", nlags="auto")
    kpss_stationary = kpss_p > alpha

    score = (
        w_adf * int(adf_stationary)
        + w_kpss * int(kpss_stationary)
    )

    stationary = score >= 0.5

    detail = {
        "adf_p": adf_p,
        "kpss_p": kpss_p,
        "score": score
    }

    return stationary, score, detail

# ==========================

def rolling_expected_return(returns, window=20):
    """
        Expected return = rolling mean of past returns
    """
    return returns.rolling(window).mean()

from statsmodels.tsa.ar_model import AutoReg

def ar_expected_return(returns, horizon=1, lags=3):
    """
    Forecast expected return using AR model
    """
    model = AutoReg(returns, lags=lags, old_names=False).fit()

    forecast = model.predict(
        start=len(returns),
        end=len(returns) + horizon - 1
    )

    return forecast.mean(), forecast


# ========================
def rolling_volatility(returns, window=20):
    """
        Rolling standard deviation of returns
    """
    return returns.rolling(window).std()

from arch import arch_model

def garch_volatility(returns, horizon=1):
    """
    Forecast volatility using GARCH(1,1)
    """
    returns = returns.dropna() * 100  # scale for stability

    model = arch_model(
        returns,
        vol="Garch",
        p=1,
        q=1,
        dist="normal"
    )

    res = model.fit(disp="off")

    forecast = res.forecast(horizon=horizon)

    # lấy volatility forecast trung bình
    vol = np.sqrt(
        forecast.variance.values[-1]
    ).mean() / 100

    return vol, forecast
