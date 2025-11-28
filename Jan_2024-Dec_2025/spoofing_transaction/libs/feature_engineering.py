import pandas as pd

def clean_missing_and_duplicates(df: pd.DataFrame, process_with_CatBoost: bool = False):
    """ This step will be appear in feature engineering process fill docstring later """

    # 1. Drop useless columns (greater than 90%) if you didnt use CatBoost
    if process_with_CatBoost:
        drop_cols = ["has_redeem_later", "Loyalty_card"]
        cat_fill_unknown = ["Payment", "exception_type", "link"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    else:
        cat_fill_unknown = ["Payment", "exception_type", "link", "has_redeem_later"]

    # 2. Handle categorical imputations
    for col in cat_fill_unknown:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown" if col != "link" else "NoLink")

    # 3. Mode-based imputations
    mode_cols = ["Channel", "City", "RegionName", "Division", "SiteName"]
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Numerical imputations
    num_median_cols = ["Total_amount", "exception_amount"]
    for col in num_median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 5. CardNo special handling
    if "CardNo" in df.columns:
        df["has_card"] = df["CardNo"].notna().astype(int)

    # 6. DVRStart: keep NaT, don’t impute
    # Many models can handle datetime NA; else convert to timestamp later

    # 7. Verify no full-row duplicates
    df = df.drop_duplicates()

    return df

def timestamp_processor(
        df: pd.DataFrame, 
        main_trans_col: str = "TransDate",
        ts_cols: list[str] = ["TransDate", "DVRStart", "start_shift", "end_shift"]
    ) -> pd.DataFrame:
    """
        Timestamp parsing + feature engineering for fraud detection.
        Extract useful seasonality from transaction timestamp.
        Add relational features between TransDate and shift timestamps.
    """
    # ---------------------------------------------------------
    # 1. Convert all timestamp columns
    # ---------------------------------------------------------
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # ---------------------------------------------------------
    # 2. Extract temporal features from transaction timestamp
    # ---------------------------------------------------------
    df["trans_hour"]       = df[main_trans_col].dt.hour
    df["trans_minute"]     = df[main_trans_col].dt.minute
    df["trans_day"]        = df[main_trans_col].dt.day
    df["trans_month"]      = df[main_trans_col].dt.month
    df["trans_dow"]        = df[main_trans_col].dt.dayofweek
    df["trans_is_weekend"] = df["trans_dow"].isin([5, 6]).astype(int)

    # ---------------------------------------------------------
    # 3. DVR alignment feature (avoid extracting month/day/hour)
    # ---------------------------------------------------------
    df["dvr_lag_sec"] = (df[main_trans_col] - df["DVRStart"]).dt.total_seconds()

    # ---------------------------------------------------------
    # 4. Shift-related behavioral features
    # ---------------------------------------------------------
    df["time_from_shift_start_hr"] = (
        (df[main_trans_col] - df["start_shift"]).dt.total_seconds() / 3600
    )

    df["time_to_shift_end_hr"] = (
        (df["end_shift"] - df[main_trans_col]).dt.total_seconds() / 3600
    )

    # Whether transaction happened outside assigned working shift
    df["is_outside_shift"] = (
        (df[main_trans_col] < df["start_shift"]) |
        (df[main_trans_col] > df["end_shift"])
    ).astype(int)

    # Shift length in hours
    df["shift_length_hr"] = (
        (df["end_shift"] - df["start_shift"]).dt.total_seconds() / 3600
    )

    return df