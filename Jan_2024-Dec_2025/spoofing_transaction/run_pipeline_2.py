""" 
        Since the pipeline 1 is strongly affected by data-drift from 2024
"""

import pandas as pd
import numpy as np
import warnings, os
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")
fpath = os.environ("lastest_version")
cur_df = pd.read_table(fpath)
cur_df_trunc = cur_df[cur_df['TransDate'] >= '2024-01-01'].reset_index(drop=True)

def build_amount_bucket(df):
    df = df.copy()

    amt = df['Total_amount']

    conditions = [
        (amt < 0) & (amt.abs() <= 50),
        (amt < 0) & (amt.abs() > 50),
        (amt >= 0) & (amt <= 50),
        (amt >= 0) & (amt > 50),
    ]

    choices = [
        'refund_small',
        'refund_large',
        'sale_small',
        'sale_large',
    ]

    df['AmountBucket'] = np.select(
        conditions,
        choices,
        default='unknown'
    )

    return df

# =========================================
# 1. TIME SPLIT
# =========================================
def time_split_new(df):
    df = df.copy()
    df['TransDate'] = pd.to_datetime(df['TransDate'])

    train = df[
        (df['TransDate'] >= '2024-01-01') &
        (df['TransDate'] <= '2025-09-30')
    ]

    valid = df[
        (df['TransDate'] >= '2025-10-01') &
        (df['TransDate'] <= '2025-11-15')
    ]

    test = df[
        (df['TransDate'] >= '2025-11-16') &
        (df['TransDate'] <= '2025-12-31')
    ]

    return train, valid, test

df = cur_df_trunc.copy()
train, valid, test = time_split_new(df)

# =========================================
# 3. FEATURE ENGINEERING (FOR CAT-BOOST)
# =========================================
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

    # 6. Timestamp missing flags (DO NOT IMPUTE TIMESTAMPS)
    ts_missing_cols = ["DVRStart", "start_shift", "end_shift"]
    for col in ts_missing_cols:
        if col in df.columns:
            df[f"is_{col}_missing"] = df[col].isna().astype(int)
            
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
    df['DVRStart'] = pd.to_datetime(df['DVRStart'])
    
    # Sort datetime w.r.t non-null columns timestamps
    df = df.sort_values("TransDate").reset_index(drop=True)

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

    # 5. Impute timestamp-derived numerical features to avoid NaN (CatBoost cannot accept NaN)
    df["dvr_lag_sec"] = df["dvr_lag_sec"].fillna(0)
    if "lag_DVR" in df.columns:
        df["lag_DVR"] = df["lag_DVR"].fillna(0)

    # Shift-related
    shift_cols_zero = ["time_from_shift_start_hr", "time_to_shift_end_hr"]
    for col in shift_cols_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "shift_length_hr" in df.columns:
        df["shift_length_hr"] = df["shift_length_hr"].fillna(-1)

    return df

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Add behavioral & transaction-level fraud features.
    """
    
    # --- 1. Value-based anomalies ---
    df["abs_amount"] = df["Total_amount"].abs()
    df["is_refund"] = (df["Total_amount"] < 0).astype(int)
    df["large_amount"] = (df["abs_amount"] > df["abs_amount"].median()).astype(int)

    # --- 2. Exception-based ---
    # Count number of exception items (split by ';')
    df["exception_count"] = df["Desc"].fillna("").apply(lambda x: len(x.split(";")) if x != "" else 0)

    # Refund text detection (string-based fraud signals)
    df["contains_discount"] = df["Desc"].str.contains("Discount", case=False, na=False).astype(int)
    df["contains_return"]   = df["Desc"].str.contains("Return", case=False, na=False).astype(int)

    # --- 3. Time behavior already extracted but add velocity features ---
    # Ensure timestamps are datetime (safe check)
    df["TransDate"] = pd.to_datetime(df["TransDate"], errors="coerce")
    df["DVRStart"]  = pd.to_datetime(df["DVRStart"], errors="coerce")
    df["start_shift"] = pd.to_datetime(df["start_shift"], errors="coerce")
    df["end_shift"]   = pd.to_datetime(df["end_shift"], errors="coerce")
    # lag
    df["lag_DVR"] = (df["TransDate"] - df["DVRStart"]).dt.total_seconds()

    df["shift_fatigue"] = (
        (df["end_shift"] - df["start_shift"]).dt.total_seconds() / 3600
    ).fillna(0)

    df["near_shift_end"] = (
        df["time_to_shift_end_hr"] < 1
    ).astype(int)

    # refund patterns
    df["abs_amount"] = df["Total_amount"].abs()
    df["is_refund"] = (df["Total_amount"] < 0).astype(int)
    df["large_amount"] = (df["abs_amount"] > df["abs_amount"].median()).astype(int)

    # text-like exception signals
    df["exception_count"] = df["exception_type"].apply(
        lambda x: 0 if isinstance(x, float) else len(str(x).split(","))
    )

    df["contains_discount"] = df["Desc"].str.contains("discount", case=False, na=False).astype(int)
    df["contains_return"]   = df["Desc"].str.contains("return",   case=False, na=False).astype(int)

    return df

def categorical_encoding_for_catboost(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
        Prepare categorical features for CatBoost: 
        - Ensure dtype = string
        - Fill missing categories
        - Return list of categorical columns for CatBoost
    """

    cat_cols = [
        "Channel", "City", "RegionName", "Division", "SiteName",
        "Payment", "exception_type", "link"
    ]

    # CardNo is categorical but high cardinality → keep as string
    if "CardNo" in df.columns:
        df["CardNo"] = df["CardNo"].astype(str)

    # Fill missing values
    for col in cat_cols:
        if col in df.columns:
            fill_value = "Unknown" if col != "link" else "NoLink"
            df[col] = df[col].fillna(fill_value).astype(str)

    # Convert numeric categories (Channel, Payment) to string
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df, cat_cols

def label_imbalance_handler(y: pd.Series) -> dict:
    """
        Compute class weights for imbalanced fraud labels.
        CatBoost accepts class_weights=[w0, w1].
    """

    fraud_rate = y.mean()  # proportion of fraud

    # weight inversely proportional to frequency
    w_fraud = 1 / fraud_rate
    w_non_fraud = 1 / (1 - fraud_rate)

    weights = {
        "class_weights": [w_non_fraud, w_fraud],
        "fraud_rate": fraud_rate
    }

    return weights

def finalize_timestamp_columns(df, ts_cols=["TransDate", "DVRStart", "start_shift", "end_shift"]):
    """
    Convert all timestamp columns into numerical format (Unix seconds).
    CatBoost cannot accept raw datetime64 or NaT.
    """
    for col in ts_cols:
        if col in df.columns:
            
            # Convert to int64 (ns), divide to get seconds
            df[col] = df[col].astype("int64") // 1_000_000_000
            
            # NaT becomes very large negative integer → replace with sentinel
            df[col] = df[col].replace(-9223372036854775808, -1)
    
    return df

def force_cat_to_string(df, cat_cols):
    for c in cat_cols:
        df[c] = df[c].astype(str)
    return df

def auto_detect_cat_cols(df: pd.DataFrame):
    """
    Auto-detect categorical columns:
    - object/string columns
    - OR integer columns with low cardinality
    """

    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    # Add numerical columns with meaningfully low cardinality
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if df[col].nunique() < 30 and df[col].nunique() > 1:
            cat_cols.append(col)

    # Remove columns that CatBoost should not treat as categorical
    remove = ["Total_amount", "exception_amount",
              "dvr_lag_sec", "lag_DVR",
              "time_to_shift_end_hr", "time_from_shift_start_hr",
              "shift_length_hr"]

    return [c for c in cat_cols if c not in remove]

# -------------------------------------------------------------------------
# UTIL: Keep Cat Features as String
# -------------------------------------------------------------------------
def force_cat_to_string(df, cat_cols):
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# -------------------------------------------------------------------------
# UTIL: Apply Entire Feature Engineering Pipeline
# -------------------------------------------------------------------------
def apply_feature_engineering(df):
    df = clean_missing_and_duplicates(df, process_with_CatBoost=True)
    df = timestamp_processor(df)
    df = finalize_timestamp_columns(df)
    df = build_transaction_features(df)
    return df

from catboost import CatBoostClassifier, Pool

drop_cols = ['Label', 'T_PACID', 'T_OperatorID']  

X_train = train.copy().drop(columns = drop_cols)
X_val = valid.copy().drop(columns = drop_cols)
X_test = test.copy().drop(columns = drop_cols)

y_train = train.copy()['Label']
y_val = valid.copy()['Label']
y_test = test.copy()['Label']

cat_cols = auto_detect_cat_cols(X_train)

X_train = apply_feature_engineering(X_train.copy())
X_val = apply_feature_engineering(X_val.copy())
X_test = apply_feature_engineering(X_test.copy())

X_train = force_cat_to_string(X_train, cat_cols)
X_val   = force_cat_to_string(X_val, cat_cols)
X_test  = force_cat_to_string(X_test, cat_cols)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

model = CatBoostClassifier(
    loss_function = "Logloss",
    eval_metric = "AUC",  # eval_metric = "AUC"
    auto_class_weights = "Balanced",
    depth = 8,
    learning_rate = 0.05,
    iterations = 2000,
    random_seed = 42
)

model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=200,
    use_best_model=True
)

# ====================================== 
# 
# ======================================
pred_proba = model.predict_proba(test_pool)[:, 1]
pred_label = (pred_proba > 0.9).astype(int)

print("proba threshold: 0.9")
print("AUC:", roc_auc_score(y_test, pred_proba))
print(confusion_matrix(y_test, pred_label))
print(classification_report(y_test, pred_label))

feat_imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending = False)

feat_imp[feat_imp['Importance'] > 0].set_index('Feature').plot(kind = 'barh', figsize=(20, 8))

# =========================================
# 2. BAYESIAN LOG-ODDS PROFILER
# =========================================
class BayesianLogOddsProfiler:
    def __init__(
        self,
        cols,
        min_count=30,
        prior_strength=20,
        cap=2.5
    ):
        self.cols = cols
        self.min_count = min_count
        self.prior_strength = prior_strength
        self.cap = cap
        self.tables = {}

    def fit(self, df):
        # ---- Global prior ----
        self.base_rate = df['Label'].mean()
        self.base_logit = np.log(
            self.base_rate / (1 - self.base_rate)
        )

        for col in self.cols:
            g = df.groupby(col)['Label'].agg(['count', 'sum'])

            # Bayesian smoothing
            g['p'] = (
                g['sum'] + self.prior_strength * self.base_rate
            ) / (
                g['count'] + self.prior_strength
            )

            # Reliability filter
            g = g[g['count'] >= self.min_count]

            # Log-odds transform
            g['logit'] = np.log(g['p'] / (1 - g['p']))

            # Delta from global prior
            g['delta'] = g['logit'] - self.base_logit

            # Cap to prevent domination
            g['delta'] = g['delta'].clip(
                -self.cap, self.cap
            )

            self.tables[col] = g['delta'].to_dict()

        return self

    def transform(self, df):
        score = np.full(len(df), self.base_logit)

        for col in self.cols:
            mapping = self.tables.get(col, {})
            score += df[col].map(mapping).fillna(0).values

        return score

# =========================================
# 3. FEATURE ENGINEERING (FOR BAYESIAN)
# =========================================
def build_features(df):
    df = df.copy()
    df['TransDate'] = pd.to_datetime(df['TransDate'])
    return df


# =========================================
# 4. SCORING + NORMALIZATION
# =========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================
# 5. EVALUATION (TIER-BASED)
# =========================================
def evaluate_bayes(df, raw_score_col, name, tier_pct=1):
    y = df['Label'].values
    s = df[raw_score_col].values

    auc = roc_auc_score(y, s)
    ks = ks_2samp(s[y == 1], s[y == 0]).statistic

    # ---- Tier B threshold (Top X%) ----
    threshold = np.percentile(s, 100 - tier_pct)
    y_pred = (s >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    report = classification_report(
        y, y_pred, digits=4, zero_division=0
    )

    lift = y[y_pred == 1].mean() / y.mean()

    print(f"\n=== {name} | BAYES STANDALONE ===")
    print(f"AUC = {auc:.4f}")
    print(f"KS  = {ks:.4f}")
    print(f"Top {tier_pct}% lift = {lift:.2f}x")
    print(f"Tier B threshold (raw log-odds) = {threshold:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

# =========================================
# 6. FULL PIPELINE
# =========================================
# ---- Minimal FE ----
train = build_features(train)
valid = build_features(valid)
test = build_features(test)

train = build_amount_bucket(train)
valid = build_amount_bucket(valid)
test  = build_amount_bucket(test)

# ---- Train Bayes (TRAIN ONLY) ----
bayes = BayesianLogOddsProfiler(
    cols=['SiteName', 'Payment', 'Channel', 'AmountBucket'],
    min_count=40,        # amount bucket cần count cao hơn
    prior_strength=20,
    cap=2.5
).fit(train)

# ---- Score ----
for d in [train, valid, test]:
    d['bayes_raw_score'] = bayes.transform(d)
    d['bayes_rank_score'] = sigmoid(d['bayes_raw_score'])

# ---- Evaluation ----
for name, d in [
    ('VALID (01-Oct-2025 to 15-Nov-2025)', valid),
    ('TEST (After 16-Nov-2025)', test),
]:
    evaluate_bayes(d, 'bayes_raw_score', name, tier_pct=1)

# ====================================== Get final result ======================================
pred_proba = model.predict_proba(test_pool)[:, 1]
pred_label = (pred_proba > 0.9).astype(int)
test['CatBoost_score'] = pred_label
test[['TransID', 'TransDate', 'bayes_raw_score', 'bayes_rank_score', 'CatBoost_score', 'Label']]