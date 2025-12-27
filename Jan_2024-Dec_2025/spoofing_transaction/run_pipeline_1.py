import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from libs.Feature_Engineering.fe_v1 import *
from libs.EDA.eda_v1 import *

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -------------------------------------------------------------------------
# 0) LOGGING SETUP
# -------------------------------------------------------------------------
LOGFILE = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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

# -------------------------------------------------------------------------
# 1) LOAD RAW DATA
# -------------------------------------------------------------------------
fpath = "data/your_data.txt"
logger.info(f"Loading dataset from {fpath}")
df = pd.read_table(fpath)

logger.info(f"Loaded dataset shape: {df.shape}")

# -------------------------------------------------------------------------
# 2) SPLIT DATA
# -------------------------------------------------------------------------
logger.info("Splitting into train/val/test...")

def select_rule_split(df: pd.DataFrame, data_drift: bool = True, test_rate: float = 0.2):
    if data_drift:
        test_size = int(test_rate * len(df))
        df_train = df.iloc[: len(df) - test_size]
        df_test = df.iloc[test_size: ]

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            df_train.drop(columns=["Label"]), df_train['Label'], test_size=0.40, random_state=42, stratify=df_train['Label']
        )

        X_test_raw = df_test.drop(columns=["Label"])
        y_test = df_test['Label']
    else:
        X = df.drop(columns=["Label"])
        y = df["Label"]

        X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
            X, y, test_size = 0.30, random_state = 42, stratify=y
        )

        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp_raw, y_temp, test_size = 0.50, random_state = 42, stratify = y_temp
        )

    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test

X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = select_rule_split(df)

logger.info(f"Train={X_train_raw.shape}  Val={X_val_raw.shape}  Test={X_test_raw.shape}")

# -------------------------------------------------------------------------
# 3) FEATURE ENGINEERING
# -------------------------------------------------------------------------
logger.info("Applying feature engineering...")
X_train = apply_feature_engineering(X_train_raw.copy())
X_val   = apply_feature_engineering(X_val_raw.copy())
X_test  = apply_feature_engineering(X_test_raw.copy())

logger.info("FE completed.")

# -------------------------------------------------------------------------
# 4) CATEGORICAL COLS
# -------------------------------------------------------------------------
logger.info("Detecting categorical columns...")
cat_cols = auto_detect_cat_cols(X_train)

X_train = force_cat_to_string(X_train, cat_cols)
X_val   = force_cat_to_string(X_val, cat_cols)
X_test  = force_cat_to_string(X_test, cat_cols)

logger.info(f"Detected cat cols: {cat_cols}")

# -------------------------------------------------------------------------
# 5) CATBOOST POOLS
# -------------------------------------------------------------------------
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

# -------------------------------------------------------------------------
# 6) TRAIN MODEL
# -------------------------------------------------------------------------
logger.info("Training CatBoost...")

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    depth=8,
    learning_rate=0.05,
    iterations=2000,
    random_seed=42
)

model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=200,
    use_best_model=True
)

# SAVE BEST MODEL
SAVE_PATH = "models/best_model.cbm"
os.makedirs("models", exist_ok=True)
model.save_model(SAVE_PATH)

logger.info(f"Best model saved to {SAVE_PATH}")

# -------------------------------------------------------------------------
# 7) TEST EVALUATION
# -------------------------------------------------------------------------
logger.info("Evaluating on TEST set...")

proba_threshval = 0.9 # if using time-drift
pred_proba = model.predict_proba(test_pool)[:, 1]
pred_label = (pred_proba > proba_threshval).astype(int)

auc = roc_auc_score(y_test, pred_proba)
cm  = confusion_matrix(y_test, pred_label)
clsr = classification_report(y_test, pred_label)

logger.info(f"AUC: {auc}")
logger.info(f"Confusion Matrix:\n{cm}")
logger.info(f"Classification Report:\n{clsr}")


# -------------------------------------------------------------------------
# 8) FEATURE IMPORTANCE
# -------------------------------------------------------------------------
feat_imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

logger.info("Top 20 Feature Importances:\n" + str(feat_imp.head(20)))

feat_imp.to_csv("models/feature_importances.csv", index=False)
logger.info("Feature importances saved.")

# -------------------------------------------------------------------------
# 9) INFERENCE PIPELINE (LOAD MODEL + NEW DATASET)
# -------------------------------------------------------------------------
def load_and_predict(new_path, model_path="models/best_model.cbm"):
    """
        Load new raw dataset, apply the exact FE pipeline,
        then predict using trained CatBoost model.
    """

    logger.info(f"[INFERENCE] Loading new data from: {new_path}")
    new_df = pd.read_table(new_path)
    logger.info(f"[INFERENCE] Shape: {new_df.shape}")

    # Apply same FE flow
    new_df_fe = apply_feature_engineering(new_df.copy())

    # Ensure cat columns exist + are strings
    missing_cats = [c for c in cat_cols if c not in new_df_fe.columns]
    if missing_cats:
        logger.warning(f"Missing categorical columns in new dataset: {missing_cats}")

    for c in cat_cols:
        if c in new_df_fe.columns:
            new_df_fe[c] = new_df_fe[c].astype(str)

    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)
    logger.info(f"[INFERENCE] Model loaded.")

    # Predict
    pool = Pool(new_df_fe, cat_features=cat_cols)
    proba = model.predict_proba(pool)[:, 1]

    logger.info("[INFERENCE] Predictions completed.")
    return pd.DataFrame({
        "prediction_proba": proba
    })

# Predict new data
pred_df = load_and_predict("data/new_unseen_data.txt")
print(pred_df.head())