"""
train.py
--------
Trains the production model on the reference/training data. XGBoost is used
because it's the realistic default for tabular fraud-style problems and
gives us permutation importance that isn't a straight line (unlike plain
logistic regression), which makes root_cause.py's weighting meaningful.
"""

import pandas as pd
from xgboost import XGBClassifier

from .scenarios import N_FEATURES_CONT, N_FEATURES_CAT

FEATURE_COLS = N_FEATURES_CONT + N_FEATURES_CAT + ["device_type_enc", "merchant_category_enc"]
NUMERIC_FEATURE_COLS = N_FEATURES_CONT


def encode(df: pd.DataFrame, device_map=None, merchant_map=None):
    df = df.copy()
    if device_map is None:
        device_map = {c: i for i, c in enumerate(sorted(df["device_type"].unique()))}
    if merchant_map is None:
        merchant_map = {c: i for i, c in enumerate(sorted(df["merchant_category"].unique()))}
    df["device_type_enc"] = df["device_type"].map(device_map).fillna(-1)
    df["merchant_category_enc"] = df["merchant_category"].map(merchant_map).fillna(-1)
    return df, device_map, merchant_map


MODEL_FEATURE_COLS = N_FEATURES_CONT + ["device_type_enc", "merchant_category_enc"]

# Canonical mapping between the "raw" feature names used everywhere in
# drift reporting (transaction_amount, device_type, ...) and the actual
# model-input column names (device_type -> device_type_enc, ...). Every
# module that needs to join a raw-feature-keyed table (drift scores) with
# a model-column-keyed object (permutation importance) MUST go through
# this mapping rather than assuming the names already match -- this is
# exactly the bug that silently zeroed out categorical root-cause scores
# in the first version of this project.
RAW_TO_MODEL_COL = {
    "transaction_amount": "transaction_amount",
    "customer_age": "customer_age",
    "account_tenure_days": "account_tenure_days",
    "hour_of_day": "hour_of_day",
    "device_type": "device_type_enc",
    "merchant_category": "merchant_category_enc",
}


def train_model(train_df: pd.DataFrame):
    df, device_map, merchant_map = encode(train_df)
    X = df[MODEL_FEATURE_COLS]
    y = df["is_fraud"]
    model = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    return model, device_map, merchant_map
