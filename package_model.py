import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import (
    FEATURES,
    clean_shipments
)


# Create models and outputs folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]


# --------------------------------------------------
# 2. RECREATE PART C SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------
# 3. READ FINAL MODEL CONFIGURATION
# --------------------------------------------------

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])

max_depth = (
    None if pd.isna(config["max_depth"])
    else int(config["max_depth"])
)

min_samples_leaf = (
    1 if pd.isna(config["min_samples_leaf"])
    else int(config["min_samples_leaf"])
)

threshold = float(config["threshold"])


# --------------------------------------------------
# 4. CREATE FINAL PIPELINE
# --------------------------------------------------

final_pipeline = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])


# Train using training data only
final_pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 5. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    final_pipeline,
    "models/delay_model.pkl"
)


# Save threshold
pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    "models/delay_threshold.csv",
    index=False
)


# Save feature order
pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    "models/feature_order.csv",
    index=False
)


# Save imputation medians
raw_shipments = pd.read_csv("data/raw_shipments.csv")

_, medians = clean_shipments(raw_shipments)

impute_values = pd.DataFrame([
    {"column": column, "median": value}
    for column, value in medians.items()
])

impute_values.to_csv(
    "models/impute_values.csv",
    index=False
)


# --------------------------------------------------
# 6. RELOAD AND CHECK MODEL
# --------------------------------------------------

loaded_model = joblib.load("models/delay_model.pkl")

original_probabilities = final_pipeline.predict_proba(X_test)[:, 1]

loaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

max_difference = np.max(
    np.abs(original_probabilities - loaded_probabilities)
)


# Save reload check
reload_check = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_difference
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("Model packaged successfully!")

print("\nFinal Configuration")
print("Selector K:", selector_k)
print("Estimators:", n_estimators)
print("Max depth:", max_depth)
print("Minimum samples leaf:", min_samples_leaf)
print("Threshold:", threshold)

print("\nReload Check")
print(reload_check.to_string(index=False))