
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load the input data
raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

# Measure raw row count and remove exact duplicates first
raw_rows = len(raw)
dedup = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(dedup)

# Record cleaning checks before missing values are filled
missing_distance_before = dedup["distance_km"].isna().sum()

sentinel_weight_rows = (dedup["parcel_weight_kg"] > 100).sum()

missing_weight_before = (
    dedup["parcel_weight_kg"].isna().sum()
)

missing_declared_value_before = (
    dedup["declared_value"].isna().sum()
)

missing_payment_mode_before = (
    dedup["payment_mode"].isna().sum()
)

service_type_spellings_before = (
    dedup["service_type"].nunique()
)

# Run the shared cleaning function
clean, medians = clean_shipments(raw)

service_type_spellings_after = clean["service_type"].nunique()

# Attach hub data and create features
modelling = build_features(clean, hubs)

rows_after_hub_merge = len(modelling)

# Create the target: delayed when delivery exceeds promised hours
modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)

# Check the required features and target for missing values
required_columns = FEATURES + ["delivery_hours", "is_delayed"]

nulls_in_features_and_target = (
    modelling[required_columns].isna().sum().sum()
)

# Calculate delivery-time quartiles and outlier bounds
hours_q1 = modelling["delivery_hours"].quantile(0.25)
hours_q3 = modelling["delivery_hours"].quantile(0.75)

iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - 1.5 * iqr
hours_upper_bound = hours_q3 + 1.5 * iqr

hours_outlier_count = (
    (modelling["delivery_hours"] < hours_lower_bound)
    | (modelling["delivery_hours"] > hours_upper_bound)
).sum()

# Delay summary
delayed_count = int(modelling["is_delayed"].sum())
delay_rate = modelling["is_delayed"].mean()

# Declared value statistics use the median-filled values
declared_value_mean_inr = modelling["declared_value_inr"].mean()
declared_value_median_inr = modelling["declared_value_inr"].median()

log_declared_value_mean = modelling["log_declared_value"].mean()
log_declared_value_median = modelling["log_declared_value"].median()

# Save the modelling table
modelling.to_csv(DATA_DIR / "modelling_table.csv", index=False)

# Save Part A checks
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df["value"] = checks_df["value"].map(
    lambda value: round(float(value), 4)
)
checks_df.to_csv(OUTPUT_DIR / "part_a_checks.csv", index=False)

# Create the distance bands required by the assignment
modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

# Calculate delay evidence by service type, hub tier and distance band
group_rows = []

group_definitions = [
    ("service_type", "service_type"),
    ("hub_tier", "hub_tier"),
    ("distance_band", "distance_band"),
]

for dimension, column in group_definitions:
    grouped = modelling.groupby(
        column, observed=False, dropna=False
    )

    for group_name, group_df in grouped:
        if pd.isna(group_name):
            continue

        group_rows.append({
            "dimension": dimension,
            "group": str(group_name),
            "shipments": len(group_df),
            "delayed": int(group_df["is_delayed"].sum()),
            "delay_rate": round(
                group_df["is_delayed"].mean(), 4
            ),
            "mean_hours": round(
                group_df["delivery_hours"].mean(), 4
            ),
        })

delay_by_group = pd.DataFrame(
    group_rows,
    columns=[
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
        "mean_hours",
    ],
)

delay_by_group.to_csv(
    OUTPUT_DIR / "part_a_delay_by_group.csv",
    index=False,
)

# Save correlations for numeric columns
numeric_columns = modelling.select_dtypes(
    include=[np.number]
)

correlations = numeric_columns.corr()
correlations.to_csv(
    OUTPUT_DIR / "part_a_correlations.csv",
    index_label="column",
    float_format="%.4f",
)

print("Part A completed.")
print("Modelling table rows:", len(modelling))
print("Delayed shipments:", delayed_count)
print("Delay rate:", round(delay_rate, 4))
print("Missing values in required columns:", nulls_in_features_and_target)
print("Generated:")
print(" - data/modelling_table.csv")
print(" - outputs/part_a_checks.csv")
print(" - outputs/part_a_delay_by_group.csv")
print(" - outputs/part_a_correlations.csv")
