import pandas as pd
import joblib

from src.shipment_prep import (
    FEATURES,
    clean_shipments,
    build_features
)


# --------------------------------------------------
# 1. LOAD FILES
# --------------------------------------------------

new_shipments = pd.read_csv("data/new_shipments.csv")

hubs = pd.read_csv("data/hubs.csv")

model = joblib.load("models/delay_model.pkl")

threshold = pd.read_csv(
    "models/delay_threshold.csv"
)["threshold"].iloc[0]

feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()

impute_values = pd.read_csv(
    "models/impute_values.csv"
)

hub_segments = pd.read_csv(
    "outputs/hub_segments.csv"
)


# --------------------------------------------------
# 2. LOAD STORED MEDIANS
# --------------------------------------------------

medians = dict(
    zip(
        impute_values["column"],
        impute_values["median"]
    )
)


# --------------------------------------------------
# 3. CLEAN NEW SHIPMENTS
# --------------------------------------------------

clean_data, _ = clean_shipments(
    new_shipments,
    medians=medians
)


# --------------------------------------------------
# 4. BUILD FEATURES
# --------------------------------------------------

new_features = build_features(
    clean_data,
    hubs
)


# Arrange columns in the saved feature order
X_new = new_features[feature_order]


# --------------------------------------------------
# 5. PREDICT DELAY PROBABILITY
# --------------------------------------------------

probabilities = model.predict_proba(X_new)[:, 1]

# Round probabilities to 4 decimal places
rounded_probabilities = probabilities.round(4)


# --------------------------------------------------
# 6. FLAG SHIPMENTS
# --------------------------------------------------

flagged = (
    probabilities >= threshold
).astype(int)


# --------------------------------------------------
# 7. CREATE PREDICTION RESULTS
# --------------------------------------------------

results = pd.DataFrame({
    "shipment_id": clean_data["shipment_id"],
    "hub_id": clean_data["hub_id"],
    "delay_probability": rounded_probabilities,
    "flagged": flagged
})


# --------------------------------------------------
# 8. JOIN HUB SEGMENTS
# --------------------------------------------------

results = results.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left"
)


# --------------------------------------------------
# 9. RECOMMEND OPERATIONS ACTION
# --------------------------------------------------

def get_action(row):

    if row["flagged"] == 1:

        if row["segment"] == "Escalate":
            return "Escalate to hub manager"

        else:
            return "Proactive customer call"

    return "No action"


results["action"] = results.apply(
    get_action,
    axis=1
)


# --------------------------------------------------
# 10. SAVE FINAL RESULTS
# --------------------------------------------------

results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]

results.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("New shipment predictions completed!")

print("\nTotal shipments:", len(results))

print("Flagged shipments:", int(results["flagged"].sum()))

print(
    "Hub manager escalations:",
    int(
        (
            (results["flagged"] == 1) &
            (results["segment"] == "Escalate")
        ).sum()
    )
)

print("\nAction Summary")
print(results["action"].value_counts())

print("\nFirst 10 Predictions")
print(results.head(10).to_string(index=False))

print("\nResults saved to outputs/new_shipment_actions.csv")