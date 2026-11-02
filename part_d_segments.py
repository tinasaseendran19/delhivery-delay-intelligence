import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


# Create outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)


# Load modelling dataset
df = pd.read_csv("data/modelling_table.csv")


# --------------------------------------------------
# 1. CREATE HUB PROFILE
# --------------------------------------------------

hub_profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

# Sort hubs
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

# Save hub profile
hub_profile.to_csv(
    "outputs/hub_profile.csv",
    index=False
)

print("Hub profile created.")
print("Total hubs:", len(hub_profile))


# --------------------------------------------------
# 2. SCALE HUB FEATURES
# --------------------------------------------------

profile_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    hub_profile[profile_features]
)


# --------------------------------------------------
# 3. K-MEANS SEPARATION EVIDENCE
# --------------------------------------------------

distance_matrix = squareform(pdist(scaled_data))

separation_results = []

for k in [2, 3, 4, 5]:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(scaled_data)

    within_distances = []
    across_distances = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            if labels[i] == labels[j]:
                within_distances.append(distance_matrix[i, j])
            else:
                across_distances.append(distance_matrix[i, j])

    ratio = (
        np.mean(across_distances) /
        np.mean(within_distances)
    )

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True
    )

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(map(str, cluster_sizes))
    })


separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False
)

print("\nK-Means Separation Results")
print(separation_df.round(4).to_string(index=False))


# --------------------------------------------------
# 4. FINAL K-MEANS WITH K = 3
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(scaled_data)


# Calculate delay rate for each cluster
cluster_delay = hub_profile.groupby("cluster")[
    "delay_rate"
].mean().sort_values()


# Assign names based on delay rate
segment_names = {
    cluster_delay.index[0]: "Standard",
    cluster_delay.index[1]: "Watch",
    cluster_delay.index[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(segment_names)


# --------------------------------------------------
# 5. ADD HUB CITY AND TIER
# --------------------------------------------------

hubs = pd.read_csv("data/hubs.csv")

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False
)


# --------------------------------------------------
# 6. CREATE SEGMENT PROFILE
# --------------------------------------------------

segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

# Keep the required order
segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values("segment")

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False
)


# --------------------------------------------------
# 7. FIND TIER AND SEGMENT MISMATCHES
# --------------------------------------------------

expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(expected_segment)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
]

mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nSegment Profile")
print(segment_profile.round(4).to_string(index=False))

print("\nHub Segments")
print(hub_segments.to_string(index=False))

print("\nTier vs Segment Mismatches")
print(mismatches.to_string(index=False))

print("\nPart D Hub Segmentation Completed!")