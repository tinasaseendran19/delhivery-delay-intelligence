
import pandas as pd

from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import NOISE_COLUMNS
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import ast


# Load the cleaned dataset from Part A
df = pd.read_csv("data/modelling_table.csv")

# Select input features
X = df[FEATURES]

# Select the target column
y = df["is_delayed"]


# Split the data into training and temporary data
# Stratify keeps the delayed and non-delayed ratio similar
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# Split temporary data into validation and test data
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# Display dataset sizes
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# Display delayed parcel counts
print("\nDelayed parcels:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))



# Predict that every validation parcel is on time
baseline_predictions = [0] * len(y_val)

# Calculate accuracy
baseline_accuracy = accuracy_score(
    y_val,
    baseline_predictions
)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_val,
    baseline_predictions,
    labels=[0, 1]
).ravel()

print("\nBlind Baseline Results")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
print("Accuracy:", round(baseline_accuracy, 4))



# Create Logistic Regression model
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train using training data only
logistic_model.fit(X_train, y_train)

# Predict on validation data
logistic_predictions = logistic_model.predict(X_val)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_val,
    logistic_predictions,
    labels=[0, 1]
).ravel()

print("\nLogistic Regression Results")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("Accuracy:", round(accuracy_score(y_val, logistic_predictions), 4))
print("Precision:", round(precision_score(y_val, logistic_predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, logistic_predictions, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_val, logistic_predictions, zero_division=0), 4))


# Depths to test
depths = [2, 3, 4, 5, 6, 8, 10]

best_depth = None
best_f1 = -1

print("\nDecision Tree Depth Results")

for depth in depths:

    # Create Decision Tree model
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    # Train the model
    tree_model.fit(X_train, y_train)

    # Predict validation data
    predictions = tree_model.predict(X_val)

    # Calculate F1 score
    score = f1_score(y_val, predictions)

    print("Depth:", depth, "F1 Score:", round(score, 4))

    # Select the best depth
    if score > best_f1:
        best_f1 = score
        best_depth = depth

print("\nBest depth:", best_depth)
print("Best F1 Score:", round(best_f1, 4))


# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict validation data
rf_predictions = rf_model.predict(X_val)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_val,
    rf_predictions,
    labels=[0, 1]
).ravel()

print("\nRandom Forest Results")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("Accuracy:", round(accuracy_score(y_val, rf_predictions), 4))
print("Precision:", round(precision_score(y_val, rf_predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, rf_predictions, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_val, rf_predictions, zero_division=0), 4))


# Train Random Forest using all training features
feature_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

feature_rf.fit(X_train, y_train)

# Calculate F-scores for all features
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)

# Store feature scores
feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector.scores_,
    "rf_importance": feature_rf.feature_importances_
})

# Rank features based on F-score
feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Identify noise features
feature_scores["is_noise"] = feature_scores["feature"].isin(
    NOISE_COLUMNS
)

# Sort by F-score rank
feature_scores = feature_scores.sort_values("f_rank")

# Save results
feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)

print("\nTop Feature Scores")
print(feature_scores.head(10).to_string(index=False))

print("\nFeature scores saved successfully.")

# Test different numbers of selected features
k_values = [5, 7, 10, 12, 17]

k_results = []

print("\nFeature Selection Results")

for k in k_values:

    # Select the top K features
    k_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Train the model
    k_model.fit(X_train, y_train)

    # Predict validation data
    k_predictions = k_model.predict(X_val)

    # Calculate F1 score
    k_f1 = f1_score(y_val, k_predictions)

    k_results.append({
        "k": k,
        "val_f1": k_f1
    })

    print("K:", k, "Validation F1:", round(k_f1, 4))


# Save results
k_results_df = pd.DataFrame(k_results)

k_results_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)

print("\nFeature selection results saved successfully.")


# Select the top 7 features using training data only
selector_7 = SelectKBest(score_func=f_classif, k=7)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

# Prepare combined training and validation data for cross-validation
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

X_train_val_7 = selector_7.transform(X_train_val)


# Random Forest parameters to test
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid_search.fit(X_train_val_7, y_train_val)

# Evaluate best parameters on validation data
grid_model = RandomForestClassifier(
    random_state=42,
    **grid_search.best_params_
)

grid_model.fit(X_train_7, y_train)
grid_predictions = grid_model.predict(X_val_7)

grid_f1 = f1_score(y_val, grid_predictions)


# Randomized Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_val_7, y_train_val)

# Evaluate best parameters on validation data
random_model = RandomForestClassifier(
    random_state=42,
    **random_search.best_params_
)

random_model.fit(X_train_7, y_train)
random_predictions = random_model.predict(X_val_7)

random_f1 = f1_score(y_val, random_predictions)


# Default Random Forest
default_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_model.fit(X_train_7, y_train)
default_predictions = default_model.predict(X_val_7)

default_f1 = f1_score(y_val, default_predictions)


# Prepare tuning results
tuning_results = pd.DataFrame([
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": None,
        "val_f1": default_f1
    }
])

# Save tuning results
tuning_results.to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)

print("\nModel Tuning Results")
print(tuning_results.to_string(index=False))

print("\nTuning results saved successfully.")


# Read the tuning results
tuning_results = pd.read_csv("outputs/part_c_tuning.csv")

# Priority for resolving ties
priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

tuning_results["priority"] = tuning_results["search"].map(priority)

# Select the model with the highest validation F1
best_row = tuning_results.sort_values(
    ["val_f1", "priority"],
    ascending=[False, True]
).iloc[0]

chosen_from = best_row["search"]

# Get selected model parameters
if chosen_from == "grid_search":
    best_params = ast.literal_eval(best_row["best_params"])

elif chosen_from == "randomized_search":
    best_params = ast.literal_eval(best_row["best_params"])

else:
    best_params = {
        "n_estimators": 100
    }

print("\nSelected Model:", chosen_from)
print("Selected Parameters:", best_params)

# Select top 7 features using training data
final_selector = SelectKBest(score_func=f_classif, k=7)

X_train_selected = final_selector.fit_transform(X_train, y_train)
X_val_selected = final_selector.transform(X_val)

# Train the selected Random Forest
final_model = RandomForestClassifier(
    random_state=42,
    **best_params
)

final_model.fit(X_train_selected, y_train)

# Get probability of delayed shipments
val_probabilities = final_model.predict_proba(X_val_selected)[:, 1]

# Test thresholds from 0.1 to 0.9
threshold_results = []

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0)
    })

# Save threshold results
threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)

# Select highest threshold with recall >= 0.80
eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    selected_threshold = eligible_thresholds["threshold"].max()
else:
    selected_threshold = 0.1

print("\nThreshold Results")
print(threshold_df.round(4).to_string(index=False))

print("\nSelected Alert Threshold:", selected_threshold)
print("Threshold results saved successfully.")

# Transform test data using the selector fitted on training data
X_test_selected = final_selector.transform(X_test)

# Predict test probabilities
test_probabilities = final_model.predict_proba(X_test_selected)[:, 1]

# Apply the selected threshold
test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

# Calculate final test metrics
final_test = pd.DataFrame([{
    "threshold": selected_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(y_test, test_predictions, zero_division=0),
    "recall": recall_score(y_test, test_predictions, zero_division=0),
    "f1": f1_score(y_test, test_predictions, zero_division=0)
}])

# Save final test results
final_test.to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)

# Save final model configuration
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": best_params["n_estimators"],
    "max_depth": best_params.get("max_depth"),
    "min_samples_leaf": best_params.get("min_samples_leaf"),
    "threshold": selected_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)

print("\nFinal Test Results")
print(final_test.round(4).to_string(index=False))

print("\nFinal Model Configuration")
print(final_config.to_string(index=False))


