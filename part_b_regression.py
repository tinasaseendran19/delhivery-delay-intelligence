
import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score


# Load the data created in Part A
df = pd.read_csv("data/modelling_table.csv")

# Select the input columns
X = df[FEATURES]

# Select the value we want to predict
y = df["delivery_hours"]

# Split 70% for training and 30% for temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Divide the temporary data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Display the number of rows in each group
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# Scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler for validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training data shape:", X_train_scaled.shape)
print("Validation data shape:", X_val_scaled.shape)
print("Test data shape:", X_test_scaled.shape)

# Create a pipeline for Linear Regression
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Train the model
linear_model.fit(X_train, y_train)

# Predict delivery hours
y_pred = linear_model.predict(X_val)

# Calculate performance metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred) ** 0.5
r2 = r2_score(y_val, y_pred)

print("\nLinear Regression Results")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Save the first model's results
results = [{
    "Model": "Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R2 Score": r2
}]

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/part_b_model_results.csv", index=False)

print("Results saved successfully.")


# Calculate the average delivery time from training data
average_hours = y_train.mean()

# Predict the same average for every validation shipment
baseline_predictions = [average_hours] * len(y_val)

# Calculate baseline performance
baseline_mae = mean_absolute_error(y_val, baseline_predictions)
baseline_mse = mean_squared_error(y_val, baseline_predictions)
baseline_rmse = mean_squared_error(y_val, baseline_predictions) ** 0.5
baseline_r2 = r2_score(y_val, baseline_predictions)

print("\nMean Baseline Results")
print("MAE:", round(baseline_mae, 2))
print("MSE:", round(baseline_mse, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2 Score:", round(baseline_r2, 3))


# Different tree depths to test
depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in depths:

    # Create a Decision Tree model
    tree_model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    # Train the model
    tree_model.fit(X_train, y_train)

    # Predict on training and validation data
    train_predictions = tree_model.predict(X_train)
    val_predictions = tree_model.predict(X_val)

    # Calculate RMSE
    train_rmse = mean_squared_error(y_train, train_predictions) ** 0.5
    val_rmse = mean_squared_error(y_val, val_predictions) ** 0.5

    # Store results
    tree_results.append({
        "max_depth": depth if depth is not None else "unlimited",
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

# Display results
tree_df = pd.DataFrame(tree_results)

print("\nDecision Tree Depth Results")
print(tree_df)

# Save results
tree_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

print("Tree depth results saved successfully.")


# --------------------------------
# STEP 7: Compare Four ML Models
# --------------------------------

# Find the best Decision Tree depth
best_depth_row = tree_df.loc[tree_df["val_rmse"].idxmin()]
best_depth = best_depth_row["max_depth"]

if best_depth == "unlimited":
    best_depth = None
else:
    best_depth = int(best_depth)

# Create the four models
models = {
    "linear_regression": linear_model,

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

model_results = []

for model_name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict validation data
    predictions = model.predict(X_val)

    # Calculate performance metrics
    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, predictions)

    model_results.append({
        "model": model_name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# Display model comparison
model_results_df = pd.DataFrame(model_results)

print("\nFour Model Comparison")
print(model_results_df)

# Save model comparison
model_results_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("\nModel comparison saved successfully.")


# --------------------------------
# STEP 8: Gradient Boosting Learning Rate Experiment
# --------------------------------

learning_rates = [0.01, 0.1, 1.0]

learning_rate_results = []

for lr in learning_rates:

    # Create Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    # Train the model
    gb_model.fit(X_train, y_train)

    # Predict validation data
    predictions = gb_model.predict(X_val)

    # Calculate performance
    rmse = mean_squared_error(y_val, predictions) ** 0.5
    r2 = r2_score(y_val, predictions)

    learning_rate_results.append({
        "learning_rate": lr,
        "val_rmse": rmse,
        "val_r2": r2
    })

# Display results
lr_df = pd.DataFrame(learning_rate_results)

print("\nGradient Boosting Learning Rate Results")
print(lr_df)

# Save results
lr_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)

print("Learning rate results saved successfully.")


# --------------------------------
# STEP 9: 5-Fold Cross-validation
# --------------------------------

# Combine training and validation data
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Select the two best models based on validation RMSE
top_two = model_results_df.nsmallest(2, "RMSE")

print("\nTop Two Models:")
print(top_two[["model", "RMSE"]])

# Create one shared 5-fold cross-validation setup
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name in top_two["model"]:

    # Get the model
    model = models[model_name]

    # Perform cross-validation
    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf,
        scoring="r2"
    )

    cv_results.append({
        "model": model_name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

# Display results
cv_df = pd.DataFrame(cv_results)

print("\n5-Fold Cross-validation Results")
print(cv_df)

# Save results
cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)

print("Cross-validation results saved successfully.")


# --------------------------------
# STEP 10: Final Model Test Evaluation
# --------------------------------

# Select the model with the lowest validation RMSE
best_model_name = model_results_df.loc[
    model_results_df["RMSE"].idxmin(), "model"
]

best_model = models[best_model_name]

print("\nSelected Final Model:", best_model_name)

# Train the selected model on training data
best_model.fit(X_train, y_train)

# Predict test data only once
test_predictions = best_model.predict(X_test)

# Calculate test performance
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\nFinal Test Results")
print("MAE:", round(test_mae, 2))
print("MSE:", round(test_mse, 2))
print("RMSE:", round(test_rmse, 2))
print("R2 Score:", round(test_r2, 3))

# Add test results to the existing metrics file
test_result = pd.DataFrame([{
    "model": best_model_name,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}])

final_metrics = pd.concat(
    [model_results_df, test_result],
    ignore_index=True
)

# Add the baseline result
baseline_result = pd.DataFrame([{
    "model": "mean_baseline",
    "split": "val",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2": baseline_r2
}])

final_metrics = pd.concat(
    [baseline_result, final_metrics],
    ignore_index=True
)

final_metrics.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("\nFinal metrics saved successfully.")
print(final_metrics)

