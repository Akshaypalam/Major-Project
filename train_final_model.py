
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

print("Loading training dataset...")

df = pd.read_csv("joined_training_dataset.csv")

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. SELECT THE REQUIRED COLUMNS
# --------------------------------------------------

df = df[["year", "state", "age_group", "incidence"]].copy()

# Remove rows with missing target or input values
df = df.dropna(
    subset=["year", "state", "age_group", "incidence"]
)

# Ensure year and incidence are numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["incidence"] = pd.to_numeric(
    df["incidence"], errors="coerce"
)

df = df.dropna(
    subset=["year", "incidence"]
)

# Convert year to integer
df["year"] = df["year"].astype(int)

# Treat state and age group as categorical text
df["state"] = df["state"].astype(str)
df["age_group"] = df["age_group"].astype(str)

print("Usable dataset shape:", df.shape)


# --------------------------------------------------
# 3. DEFINE INPUTS AND TARGET
# --------------------------------------------------

# Context-only model: year, state and age group
# Pollution readings are intentionally excluded.

X = df[["year", "state", "age_group"]]
y = df["incidence"]


# --------------------------------------------------
# 4. SPLIT DATA CHRONOLOGICALLY
# --------------------------------------------------

# Train on 1990–2014
# Test on 2015–2019

train_mask = df["year"] <= 2014
test_mask = df["year"] >= 2015

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError(
        "Training or testing data is empty. "
        "Check the year values in your dataset."
    )

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Training years:",
    df.loc[train_mask, "year"].min(),
    "to",
    df.loc[train_mask, "year"].max()
)

print(
    "Testing years:",
    df.loc[test_mask, "year"].min(),
    "to",
    df.loc[test_mask, "year"].max()
)


# --------------------------------------------------
# 5. PREPROCESS THE INPUTS
# --------------------------------------------------

numeric_features = ["year"]
categorical_features = ["state", "age_group"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 6. CREATE THE RANDOM FOREST MODEL
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# --------------------------------------------------
# 7. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining the context-only Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 8. EVALUATE THE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
print("==================================")


# --------------------------------------------------
# 9. SAVE THE TRAINED MODEL
# --------------------------------------------------

model_filename = "incidence_context_model.joblib"

joblib.dump(model, model_filename)

print(f"\nModel saved successfully as: {model_filename}")


# --------------------------------------------------
# 10. SAVE TEST PREDICTIONS
# --------------------------------------------------

results = df.loc[
    test_mask,
    ["year", "state", "age_group", "incidence"]
].copy()

results["predicted_incidence"] = predictions
results["absolute_error"] = (
    results["incidence"] - results["predicted_incidence"]
).abs()

results.to_csv(
    "context_model_predictions.csv",
    index=False
)

print("Predictions saved as: context_model_predictions.csv")

print("\nFirst five predictions:")
print(results.head().to_string(index=False))

print("\nAll steps completed successfully.")

