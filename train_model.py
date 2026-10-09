
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("cleaned_training_dataset.csv")
MODEL_FILE = Path("incidence_model.joblib")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_FILE)

# Identify the environmental features
pollution_features = [
    col for col in df.columns
    if col.startswith(("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__"))
]

# Features available to the model
numeric_features = ["year"] + pollution_features
categorical_features = ["state", "age_group"]
feature_columns = numeric_features + categorical_features

# Ensure essential columns are present
required = feature_columns + ["incidence"]
missing_columns = [col for col in required if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Keep valid target values
df = df.dropna(subset=["incidence", "year"]).copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["incidence"] = pd.to_numeric(df["incidence"], errors="coerce")
df = df.dropna(subset=["year", "incidence"]).copy()

# Chronological split: earlier years for training, later years for testing
train_df = df[df["year"] <= 2014].copy()
test_df = df[df["year"].between(2015, 2019)].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "The chronological split produced an empty dataset. "
        "Check the available years in the CSV."
    )

X_train = train_df[feature_columns]
y_train = train_df["incidence"]

X_test = test_df[feature_columns]
y_test = test_df["incidence"]

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training years:", int(train_df["year"].min()), "-", int(train_df["year"].max()))
print("Testing years:", int(test_df["year"].min()), "-", int(test_df["year"].max()))
print("States in training:", train_df["state"].nunique())
print("States in testing:", test_df["state"].nunique())
print("Pollution features:", len(pollution_features))

# Fill missing pollution measurements using training-set medians.
# Encode state and age group as categorical variables.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Simple reference model: predict the training-set median for every row
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyRegressor(strategy="median"))
])

# Main model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining baseline model...")
baseline.fit(X_train, y_train)

print("Training Random Forest model...")
model.fit(X_train, y_train)

def evaluate(name, fitted_model, X, y):
    predictions = fitted_model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    return predictions

baseline_predictions = evaluate(
    "BASELINE TEST RESULTS",
    baseline,
    X_test,
    y_test
)

predictions = evaluate(
    "RANDOM FOREST TEST RESULTS",
    model,
    X_test,
    y_test
)

# Save test predictions for inspection
results = test_df[["state", "year", "age_group", "incidence"]].copy()
results["predicted_incidence"] = predictions
results["absolute_error"] = (
    results["incidence"] - results["predicted_incidence"]
).abs()

results.to_csv("model_test_predictions.csv", index=False)

# Save the fitted pipeline for later integration
joblib.dump(model, MODEL_FILE)

print("\n========== FILES SAVED ==========")
print("Model:", MODEL_FILE.resolve())
print("Predictions:", Path("model_test_predictions.csv").resolve())

print("\nFirst 10 test predictions:")
print(results.head(10).to_string(index=False))

print("\nTraining and evaluation complete.")