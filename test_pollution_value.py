
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("joined_training_dataset.csv")
OUTPUT_DIR = Path("pollution_investigation")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing file: {DATA_FILE.resolve()}")

df = pd.read_csv(DATA_FILE)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["incidence"] = pd.to_numeric(df["incidence"], errors="coerce")
df = df.dropna(
    subset=["year", "incidence", "state", "age_group"]
).copy()

pollution_features = [
    col for col in df.columns
    if col.startswith(("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__"))
]

context_features = ["year", "state", "age_group"]

# Select features using the training period only.
train = df[df["year"] <= 2014].copy()
test = df[df["year"].between(2015, 2019)].copy()

if train.empty or test.empty:
    raise ValueError("Training or test set is empty.")

missing_pct = train[pollution_features].isna().mean() * 100

# Keep features with <= 40% missing values in training.
selected = missing_pct[missing_pct <= 40].index.tolist()

# Check whether retained columns are duplicates or nearly duplicates.
print("=" * 65)
print("1. CHECK FOR DUPLICATE POLLUTION FEATURES")
print("=" * 65)

duplicate_pairs = []
for i, first in enumerate(selected):
    for second in selected[i + 1:]:
        a = train[first]
        b = train[second]
        both = a.notna() & b.notna()

        if both.sum() == 0:
            continue

        equal_fraction = (a[both] == b[both]).mean()

        if equal_fraction >= 0.99:
            duplicate_pairs.append({
                "feature_1": first,
                "feature_2": second,
                "fraction_equal": equal_fraction,
                "overlapping_records": int(both.sum())
            })

if duplicate_pairs:
    duplicates_df = pd.DataFrame(duplicate_pairs)
    print(duplicates_df.to_string(index=False))
    duplicates_df.to_csv(
        OUTPUT_DIR / "duplicate_pollution_features.csv",
        index=False
    )
else:
    print("No pairs with at least 99% exact equality found.")

# Keep features with actual variability.
selected = [
    col for col in selected
    if train[col].nunique(dropna=True) > 1
]

print("\nSelected pollution features:", len(selected))
print(selected)

# Make indicators explicit: 1 means measurement is missing.
indicator_names = {
    col: col + "__missing"
    for col in selected
}

for col, indicator in indicator_names.items():
    train[indicator] = train[col].isna().astype(int)
    test[indicator] = test[col].isna().astype(int)

pollution_plus_missing = selected + list(indicator_names.values())

# Build model pipelines.
def make_model(numeric, categorical, model_kind="extra"):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])

    if model_kind == "extra":
        estimator = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

    return Pipeline([
        ("preprocessor", prep),
        ("model", estimator)
    ])

experiments = [
    (
        "Pollution only",
        selected,
        [],
        "extra"
    ),
    (
        "Pollution + missingness indicators",
        pollution_plus_missing,
        [],
        "extra"
    ),
    (
        "Context only",
        ["year"],
        ["state", "age_group"],
        "extra"
    ),
    (
        "Context + pollution",
        ["year"] + selected,
        ["state", "age_group"],
        "extra"
    ),
    (
        "Context + pollution + missingness",
        ["year"] + pollution_plus_missing,
        ["state", "age_group"],
        "extra"
    )
]

results = []
y_train = train["incidence"]
y_test = test["incidence"]

print("\n" + "=" * 65)
print("2. MODEL COMPARISON")
print("=" * 65)
print("Training years: 1990–2014")
print("Testing years: 2015–2019")
print("Training rows:", len(train))
print("Testing rows:", len(test))

for name, numeric, categorical, kind in experiments:
    features = numeric + categorical
    print(f"\nTraining: {name}")

    model = make_model(numeric, categorical, kind)
    model.fit(train[features], y_train)
    predictions = model.predict(test[features])

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    results.append({
        "model": name,
        "pollution_features": len(selected),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv(
    OUTPUT_DIR / "pollution_value_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

print("\nSaved report:")
print((OUTPUT_DIR / "pollution_value_comparison.csv").resolve())
print("\nExperiment complete.")