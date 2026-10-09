
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

JOINED_FILE = Path("joined_training_dataset.csv")
CLEAN_FILE = Path("cleaned_training_dataset.csv")

REPORT_DIR = Path("pollution_investigation")
REPORT_DIR.mkdir(exist_ok=True)

TRAIN_END_YEAR = 2014
TEST_START_YEAR = 2015
TEST_END_YEAR = 2019

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

if not JOINED_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {JOINED_FILE.resolve()}"
    )

if not CLEAN_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {CLEAN_FILE.resolve()}"
    )

print("=" * 65)
print("POLLUTION FEATURE INVESTIGATION")
print("=" * 65)

joined = pd.read_csv(JOINED_FILE)
clean = pd.read_csv(CLEAN_FILE)

pollution_all = [
    col for col in joined.columns
    if col.startswith(("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__"))
]

pollution_selected = [
    col for col in clean.columns
    if col.startswith(("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__"))
]

print("\n1. DATASET OVERVIEW")
print("-" * 40)
print("Joined dataset rows:", len(joined))
print("Cleaned dataset rows:", len(clean))
print("All pollution features:", len(pollution_all))
print("Selected pollution features:", len(pollution_selected))
print("Target: incidence")

# ============================================================
# INVESTIGATION A: MISSINGNESS IN THE FULL DATASET
# ============================================================

print("\n2. POLLUTION FEATURE MISSINGNESS")
print("-" * 40)

missing_percent = (
    joined[pollution_all].isna().mean() * 100
).sort_values(ascending=False)

missing_report = pd.DataFrame({
    "missing_percent": missing_percent,
    "available_records": joined[pollution_all].notna().sum(),
    "distinct_values": joined[pollution_all].nunique(dropna=True),
    "zero_percent_of_available": [
        (
            (joined[col].dropna() == 0).mean() * 100
            if joined[col].notna().any()
            else np.nan
        )
        for col in missing_percent.index
    ]
})

missing_report.to_csv(
    REPORT_DIR / "pollution_missingness.csv"
)

print("\nMost missing features:")
print(missing_report.head(15).round(2).to_string())

print("\nFeatures with at least 70% missing:")
print(int((missing_percent >= 70).sum()))

print("Features with at most 40% missing:")
print(int((missing_percent <= 40).sum()))

# ============================================================
# INVESTIGATION B: COVERAGE BY YEAR
# ============================================================

print("\n3. POLLUTION COVERAGE BY YEAR")
print("-" * 40)

year_coverage = (
    joined.groupby("year")[pollution_all]
    .apply(lambda frame: frame.notna().mean().mean() * 100)
    .rename("average_feature_coverage_percent")
    .reset_index()
)

year_coverage.to_csv(
    REPORT_DIR / "pollution_coverage_by_year.csv",
    index=False
)

print(year_coverage.to_string(index=False))

# ============================================================
# INVESTIGATION C: COVERAGE BY STATE
# ============================================================

print("\n4. POLLUTION COVERAGE BY STATE")
print("-" * 40)

state_coverage = (
    joined.groupby("state")[pollution_all]
    .apply(lambda frame: frame.notna().mean().mean() * 100)
    .rename("average_feature_coverage_percent")
    .reset_index()
    .sort_values("average_feature_coverage_percent")
)

state_coverage.to_csv(
    REPORT_DIR / "pollution_coverage_by_state.csv",
    index=False
)

print("Lowest-coverage states/regions:")
print(state_coverage.head(10).round(2).to_string(index=False))

print("\nHighest-coverage states/regions:")
print(state_coverage.tail(10).round(2).to_string(index=False))

# ============================================================
# INVESTIGATION D: CHECK SELECTED FEATURES
# ============================================================

print("\n5. SELECTED FEATURE QUALITY")
print("-" * 40)

selected_report = pd.DataFrame({
    "missing_percent": clean[pollution_selected].isna().mean() * 100,
    "distinct_values": clean[pollution_selected].nunique(dropna=True),
    "standard_deviation": clean[pollution_selected].std()
}).sort_values("missing_percent", ascending=False)

selected_report.to_csv(
    REPORT_DIR / "selected_feature_quality.csv"
)

print(selected_report.round(3).to_string())

# ============================================================
# PREPARE CHRONOLOGICAL TRAIN/TEST SETS
# ============================================================

def prepare_frame(frame):
    frame = frame.copy()

    frame["year"] = pd.to_numeric(
        frame["year"], errors="coerce"
    )
    frame["incidence"] = pd.to_numeric(
        frame["incidence"], errors="coerce"
    )

    frame = frame.dropna(
        subset=["year", "incidence", "state", "age_group"]
    )

    return frame


data = prepare_frame(joined)

train = data[data["year"] <= TRAIN_END_YEAR].copy()
test = data[
    data["year"].between(TEST_START_YEAR, TEST_END_YEAR)
].copy()

if train.empty or test.empty:
    raise ValueError(
        "Training or test data is empty. Check the year values."
    )

print("\n6. MODEL DATA SPLIT")
print("-" * 40)
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training years:", train["year"].min(), "-", train["year"].max())
print("Test years:", test["year"].min(), "-", test["year"].max())

# Context features are the non-pollution variables.
context_features = ["year", "state", "age_group"]

# ============================================================
# MODEL FACTORY
# ============================================================

def build_pipeline(features, model_type="random_forest"):
    numeric_features = [
        col for col in features
        if col == "year" or col in pollution_all
    ]

    categorical_features = [
        col for col in features
        if col in ["state", "age_group"]
    ]

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

    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == "baseline":
        estimator = DummyRegressor(strategy="median")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])


# ============================================================
# TEST DIFFERENT POLLUTION FEATURE SETS
# ============================================================

# Use features from the full joined dataset, but retain only
# those with sufficient coverage in the TRAINING DATA.
training_missing = train[pollution_all].isna().mean() * 100

features_40 = training_missing[
    training_missing <= 40
].index.tolist()

features_60 = training_missing[
    training_missing <= 60
].index.tolist()

features_80 = training_missing[
    training_missing <= 80
].index.tolist()

experiments = [
    ("Baseline median", [], "baseline"),
    ("Context only", context_features, "random_forest"),
    (
        "Context + pollution <=40% missing",
        context_features + features_40,
        "random_forest"
    ),
    (
        "Context + pollution <=60% missing",
        context_features + features_60,
        "random_forest"
    ),
    (
        "Context + pollution <=80% missing",
        context_features + features_80,
        "random_forest"
    ),
    (
        "Context + selected 10 pollution features",
        context_features + [
            col for col in pollution_selected
            if col in pollution_all
        ],
        "random_forest"
    ),
    (
        "Context + Extra Trees <=40% missing",
        context_features + features_40,
        "extra_trees"
    )
]

# Remove accidental duplicate feature sets.
seen = set()
unique_experiments = []

for name, features, model_type in experiments:
    key = (tuple(features), model_type)
    if key not in seen:
        seen.add(key)
        unique_experiments.append((name, features, model_type))

experiments = unique_experiments

# ============================================================
# TRAIN, EVALUATE AND SAVE RESULTS
# ============================================================

print("\n7. MODEL COMPARISON")
print("-" * 40)

y_train = train["incidence"]
y_test = test["incidence"]

results = []

for name, features, model_type in experiments:
    print(f"\nTraining: {name}")

    if model_type == "baseline":
        X_train = train[["year"]]
        X_test = test[["year"]]
    else:
        X_train = train[features]
        X_test = test[features]

    model = build_pipeline(
        features,
        model_type=model_type
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    result = {
        "model": name,
        "pollution_feature_count": sum(
            col in pollution_all for col in features
        ),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    results.append(result)

    print(f"Pollution features: {result['pollution_feature_count']}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("MAE")

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON — LOWER MAE/RMSE, HIGHER R² IS BETTER")
print("=" * 65)
print(results_df.round(4).to_string(index=False))

print("\nReports saved in:")
print(REPORT_DIR.resolve())

print("\nInvestigation complete.")