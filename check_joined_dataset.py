
import pandas as pd
from pathlib import Path

FILE = Path("joined_training_dataset.csv")

if not FILE.exists():
    raise FileNotFoundError(f"File not found: {FILE.resolve()}")

df = pd.read_csv(FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:")
print(df.columns.tolist())

print("\n========== TARGET CHECK ==========")
print(df["incidence"].describe())

print("\n========== DUPLICATE CHECK ==========")
keys = ["state", "year", "age_group"]
print("Duplicate state-year-age records:", df.duplicated(keys).sum())

print("\n========== MISSING VALUES ==========")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values found.")
else:
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / len(df) * 100).round(2)
    })
    print(report.to_string())

print("\n========== ENVIRONMENTAL FEATURE QUALITY ==========")
features = [
    col for col in df.columns
    if col.startswith(("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__"))
]

print("Environmental features:", len(features))

if features:
    feature_missing = df[features].isna().mean().mul(100)
    print("\nFeatures with most missing data:")
    print(feature_missing.sort_values(ascending=False).head(20).round(2).to_string())

    completely_missing = feature_missing[feature_missing == 100]
    print("\nCompletely missing features:", len(completely_missing))
    if len(completely_missing):
        print(completely_missing.index.tolist())

    no_environment = df[features].isna().all(axis=1)
    print("\nRows with no environmental features:", int(no_environment.sum()))

    print("\nEnvironmental feature summary:")
    print(df[features].describe().loc[["min", "50%", "max"]].T.head(10))

print("\n========== YEAR AND AGE COVERAGE ==========")
print("Year range:", df["year"].min(), "-", df["year"].max())
print("Age groups:", sorted(df["age_group"].dropna().unique()))
print("States/regions:", df["state"].nunique())

print("\n========== CHECK COMPLETE ==========")