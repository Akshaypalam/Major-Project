
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("joined_training_dataset.csv")
OUTPUT_FILE = Path("cleaned_training_dataset.csv")
REPORT_FILE = Path("cleaning_report.txt")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE.resolve()}"
    )

print("Loading joined dataset...")
df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

# Identify environmental features
environment_features = [
    col for col in df.columns
    if col.startswith(
        ("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__")
    )
]

# Keep features with at most 40% missing values.
# This is a practical starting point, not a universal rule.
missing_percent = df[environment_features].isna().mean() * 100

selected_features = missing_percent[
    missing_percent <= 40
].index.tolist()

# Keep essential health and grouping columns
essential_columns = [
    "ID", "state", "year", "age_group", "incidence"
]

# Remove records without a target or essential grouping values
df = df.dropna(
    subset=["state", "year", "age_group", "incidence"]
).copy()

# Remove records with no usable environmental measurements
df = df.dropna(
    subset=selected_features,
    how="all"
).copy()

# Keep only selected environmental features
final_columns = essential_columns + selected_features
final_columns = [
    col for col in final_columns if col in df.columns
]

cleaned = df[final_columns].copy()

# Save the cleaned dataset.
# Missing values in individual pollution features are preserved.
cleaned.to_csv(OUTPUT_FILE, index=False)

report = []
report.append("DATA CLEANING REPORT")
report.append("====================")
report.append(f"Original rows: {original_rows}")
report.append(f"Original columns: {original_columns}")
report.append(f"Rows after cleaning: {len(cleaned)}")
report.append(f"Columns after cleaning: {len(cleaned.columns)}")
report.append(f"Environmental features originally: {len(environment_features)}")
report.append(f"Environmental features retained: {len(selected_features)}")
report.append(f"Features removed for excessive missingness: {len(environment_features) - len(selected_features)}")
report.append(f"Rows removed: {original_rows - len(cleaned)}")
report.append("")
report.append("Retained environmental features and missing percentages:")

for col in selected_features:
    report.append(
        f"{col}: {missing_percent[col]:.2f}% missing"
    )

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print("\nCleaning complete.")
print("Original rows:", original_rows)
print("Rows after cleaning:", len(cleaned))
print("Original environmental features:", len(environment_features))
print("Retained environmental features:", len(selected_features))
print("Final columns:", len(cleaned.columns))
print("Saved:", OUTPUT_FILE.resolve())
print("Saved:", REPORT_FILE.resolve())

print("\nFirst five rows:")
print(cleaned.head().to_string(index=False))