
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE = Path("cleaned_training_dataset.csv")
df = pd.read_csv(FILE)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["incidence"] = pd.to_numeric(df["incidence"], errors="coerce")
df = df.dropna(
    subset=["year", "incidence", "state", "age_group"]
).copy()

context_features = ["year", "state", "age_group"]

pollution_features = [
    col for col in df.columns
    if col.startswith(("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__"))
]

# Use exactly the same chronological test split
train = df[df["year"] <= 2014].copy()
test = df[df["year"].between(2015, 2019)].copy()

y_train = train["incidence"]
y_test = test["incidence"]

def build_model(features):
    numeric = [
        col for col in features
        if col == "year" or col in pollution_features
    ]
    categorical = [
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

    prep = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])

    return Pipeline([
        ("preprocessor", prep),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])

experiments = {
    "Context only": context_features,
    "Context + pollution": context_features + pollution_features
}

print("TRAINING AND COMPARING MODELS")
print("Training years: 1990–2014")
print("Testing years: 2015–2019")
print("Training records:", len(train))
print("Testing records:", len(test))
print()

for name, features in experiments.items():
    model = build_model(features)

    model.fit(train[features], y_train)
    predictions = model.predict(test[features])

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(name)
    print("-" * len(name))
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    print()

print("Comparison complete.")