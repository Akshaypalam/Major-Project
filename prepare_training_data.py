
import zipfile
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ZIP_FILE = Path("air-toxicity-and-chronic-respiratory-diseases-us.zip")
DATA_DIR = Path("pollution_data")
OUTPUT_FILE = Path("joined_training_dataset.csv")
REPORT_FILE = Path("joined_dataset_report.txt")

POLLUTION_FILES = {
    "HAPS": "supplementary_HAPS.parquet",
    "LEAD": "supplementary_LEAD.parquet",
    "NONOxNOy": "supplementary_NONOxNOy.parquet",
    "VOCS": "supplementary_VOCS.parquet",
}

START_YEAR = 1990
END_YEAR = 2019
BATCH_SIZE = 500_000


def clean_feature_name(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value))
    return value.strip("_").lower()


def extract_pollution_files():
    DATA_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        for filename in POLLUTION_FILES.values():
            destination = DATA_DIR / filename

            if destination.exists():
                print("Already extracted:", filename)
                continue

            print("Extracting:", filename)
            with archive.open(filename) as source:
                with open(destination, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)

    print("All pollution files extracted.")


def aggregate_pollution(file_path, prefix):
    print("\nProcessing:", file_path.name)

    parquet_file = pq.ParquetFile(file_path)
    partial_results = []

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[
                "State.Name",
                "Date.Local",
                "Parameter.Name",
                "Arithmetic.Mean",
            ],
        ),
        start=1,
    ):
        df = batch.to_pandas()

        df["Date.Local"] = pd.to_datetime(
            df["Date.Local"], errors="coerce"
        )
        df["Arithmetic.Mean"] = pd.to_numeric(
            df["Arithmetic.Mean"], errors="coerce"
        )

        df = df.dropna(
            subset=[
                "State.Name",
                "Date.Local",
                "Parameter.Name",
                "Arithmetic.Mean",
            ]
        )

        df["Year"] = df["Date.Local"].dt.year

        df = df[
            df["Year"].between(START_YEAR, END_YEAR)
        ]

        if df.empty:
            continue

        grouped = (
            df.groupby(
                ["State.Name", "Year", "Parameter.Name"],
                observed=True,
            )["Arithmetic.Mean"]
            .agg(["sum", "count"])
            .reset_index()
        )

        partial_results.append(grouped)

        if batch_number % 10 == 0:
            print(
                f"  Processed {batch_number} batches; "
                f"kept {len(partial_results)} batch summaries."
            )

        del df, grouped

    if not partial_results:
        raise ValueError(
            f"No usable measurements found in {file_path.name}"
        )

    combined = pd.concat(partial_results, ignore_index=True)

    annual = (
        combined.groupby(
            ["State.Name", "Year", "Parameter.Name"],
            observed=True,
        )[["sum", "count"]]
        .sum()
        .reset_index()
    )

    annual["annual_mean"] = annual["sum"] / annual["count"]

    annual["feature"] = annual["Parameter.Name"].map(
        lambda name: f"{prefix}__{clean_feature_name(name)}"
    )

    wide = annual.pivot(
        index=["State.Name", "Year"],
        columns="feature",
        values="annual_mean",
    ).reset_index()

    wide.columns.name = None

    print("Annual state-year rows:", len(wide))
    print("Pollution features:", len(wide.columns) - 2)

    return wide


def main():
    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"ZIP file not found: {ZIP_FILE.resolve()}"
        )

    print("Step 1: Extract pollution files")
    extract_pollution_files()

    print("\nStep 2: Load health training data")

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        train = pd.read_csv(archive.open("train.csv"))

    train["State.Name"] = train["State.Name"].astype(str).str.strip()
    train["Year"] = pd.to_numeric(train["Year"], errors="coerce")
    train["Incidence"] = pd.to_numeric(
        train["Incidence"], errors="coerce"
    )

    train = train.dropna(
        subset=["State.Name", "Year", "Age", "Incidence"]
    )
    train["Year"] = train["Year"].astype(int)

    print("Health records:", len(train))

    print("\nStep 3: Aggregate and combine pollution data")

    pollution_tables = []

    for prefix, filename in POLLUTION_FILES.items():
        table = aggregate_pollution(
            DATA_DIR / filename,
            prefix,
        )
        pollution_tables.append(table)

    pollution = pollution_tables[0]

    for table in pollution_tables[1:]:
        pollution = pollution.merge(
            table,
            on=["State.Name", "Year"],
            how="outer",
            validate="one_to_one",
        )

    print("\nCombined environmental rows:", len(pollution))
    print("Environmental features:", len(pollution.columns) - 2)

    print("\nStep 4: Join health and pollution data")

    joined = train.merge(
        pollution,
        on=["State.Name", "Year"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched = int((joined["_merge"] == "both").sum())
    unmatched = int((joined["_merge"] == "left_only").sum())

    joined = joined.drop(columns=["_merge"])
    joined = joined.rename(
        columns={
            "State.Name": "state",
            "Year": "year",
            "Age": "age_group",
            "Incidence": "incidence",
        }
    )

    joined.to_csv(OUTPUT_FILE, index=False)

    feature_columns = [
        col for col in joined.columns
        if col.startswith(
            ("HAPS__", "LEAD__", "NONOxNOy__", "VOCS__")
        )
    ]

    missing_rates = joined[feature_columns].isna().mean()

    report = [
        "JOINED DATASET REPORT",
        f"Health records: {len(train)}",
        f"Joined records: {len(joined)}",
        f"Matched records: {matched}",
        f"Unmatched records: {unmatched}",
        f"Environmental features: {len(feature_columns)}",
        f"Year range: {joined['year'].min()}-{joined['year'].max()}",
        "",
        "Missing values by column:",
        joined.isna().sum().to_string(),
        "",
        "Features with the most missing data:",
        missing_rates.sort_values(ascending=False)
        .head(20)
        .to_string(),
        "",
        "IMPORTANT:",
        "The target is population-level incidence, not individual disease risk.",
        "Pollution measurements are state-year aggregates.",
        "Review feature missingness and target documentation before model training.",
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\n========== RESULTS ==========")
    print("Saved:", OUTPUT_FILE.resolve())
    print("Saved:", REPORT_FILE.resolve())
    print("Joined rows:", len(joined))
    print("Environmental features:", len(feature_columns))
    print("Matched records:", matched)
    print("Unmatched records:", unmatched)
    print("\nNext: inspect the report before training a model.")


if __name__ == "__main__":
    main()