
import zipfile
import pandas as pd
from pathlib import Path
from io import BytesIO

ZIP_FILE = "air-toxicity-and-chronic-respiratory-diseases-us.zip"


def main():
    zip_path = Path(ZIP_FILE)

    if not zip_path.exists():
        print("ERROR: ZIP file not found.")
        print("Place the downloaded ZIP in your project folder.")
        print("Expected file:", zip_path.resolve())
        return

    with zipfile.ZipFile(zip_path, "r") as archive:
        files = archive.namelist()

        print("\n========== FILES IN ZIP ==========")
        for filename in files:
            print(filename)

        print("\n========== TRAIN.CSV ==========")
        train_name = next(
            (f for f in files if Path(f).name == "train.csv"),
            None
        )

        if train_name:
            train = pd.read_csv(archive.open(train_name))
            print("Shape:", train.shape)
            print("Columns:", train.columns.tolist())
            print("\nSample rows:")
            print(train.head(3).to_string(index=False))

        parquet_files = [
            f for f in files if f.lower().endswith(".parquet")
        ]

        for filename in parquet_files:
            print("\n" + "=" * 60)
            print("FILE:", filename)
            print("=" * 60)

            with archive.open(filename) as file:
                content = file.read()

            df = pd.read_parquet(BytesIO(content))

            print("Shape:", df.shape)
            print("\nColumns and data types:")
            print(df.dtypes.to_string())

            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))

            print("\nMissing values:")
            print(df.isna().sum().to_string())

            print("\nPotential join columns:")
            for column in df.columns:
                if any(word in column.lower() for word in [
                    "state", "year", "date", "fips", "county",
                    "geoid", "id", "name", "age"
                ]):
                    print(
                        f"{column}: "
                        f"{df[column].nunique(dropna=True)} unique values"
                    )

    print("\nInspection complete.")
    print("Send me the terminal output so we can identify the join keys.")


if __name__ == "__main__":
    main()