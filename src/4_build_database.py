import sqlite3
import pandas as pd
from pathlib import Path

CATEGORY_CSV = Path("data/labels/category_labels.csv")
EXTRACTED_CSV = Path("data/labels/extracted_data.csv")
DB_PATH = Path("expenses.db")


def clean_total(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def clean_date(value):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(value, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


def main():
    categories = pd.read_csv(CATEGORY_CSV)
    extracted = pd.read_csv(EXTRACTED_CSV)

    merged = pd.merge(extracted, categories, on="image_filename", how="left")
    merged["total"] = merged["total_ocr"].apply(clean_total)
    merged["date"] = merged["date_ocr"].apply(clean_date)
    merged["category"] = merged["category"].fillna("Retail")

    clean_df = merged.dropna(subset=["total", "date"]).copy()
    clean_df = clean_df[["image_filename", "vendor", "date", "total", "category"]]
    clean_df.columns = ["filename", "vendor", "date", "total", "category"]

    conn = sqlite3.connect(DB_PATH)
    clean_df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(clean_df)} clean transactions into {DB_PATH}")
    print(f"Dropped {len(merged) - len(clean_df)} rows with missing date or total")
    print("\nSample:")
    print(clean_df.head())


if __name__ == "__main__":
    main()