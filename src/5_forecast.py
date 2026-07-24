import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pathlib import Path

DB_PATH = Path("expenses.db")
FORECAST_CSV = Path("data/labels/forecast.csv")
CHART_OUT = Path("static/forecast_chart.png")


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT date, total, category FROM transactions", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.groupby([pd.Grouper(key="date", freq="MS"), "category"])["total"]
        .sum()
        .reset_index()
    )

    print("Months of data available per category:")
    print(monthly.groupby("category").size())

    all_forecasts = []
    plt.figure(figsize=(10, 6))

    for category in monthly["category"].unique():
        cat_df = monthly[monthly["category"] == category][["date", "total"]]
        cat_df.columns = ["ds", "y"]

        if len(cat_df) < 3:
            print(f"Skipping {category}, only {len(cat_df)} month(s) of data, need at least 3 to forecast")
            continue

        model = Prophet(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(cat_df)
        future = model.make_future_dataframe(periods=3, freq="MS")
        forecast = model.predict(future)

        forecast["category"] = category
        all_forecasts.append(forecast[["ds", "yhat", "category"]])

        plt.plot(cat_df["ds"], cat_df["y"], marker="o", label=f"{category} actual")
        plt.plot(forecast["ds"], forecast["yhat"], linestyle="--", label=f"{category} forecast")

    plt.legend(fontsize=8)
    plt.title("Spend by Category, Actual and 3 Month Forecast")
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.tight_layout()
    plt.savefig(CHART_OUT)
    print(f"Saved forecast chart to {CHART_OUT}")

    if all_forecasts:
        result = pd.concat(all_forecasts)
        result.to_csv(FORECAST_CSV, index=False)
        print(f"Saved forecast data to {FORECAST_CSV}")
    else:
        print("No categories had enough data to forecast.")


if __name__ == "__main__":
    main()