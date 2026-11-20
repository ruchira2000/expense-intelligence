import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.environ.get("CHATBOT_MODEL", "gemini-3.5-flash-lite")
DB_PATH = (
    Path("demo/expenses_demo.db")
    if os.environ.get("USE_DEMO_DB") == "1" or not Path("expenses.db").exists()
    else Path("expenses.db")
)

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY.")
        _client = genai.Client(api_key=key)
    return _client


def get_context_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT date, vendor, total, category FROM transactions", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2019)]
    df["month"] = df["date"].dt.to_period("M").astype(str)

    overview = (
        f"Transactions: {len(df)}\n"
        f"Date range: {df['date'].min().date()} to {df['date'].max().date()}\n"
        f"Total spend: {df['total'].sum():.2f}\n"
        f"Transactions with a total of 0 (usually an amount the OCR could not read): "
        f"{int((df['total'] == 0).sum())}"
    )
    category_totals = (
        df.groupby("category")["total"]
        .agg(total_spend="sum", transactions="count")
        .sort_values("total_spend", ascending=False)
        .round(2)
        .reset_index()
    )
    monthly_totals = df.groupby("month")["total"].sum().round(2).reset_index()
    monthly_by_category = df.groupby(["month", "category"])["total"].sum().round(2).reset_index()
    top_vendors = (
        df.groupby("vendor")["total"].sum().sort_values(ascending=False).head(10).round(2).reset_index()
    )
    return overview, category_totals, monthly_totals, monthly_by_category, top_vendors


def answer_question(question):
    overview, category_totals, monthly_totals, monthly_by_category, top_vendors = get_context_data()

    context = (
        f"Overview:\n{overview}\n\n"
        f"Total spend by category (largest first):\n{category_totals.to_string(index=False)}\n\n"
        f"Total spend by month:\n{monthly_totals.to_string(index=False)}\n\n"
        f"Monthly spend by category:\n{monthly_by_category.to_string(index=False)}\n\n"
        f"Top 10 vendors by total spend:\n{top_vendors.to_string(index=False)}"
    )

    prompt = (
        "You are a business finance assistant. Using only the data below, answer "
        "the user's question in plain, direct English, in 3 to 5 sentences. "
        "Quote the exact figures from the tables and do not add numbers up yourself "
        "if a total is already given. The currency is not recorded, so do not state one. "
        "If the data doesn't clearly support an answer, say so honestly rather "
        "than guessing.\n\n"
        f"DATA:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    response = get_client().models.generate_content(model=MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What category am I spending the most on?"
    print(f"Question: {q}\n")
    print(answer_question(q))