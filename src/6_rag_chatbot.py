import sys
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

DB_PATH = Path("expenses.db")


def get_context_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT date, vendor, total, category FROM transactions", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2019)]
    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly_by_category = df.groupby(["month", "category"])["total"].sum().reset_index()
    top_vendors = (
        df.groupby("vendor")["total"].sum().sort_values(ascending=False).head(10).reset_index()
    )

    return monthly_by_category, top_vendors


def answer_question(question):
    monthly_by_category, top_vendors = get_context_data()

    context = (
        "Monthly spend by category:\n"
        f"{monthly_by_category.to_string(index=False)}\n\n"
        "Top 10 vendors by total spend:\n"
        f"{top_vendors.to_string(index=False)}"
    )

    prompt = (
        "You are a business finance assistant. Using only the data below, answer "
        "the user's question in plain, direct English, in 3 to 5 sentences. "
        "If the data doesn't clearly support an answer, say so honestly rather "
        "than guessing.\n\n"
        f"DATA:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What category am I spending the most on?"
    print(f"Question: {q}\n")
    print(answer_question(q))