from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
from pathlib import Path
import sys

sys.path.append("src")
from importlib import import_module

DB_PATH = Path("expenses.db")

app = Flask(__name__)


def get_summary():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT date, total, category FROM transactions", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2019)]

    total_spend = df["total"].sum()
    by_category = df.groupby("category")["total"].sum().sort_values(ascending=False)
    return {
        "total_spend": round(total_spend, 2),
        "by_category": by_category.round(2).to_dict(),
        "transaction_count": len(df),
    }


@app.route("/")
def home():
    summary = get_summary()
    return render_template("index.html", summary=summary)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    chatbot = import_module("6_rag_chatbot")
    answer = chatbot.answer_question(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)