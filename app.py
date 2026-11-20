import os
import sqlite3
import sys
import time
from collections import defaultdict, deque
from importlib import import_module
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
sys.path.append("src")

# Uses the real database on your laptop. On the web, only the demo copy exists.
DB_PATH = (
    Path("demo/expenses_demo.db")
    if os.environ.get("USE_DEMO_DB") == "1" or not Path("expenses.db").exists()
    else Path("expenses.db")
)
PER_VISITOR_LIMIT = 8   # questions per hour for each visitor
DAILY_LIMIT = 300       # questions per 24 hours in total, to protect the free AI quota

app = Flask(__name__)
visitor_log = defaultdict(deque)
daily_log = deque()


def check_limits(visitor):
    now = time.time()
    while daily_log and now - daily_log[0] > 86400:
        daily_log.popleft()
    log = visitor_log[visitor]
    while log and now - log[0] > 3600:
        log.popleft()
    if len(log) >= PER_VISITOR_LIMIT:
        return "You have reached the hourly question limit for this demo. Please try again later."
    if len(daily_log) >= DAILY_LIMIT:
        return "This demo has reached its free daily limit. Please try again tomorrow."
    log.append(now)
    daily_log.append(now)
    return None


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
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"answer": "Please type a question first."})
    if len(question) > 300:
        return jsonify({"answer": "Please keep questions under 300 characters."})

    visitor = request.access_route[0] if request.access_route else (request.remote_addr or "unknown")
    blocked = check_limits(visitor)
    if blocked:
        return jsonify({"answer": blocked})

    try:
        chatbot = import_module("6_rag_chatbot")
        answer = chatbot.answer_question(question)
    except Exception as e:
        app.logger.error("Chatbot error: %s", e)
        answer = ("Sorry, the AI part of this demo could not answer right now. "
                  "The free AI service may be busy or at its daily limit. Please try again later.")
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=False, port=5000)