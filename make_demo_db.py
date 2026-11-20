import sqlite3
from pathlib import Path

SRC = Path("expenses.db")
DST = Path("demo/expenses_demo.db")
DST.parent.mkdir(exist_ok=True)
if DST.exists():
    DST.unlink()

src = sqlite3.connect(SRC)
rows = src.execute("SELECT date, vendor, total, category FROM transactions").fetchall()
src.close()

vendors = sorted({r[1] for r in rows if r[1] is not None})
alias = {v: f"Vendor {i + 1:03d}" for i, v in enumerate(vendors)}

dst = sqlite3.connect(DST)
dst.execute("CREATE TABLE transactions (date TEXT, vendor TEXT, total REAL, category TEXT)")
dst.executemany(
    "INSERT INTO transactions VALUES (?, ?, ?, ?)",
    [(d, alias.get(v, "Unknown vendor"), t, c) for d, v, t, c in rows],
)
dst.commit()

print("Rows copied:", dst.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
print("Vendors anonymised:", len(alias))
for row in dst.execute("SELECT * FROM transactions LIMIT 3"):
    print("  ", row)
dst.close()