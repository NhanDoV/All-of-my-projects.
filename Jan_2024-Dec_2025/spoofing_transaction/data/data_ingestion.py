# fetch_joined_transactions_from_sql.py
import os
import psycopg2
import psycopg2.extras
from datetime import datetime

PG_HOST = os.environ["PG_HOST"]
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB   = os.environ["PG_DB_NAME"]
PG_USER = os.environ["PG_USER"]
PG_PASS = os.environ["PG_PASSWORD"]

SQL_FILE = "query_joined_transactions.sql"   # SQL cùng thư mục
OUTPUT_FILE = "data.txt"                    # file output cùng thư mục

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
        connect_timeout=10,
    )

def load_sql(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fetch_data(start_dt, end_dt):
    sql = load_sql(SQL_FILE)
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql, {"start_dt": start_dt, "end_dt": end_dt})
        rows = cur.fetchall()
    return [dict(r) for r in rows]

def save_to_txt(rows, path=OUTPUT_FILE):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("NO DATA\n")
        return

    # ghi header + từng dòng dạng TSV
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) if r[c] is not None else "" for c in cols) + "\n")

if __name__ == "__main__":
    import sys

    def ask_dt(prompt):
        s = input(prompt).strip()
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            print("Invalid format. Use YYYY-MM-DD")
            sys.exit(1)

    print("=== Fetch Transactions ===")
    start = ask_dt("Start date (YYYY-MM-DD): ")
    end   = ask_dt("End date (YYYY-MM-DD):   ")

    if end <= start:
        print("End date must be > start date")
        sys.exit(1)

    data = fetch_data(start, end)
    print(f"Fetched {len(data)} rows")

    save_to_txt(data)
    print(f"Saved to {OUTPUT_FILE}")
