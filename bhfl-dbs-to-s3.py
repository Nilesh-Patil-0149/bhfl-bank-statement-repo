# pip install sqlalchemy pymysql pandas s3fs  # run if needed
import pyarrow as pa
import pyarrow.parquet as pq

from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

# -------------------------
# Configuration (replace or use env vars)
# -------------------------
DB_USER = "admin"
DB_PASSWORD = "Admin#2025"
DB_HOST = "mysql.cfumki2oembm.ap-south-1.rds.amazonaws.com"
DB_PORT = 3306
DB_NAME = "bank_db"

TABLE_NAME = "transactions"
DATE_COL = "transaction_date"
ACCOUNT_COL = "customer_id"

# date range (half-open: inclusive start, exclusive end)
today = pd.Timestamp.today().normalize()
current_month_start = today.replace(day=1)

# previous month range (dates only)
previous_month_start = (current_month_start - pd.offsets.MonthBegin(1)).date()
previous_month_end   = current_month_start.date()
start_date = previous_month_start
end_date   = previous_month_end

print("Using date range (inclusive start, exclusive end):", start_date, end_date)

# chunking params
chunk_size = 500   # number of ids per IN(...) chunk - tune for your DB

# output paths
out_csv = "transactions_filtered.csv"
out_parquet = "transactions_filtered.parquet"
# optional S3 path (requires s3fs and AWS credentials in env or IAM role)
s3_path = "s3://bhfl-pipeline000/data/transactions_filtered.parquet"

# -------------------------
# Build SQLAlchemy engine
# -------------------------
p = quote_plus(DB_PASSWORD)
engine_url = f"mysql+pymysql://{DB_USER}:{p}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(engine_url, pool_pre_ping=True)

# -------------------------
# Fetch account ids (customers)
# -------------------------
sql2 = "SELECT customer_id FROM customers WHERE customer_id IS NOT NULL"
pdf2 = pd.read_sql(sql2, con=engine)   # no params passed here
account_ids = list(pdf2["customer_id"].astype(str).unique())
print(f"Found {len(account_ids)} unique account_ids")

# -------------------------
# Helpers
# -------------------------
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def build_in_clause_param_names(col_name, ids_chunk, prefix="id"):
    """
    Return an IN(...) fragment using named parameters (safe) and the params dict.
    Example: "account_id IN (:id0,:id1)" and {"id0": "ACC1", "id1": "ACC2"}
    """
    param_names = []
    params = {}
    for idx, val in enumerate(ids_chunk):
        name = f"{prefix}{idx}"
        param_names.append(f":{name}")
        params[name] = val
    in_clause = f"{col_name} IN ({','.join(param_names)})"
    return in_clause, params

# -------------------------
# Quick debug checks (single account + count)
# -------------------------
if account_ids:
    single = account_ids[0]
    debug_sql = text(f"""
        SELECT COUNT(*) AS cnt
        FROM {TABLE_NAME}
        WHERE {ACCOUNT_COL} = :acct
          AND DATE({DATE_COL}) >= :start_date
          AND DATE({DATE_COL}) <  :end_date
    """)
    debug_cnt = pd.read_sql(debug_sql, con=engine, params={"acct": single, "start_date": start_date, "end_date": end_date})
    print(f"Debug: account {single} has {int(debug_cnt['cnt'].iloc[0])} transactions in range")
else:
    print("No accounts found; exiting.")
    raise SystemExit(0)

# -------------------------
# Main: iterate chunks & collect pandas DataFrames
# -------------------------
dfs = []
total = 0
chunks = list(chunk_list(account_ids, chunk_size))
if not chunks:
    print("No account_ids provided — exiting.")
else:
    for chunk_idx, ids_chunk in enumerate(chunks, start=1):
        in_clause, params = build_in_clause_param_names(ACCOUNT_COL, ids_chunk, prefix=f"c{chunk_idx}_")
        params["start_date"] = start_date
        params["end_date"] = end_date

        # Use DATE() in SQL to ignore time component in DB column
        sql = text(f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE {in_clause}
              AND DATE({DATE_COL}) >= :start_date
              AND DATE({DATE_COL}) <  :end_date
        """)

        pdf = pd.read_sql(sql, con=engine, params=params)
        nrows = len(pdf)
        total += nrows
        print(f"Chunk {chunk_idx}/{len(chunks)} fetched {nrows} rows (accum: {total})")
        if nrows > 0:
            dfs.append(pdf)

    # concat all chunks into single DataFrame
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=[])  # empty

    print("Total rows fetched after concatenation:", len(result_df))

    result_df['transaction_date'] = pd.to_datetime(result_df['transaction_date']).dt.date
    
    #join
    
    joined_df = result_df.merge(pdf2,how = "left", on = "customer_id")
    print(joined_df)


# write to S3 (ensure AWS creds in env or instance role)
    try:
        # ensure date-only column (no time)
        
        # write partitioned dataset directly to S3
        s3_out_dir = "s3://bhfl-pipeline000/data2/transactions_partitioned"
        joined_df.to_parquet(
            s3_out_dir,
            engine="pyarrow",
            partition_cols=["customer_id"],
            index=False,
            storage_options={"anon": False}  # add aws_access_key_id/secret_access_key if required
        )
        print("Wrote S3 partitioned parquet to", s3_out_dir)
    except Exception as e:
        print("S3 write failed:", e)
