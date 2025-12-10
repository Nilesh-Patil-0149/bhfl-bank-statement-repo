"""
statements_from_customer_file.py

Generates monthly PDF bank statements (one PDF per account per month) from a file that has
columns like:
transaction_id, account_id, transaction_date, amount, currency, txn_type, description, status, availablebalance

Supports CSV (auto-encoding fallback), Parquet, JSON/NDJSON.

Usage:
    python statements_from_customer_file.py /path/to/transactions.csv --output-dir statements
"""

import os
from pathlib import Path
from datetime import datetime
import locale
import argparse

import pandas as pd

from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph


# attempt locale currency formatting
try:
    locale.setlocale(locale.LC_ALL, "")
except Exception:
    pass

def fmt_currency(x, curr_symbol=None):
    try:
        # prefer locale formatting if possible
        s = locale.currency(x, grouping=True)
        if curr_symbol:
            return f"{curr_symbol} {abs(x):,.2f}"
        return s
    except Exception:
        if curr_symbol:
            return f"{curr_symbol} {x:,.2f}"
        return f"{x:,.2f}"

def parse_possible_epoch(ts):
    if pd.isnull(ts):
        return pd.NaT
    if isinstance(ts, (pd.Timestamp, datetime)):
        return pd.to_datetime(ts)
    if isinstance(ts, str):
        ts_strip = ts.strip()
        if ts_strip.isdigit():
            ts = int(ts_strip)
        else:
            try:
                return pd.to_datetime(ts_strip)
            except Exception:
                return pd.NaT
    if isinstance(ts, (int, float)):
        val = int(ts)
        l = len(str(abs(val)))
        if l <= 10:
            return pd.to_datetime(val, unit="s")
        if l <= 13:
            return pd.to_datetime(val, unit="ms")
        if l <= 16:
            return pd.to_datetime(val, unit="us")
        return pd.to_datetime(val, unit="ns")
    return pd.to_datetime(ts, errors="coerce")

def read_table_with_fallback(path):
    """
    Read CSV/Parquet/JSON with sensible fallbacks.
    Returns a pandas DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)

   
def read_table_with_fallback(path, repair_output_dir=None):
    """
    Read CSV/Parquet/JSON with sensible fallbacks.
    For parquet files, try pyarrow, fastparquet, and a low-level pyarrow row-group read.
    If 'repair_output_dir' is provided and a low-level read succeeds, write a repaired copy there.
    Returns a pandas DataFrame.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        # Try pandas with pyarrow first
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as e_py:
            print(f"pd.read_parquet(..., engine='pyarrow') failed: {e_py!r}")
        # Try fastparquet
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e_fp:
            print(f"pd.read_parquet(..., engine='fastparquet') failed: {e_fp!r}")

        # Try low-level pyarrow read of row-groups (more tolerant / gives control)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e_imp:
            raise RuntimeError(
                "pyarrow is required to attempt low-level parquet repair/read. Install pyarrow and try again."
            ) from e_imp

        try:
            pf = pq.ParquetFile(path)
            # read row-groups one by one and concat — this may succeed where a single read fails
            tables = []
            for rg in range(pf.num_row_groups):
                try:
                    rg_table = pf.read_row_group(rg, columns=None)  # None -> all columns
                    tables.append(rg_table)
                except Exception as e_rg:
                    # skip or rethrow depending on severity; here we rethrow to alert the user
                    raise RuntimeError(f"Failed reading row-group {rg}: {e_rg!r}")
            if not tables:
                raise RuntimeError("No row-groups could be read from the parquet file.")
            combined = pa.concat_tables(tables, promote=True)
            df = combined.to_pandas()
            # Optionally write out a repaired parquet so future reads are fine
            if repair_output_dir:
                try:
                    repair_output_dir = Path(repair_output_dir)
                    repair_output_dir.mkdir(parents=True, exist_ok=True)
                    repaired_path = repair_output_dir / (p.stem + "_repaired.parquet")
                    pq.write_table(combined, str(repaired_path))
                    print(f"Wrote repaired parquet to: {repaired_path}")
                except Exception as e_w:
                    print(f"Warning: unable to write repaired parquet: {e_w!r}")
            return df
        except Exception as e_low:
            # final fallback: give user actionable error
            raise RuntimeError(
                "Attempted pyarrow/fastparquet and low-level pyarrow read, but all failed. "
                "This often means the parquet file is corrupted or uses an exotic writer/encoding. "
                "Try one of:\n"
                " - upgrading/downgrading pyarrow or fastparquet\n"
                " - opening & rewriting the file with the original writer if available\n"
                " - inspecting the file for corruption\n"
                f"\nUnderlying error: {e_low!r}"
            ) from e_low

    # JSON-like
    if suffix in [".json", ".ndjson", ".lines"]:
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path)

    # assume CSV-like
    encodings_to_try = ["utf-8", "latin1", "cp1252", "utf-16"]
    last_exc = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            print(f"Loaded file using encoding: {enc}")
            return df
        except Exception as e:
            last_exc = e
            continue
    # final attempt without dtype override
    try:
        df = pd.read_csv(path)
        print("Loaded CSV without forced dtype")
        return df
    except Exception:
        raise last_exc or Exception("Unable to read CSV file with tried encodings.")

def compute_opening_balance_for_month(account_df, year, month):
    month_start = pd.Timestamp(year=year, month=month, day=1)
    prior_ab = account_df[account_df["transaction_date_parsed"] < month_start].copy()
    ab_non_null = prior_ab[prior_ab["availablebalance"].notna()]
    if not ab_non_null.empty:
        last_row = ab_non_null.sort_values("transaction_date_parsed").iloc[-1]
        return float(last_row["availablebalance"])
    return float(prior_ab["signed_amount"].sum() if not prior_ab.empty else 0.0)



def build_statement_flowables(account_df, year, month, opening_balance, currency_hint=None):
    # use same margins as when creating SimpleDocTemplate (see below)
    left_margin_mm = 12
    right_margin_mm = 12
    top_margin_mm = 12
    bottom_margin_mm = 12

    page_size = landscape(A4)
    page_width_pts, page_height_pts = page_size
    # convert mm to points (1 mm = 2.8346456693 points)
    mm_to_pt = 2.8346456693
    usable_width = page_width_pts - (left_margin_mm + right_margin_mm) * mm_to_pt

    month_start = pd.Timestamp(year=year, month=month, day=1)
    next_month = (month_start + pd.offsets.MonthBegin(1))
    txs = account_df[(account_df["transaction_date_parsed"] >= month_start) & (account_df["transaction_date_parsed"] < next_month)].copy()
    txs = txs.sort_values("transaction_date_parsed")

    # running balance
    txs["running_balance"] = txs["signed_amount"].cumsum() + opening_balance

    total_cr = txs.loc[txs["txn_type"]=="CR", "amount"].sum()
    total_dr = txs.loc[txs["txn_type"]=="DR", "amount"].sum()
    closing_balance = opening_balance + txs["signed_amount"].sum()

    # Styles
    header_style = ParagraphStyle("h", fontSize=12, leading=14)
    small = ParagraphStyle("small", fontSize=8, leading=10)            # for table cells
    small_bold = ParagraphStyle("small_bold", fontSize=9, leading=11)
    normal = ParagraphStyle("n", fontSize=10, leading=12)

    flowables = []
    acc_id = account_df["account_id"].iloc[0]
    cust_name = None
    for c in ["customer_name", "name", "account_name"]:
        if c in account_df.columns and account_df[c].notna().any():
            cust_name = account_df[c].iloc[0]
            break

    period_str = month_start.strftime("%B %Y")
    header_text = f"<b>Account:</b> {acc_id}"
    if cust_name:
        header_text = f"<b>{cust_name}</b><br/>{header_text}"
    header_text += f"<br/><b>Statement period:</b> {period_str}<br/><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    flowables.append(Paragraph(header_text, header_style))
    flowables.append(Spacer(1, 6))

    # Summary table (unchanged)
    summary = [
        ["Opening balance", fmt_currency(opening_balance, currency_hint)],
        ["Total credits", fmt_currency(total_cr, currency_hint)],
        ["Total debits", fmt_currency(total_dr, currency_hint)],
        ["Closing balance", fmt_currency(closing_balance, currency_hint)]
    ]
    tbl = Table(summary, colWidths=[90*mm, 60*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
    ]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 8))

    # Prepare transactions table rows using Paragraphs for wrap
    tx_rows = [["Date", "Txn ID", "Type", "Description", "Debit", "Credit", "Balance"]]

    for _, r in txs.iterrows():
        date_s = r["transaction_date_parsed"].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(r["transaction_date_parsed"]) else ""
        txn_id = str(r.get("transaction_id",""))
        ttype = str(r.get("txn_type",""))
        desc = (r.get("description") or "")
        debit = fmt_currency(r["amount"], currency_hint) if r["txn_type"]=="DR" else ""
        credit = fmt_currency(r["amount"], currency_hint) if r["txn_type"]=="CR" else ""
        bal = fmt_currency(r["running_balance"], currency_hint)

        # wrap long fields into Paragraph so they will not overflow
        date_p = Paragraph(date_s, small)
        txnid_p = Paragraph(txn_id, small)
        type_p = Paragraph(ttype, small)
        desc_p = Paragraph(desc.replace("\n", " "), small)   # remove newlines or keep as needed
        debit_p = Paragraph(debit, small)
        credit_p = Paragraph(credit, small)
        bal_p = Paragraph(bal, small)

        tx_rows.append([date_p, txnid_p, type_p, desc_p, debit_p, credit_p, bal_p])

    if len(tx_rows) == 1:
        tx_rows.append([Paragraph(f"No transactions during {period_str}", small), "", "", "", "", "", Paragraph(fmt_currency(opening_balance, currency_hint), small)])

    # Choose column widths that fit the usable width. Adjust the description width to take most space.
    # Example proportions (sum should be ~ usable_width):
    col_widths_pts = [
        40 * mm_to_pt,   # Date
        35 * mm_to_pt,   # Txn ID
        18 * mm_to_pt,   # Type
        (usable_width - ((40+35+18+22+22+28) * mm_to_pt)) ,  # Description: take remaining width
        22 * mm_to_pt,   # Debit
        22 * mm_to_pt,   # Credit
        28 * mm_to_pt    # Balance
    ]
    # In case remaining calc gives negative or too-small, fallback to fixed widths
    if col_widths_pts[3] < 60 * mm_to_pt:
        col_widths_pts = [40*mm_to_pt, 35*mm_to_pt, 18*mm_to_pt, 110*mm_to_pt, 22*mm_to_pt, 22*mm_to_pt, 28*mm_to_pt]

    tx_table = Table(tx_rows, colWidths=col_widths_pts, repeatRows=1)
    tx_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (4,1), (6,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    flowables.append(tx_table)
    flowables.append(PageBreak())
    return flowables

def generate_statements(df_path, output_dir="statements"):
    df = load_and_normalize(df_path)
    os.makedirs(output_dir, exist_ok=True)

    for account_id, account_df in df.groupby("account_id"):
        account_df = account_df.sort_values("transaction_date_parsed").reset_index(drop=True)
        account_df["year"] = account_df["transaction_date_parsed"].dt.year
        account_df["month"] = account_df["transaction_date_parsed"].dt.month
        ym_pairs = account_df.dropna(subset=["year","month"]).groupby(["year","month"]).size().index.tolist()

        currency_hint = account_df["currency"].dropna().mode().iloc[0] if "currency" in account_df.columns and account_df["currency"].notna().any() else None

        for year, month in sorted(ym_pairs):
            opening_balance = compute_opening_balance_for_month(account_df, year, month)
            flowables = build_statement_flowables(account_df, year, month, opening_balance, currency_hint)
            fname = f"{account_id}_{year}_{int(month):02d}_statement.pdf"
            outpath = Path(output_dir) / fname
            doc = SimpleDocTemplate(str(outpath), pagesize=A4,
                                    rightMargin=12*mm, leftMargin=12*mm,
                                    topMargin=12*mm, bottomMargin=12*mm)
            doc.build(flowables)
            print(f"Generated {outpath}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("transactions_csv", help="CSV file (per-account transactions) with the columns shown in the screenshot")
    # parser.add_argument("--output-dir", default="statements", help="Directory to write PDFs")
    # args = parser.parse_args()
    generate_statements(r"C:\Users\niles\Downloads\2dc.parquet", "statements")
