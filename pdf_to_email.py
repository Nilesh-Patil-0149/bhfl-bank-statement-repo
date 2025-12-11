#!/usr/bin/env python3
"""
statements_send_email_job.py

Second processing job: sequentially read partitioned parquet from S3,
generate monthly PDF statements for each unique customer partition,
upload PDFs to S3, create presigned URLs and send emails via SES.

Requirements:
  pip install pandas pyarrow s3fs boto3 reportlab

Run in SageMaker Processing container or any environment with AWS access / IAM role.
"""

import os
import io
import json
import tempfile
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import boto3
import pandas as pd

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# -------------------
# CONFIG - override via environment variables
# -------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "bhfl-pipeline000")
PARQUET_ROOT = os.environ.get("PARQUET_ROOT", "data2/transactions_partitioned")  # relative path in bucket
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "job-outputs/statements")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# Sender must be verified in SES for the region (or domain must be verified).
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "nileshkcp@gmail.com")

# expiry - accept numeric string or simple expression like "7*24*3600"
_raw_expiry = os.environ.get("PRESIGNED_EXPIRY", "604800")  # default 7 days in seconds
try:
    PRESIGNED_EXPIRY = int(_raw_expiry)
except Exception:
    try:
        PRESIGNED_EXPIRY = int(eval(_raw_expiry, {"__builtins__": None}, {}))
    except Exception:
        PRESIGNED_EXPIRY = 7 * 24 * 3600

# Logo settings
LOGO_S3_PATH = os.environ.get("LOGO_S3_PATH", f"s3://{S3_BUCKET}/assets/bhfl_logo.png")
LOCAL_LOGO_PATH = os.environ.get("LOCAL_LOGO_PATH", "/opt/ml/processing/input/bhfl_logo.png")
LOGO_MAX_HEIGHT_MM = float(os.environ.get("LOGO_MAX_HEIGHT_MM", "18"))

# DLQ prefix to write failed email records
DLQ_PREFIX = os.environ.get("DLQ_PREFIX", "job-outputs/email_dlq")

# SES region
AWS_REGION = os.environ.get("AWS_REGION", AWS_REGION)

# Logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("statements_job")

# boto3 clients
s3 = boto3.client("s3", region_name=AWS_REGION)
s3_resource = boto3.resource("s3", region_name=AWS_REGION)
ses = boto3.client("ses", region_name=AWS_REGION)

# Try to import s3fs so pandas can read s3:// paths directly
try:
    import s3fs  # noqa: F401
    S3FS_AVAILABLE = True
except Exception:
    S3FS_AVAILABLE = False

# -------------------
# Utilities: S3, file listing, reading parquet
# -------------------
def list_customer_prefixes(bucket: str, parquet_root: str) -> List[str]:
    """
    List discovered partition prefixes in the form: 'data2/.../customer_id=XXX/' (S3 key prefix)
    Returns sorted list of prefix strings (relative key under bucket, including trailing '/').
    """
    prefix = parquet_root.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    prefixes = set()
    logger.info("Listing objects under s3://%s/%s", bucket, prefix)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            for seg in parts:
                if seg.startswith("customer_id="):
                    parent = "/".join(parts[:parts.index(seg)+1]) + "/"
                    prefixes.add(parent)
    return sorted(prefixes)

def read_customer_parquet(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Read all parquet files under s3://bucket/{prefix} into pandas DataFrame.
    prefix example: 'data2/.../customer_id=ACC101/'
    """
    s3_uri = f"s3://{bucket}/{prefix}"
    # Try direct pandas read_parquet (requires s3fs + pyarrow)
    if S3FS_AVAILABLE:
        try:
            df = pd.read_parquet(s3_uri, engine="pyarrow")
            return df
        except Exception as e:
            logger.debug("pd.read_parquet direct failed for %s: %s", s3_uri, e)

    # Fallback: list parquet files and read each via boto3
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [o["Key"] for o in objs.get("Contents", []) if o["Key"].endswith(".parquet")]
    parts = []
    import io
    for key in files:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        parts.append(pd.read_parquet(io.BytesIO(body), engine="pyarrow"))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()

def upload_bytes_to_s3(bucket: str, key: str, data: bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    logger.info("Uploaded: s3://%s/%s", bucket, key)

def make_presigned_url(bucket: str, key: str, expiry: int = PRESIGNED_EXPIRY) -> str:
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expiry)
    )
    return url

# -------------------
# PDF template + generator (BHFL standard)
# -------------------
def fmt_currency(x, curr_symbol=None):
    try:
        return f"{curr_symbol or 'INR'} {float(x):,.2f}"
    except Exception:
        return str(x)

def load_logo_to_temp(logo_s3_path: str = LOGO_S3_PATH, local_path: str = LOCAL_LOGO_PATH) -> Optional[str]:
    """
    Try to fetch image from S3 and write to a temp file. If fails, fall back to local path.
    Returns a filesystem path to the image or None.
    """
    try:
        if logo_s3_path and logo_s3_path.startswith("s3://"):
            _, rest = logo_s3_path.split("s3://", 1)
            bucket, key = rest.split("/", 1)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix)
            tmp.write(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            tmp.flush()
            tmp.close()
            return tmp.name
    except Exception as e:
        logger.debug("Logo fetch from S3 failed: %s", e)
    # fallback local
    try:
        if local_path and Path(local_path).exists():
            return str(Path(local_path))
    except Exception:
        pass
    return None

def build_standard_statement_flowables(account_df: pd.DataFrame, year: int, month: int, opening_balance: float, currency_hint=None) -> List:
    """
    Returns a list of ReportLab flowables forming a polished statement page (landscape A4).
    """
    page_size = landscape(A4)
    page_width_pts, page_height_pts = page_size
    left_margin_mm = 12
    right_margin_mm = 12
    mm_to_pt = 2.8346456693
    usable_width = page_width_pts - (left_margin_mm + right_margin_mm) * mm_to_pt

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle("header", parent=styles["Heading2"], fontSize=14, leading=16)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    normal = ParagraphStyle("normal", parent=styles["BodyText"], fontSize=10, leading=12)
    muted = ParagraphStyle("muted", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.grey)

    flowables = []

    # Header with logo and account meta
    logo_path = load_logo_to_temp()
    logo_height = LOGO_MAX_HEIGHT_MM * mm_to_pt
    if logo_path:
        try:
            img = Image(logo_path)
            img.drawHeight = logo_height
            img.drawWidth = img.drawHeight * (img.imageWidth / img.imageHeight)
            left_cell = [img]
        except Exception:
            left_cell = [Paragraph("BHFL", header_style)]
    else:
        left_cell = [Paragraph("BHFL", header_style)]

    account_id = str(account_df.get("account_id", account_df.get("customer_id")).iloc[0])
    cust_name = None
    for c in ["customer_name", "name", "account_name"]:
        if c in account_df.columns and account_df[c].notna().any():
            cust_name = str(account_df[c].dropna().iloc[0])
            break

    right_lines = []
    right_lines.append(f"<b>BHFL</b>")
    if cust_name:
        right_lines.append(cust_name)
    right_lines.append(f"<b>Account:</b> {account_id}")
    period_str = pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")
    right_lines.append(f"<b>Statement period:</b> {period_str}")
    right_lines.append(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")

    right_cell = [Paragraph("<br/>".join(right_lines), normal)]

    header_tbl = Table([[left_cell, right_cell]], colWidths=[usable_width * 0.25, usable_width * 0.75])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    flowables.append(header_tbl)
    flowables.append(Spacer(1, 6))

    # Summary box
    df = account_df.copy()
    df = df.sort_values("transaction_date_parsed")
    txs_month = df[(df["transaction_date_parsed"].dt.year == year) & (df["transaction_date_parsed"].dt.month == month)]
    total_cr = txs_month.loc[txs_month["txn_type"]=="CR", "amount"].sum() if "amount" in txs_month.columns else 0.0
    total_dr = txs_month.loc[txs_month["txn_type"]=="DR", "amount"].sum() if "amount" in txs_month.columns else 0.0
    closing_balance = opening_balance + (txs_month["signed_amount"].sum() if "signed_amount" in txs_month.columns else 0.0)

    sum_table = [
        ["Opening balance", fmt_currency(opening_balance, currency_hint)],
        ["Total credits", fmt_currency(total_cr, currency_hint)],
        ["Total debits", fmt_currency(total_dr, currency_hint)],
        ["Closing balance", fmt_currency(closing_balance, currency_hint)]
    ]
    sum_tbl = Table(sum_table, colWidths=[usable_width*0.6, usable_width*0.4])
    sum_tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    flowables.append(sum_tbl)
    flowables.append(Spacer(1, 8))

    # Transaction table
    tx_rows = [["Date", "Txn ID", "Type", "Description", "Debit", "Credit", "Balance"]]
    for _, r in txs_month.iterrows():
        date_s = r["transaction_date_parsed"].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(r["transaction_date_parsed"]) else ""
        txn_id = str(r.get("transaction_id", ""))
        ttype = str(r.get("txn_type", ""))
        desc = (r.get("description") or "")
        # numeric amount (if present)
        amt = None
        if "amount" in r and r.get("amount") is not None:
            try:
                amt = float(r.get("amount"))
            except Exception:
                amt = None
        # signed amount (preferred)
        signed = None
        if "signed_amount" in r and r.get("signed_amount") is not None:
            try:
                signed = float(r.get("signed_amount"))
            except Exception:
                signed = None
    
        debit = credit = ""
        if signed is not None:
            if signed < 0:
                debit = fmt_currency(abs(signed), currency_hint)
            elif signed > 0:
                credit = fmt_currency(signed, currency_hint)
        elif amt is not None:
            if amt < 0:
                debit = fmt_currency(abs(amt), currency_hint)
            else:
                credit = fmt_currency(amt, currency_hint)
    
        bal = fmt_currency(r.get("running_balance"), currency_hint) if "running_balance" in r else ""
        tx_rows.append([Paragraph(date_s, small), Paragraph(txn_id, small), Paragraph(ttype, small),
                        Paragraph(desc.replace("\n", " "), small), Paragraph(debit, small),
                        Paragraph(credit, small), Paragraph(bal, small)])

    if len(tx_rows) == 1:
        tx_rows.append([Paragraph(f"No transactions during {period_str}", small), "", "", "", "", "", Paragraph(fmt_currency(opening_balance, currency_hint), small)])

    col_widths_pts = [
        80, 75, 45, usable_width - (80+75+45+70+70+90), 70, 70, 90
    ]
    if col_widths_pts[3] < 110:
        col_widths_pts = [80, 75, 45, 160, 70, 70, 90]

    tx_table = Table(tx_rows, colWidths=col_widths_pts, repeatRows=1)
    tx_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (4,1), (6,-1), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    flowables.append(tx_table)
    flowables.append(Spacer(1, 10))

    footer = Paragraph("BHFL Customer Care | +91-22-XXXX-XXXX | support@bhfl.example.com", muted)
    flowables.append(Spacer(1, 6))
    flowables.append(footer)
    flowables.append(PageBreak())
    return flowables

def generate_standard_statement_pdf_bytes(account_df: pd.DataFrame, year: int, month: int, opening_balance: float, currency_hint=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            rightMargin=12*mm, leftMargin=12*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    flowables = build_standard_statement_flowables(account_df, year, month, opening_balance, currency_hint)
    doc.build(flowables)
    buf.seek(0)
    return buf.read()

# -------------------
# Data normalization & helpers
# -------------------
def normalize_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure customer_id column
    if "customer_id" not in df.columns and "account_id" in df.columns:
        df = df.rename(columns={"account_id": "customer_id"})
    # Ensure account_id exists for naming
    if "account_id" not in df.columns:
        df["account_id"] = df.get("account_id", df["customer_id"].astype(str))
    # Parse transaction_date into transaction_date_parsed
    if "transaction_date" in df.columns:
        df["transaction_date_parsed"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    else:
        for c in ["txn_date", "date", "created_at"]:
            if c in df.columns:
                df["transaction_date_parsed"] = pd.to_datetime(df[c], errors="coerce")
                break
    # signed_amount
    if "signed_amount" not in df.columns:
        if "amount" in df.columns and "txn_type" in df.columns:
            def signed(row):
                try:
                    v = float(row["amount"])
                except Exception:
                    return 0.0
                return v if str(row["txn_type"]).upper()=="CR" else -abs(v)
            df["signed_amount"] = df.apply(signed, axis=1)
        elif "amount" in df.columns:
            df["signed_amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        else:
            df["signed_amount"] = 0.0
    return df

def compute_opening_balance_for_month(account_df: pd.DataFrame, year: int, month: int) -> float:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    prior_ab = account_df[account_df["transaction_date_parsed"] < month_start].copy()
    if "availablebalance" in prior_ab.columns and prior_ab["availablebalance"].notna().any():
        last_row = prior_ab.sort_values("transaction_date_parsed").iloc[-1]
        try:
            return float(last_row["availablebalance"])
        except Exception:
            pass
    if "signed_amount" in prior_ab.columns and not prior_ab.empty:
        return float(prior_ab["signed_amount"].sum())
    return 0.0

def find_email_in_df(df: pd.DataFrame) -> Optional[str]:
    for c in ["customer_email", "email", "contact_email", "email_address"]:
        if c in df.columns and df[c].notna().any():
            return str(df[c].dropna().iloc[0])
    return None

# -------------------
# Email send + DLQ
# -------------------
def write_dlq_record(bucket: str, customer_id: str, pdf_keys: List[str], recipient: Optional[str], error_msg: str):
    key = f"{DLQ_PREFIX.rstrip('/')}/{customer_id}/{int(time.time())}.json"
    payload = {
        "customer_id": customer_id,
        "recipient": recipient,
        "pdf_keys": pdf_keys,
        "error": error_msg,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode("utf-8"))
    logger.info("Wrote DLQ record s3://%s/%s", bucket, key)

def send_email_with_links(recipient: str, subject: str, html_body: str, text_body: Optional[str], customer_id: str, pdf_keys: List[str]):
    if text_body is None:
        text_body = "Please view your statement(s) using the links in the HTML email."

    try:
        resp = ses.send_email(
            Source=SENDER_EMAIL,
            Destination={"ToAddresses":[recipient]},
            Message={
                "Subject": {"Data": subject},
                "Body": {
                    "Text": {"Data": text_body},
                    "Html": {"Data": html_body}
                }
            }
        )
        logger.info("Sent email to %s MessageId=%s", recipient, resp.get("MessageId"))
        return True
    except Exception as e:
        # log and write DLQ
        logger.exception("Failed to send email to %s", recipient)
        write_dlq_record(S3_BUCKET, customer_id, pdf_keys, recipient, str(e))
        return False

# -------------------
# Core processing loop
# -------------------
def generate_pdf_files_for_account(df: pd.DataFrame, account_id: str) -> List[Tuple[str, bytes]]:
    """
    Returns list of tuples (filename, pdf_bytes)
    """
    df = normalize_transactions_df(df)
    df = df.sort_values("transaction_date_parsed").reset_index(drop=True)
    df["year"] = df["transaction_date_parsed"].dt.year
    df["month"] = df["transaction_date_parsed"].dt.month
    ym_pairs = df.dropna(subset=["year","month"]).groupby(["year","month"]).size().index.tolist()
    generated = []
    currency_hint = df["currency"].dropna().mode().iloc[0] if "currency" in df.columns and df["currency"].notna().any() else "INR"
    for year, month in sorted(ym_pairs):
        opening_balance = compute_opening_balance_for_month(df, int(year), int(month))
        pdf_bytes = generate_standard_statement_pdf_bytes(df, int(year), int(month), opening_balance, currency_hint)
        fname = f"{account_id}_{year}_{int(month):02d}_statement.pdf"
        generated.append((fname, pdf_bytes))
    return generated

def main():
    logger.info("Starting sequential statement + email job")
    prefixes = list_customer_prefixes(S3_BUCKET, PARQUET_ROOT)
    logger.info("Found %d customer prefixes", len(prefixes))
    if not prefixes:
        logger.error("No customer partitions found under s3://%s/%s", S3_BUCKET, PARQUET_ROOT)
        return

    for idx, prefix in enumerate(prefixes, start=1):
        try:
            logger.info("[%d/%d] Processing prefix: %s", idx, len(prefixes), prefix)
            df = read_customer_parquet(S3_BUCKET, prefix)
            if df is None or df.empty:
                logger.info("No rows for prefix %s — skipping", prefix)
                continue

            # normalize/rename
            if "customer_id" not in df.columns and "account_id" in df.columns:
                df = df.rename(columns={"account_id":"customer_id"})
            cust_id = str(df["customer_id"].dropna().iloc[0])
            if "account_id" not in df.columns:
                df["account_id"] = cust_id

            # generate PDFs
            pdfs = generate_pdf_files_for_account(df, account_id=cust_id)
            if not pdfs:
                logger.info("No statements generated for customer %s", cust_id)
                continue

            # upload pdf(s) and create presigned urls
            links = []
            pdf_s3_keys = []
            for fname, pdf_bytes in pdfs:
                key = f"{OUTPUT_PREFIX.rstrip('/')}/customer_id={cust_id}/{fname}"
                upload_bytes_to_s3(S3_BUCKET, key, pdf_bytes)
                pdf_s3_keys.append(key)
                url = make_presigned_url(S3_BUCKET, key, expiry=PRESIGNED_EXPIRY)
                links.append((fname, url))

            # find recipient email
            recipient = "nileshkcp@gmail.com" #find_email_in_df(df)
            if not recipient:
                logger.warning("No email found for customer %s — writing DLQ record", cust_id)
                write_dlq_record(S3_BUCKET, cust_id, pdf_s3_keys, None, "no-email-found")
                continue

            # compose email
            subject = f"Your monthly BHFL statement(s) - Account {cust_id}"
            html_lines = [
                "<p>Dear Customer,</p>",
                "<p>Your monthly statement(s) are ready. Click the links below to download:</p>",
                "<ul>"
            ]
            for fname, url in links:
                html_lines.append(f"<li><a href='{url}' target='_blank'>{fname}</a></li>")
            html_lines.append(f"</ul><p>These links will expire in {int(PRESIGNED_EXPIRY//86400)} days.</p>")
            html_lines.append("<p>Regards,<br/>BHFL</p>")
            html_body = "\n".join(html_lines)
            text_body = "Please download your statements from the links in the email."

            ok = send_email_with_links(recipient, subject, html_body, text_body, cust_id, pdf_s3_keys)
            if not ok:
                logger.error("Email send failed for %s (DLQ written).", cust_id)
            else:
                logger.info("Processed customer %s successfully.", cust_id)

        except Exception:
            logger.exception("Failed processing prefix %s — continuing to next", prefix)
            continue

    logger.info("All done")

if __name__ == "__main__":
    main()
