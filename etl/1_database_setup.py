import logging
import os
import sqlite3
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    APP_DIR    = os.path.join(ROOT_DIR, "app")
    DATA_DIR   = os.path.join(ROOT_DIR, "data")
    DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")

    log.info("[1/4] Loading E-Commerce Data...")

    excel_path = os.path.join(DATA_DIR, "Online Retail.xlsx")
    csv_path   = os.path.join(DATA_DIR, "Online Retail.csv")

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="unicode_escape")
    else:
        raise FileNotFoundError(f"No data file found in {DATA_DIR}")

    log.info("[2/4] Executing Data Reconciliation...")
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "TransactionNo": "InvoiceNo",
        "OrderID":       "InvoiceNo",
        "Date":          "InvoiceDate",
        "OrderDate":     "InvoiceDate",
        "TotalAmount":   "TotalSpend",
        "Price":         "UnitPrice",
        "CustomerNo":    "CustomerID",
        "Customer ID":   "CustomerID",
    })

    if "TotalSpend" not in df.columns:
        if "UnitPrice" in df.columns and "Quantity" in df.columns:
            log.info("TotalSpend missing — calculating from UnitPrice * Quantity")
            df["TotalSpend"] = df["UnitPrice"] * df["Quantity"]
        else:
            raise ValueError("Missing financial columns (TotalSpend or UnitPrice/Quantity)")

    if "OrderStatus" in df.columns:
        df = df[df["OrderStatus"].isin(["Delivered"])]

    df = df.dropna(subset=["CustomerID"])
    df["Quantity"]    = df["Quantity"].clip(lower=1)
    df                = df[df["TotalSpend"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["CustomerID"]  = df["CustomerID"].astype(str)

    log.info(f"Clean rows after reconciliation: {len(df):,}")

    log.info("[3/4] Saving to Database...")

    df_final = df[["InvoiceNo", "InvoiceDate", "CustomerID", "TotalSpend", "Quantity"]]
    with sqlite3.connect(DB_PATH) as conn:
        df_final.to_sql("raw_transactions", conn, if_exists="replace", index=False)

    log.info("[4/4] SUCCESS: Data pipeline complete. DB saved to: %s", DB_PATH)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log.error("Pipeline failed: %s", e)
        sys.exit(1)
