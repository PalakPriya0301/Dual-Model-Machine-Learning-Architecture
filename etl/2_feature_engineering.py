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
    DB_PATH    = os.path.join(ROOT_DIR, "app", "enterprise_crm.db")

    log.info("[1/4] Loading raw transactions from database: %s", DB_PATH)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Did you run etl/1_database_setup.py first?"
        )

    with sqlite3.connect(DB_PATH) as conn:

        try:
            df = pd.read_sql("SELECT * FROM raw_transactions", conn)
        except Exception as e:
            raise RuntimeError(f"Could not read database: {e}") from e

        log.info("[2/4] Engineering Enterprise RFM Features...")

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        if "TotalSpend" not in df.columns:
            raise ValueError("'TotalSpend' missing. Re-run 1_database_setup.py first.")

        current_date = df["InvoiceDate"].max()

        df_rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (current_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),   # FIX: nunique on InvoiceNo, not CustomerID
            Monetary=("TotalSpend", "sum"),
            TotalQuantity=("Quantity", "sum"),
        ).reset_index()

        df_rfm["AvgOrderValue"] = df_rfm["Monetary"] / df_rfm["Frequency"]

      

        log.info("[3/4] Generating AI Training Labels (Churn)...")

        for pct in [0.60, 0.70, 0.80]:
            threshold = df_rfm["Recency"].quantile(pct)
            churned   = (df_rfm["Recency"] > threshold).sum()
            loyal     = len(df_rfm) - churned
            log.info(
                "  Threshold @ %.0f%%ile = %.0f days → Churned: %d | Loyal: %d",
                pct * 100, threshold, churned, loyal,
            )

        churn_threshold     = df_rfm["Recency"].quantile(0.70)
        df_rfm["Churn_Label"] = (df_rfm["Recency"] > churn_threshold).astype(int)

        log.info(
            "Selected threshold (70th pct): %.0f days — Label split: %s",
            churn_threshold,
            df_rfm["Churn_Label"].value_counts().to_dict(),
        )

        log.info("[4/4] Saving engineered features to database...")
        df_rfm.to_sql("customer_features", conn, if_exists="replace", index=False)

    log.info("=" * 60)
    log.info("SUCCESS: Feature Engineering complete!")
    log.info("%d customer profiles saved to: %s", len(df_rfm), DB_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log.error("Feature engineering failed: %s", e)
        sys.exit(1)
