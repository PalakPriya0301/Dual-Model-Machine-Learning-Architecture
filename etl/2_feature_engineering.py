# import pandas as pd
# import sqlite3
# import sys

# print("⏳ [1/4] Loading raw transactions from database...")
# try:
#     conn = sqlite3.connect("enterprise_crm.db")
#     df = pd.read_sql("SELECT * FROM raw_transactions", conn)
# except Exception as e:
#     print(f"❌ ERROR: Could not read database. Did you run Step 1? Details: {e}")
#     sys.exit()

# print("⚙️ [2/4] Engineering Enterprise RFM Features...")
# # Convert string dates to datetime objects safely
# df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# # 💰 REVENUE FIX: Calculate actual spend per row safely
# if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
#     df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
# elif 'UnitPrice' in df.columns:
#     df['TotalSpend'] = df['UnitPrice']
# else:
#     print("❌ ERROR: Missing pricing columns in the database!")
#     sys.exit()

# # ⏱️ TIME TRAVEL FIX: Set "Today" to the newest date in the dataset
# current_date = df['InvoiceDate'].max()

# # Group by customer to calculate Recency, Frequency, and Monetary value
# df_rfm = df.groupby('CustomerID').agg({
#     'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
#     'InvoiceNo': 'nunique',                                  # Frequency
#     'TotalSpend': 'sum'                                      # Monetary (True Revenue)
# }).reset_index()

# # Rename columns to standard pipeline names
# df_rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# print("🏷️ [3/4] Generating AI Training Labels (Churn)...")
# # ⚖️ THE REALITY FIX: Shift Churn to the 70th percentile (30% Churn Rate)
# churn_threshold = df_rfm['Recency'].quantile(0.70)
# df_rfm['Churn_Label'] = (df_rfm['Recency'] > churn_threshold).astype(int)

# # Print out the split so we can prove it's balanced
# print(f"   AI Training Split (0=Loyal, 1=Churned): {df_rfm['Churn_Label'].value_counts().to_dict()}")

# print("💾 [4/4] Saving engineered features to database...")
# df_rfm.to_sql("customer_features", conn, if_exists="replace", index=False)
# conn.close()

# print("=" * 60)
# print("✅ SUCCESS: Feature Engineering complete! Data is ready for the AI.")
# print(f"   {len(df_rfm)} customer profiles saved to:")
# print("=" * 60)


import pandas as pd
import sqlite3
import sys
import os

# ── PATH RESOLUTION ───────────────────────────────────────────
# This script lives in: MAJOR PROJECT-B.../etl/
# DB lives in:          MAJOR PROJECT-B.../app/enterprise_crm.db
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../etl/
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # .../MAJOR PROJECT-B.../
DB_PATH    = os.path.join(ROOT_DIR, "app", "enterprise_crm.db")

# ── 1. LOAD RAW TRANSACTIONS ──────────────────────────────────
print("⏳ [1/4] Loading raw transactions from database...")
print(f"   DB path: {DB_PATH}")

if not os.path.exists(DB_PATH):
    print(f"❌ ERROR: Database not found at {DB_PATH}")
    print("   Did you run etl/1_database_setup.py first?")
    sys.exit()

try:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM raw_transactions", conn)
except Exception as e:
    print(f"❌ ERROR: Could not read database. Details: {e}")
    sys.exit()

# ── 2. FEATURE ENGINEERING ────────────────────────────────────
print("⚙️ [2/4] Engineering Enterprise RFM Features...")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
elif 'UnitPrice' in df.columns:
    df['TotalSpend'] = df['UnitPrice']
else:
    print("❌ ERROR: Missing pricing columns in the database!")
    sys.exit()

# Set "today" to the newest date in the dataset (avoids future-date issues)
current_date = df['InvoiceDate'].max()

df_rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'InvoiceNo':   'nunique',                                 # Frequency
    'TotalSpend':  'sum'                                      # Monetary
}).reset_index()

df_rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# ── 3. CHURN LABELS ───────────────────────────────────────────
print("🏷️ [3/4] Generating AI Training Labels (Churn)...")

# 70th percentile Recency threshold → ~30% churn rate
churn_threshold = df_rfm['Recency'].quantile(0.70)
df_rfm['Churn_Label'] = (df_rfm['Recency'] > churn_threshold).astype(int)

print(f"   Churn threshold (Recency): {churn_threshold:.0f} days")
print(f"   Label split (0=Loyal, 1=Churned): {df_rfm['Churn_Label'].value_counts().to_dict()}")

# ── 4. SAVE TO DATABASE ───────────────────────────────────────
print("💾 [4/4] Saving engineered features to database...")
df_rfm.to_sql("customer_features", conn, if_exists="replace", index=False)
conn.close()

print("=" * 60)
print("✅ SUCCESS: Feature Engineering complete! Data is ready for the AI.")
print(f"   {len(df_rfm)} customer profiles saved to: {DB_PATH}")
print("=" * 60)