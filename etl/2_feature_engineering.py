import pandas as pd
import sqlite3
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  
DB_PATH    = os.path.join(ROOT_DIR, "app", "enterprise_crm.db")


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


print("⚙️ [2/4] Engineering Enterprise RFM Features...")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
elif 'UnitPrice' in df.columns:
    df['TotalSpend'] = df['UnitPrice']
else:
    print("❌ ERROR: Missing pricing columns in the database!")
    sys.exit()


current_date = df['InvoiceDate'].max()

df_rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'InvoiceNo':   'nunique',                                 # Frequency
    'TotalSpend':  'sum'                                      # Monetary
}).reset_index()

df_rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']


print("🏷️ [3/4] Generating AI Training Labels (Churn)...")


churn_threshold = df_rfm['Recency'].quantile(0.70)
df_rfm['Churn_Label'] = (df_rfm['Recency'] > churn_threshold).astype(int)

print(f"   Churn threshold (Recency): {churn_threshold:.0f} days")
print(f"   Label split (0=Loyal, 1=Churned): {df_rfm['Churn_Label'].value_counts().to_dict()}")

print("💾 [4/4] Saving engineered features to database...")
df_rfm.to_sql("customer_features", conn, if_exists="replace", index=False)
conn.close()

print("=" * 60)
print("✅ SUCCESS: Feature Engineering complete! Data is ready for the AI.")
print(f"   {len(df_rfm)} customer profiles saved to: {DB_PATH}")
print("=" * 60)