import pandas as pd
import sqlite3
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  
APP_DIR    = os.path.join(ROOT_DIR, "app")
DATA_DIR   = os.path.join(ROOT_DIR, "data")
DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")


print("⏳ [1/4] Loading E-Commerce Data...")

excel_path = os.path.join(DATA_DIR, "Online Retail.xlsx")
csv_path   = os.path.join(DATA_DIR, "Online Retail.csv")

if os.path.exists(excel_path):
    print("   ✅ Detected Excel file (.xlsx)...")
    df = pd.read_excel(excel_path)

elif os.path.exists(csv_path):
    print("   ✅ Detected CSV file (.csv)...")
    df = pd.read_csv(csv_path, encoding='unicode_escape')

else:
    print(f"❌ ERROR: Could not find 'Online Retail.xlsx' or 'Online Retail.csv' in: {DATA_DIR}")
    sys.exit()

print(f"   Raw rows found: {len(df)}")


print("⚙️ [2/4] Executing Data Reconciliation (Cleaning)...")

df.columns = df.columns.str.strip()

if 'OrderStatus' in df.columns:
    df = df[df['OrderStatus'].isin(['Delivered'])]

if 'TotalAmount' in df.columns and 'UnitPrice' in df.columns:
    df = df.drop(columns=['UnitPrice'])

df = df.rename(columns={
    'TransactionNo': 'InvoiceNo',
    'OrderID':       'InvoiceNo',
    'Date':          'InvoiceDate',
    'OrderDate':     'InvoiceDate',
    'TotalAmount':   'TotalSpend', 
    'CustomerNo':    'CustomerID',
    'Customer ID':   'CustomerID'
})

if 'CustomerID' not in df.columns:
    print(f"❌ ERROR: 'CustomerID' column missing! Available columns: {list(df.columns)}")
    sys.exit()

df = df.dropna(subset=['CustomerID'])

if 'Quantity' in df.columns:
    df = df[df['Quantity'] > 0]
else:
    df['Quantity'] = 1

if 'TotalSpend' in df.columns:
    df = df[df['TotalSpend'] > 0]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
df['CustomerID']  = df['CustomerID'].astype(str)

print(f"   Clean rows remaining: {len(df)}")

print(f"💾 [3/4] Saving database to: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
print("⏳ [4/4] Writing to 'raw_transactions' table...")
df_final = df[['InvoiceNo', 'InvoiceDate', 'CustomerID', 'TotalSpend', 'Quantity']]
df_final.to_sql("raw_transactions", conn, if_exists="replace", index=False)

cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM raw_transactions")
row_count = cursor.fetchone()[0]
conn.close()

print("=" * 60)
print(f"✅ SUCCESS: {row_count} transactions saved to {DB_PATH}")
print("=" * 60)