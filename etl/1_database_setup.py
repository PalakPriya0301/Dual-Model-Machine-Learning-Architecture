import pandas as pd
import sqlite3
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  
APP_DIR    = os.path.join(ROOT_DIR, "app")
DATA_DIR   = os.path.join(ROOT_DIR, "data")
DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")

print("[1/4] Loading E-Commerce Data...")

excel_path = os.path.join(DATA_DIR, "Online Retail.xlsx")
csv_path   = os.path.join(DATA_DIR, "Online Retail.csv")

if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
elif os.path.exists(csv_path):
    df = pd.read_csv(csv_path, encoding='unicode_escape')
else:
    print(f" ERROR: No data file found in {DATA_DIR}")
    sys.exit()

print("⚙️ [2/4] Executing Data Reconciliation...")
df.columns = df.columns.str.strip()

# 1. Standardize column names
df = df.rename(columns={
    'TransactionNo': 'InvoiceNo',
    'OrderID':       'InvoiceNo',
    'Date':          'InvoiceDate',
    'OrderDate':     'InvoiceDate',
    'TotalAmount':   'TotalSpend', 
    'Price':         'UnitPrice',
    'CustomerNo':    'CustomerID',
    'Customer ID':   'CustomerID'
})


if 'TotalSpend' not in df.columns:
    if 'UnitPrice' in df.columns and 'Quantity' in df.columns:
        print("    TotalSpend missing! Calculating from UnitPrice * Quantity...")
        df['TotalSpend'] = df['UnitPrice'] * df['Quantity']
    else:
        print(" ERROR: Missing financial columns (TotalSpend or UnitPrice/Quantity).")
        sys.exit()

if 'OrderStatus' in df.columns:
    df = df[df['OrderStatus'].isin(['Delivered'])]

df = df.dropna(subset=['CustomerID'])
df['Quantity'] = df['Quantity'].clip(lower=1) # Ensure at least 1
df = df[df['TotalSpend'] > 0] # Filter out free/invalid orders

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
df['CustomerID']  = df['CustomerID'].astype(str)

print(f"   Clean rows: {len(df)}")

print("[3/4] Saving to Database...")
conn = sqlite3.connect(DB_PATH)
df_final = df[['InvoiceNo', 'InvoiceDate', 'CustomerID', 'TotalSpend', 'Quantity']]
df_final.to_sql("raw_transactions", conn, if_exists="replace", index=False)
conn.close()

print("✅ SUCCESS: Data pipeline stabilized.")