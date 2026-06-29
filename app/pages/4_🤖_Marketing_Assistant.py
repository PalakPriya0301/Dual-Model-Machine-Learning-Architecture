import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Marketing Assistant", layout="wide")

try:
    SENDER_EMAIL = st.secrets["SENDER_EMAIL"]
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
except Exception:
    load_dotenv()
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
    APP_PASSWORD  = os.getenv("APP_PASSWORD", "")

st.title("🤖 Smart Marketing Assistant (Live Dispatch)")
st.write("Search for a Customer ID to generate a targeted campaign and dispatch it.")

if "historical_df" not in st.session_state:
    st.error("⚠️ Please go to the Home page first to initialize the database.")
    st.stop()

st.markdown("---")

search_id = st.text_input("🔍 Enter Customer ID :")

if st.button("Fetch Live Data", type="primary"):
    if "active_customer" in st.session_state:
        del st.session_state["active_customer"]
 
    if "email_draft" in st.session_state:
        del st.session_state["email_draft"]

    hist_df = st.session_state["historical_df"]
    match   = hist_df[hist_df["CustomerID"].astype(str) == str(search_id)]

    if not match.empty:
        st.session_state["active_customer"] = match.iloc[0]
        st.success(f"✅ Found Customer {search_id} in Enterprise Database!")
    else:
        if "dynamic_rfm" in st.session_state:
            dyn_df    = st.session_state["dynamic_rfm"]
            dyn_match = dyn_df[dyn_df["Customer ID"].astype(str) == str(search_id)]
            if not dyn_match.empty:
                row = dyn_match.iloc[0].copy()
                row["Persona"] = row.get("Assigned_Persona", "Unknown")
                st.session_state["active_customer"] = row
                st.success(f"✅ Found Customer {search_id} in Live SaaS Upload!")
            else:
                st.error("❌ Customer ID not found in either the database or the live upload.")
        else:
            st.error("❌ Customer ID not found.")

if "active_customer" in st.session_state:
    customer = st.session_state["active_customer"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Persona",     customer["Persona"])
    col2.metric("Recency",     f"{int(customer['Recency'])} days")
    col3.metric("Frequency",   int(customer["Frequency"]))
    col4.metric("Total Spend", f"${customer['Monetary']:.2f}")

    st.markdown("---")
    st.subheader("📧 Campaign Dispatcher")
    target_inbox = st.text_input("Enter destination email address:", placeholder="customer_email@example.com")

    # Build the draft based on persona
    persona = customer["Persona"]
    recency = int(customer["Recency"])

    if "Sleeper" in persona:
        subject = "We miss you! A 20% discount awaits 🎁"
        body    = (
            f"Hi,\n\nIt's been {recency} days since your last purchase. "
            "We'd love to see you back! Use code WINBACK20 for 20% off your next order.\n\nBest regards,\nThe Team"
        )
    elif "Top-Tier" in persona:
        subject = "VIP Early Access: New Collection inside 🌟"
        body    = (
            "Hi,\n\nAs one of our most valued customers, you've earned exclusive early access "
            "to our latest collection — 24 hours before everyone else.\n\nThank you for your loyalty.\nThe Team"
        )
    else:
        subject = "Thinking of leveling up? Check out our Gold Tier 🥇"
        body    = (
            f"Hi,\n\nThanks for your {int(customer['Frequency'])} purchases! "
            "You're almost at VIP status. One more order unlocks exclusive Gold Tier benefits.\n\nThe Team"
        )


    st.markdown("#### ✉️ Email Draft Preview")
    st.info(f"**To:** {target_inbox or '(enter email above)'}\n\n**Subject:** {subject}\n\n**Body:**\n\n{body}")

    
    if st.button("📤 Confirm & Send Email"):
        if not target_inbox:
            st.error("Please enter a destination email address.")
        elif not SENDER_EMAIL or not APP_PASSWORD:
            st.error("Email credentials missing in Streamlit Secrets or .env file.")
        else:
            with st.spinner("Transmitting email via SMTP..."):
                try:
                    msg = MIMEMultipart()
                    msg["From"]    = SENDER_EMAIL
                    msg["To"]      = target_inbox
                    msg["Subject"] = subject
                    msg.attach(MIMEText(body, "plain"))

                    server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
                    server.starttls()
                    server.login(SENDER_EMAIL, APP_PASSWORD)
                    server.send_message(msg)
                    server.quit()

                    st.success(f"✅ Campaign delivered to {target_inbox}!")
                except smtplib.SMTPAuthenticationError:
                    st.error("❌ SMTP Authentication failed. Check your SENDER_EMAIL and APP_PASSWORD secrets.")
                except TimeoutError:
                    st.error("❌ SMTP connection timed out. Gmail may be unreachable from this server.")
                except Exception as e:
                    st.error(f"SMTP Error: {e}")
