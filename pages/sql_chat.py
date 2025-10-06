import streamlit as st
import hmac
import pandas as pd
import numpy as np
import re
from google.cloud import bigquery
from google.oauth2 import service_account

# Page config
st.set_page_config(page_title="SQL Chat", layout="wide")

def check_password():
    """Password protection for individual pages"""
    try:
        correct_password = st.secrets["password"]
    except:
        correct_password = "ford2024"
    
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
        
    # If not authenticated, redirect to main app
    st.warning("🔒 Please authenticate through the main app")
    if st.button("Go to Login"):
        st.switch_page("app.py")
    st.stop()

def get_bigquery_client():
    try:
        secrets = st.secrets["gcp_service_account"]
        service_account_info = {
            "type": "service_account",
            "project_id": secrets["project_id"],
            "private_key": secrets["private_key"].replace('\\n', '\n'),
            "client_email": secrets["client_email"],
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        return bigquery.Client(credentials=credentials, project=secrets["project_id"])
    except Exception as e:
        st.error(f"BigQuery connection failed: {str(e)}")
        return None

def main():
    # Check password first
    check_password()
    
    # Set current page
    st.session_state.current_page = "sql_chat"
    
    # Add navigation back to main app
    with st.sidebar:
        if st.button("← Back to Main Menu"):
            st.switch_page("app.py")
    
    # Your existing SQL Chat code here
    st.title("💬 Intelligent SQL Generator")
    st.markdown("**Natural Language to SQL** - Describe your analysis in plain English")
    
    # Add your SQL Chat interface here
    st.info("SQL Chat interface would be implemented here")
    
    # Example content
    st.text_area(
        "Ask your question...",
        placeholder="e.g., 'Show me the top 5 customers by spending'",
        height=100
    )
    
    if st.button("Generate SQL"):
        st.code("SELECT * FROM consumer_sales LIMIT 10", language='sql')

if __name__ == "__main__":
    main()
