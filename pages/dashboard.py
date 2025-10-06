import streamlit as st
import hmac
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# Page config
st.set_page_config(page_title="Ford Dashboard", layout="wide")

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
    st.session_state.current_page = "dashboard"
    
    # Add navigation back to main app
    with st.sidebar:
        if st.button("← Back to Main Menu"):
            st.switch_page("app.py")
    
    client = get_bigquery_client()

    st.title("📊 Ford Analytics Dashboard")
    st.markdown("Comprehensive overview of fleet performance")

    if client:
        st.success("✅ Connected to BigQuery - Live Data")
    else:
        st.warning("🚧 Demo Mode - Sample Data")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", "$4.2M", "+12%")
    col2.metric("Active Loans", "1,847", "+8%")
    col3.metric("Delinquency Rate", "2.3%", "-0.4%")
    col4.metric("Customer Satisfaction", "4.2/5", "+0.3")

    # Data Preview
    st.markdown("---")
    st.subheader("Live Data Preview")

    if client:
        try:
            preview_query = "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10"
            query_job = client.query(preview_query)
            data = query_job.to_dataframe()
            st.dataframe(data)
            st.success(f"✅ Loaded {len(data)} rows from BigQuery")
        except Exception as e:
            st.error(f"Could not load data: {str(e)}")
    else:
        st.info("Connect to BigQuery to see live data")

if __name__ == "__main__":
    main()
