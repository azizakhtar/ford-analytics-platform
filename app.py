import streamlit as st
import hmac
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv('ford-credit-key.env')

# Page config
st.set_page_config(page_title="Ford Analytics", page_icon="🚗", layout="wide")

def check_password():
    correct_password = "ford2024"
    
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
        
    st.title("🔐 Ford Analytics Portal")
    st.markdown("### Enter the access password")
    pwd = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Login"):
        if hmac.compare_digest(pwd, correct_password):
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False

if not check_password():
    st.stop()

def get_bigquery_client():
    """Get BigQuery client using environment variables"""
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        private_key = os.getenv('GCP_PRIVATE_KEY')
        client_email = os.getenv('GCP_CLIENT_EMAIL')
        
        if not all([project_id, private_key, client_email]):
            st.warning("Environment variables not set. Using demo mode.")
            return None
        
        # Fix private key formatting
        private_key = private_key.replace('\\n', '\n')
        
        service_account_info = {
            "type": "service_account",
            "project_id": project_id,
            "private_key": private_key,
            "client_email": client_email,
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(credentials=credentials, project=project_id)
        st.success("✅ Connected to BigQuery")
        return client
        
    except Exception as e:
        st.error(f"❌ BigQuery connection failed: {str(e)}")
        return None

# Initialize connection
client = get_bigquery_client()

def execute_query(query):
    """Execute query or return demo data"""
    try:
        if client:
            query_job = client.query(query)
            return query_job.to_dataframe()
        else:
            return create_demo_data()
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return create_demo_data()

def create_demo_data():
    """Create realistic demo data"""
    np.random.seed(42)
    return pd.DataFrame({
        'customer_id': [f'CUST_{i:04d}' for i in range(20)],
        'name': [f'Customer {i}' for i in range(20)],
        'credit_tier': np.random.choice(['Gold', 'Silver', 'Bronze'], 20),
        'state': np.random.choice(['CA', 'TX', 'FL', 'NY'], 20),
        'total_loans': np.random.randint(1, 8, 20),
        'avg_loan_amount': np.random.normal(25000, 5000, 20)
    })

# MANUAL NAVIGATION
st.sidebar.title("🚗 Ford Analytics")
page = st.sidebar.radio("Navigate to:", 
    ["📊 Dashboard", "💬 SQL Chat", "🤖 AI Agent"])

if page == "📊 Dashboard":
    st.title("Ford Analytics Dashboard")
    
    if client:
        st.success("✅ Connected to BigQuery - Live Data")
        # Test connection
        try:
            test_query = "SELECT 1 as test"
            execute_query(test_query)
        except Exception as e:
            st.error(f"Connection test failed: {e}")
    else:
        st.warning("🚧 Demo Mode - Sample Data")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "1,847", "+8%")
    col2.metric("Active Loans", "4,292", "+12%")
    col3.metric("Revenue", "$42.5M", "+15%")
    col4.metric("Satisfaction", "4.3/5", "+0.2")
    
    # Data Preview
    st.markdown("---")
    st.subheader("Data Preview")
    data = execute_query("SELECT * FROM customers LIMIT 10")
    st.dataframe(data)

elif page == "💬 SQL Chat":
    st.title("🤖 SQL Query Interface")
    
    if client:
        st.success("✅ Connected to BigQuery")
    else:
        st.warning("🔒 Connect to BigQuery to run real queries")
    
    query = st.text_area("Enter SQL Query:", "SELECT * FROM customers LIMIT 10", height=150)
    
    if st.button("Execute Query"):
        results = execute_query(query)
        st.dataframe(results)
        st.metric("Rows Returned", len(results))

elif page == "🤖 AI Agent":
    st.title("🧠 AI Analytics")
    
    # Demo analysis
    st.subheader("Customer Segmentation")
    fig, ax = plt.subplots(figsize=(10, 6))
    segments = ['Gold', 'Silver', 'Bronze']
    counts = [25, 45, 30]
    ax.bar(segments, counts, color=['gold', 'silver', 'brown'])
    ax.set_title('Customer Distribution by Credit Tier')
    ax.set_ylabel('Percentage (%)')
    st.pyplot(fig)
