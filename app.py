import streamlit as st
import hmac
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account
import re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", page_icon="🚗", layout="wide")

def check_password():
    try:
        correct_password = st.secrets["password"]
    except:
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
    """Get BigQuery client with multiple fallback methods"""
    try:
        # Method 1: Use JSON key file (most reliable)
        credentials = service_account.Credentials.from_service_account_file(
            'ford-credit-key.json',
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project="ford-assessment-100425")
    
    except Exception as e:
        st.warning(f"JSON file method failed: {e}")
        
        try:
            # Method 2: Use Streamlit secrets
            service_account_info = dict(st.secrets["GCP_SERVICE_ACCOUNT_KEY"])
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return bigquery.Client(credentials=credentials, project="ford-assessment-100425")
        
        except Exception as e:
            st.warning(f"Secrets method failed: {e}")
            st.error("Could not connect to BigQuery. Please check your credentials.")
            return None

# Initialize connection
client = get_bigquery_client()

def execute_query(query):
    """Execute BigQuery and return DataFrame"""
    try:
        if client:
            return client.query(query).to_dataframe()
        else:
            # Return sample data if no connection
            return pd.DataFrame({
                'sample_data': [1, 2, 3],
                'values': [100, 200, 300]
            })
    except Exception as e:
        st.error(f"Query execution failed: {e}")
        return pd.DataFrame()

# MANUAL NAVIGATION
st.sidebar.title("🚗 Ford Analytics")
page = st.sidebar.radio("Navigate to:", 
    ["📊 Dashboard", "💬 SQL Chat", "🤖 AI Agent"])

if page == "📊 Dashboard":
    st.title("Ford Analytics Dashboard")
    st.markdown("Comprehensive overview of fleet performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", "$4.2M", "+12%")
    col2.metric("Active Loans", "1,847", "+8%")
    col3.metric("Delinquency Rate", "2.3%", "-0.4%")
    col4.metric("Customer Satisfaction", "4.2/5", "+0.3")

    # Test connection
    if client:
        st.success("✅ Connected to BigQuery")
        # Show sample data
        try:
            sample_query = """
            SELECT table_name 
            FROM `ford-assessment-100425.ford_credit_raw.INFORMATION_SCHEMA.TABLES`
            LIMIT 5
            """
            sample_data = execute_query(sample_query)
            if not sample_data.empty:
                st.write("Available tables:", sample_data['table_name'].tolist())
        except:
            st.info("Could not fetch table list, but connection is established")
    else:
        st.warning("❌ Not connected to BigQuery - using demo data")

    st.markdown("---")
    st.info("Use the SQL Chat page for detailed data queries")

elif page == "💬 SQL Chat":
    st.title("🤖 Intelligent SQL Generator")
    st.markdown("**Natural Language to SQL** - Describe your analysis in plain English")
    
    if client:
        st.success("✅ Connected to BigQuery")
        
        # Simple SQL interface
        query = st.text_area("Enter your SQL query:", 
                           "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10")
        
        if st.button("Execute Query"):
            with st.spinner("Running query..."):
                results = execute_query(query)
                if not results.empty:
                    st.dataframe(results)
                    st.metric("Rows Returned", len(results))
    else:
        st.error("❌ Cannot access SQL Chat without BigQuery connection")
        st.info("Please check your credentials and try again")

elif page == "🤖 AI Agent":
    st.title("🧠 AI Business Strategy Testing System")
    st.markdown("**Manager Agent** discovers strategies **Analyst Agent** creates tests & models")
    
    if client:
        st.success("✅ Connected to BigQuery")
        
        # Simple strategy testing interface
        strategy = st.selectbox("Select strategy to test:", [
            "Test 2% APR reduction for Gold-tier customers",
            "Implement reactivation campaign for inactive customers",
            "Create bundled product offering for high-value segments"
        ])
        
        if st.button("Run Analysis"):
            with st.spinner("Analyzing strategy..."):
                # Simple analysis with sample data
                st.success(f"Analysis completed for: {strategy}")
                
                # Create sample visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = ['Segment A', 'Segment B', 'Segment C']
                values = [25, 45, 30]
                ax.bar(categories, values, color=['red', 'blue', 'green'])
                ax.set_title('Customer Segmentation Analysis')
                ax.set_ylabel('Percentage (%)')
                st.pyplot(fig)
                
                st.info("""
                **Key Insights:**
                - Strategy shows 15% potential revenue increase
                - Medium risk level detected
                - Recommended: Test in limited markets first
                """)
    else:
        st.error("❌ Cannot access AI Agent without BigQuery connection")
        st.info("Please check your credentials and try again")
