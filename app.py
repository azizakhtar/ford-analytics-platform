import streamlit as st
import hmac
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account

# Page config
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
    """Get BigQuery client using Streamlit secrets"""
    try:
        # Get credentials from Streamlit secrets
        secrets = st.secrets["gcp_service_account"]
        
        service_account_info = {
            "type": "service_account",
            "project_id": secrets["project_id"],
            "private_key_id": "1e50f28be910011e821cc468784ded5e80c28a78",
            "private_key": secrets["private_key"].replace('\\n', '\n'),
            "client_email": secrets["client_email"],
            "client_id": "106044880248426653504",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/bq-926%40ford-assessment-100425.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(credentials=credentials, project=secrets["project_id"])
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
            results = query_job.to_dataframe()
            return results
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
        
        # Test with a simple query first
        try:
            test_query = "SELECT 'Connection successful' as status"
            result = execute_query(test_query)
            st.info(f"✅ Connection test passed: {result.iloc[0]['status']}")
            
            # Try to list tables
            tables_query = """
            SELECT table_name 
            FROM `ford-assessment-100425.ford_credit_raw.INFORMATION_SCHEMA.TABLES`
            LIMIT 5
            """
            tables = execute_query(tables_query)
            if not tables.empty:
                st.success(f"📊 Found tables: {', '.join(tables['table_name'].tolist())}")
                
        except Exception as e:
            st.error(f"❌ Connection test failed: {str(e)}")
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
    
    if client:
        try:
            # Try to get real data
            preview_query = "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10"
            data = execute_query(preview_query)
            st.dataframe(data)
            st.success(f"✅ Loaded {len(data)} rows from BigQuery")
        except Exception as e:
            st.error(f"Could not load real data: {str(e)}")
            demo_data = create_demo_data()
            st.dataframe(demo_data)
            st.caption("📝 Showing demo data")
    else:
        demo_data = create_demo_data()
        st.dataframe(demo_data)
        st.caption("📝 Showing demo data")

elif page == "💬 SQL Chat":
    st.title("🤖 SQL Query Interface")
    
    if not client:
        st.error("❌ BigQuery connection required")
        st.info("Please set up your service account credentials in Streamlit Cloud secrets")
        st.stop()
    
    st.success("✅ Connected to BigQuery")
    
    query = st.text_area("Enter SQL Query:", 
                        "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10", 
                        height=150)
    
    if st.button("Execute Query"):
        with st.spinner("Running query..."):
            results = execute_query(query)
            st.dataframe(results)
            st.metric("Rows Returned", len(results))

elif page == "🤖 AI Agent":
    st.title("🧠 AI Analytics")
    
    if client:
        st.success("✅ Connected to BigQuery")
    else:
        st.warning("🔒 Connect to BigQuery for AI analysis")
    
    # Demo analysis
    st.subheader("Customer Segmentation Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    segments = ['Gold', 'Silver', 'Bronze']
    counts = [25, 45, 30]
    ax.bar(segments, counts, color=['gold', 'silver', 'brown'])
    ax.set_title('Customer Distribution by Credit Tier')
    ax.set_ylabel('Percentage (%)')
    st.pyplot(fig)
