import streamlit as st
import hmac
import pandas as pd
import numpy as np
import re
from google.cloud import bigquery
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", layout="wide")

# Hide ALL Streamlit default elements
st.markdown("""
    <style>
        /* Hide the default Streamlit sidebar navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Hide the default page navigation in sidebar */
        .st-emotion-cache-1oe5cao {
            display: none !important;
        }
        
        /* Hide hamburger menu */
        #MainMenu {
            visibility: hidden;
        }
        
        /* Hide footer */
        footer {
            visibility: hidden;
        }
        
        /* Hide deploy button */
        .stDeployButton {
            display: none !important;
        }
        
        /* Ensure our sidebar content is visible */
        .sidebar .sidebar-content {
            display: block !important;
        }
    </style>
""", unsafe_allow_html=True)

def check_password():
    try:
        correct_password = st.secrets["password"]
    except:
        correct_password = "ford2024"
    
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True
        
    st.title("Ford Analytics Portal")
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

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

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

# Custom sidebar navigation
with st.sidebar:
    st.title("Ford Analytics")
    st.markdown("---")
    
    # Page selection buttons
    if st.button("Dashboard", use_container_width=True):
        st.session_state.page = 'Dashboard'
        st.rerun()
        
    if st.button("SQL Chat", use_container_width=True):
        st.session_state.page = 'SQL Chat'
        st.rerun()
        
    if st.button("AI Agent", use_container_width=True):
        st.session_state.page = 'AI Agent'
        st.rerun()

# Display the selected page
if st.session_state.page == 'Dashboard':
    client = get_bigquery_client()

    st.title("Ford Analytics Dashboard")
    st.markdown("Comprehensive overview of fleet performance")

    if client:
        st.success("Connected to BigQuery - Live Data")
    else:
        st.warning("Demo Mode - Sample Data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", "$4.2M", "+12%")
    col2.metric("Active Loans", "1,847", "+8%")
    col3.metric("Delinquency Rate", "2.3%", "-0.4%")
    col4.metric("Customer Satisfaction", "4.2/5", "+0.3")

    st.markdown("---")
    st.subheader("Live Data Preview")

    if client:
        try:
            preview_query = "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10"
            query_job = client.query(preview_query)
            data = query_job.to_dataframe()
            st.dataframe(data)
            st.success(f"Loaded {len(data)} rows from BigQuery")
        except Exception as e:
            st.error(f"Could not load data: {str(e)}")
    else:
        st.info("Connect to BigQuery to see live data")

elif st.session_state.page == 'SQL Chat':
    client = get_bigquery_client()
    
    # Your complete SQL Chat classes and functionality
    class SchemaManager:
        def __init__(self, client):
            self.client = client
            self.tables = {
                'customer_360_view': {
                    'description': 'Customer 360 view with comprehensive profile data',
                    'primary_key': 'customer_id',
                    'columns': [
                        'customer_id', 'first_name', 'last_name', 'credit_tier', 
                        'household_income_range', 'state', 'vehicles_owned', 
                        'total_loans', 'avg_loan_amount', 'total_payments', 
                        'late_payment_rate', 'service_interactions'
                    ]
                },
                'loan_originations': {
                    'description': 'Loan applications and originations',
                    'primary_key': 'contract_id', 
                    'columns': ['contract_id', 'customer_id', 'vin', 'contract_type', 'origination_date', 'loan_amount', 'interest_rate_apr', 'term_months', 'monthly_payment', 'remaining_balance', 'risk_tier', 'loan_status']
                },
                'consumer_sales': {
                    'description': 'Individual vehicle sales transactions',
                    'primary_key': 'vin',
                    'columns': ['vin', 'customer_id', 'dealer_id', 'sale_timestamp', 'vehicle_model', 'vehicle_year', 
                               'trim_level', 'powertrain', 'sale_type', 'sale_price', 'dealer_state', 'warranty_type', 'purchase_financed']
                },
                'billing_payments': {
                    'description': 'Customer billing and payment records',
                    'primary_key': 'payment_id',
                    'columns': ['payment_id', 'customer_id', 'payment_amount', 'payment_date', 'payment_status', 'due_date']
                },
                'fleet_sales': {
                    'description': 'Fleet vehicle sales to businesses',
                    'primary_key': 'fleet_id',
                    'columns': ['fleet_id', 'fleet_manager_id', 'business_name', 'business_type', 'fleet_size', 'primary_vehicle_type', 'total_vehicles_owned', 'fleet_contact_email', 'contract_start_date', 'preferred_dealer_network']
                },
                'customer_service': {
                    'description': 'Customer service interactions and support tickets',
                    'primary_key': 'interaction_id',
                    'columns': [
                        'interaction_id', 'customer_id', 'interaction_timestamp', 'channel', 
                        'interaction_type', 'sentiment_score', 'issue_resolved', 
                        'interaction_duration_min', 'agent_id', 'follow_up_required'
                    ]
                },
                'vehicle_telemetry': {
                    'description': 'Vehicle usage and performance data',
                    'primary_key': 'vin',
                    'columns': ['vin', 'aggregation_date', 'total_miles_driven', 'average_mpg', 'average_kwh_per_mile', 
                              'hard_braking_events', 'rapid_accelerations', 'average_speed', 'ev_charging_sessions', 
                              'average_state_of_charge', 'primary_usage_time', 'location_state']
                }
            }

    class IntelligentSQLGenerator:
        def __init__(self, schema_manager):
            self.schema_manager = schema_manager
            
        def generate_sql(self, natural_language):
            nl_lower = natural_language.lower()
            
            if any(word in nl_lower for word in ['average', 'avg']) and 'price' in nl_lower:
                return """
                SELECT AVG(sale_price) as average_sale_price
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                WHERE sale_price IS NOT NULL
                """
            
            elif 'count of sales' in nl_lower or 'number of sales' in nl_lower:
                return """
                SELECT COUNT(*) as total_sales
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                """
            
            elif 'count of customers' in nl_lower or 'number of customers' in nl_lower:
                return """
                SELECT COUNT(*) as total_customers
                FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
                """
            
            elif 'top' in nl_lower and 'customer' in nl_lower and any(word in nl_lower for word in ['spend', 'purchase', 'sale']):
                limit_match = re.search(r'top\s+(\d+)', nl_lower)
                limit = limit_match.group(1) if limit_match else '10'
                
                return f"""
                SELECT 
                    customer_id,
                    SUM(sale_price) as total_spending
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                GROUP BY customer_id
                ORDER BY total_spending DESC
                LIMIT {limit}
                """
            
            elif 'payment status' in nl_lower or 'payment distribution' in nl_lower:
                return """
                SELECT 
                    payment_status,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM `ford-assessment-100425.ford_credit_raw.billing_payments`
                GROUP BY payment_status
                ORDER BY count DESC
                """
            
            elif 'vehicle usage by state' in nl_lower or 'usage by state' in nl_lower:
                return """
                SELECT 
                    location_state,
                    COUNT(DISTINCT vin) as vehicle_count,
                    AVG(total_miles_driven) as avg_miles,
                    AVG(average_mpg) as avg_mpg
                FROM `ford-assessment-100425.ford_credit_raw.vehicle_telemetry`
                WHERE location_state IS NOT NULL
                GROUP BY location_state
                ORDER BY vehicle_count DESC
                """
            
            elif 'monthly sales' in nl_lower or 'sales trend' in nl_lower:
                return """
                SELECT 
                    EXTRACT(YEAR FROM sale_timestamp) as year,
                    EXTRACT(MONTH FROM sale_timestamp) as month,
                    COUNT(*) as monthly_sales,
                    SUM(sale_price) as monthly_revenue,
                    AVG(sale_price) as avg_sale_price
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                GROUP BY year, month
                ORDER BY year, month
                """
            
            elif 'credit tier' in nl_lower and any(word in nl_lower for word in ['sales', 'revenue']):
                return """
                SELECT 
                    cp.credit_tier,
                    COUNT(cs.vin) as total_sales,
                    SUM(cs.sale_price) as total_revenue,
                    AVG(cs.sale_price) as avg_sale_price
                FROM `ford-assessment-100425.ford_credit_curated.customer_360_view` cp
                JOIN `ford-assessment-100425.ford_credit_raw.consumer_sales` cs
                    ON cp.customer_id = cs.customer_id
                GROUP BY cp.credit_tier
                ORDER BY total_revenue DESC
                """
            
            elif 'service' in nl_lower and any(word in nl_lower for word in ['type', 'request', 'interaction']):
                return """
                SELECT 
                    interaction_type,
                    COUNT(*) as request_count,
                    AVG(interaction_duration_min) as avg_duration_minutes,
                    AVG(sentiment_score) as avg_sentiment,
                    SUM(CASE WHEN issue_resolved THEN 1 ELSE 0 END) as resolved_issues,
                    ROUND(SUM(CASE WHEN issue_resolved THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as resolution_rate
                FROM `ford-assessment-100425.ford_credit_raw.customer_service`
                GROUP BY interaction_type
                ORDER BY request_count DESC
                """
            
            elif 'fleet' in nl_lower and any(word in nl_lower for word in ['sales', 'summary']):
                return """
                SELECT 
                    EXTRACT(YEAR FROM contract_start_date) as year,
                    COUNT(*) as fleet_contracts,
                    SUM(fleet_size) as total_fleet_vehicles,
                    AVG(fleet_size) as avg_fleet_size
                FROM `ford-assessment-100425.ford_credit_raw.fleet_sales`
                WHERE contract_start_date IS NOT NULL
                GROUP BY year
                ORDER BY year
                """
            
            elif 'loan' in nl_lower and any(word in nl_lower for word in ['portfolio', 'status']):
                return """
                SELECT 
                    loan_status,
                    COUNT(*) as loan_count,
                    SUM(loan_amount) as total_portfolio,
                    AVG(loan_amount) as avg_loan_size,
                    AVG(interest_rate_apr) as avg_interest_rate
                FROM `ford-assessment-100425.ford_credit_raw.loan_originations`
                GROUP BY loan_status
                ORDER BY total_portfolio DESC
                """
            
            else:
                return """
                SELECT *
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                LIMIT 10
                """

    class SQLGeneratorApp:
        def __init__(self, client):
            self.client = client
            self.schema_manager = SchemaManager(client)
            self.sql_generator = IntelligentSQLGenerator(self.schema_manager)
        
        def execute_query(self, query):
            try:
                if self.client:
                    query_job = self.client.query(query)
                    return query_job.to_dataframe()
                else:
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Query execution failed: {e}")
                return pd.DataFrame()
        
        def render_interface(self):
            st.title("Intelligent SQL Generator")
            st.markdown("Natural Language to SQL - Describe your analysis in plain English")
            
            st.sidebar.header("Quick Analysis Templates")
            
            template_options = {
                "Customer Count": "SELECT COUNT(*) as total_customers FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`",
                "Top Customers by Sales": "SELECT customer_id, SUM(sale_price) as total_spent FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` GROUP BY customer_id ORDER BY total_spent DESC LIMIT 10",
                "Sales by Credit Tier": "SELECT cp.credit_tier, COUNT(cs.vin) as total_sales, SUM(cs.sale_price) as total_revenue FROM `ford-assessment-100425.ford_credit_curated.customer_360_view` cp JOIN `ford-assessment-100425.ford_credit_raw.consumer_sales` cs ON cp.customer_id = cs.customer_id GROUP BY cp.credit_tier ORDER BY total_revenue DESC",
                "Monthly Sales Trends": "SELECT EXTRACT(YEAR FROM sale_timestamp) as year, EXTRACT(MONTH FROM sale_timestamp) as month, COUNT(*) as monthly_sales, SUM(sale_price) as monthly_revenue FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` GROUP BY year, month ORDER BY year, month",
                "Payment Behavior": "SELECT payment_status, COUNT(*) as transaction_count, AVG(payment_amount) as avg_amount FROM `ford-assessment-100425.ford_credit_raw.billing_payments` GROUP BY payment_status ORDER BY transaction_count DESC"
            }
            
            for display_name, sql in template_options.items():
                if st.sidebar.button(display_name):
                    st.session_state.generated_sql = sql
                    st.session_state.natural_language_query = display_name
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                natural_language = st.text_area(
                    "Tell me what you want to analyze...",
                    placeholder="e.g., Show me the top 5 customers by spending or What is the average sale price?",
                    height=100,
                    key="nl_input"
                )
                
                if st.button("Generate SQL", type="primary") and natural_language:
                    generated_sql = self.sql_generator.generate_sql(natural_language)
                    st.session_state.generated_sql = generated_sql
                    st.session_state.natural_language_query = natural_language
            
            if hasattr(st.session_state, 'generated_sql'):
                st.markdown("---")
                st.subheader("Generated SQL")
                st.code(st.session_state.generated_sql, language='sql')
                
                if st.button("Execute Query"):
                    with st.spinner("Executing query..."):
                        results = self.execute_query(st.session_state.generated_sql)
                        
                        if not results.empty:
                            st.subheader("Results")
                            st.dataframe(results, use_container_width=True)
                            
                            numeric_cols = results.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                with st.expander("Quick Statistics"):
                                    st.write(results[numeric_cols].describe())
                            
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No results returned from the query.")

    if not client:
        st.error("BigQuery connection required for SQL Chat")
        st.info("Please check your BigQuery credentials")
    else:
        sql_app = SQLGeneratorApp(client)
        sql_app.render_interface()

elif st.session_state.page == 'AI Agent':
    client = get_bigquery_client()
    
    st.title("AI Business Strategy Testing System")
    st.markdown("Manager Agent discovers strategies Analyst Agent creates tests and models")
    
    if not client:
        st.error("BigQuery connection required for AI Agent")
        st.info("Please check your BigQuery credentials")
    else:
        # Your AI Agent functionality would go here
        st.info("AI Agent functionality would be implemented here")
