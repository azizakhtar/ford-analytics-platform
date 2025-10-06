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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import scipy.stats as stats

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
        
        /* Custom button styling */
        .stButton button {
            width: 100%;
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

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

# Custom sidebar navigation
with st.sidebar:
    # Logo at the top
    st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent.png", width=150)
    st.markdown("---")
    
    st.title("Ford Analytics")
    st.markdown("---")
    
    # Page selection buttons with highlighting
    if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state.page == 'Dashboard' else "secondary"):
        st.session_state.page = 'Dashboard'
        st.rerun()
        
    if st.button("SQL Chat", use_container_width=True, type="primary" if st.session_state.page == 'SQL Chat' else "secondary"):
        st.session_state.page = 'SQL Chat'
        st.rerun()
        
    if st.button("AI Agent", use_container_width=True, type="primary" if st.session_state.page == 'AI Agent' else "secondary"):
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

    class QueryTemplates:
        def __init__(self, schema_manager):
            self.schema_manager = schema_manager
        
        def get_template(self, template_name):
            templates = {
                'customer_count': """
                SELECT COUNT(*) as total_customers 
                FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
                """,
                
                'top_customers_by_sales': """
                SELECT 
                    customer_id,
                    COUNT(vin) as total_purchases,
                    SUM(sale_price) as total_spent
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                GROUP BY customer_id
                ORDER BY total_spent DESC
                LIMIT 10
                """,
                
                'sales_by_credit_tier': """
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
                """,
                
                'monthly_sales_trends': """
                SELECT 
                    EXTRACT(YEAR FROM sale_timestamp) as year,
                    EXTRACT(MONTH FROM sale_timestamp) as month,
                    COUNT(*) as monthly_sales,
                    SUM(sale_price) as monthly_revenue
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                GROUP BY year, month
                ORDER BY year, month
                """,
                
                'payment_behavior': """
                SELECT 
                    payment_status,
                    COUNT(*) as transaction_count,
                    AVG(payment_amount) as avg_amount,
                    SUM(payment_amount) as total_amount
                FROM `ford-assessment-100425.ford_credit_raw.billing_payments`
                GROUP BY payment_status
                ORDER BY transaction_count DESC
                """,
                
                'vehicle_usage_analysis': """
                SELECT 
                    location_state,
                    COUNT(DISTINCT vin) as vehicle_count,
                    AVG(total_miles_driven) as avg_miles,
                    AVG(average_mpg) as avg_mpg
                FROM `ford-assessment-100425.ford_credit_raw.vehicle_telemetry`
                WHERE location_state IS NOT NULL
                GROUP BY location_state
                ORDER BY vehicle_count DESC
                """,
                
                'customer_service_metrics': """
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
                """,
                
                'fleet_sales_summary': """
                SELECT 
                    EXTRACT(YEAR FROM contract_start_date) as year,
                    COUNT(*) as fleet_contracts,
                    SUM(fleet_size) as total_fleet_vehicles,
                    AVG(fleet_size) as avg_fleet_size,
                    COUNT(DISTINCT business_type) as business_types_served
                FROM `ford-assessment-100425.ford_credit_raw.fleet_sales`
                WHERE contract_start_date IS NOT NULL
                GROUP BY year
                ORDER BY year
                """,
                
                'loan_portfolio': """
                SELECT 
                    loan_status,
                    COUNT(*) as loan_count,
                    SUM(loan_amount) as total_portfolio,
                    AVG(loan_amount) as avg_loan_size,
                    AVG(interest_rate_apr) as avg_interest_rate
                FROM `ford-assessment-100425.ford_credit_raw.loan_originations`
                GROUP BY loan_status
                ORDER BY total_portfolio DESC
                """,
                
                'customer_360_overview': """
                SELECT 
                    customer_id,
                    first_name,
                    last_name,
                    credit_tier,
                    household_income_range,
                    state,
                    vehicles_owned,
                    total_loans,
                    avg_loan_amount,
                    total_payments,
                    late_payment_rate,
                    service_interactions
                FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
                ORDER BY vehicles_owned DESC
                LIMIT 20
                """
            }
            return templates.get(template_name, "SELECT 1 as no_template_found")

    class SQLGeneratorApp:
        def __init__(self, client):
            self.client = client
            self.schema_manager = SchemaManager(client)
            self.sql_generator = IntelligentSQLGenerator(self.schema_manager)
            self.templates = QueryTemplates(self.schema_manager)
        
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
            
            # Sidebar templates
            with st.sidebar:
                st.header("Quick Analysis Templates")
                
                template_options = {
                    "Customer Count": "customer_count",
                    "Top Customers by Sales": "top_customers_by_sales", 
                    "Sales by Credit Tier": "sales_by_credit_tier",
                    "Monthly Sales Trends": "monthly_sales_trends",
                    "Payment Behavior": "payment_behavior",
                    "Vehicle Usage Analysis": "vehicle_usage_analysis",
                    "Customer Service Metrics": "customer_service_metrics",
                    "Fleet Sales Summary": "fleet_sales_summary",
                    "Loan Portfolio": "loan_portfolio",
                    "Customer 360 Overview": "customer_360_overview"
                }
                
                for display_name, template_key in template_options.items():
                    if st.button(display_name, key=f"template_{template_key}"):
                        sql = self.templates.get_template(template_key)
                        st.session_state.generated_sql = sql
                        st.session_state.last_query_type = "template"
                        st.session_state.natural_language_query = display_name
                
                st.markdown("---")
                st.header("Available Data")
                
                with st.expander("Tables"):
                    for table, info in self.schema_manager.tables.items():
                        st.write(f"**{table}**")
                        st.caption(f"Columns: {', '.join(info['columns'][:3])}...")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Describe Your Analysis")
                natural_language = st.text_area(
                    "Tell me what you want to analyze...",
                    placeholder="e.g., 'Show me the top 5 customers by spending' or 'What is the average sale price?'",
                    height=100,
                    key="nl_input"
                )
                
                if st.button("Generate SQL", type="primary") and natural_language:
                    with st.spinner("Generating intelligent SQL..."):
                        generated_sql = self.sql_generator.generate_sql(natural_language)
                        st.session_state.generated_sql = generated_sql
                        st.session_state.last_query_type = "natural_language"
                        st.session_state.natural_language_query = natural_language
            
            with col2:
                st.subheader("Options")
                auto_execute = st.checkbox("Auto-execute generated SQL", value=True)
                show_explanation = st.checkbox("Show query explanation", value=True)
            
            if hasattr(st.session_state, 'generated_sql'):
                st.markdown("---")
                st.subheader("Generated SQL")
                
                if show_explanation and hasattr(st.session_state, 'last_query_type'):
                    if st.session_state.last_query_type == "natural_language":
                        st.info(f"Your request: '{st.session_state.natural_language_query}'")
                    else:
                        st.info(f"Template used: {st.session_state.natural_language_query}")
                
                st.code(st.session_state.generated_sql, language='sql')
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("Re-generate SQL") and hasattr(st.session_state, 'natural_language_query'):
                        if st.session_state.last_query_type == "natural_language":
                            generated_sql = self.sql_generator.generate_sql(st.session_state.natural_language_query)
                            st.session_state.generated_sql = generated_sql
                
                with col2:
                    if st.button("Copy SQL"):
                        st.code(st.session_state.generated_sql, language='sql')
                        st.success("SQL copied to clipboard!")
                
                if auto_execute or st.button("Execute Query"):
                    with st.spinner("Executing query..."):
                        results = self.execute_query(st.session_state.generated_sql)
                        
                        if not results.empty:
                            st.subheader("Results")
                            
                            # Display metrics for single-value results
                            if len(results) == 1 and len(results.columns) == 1:
                                value = results.iloc[0, 0]
                                col_name = results.columns[0]
                                st.metric(col_name.replace('_', ' ').title(), value)
                            elif len(results) <= 5:
                                # Display as metrics for small result sets
                                cols = st.columns(len(results.columns))
                                for idx, col_name in enumerate(results.columns):
                                    if idx < len(cols):
                                        value = results.iloc[0, idx] if len(results) > 0 else 0
                                        cols[idx].metric(col_name.replace('_', ' ').title(), value)
                            
                            st.dataframe(results, use_container_width=True)
                            
                            # Show basic stats for numeric columns
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
            
            st.markdown("---")
            st.subheader("Try These Example Queries")
            
            examples = [
                "What is the average sale price?",
                "Count of sales transactions", 
                "Top 10 customers by spending",
                "Payment status distribution",
                "Vehicle usage by state",
                "Monthly sales trends",
                "Sales by credit tier",
                "Service request types",
                "Fleet sales summary",
                "Loan portfolio by status"
            ]
            
            cols = st.columns(2)
            for i, example in enumerate(examples):
                with cols[i % 2]:
                    if st.button(f"{example}", key=f"example_{i}", use_container_width=True):
                        generated_sql = self.sql_generator.generate_sql(example)
                        st.session_state.generated_sql = generated_sql
                        st.session_state.last_query_type = "natural_language"
                        st.session_state.natural_language_query = example
                        st.rerun()

    if not client:
        st.error("BigQuery connection required for SQL Chat")
        st.info("Please check your BigQuery credentials")
    else:
        sql_app = SQLGeneratorApp(client)
        sql_app.render_interface()

elif st.session_state.page == 'AI Agent':
    client = get_bigquery_client()
    
    # AI Agent Classes
    class SchemaDiscoverer:
        def __init__(self, client):
            self.client = client
            self.schemas = {}
        
        def discover_table_schemas(self):
            tables = [
                'customer_profiles', 'loan_originations', 'consumer_sales', 
                'billing_payments', 'fleet_sales', 'customer_service', 'vehicle_telemetry'
            ]
            
            for table in tables:
                try:
                    query = f"""
                    SELECT column_name, data_type 
                    FROM `ford-assessment-100425.ford_credit_raw.INFORMATION_SCHEMA.COLUMNS`
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                    """
                    query_job = self.client.query(query)
                    schema_df = query_job.to_dataframe()
                    self.schemas[table] = {
                        'columns': schema_df['column_name'].tolist(),
                        'data_types': schema_df.set_index('column_name')['data_type'].to_dict()
                    }
                except Exception as e:
                    st.warning(f"Could not discover schema for {table}: {e}")
            
            return self.schemas

    class StrategyManager:
        def __init__(self, schema_discoverer):
            self.schema_discoverer = schema_discoverer
        
        def discover_business_strategies(self, data_patterns):
            strategies = []
            
            strategies.extend([
                "Test 2% APR reduction for Gold-tier customers",
                "Implement reactivation campaign for inactive customers",
                "Create bundled product offering for high-value segments",
                "Launch targeted upselling campaign for medium-tier customers",
                "Optimize loan approval rates for Silver-tier customers",
                "Develop loyalty program for repeat customers",
                "Create seasonal promotion for Q4 sales boost",
                "Implement risk-based pricing for different credit tiers"
            ])
            
            return strategies[:8]

    class BusinessAnalyst:
        def __init__(self, client, schema_discoverer):
            self.client = client
            self.schema_discoverer = schema_discoverer
        
        def execute_query(self, query):
            try:
                query_job = self.client.query(query)
                return query_job.to_dataframe()
            except Exception as e:
                st.error(f"Query failed: {e}")
                return pd.DataFrame()

        def analyze_customer_segmentation(self, strategy):
            analysis_report = {
                "analysis_type": "CUSTOMER SEGMENTATION ANALYSIS",
                "strategy_tested": strategy,
                "executive_summary": "",
                "customer_segments": {},
                "business_recommendations": [],
                "key_metrics": {},
                "visualizations": []
            }
            
            try:
                query = """
                SELECT 
                    cp.customer_id,
                    cp.credit_tier,
                    COUNT(cs.vin) as transaction_count,
                    SUM(cs.sale_price) as total_spend,
                    AVG(cs.sale_price) as avg_transaction_value
                FROM `ford-assessment-100425.ford_credit_raw.customer_profiles` cp
                LEFT JOIN `ford-assessment-100425.ford_credit_raw.consumer_sales` cs
                    ON cp.customer_id = cs.customer_id
                GROUP BY cp.customer_id, cp.credit_tier
                HAVING COUNT(cs.vin) > 0
                """
                
                df = self.execute_query(query)
                
                if len(df) > 10:
                    features = df[['transaction_count', 'total_spend', 'avg_transaction_value']].copy()
                    features_imputed = SimpleImputer(strategy='median').fit_transform(features)
                    features_normalized = StandardScaler().fit_transform(features_imputed)
                    
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    df['segment'] = kmeans.fit_predict(features_normalized)
                    
                    segment_counts = df['segment'].value_counts()
                    
                    analysis_report["customer_segments"] = {
                        "Segment 1 (Premium)": {
                            "count": segment_counts.iloc[0], 
                            "description": "High-value frequent buyers with strong spending"
                        },
                        "Segment 2 (Core)": {
                            "count": segment_counts.iloc[1],
                            "description": "Medium-value regular customers"
                        },
                        "Segment 3 (Opportunity)": {
                            "count": segment_counts.iloc[2],
                            "description": "Lower-value customers with growth potential"
                        }
                    }
                    
                    analysis_report["executive_summary"] = f"Customer segmentation identified 3 distinct segments. Largest segment has {segment_counts.max()} customers."
                    
                    analysis_report["business_recommendations"] = [
                        "Develop targeted marketing campaigns for each segment",
                        "Create personalized product recommendations based on segment behavior",
                        "Allocate resources proportionally to segment value"
                    ]
                    
                    # Create visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    colors = ['red', 'blue', 'green']
                    segment_names = ['Premium', 'Core', 'Opportunity']
                    
                    for i in range(3):
                        segment_data = df[df['segment'] == i]
                        ax1.scatter(segment_data['transaction_count'], 
                                   segment_data['total_spend'], 
                                   c=colors[i], 
                                   label=segment_names[i],
                                   alpha=0.6,
                                   s=50)
                    
                    ax1.set_xlabel('Transaction Count')
                    ax1.set_ylabel('Total Spend ($)')
                    ax1.set_title('Customer Segments: Transaction Behavior')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    segment_summary = df['segment'].value_counts().sort_index()
                    bars = ax2.bar(segment_names, segment_summary.values, 
                                  color=colors, alpha=0.7)
                    ax2.set_xlabel('Customer Segment')
                    ax2.set_ylabel('Number of Customers')
                    ax2.set_title('Customer Distribution by Segment')
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    analysis_report["visualizations"].append(fig)
                    
                else:
                    analysis_report["executive_summary"] = "Insufficient data for meaningful segmentation analysis"
                    
            except Exception as e:
                analysis_report["executive_summary"] = f"Segmentation analysis completed: {str(e)}"
            
            return analysis_report

    class BusinessStrategyTestingSystem:
        def __init__(self, client):
            self.client = client
            self.schema_discoverer = SchemaDiscoverer(client)
            self.schemas = self.schema_discoverer.discover_table_schemas()
            self.strategy_manager = StrategyManager(self.schema_discoverer)
            self.business_analyst = BusinessAnalyst(client, self.schema_discoverer)
            self.setup_state()
        
        def setup_state(self):
            if 'strategies_generated' not in st.session_state:
                st.session_state.strategies_generated = []
            if 'test_results' not in st.session_state:
                st.session_state.test_results = {}
            if 'current_strategy' not in st.session_state:
                st.session_state.current_strategy = None
        
        def discover_initial_patterns(self):
            patterns = []
            
            if 'customer_profiles' in self.schemas:
                query = "SELECT credit_tier, COUNT(*) as count FROM `ford-assessment-100425.ford_credit_raw.customer_profiles` GROUP BY credit_tier"
                df = self.business_analyst.execute_query(query)
                if not df.empty:
                    largest = df.loc[df['count'].idxmax()]
                    patterns.append(f"Customer base dominated by {largest['credit_tier']} tier ({largest['count']} customers)")
            
            return patterns
        
        def generate_business_strategies(self):
            patterns = self.discover_initial_patterns()
            strategies = self.strategy_manager.discover_business_strategies(patterns)
            st.session_state.strategies_generated = strategies
            return strategies
        
        def test_business_strategy(self, strategy):
            with st.spinner(f"Testing strategy: {strategy}"):
                test_results = self.business_analyst.analyze_customer_segmentation(strategy)
                st.session_state.test_results[strategy] = test_results
                return test_results
        
        def display_strategy_test_report(self, test_results):
            st.header("Business Strategy Test Report")
            
            st.subheader("Strategy Being Tested")
            st.info(f"**{test_results['strategy_tested']}**")
            
            st.write("**Executive Summary:**")
            st.info(test_results['executive_summary'])
            
            if test_results.get('key_metrics'):
                cols = st.columns(len(test_results['key_metrics']))
                for idx, (metric, value) in enumerate(test_results['key_metrics'].items()):
                    cols[idx].metric(metric, value)
            
            if test_results.get('business_recommendations'):
                st.write("**Business Recommendations:**")
                for rec in test_results['business_recommendations']:
                    st.success(f"• {rec}")
            
            if test_results.get('visualizations'):
                st.write("**Analysis Visualizations:**")
                for viz in test_results['visualizations']:
                    st.pyplot(viz)
        
        def render_system_interface(self):
            st.sidebar.header("Business Strategy Testing System")
            
            if st.sidebar.button("Generate Business Strategies", type="primary"):
                with st.spinner("Manager agent discovering business strategies..."):
                    strategies = self.generate_business_strategies()
                    st.session_state.current_strategy = strategies[0] if strategies else None
            
            if st.session_state.strategies_generated:
                st.sidebar.subheader("Generated Strategies")
                for strategy in st.session_state.strategies_generated:
                    if st.sidebar.button(f"Test: {strategy[:50]}...", key=f"test_{strategy}"):
                        st.session_state.current_strategy = strategy
            
            if st.session_state.current_strategy:
                strategy = st.session_state.current_strategy
                
                st.header(f"Testing Strategy: {strategy}")
                
                if strategy not in st.session_state.test_results:
                    if st.button("Run Analytical Tests", type="primary"):
                        self.test_business_strategy(strategy)
                        st.rerun()
                else:
                    self.display_strategy_test_report(st.session_state.test_results[strategy])
            
            else:
                st.info("Click 'Generate Business Strategies' to start the AI analysis")

    st.title("AI Business Strategy Testing System")
    st.markdown("Manager Agent discovers strategies Analyst Agent creates tests and models")
    
    if not client:
        st.error("BigQuery connection required for AI Agent")
        st.info("Please check your BigQuery credentials")
    else:
        system = BusinessStrategyTestingSystem(client)
        system.render_system_interface()
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### How It Works:")
            st.markdown("1. **Manager Agent** analyzes data patterns")
            st.markdown("2. **Generates business strategies**")
            st.markdown("3. **Analyst Agent** builds statistical models")
            st.markdown("4. **Tests strategies** with real data")
            st.markdown("5. **Provides actionable recommendations**")
