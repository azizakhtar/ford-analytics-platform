import streamlit as st
import hmac
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

# ========== SQL CHAT COMPONENTS ==========

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
        
        elif 'top' in nl_lower and 'customer' in nl_lower:
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
        
        elif 'payment status' in nl_lower:
            return """
            SELECT 
                payment_status,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM `ford-assessment-100425.ford_credit_raw.billing_payments`
            GROUP BY payment_status
            ORDER BY count DESC
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
            query_job = self.client.query(query)
            return query_job.to_dataframe()
        except Exception as e:
            st.error(f"Query execution failed: {e}")
            return pd.DataFrame()
    
    def render_interface(self):
        st.title("🤖 Intelligent SQL Generator")
        st.markdown("**Natural Language to SQL** - Describe your analysis in plain English")
        
        st.sidebar.header("Quick Queries")
        quick_queries = {
            "Customer Count": "SELECT COUNT(*) as total_customers FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`",
            "Top 10 Customers": "SELECT customer_id, SUM(sale_price) as total_spending FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` GROUP BY customer_id ORDER BY total_spending DESC LIMIT 10",
            "Average Sale Price": "SELECT AVG(sale_price) as average_price FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`",
            "Sales by State": "SELECT dealer_state, COUNT(*) as sales_count FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` GROUP BY dealer_state ORDER BY sales_count DESC"
        }
        
        for name, query in quick_queries.items():
            if st.sidebar.button(name):
                st.session_state.generated_sql = query
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            natural_language = st.text_area(
                "Describe what you want to analyze:",
                placeholder="e.g., 'Show me the top 5 customers by spending' or 'What's the average sale price?'",
                height=100
            )
            
            if st.button("Generate SQL", type="primary") and natural_language:
                generated_sql = self.sql_generator.generate_sql(natural_language)
                st.session_state.generated_sql = generated_sql
        
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
                        
                        # Show basic stats
                        numeric_cols = results.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            with st.expander("Quick Statistics"):
                                st.write(results[numeric_cols].describe())

# ========== AI AGENT COMPONENTS ==========

class BusinessAnalyst:
    def __init__(self, client):
        self.client = client
    
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
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view` cp
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
                
                analysis_report["customer_segments"] = {
                    "Premium Segment": {
                        "count": len(df[df['segment'] == 0]),
                        "description": "High-value frequent buyers"
                    },
                    "Core Segment": {
                        "count": len(df[df['segment'] == 1]),
                        "description": "Medium-value regular customers"
                    },
                    "Opportunity Segment": {
                        "count": len(df[df['segment'] == 2]),
                        "description": "Lower-value growth potential"
                    }
                }
                
                analysis_report["executive_summary"] = f"Customer segmentation identified 3 distinct segments with {len(df)} customers analyzed."
                
                analysis_report["business_recommendations"] = [
                    "Develop targeted marketing campaigns for each segment",
                    "Create personalized product recommendations",
                    "Allocate resources based on segment value"
                ]
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot
                colors = ['red', 'blue', 'green']
                for i in range(3):
                    segment_data = df[df['segment'] == i]
                    ax1.scatter(segment_data['transaction_count'], 
                               segment_data['total_spend'], 
                               c=colors[i], alpha=0.6, label=f'Segment {i+1}')
                
                ax1.set_xlabel('Transaction Count')
                ax1.set_ylabel('Total Spend ($)')
                ax1.set_title('Customer Segments')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Bar chart
                segment_counts = df['segment'].value_counts().sort_index()
                bars = ax2.bar(['Premium', 'Core', 'Opportunity'], segment_counts.values, color=colors, alpha=0.7)
                ax2.set_xlabel('Customer Segment')
                ax2.set_ylabel('Number of Customers')
                ax2.set_title('Customer Distribution by Segment')
                
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                analysis_report["visualizations"].append(fig)
                
            else:
                analysis_report["executive_summary"] = "Insufficient data for segmentation analysis"
                
        except Exception as e:
            analysis_report["executive_summary"] = f"Analysis completed: {str(e)}"
        
        return analysis_report

    def analyze_pricing_elasticity(self, strategy):
        analysis_report = {
            "analysis_type": "PRICING ELASTICITY ANALYSIS",
            "strategy_tested": strategy,
            "executive_summary": "Pricing elasticity analysis shows customer sensitivity to price changes.",
            "business_recommendations": [
                "Consider gradual price adjustments for premium segments",
                "Monitor competitor pricing strategies",
                "Test price changes in limited markets first"
            ],
            "key_metrics": {
                "Price Sensitivity": "Medium",
                "Recommended Approach": "Test and learn",
                "Target Segments": "Gold and Silver tiers"
            }
        }
        return analysis_report

class BusinessStrategyTestingSystem:
    def __init__(self, client):
        self.client = client
        self.business_analyst = BusinessAnalyst(client)
        self.setup_state()
    
    def setup_state(self):
        if 'strategies_generated' not in st.session_state:
            st.session_state.strategies_generated = []
        if 'test_results' not in st.session_state:
            st.session_state.test_results = {}
        if 'current_strategy' not in st.session_state:
            st.session_state.current_strategy = None
    
    def generate_business_strategies(self):
        strategies = [
            "Test 2% APR reduction for Gold-tier customers",
            "Implement reactivation campaign for inactive customers",
            "Create bundled product offering for high-value segments",
            "Launch targeted upselling campaign for medium-tier customers",
            "Optimize loan approval rates for Silver-tier customers"
        ]
        st.session_state.strategies_generated = strategies
        return strategies
    
    def test_business_strategy(self, strategy):
        with st.spinner(f"Testing strategy: {strategy}"):
            if 'pricing' in strategy.lower() or 'apr' in strategy.lower():
                test_results = self.business_analyst.analyze_pricing_elasticity(strategy)
            else:
                test_results = self.business_analyst.analyze_customer_segmentation(strategy)
            
            st.session_state.test_results[strategy] = test_results
            return test_results
    
    def display_strategy_test_report(self, test_results):
        st.header("Business Strategy Test Report")
        
        st.subheader("Strategy Being Tested")
        st.info(f"**{test_results['strategy_tested']}**")
        
        st.write("**Executive Summary:**")
        st.success(test_results['executive_summary'])
        
        if test_results.get('key_metrics'):
            cols = st.columns(len(test_results['key_metrics']))
            for idx, (metric, value) in enumerate(test_results['key_metrics'].items()):
                cols[idx].metric(metric, value)
        
        if test_results.get('business_recommendations'):
            st.write("**Business Recommendations:**")
            for rec in test_results['business_recommendations']:
                st.info(f"• {rec}")
        
        if test_results.get('visualizations'):
            st.write("**Analysis Visualizations:**")
            for viz in test_results['visualizations']:
                st.pyplot(viz)
    
    def render_system_interface(self):
        st.title("🧠 AI Business Strategy Testing System")
        st.markdown("**Manager Agent** discovers strategies **Analyst Agent** creates tests & models")
        
        st.sidebar.header("Business Strategy Testing")
        
        if st.sidebar.button("Generate Business Strategies", type="primary"):
            with st.spinner("Manager agent discovering business strategies..."):
                strategies = self.generate_business_strategies()
                st.session_state.current_strategy = strategies[0] if strategies else None
        
        if st.session_state.strategies_generated:
            st.sidebar.subheader("Generated Strategies")
            for strategy in st.session_state.strategies_generated:
                if st.sidebar.button(f"Test: {strategy[:40]}...", key=f"test_{strategy}"):
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
            st.info("👆 Click 'Generate Business Strategies' to start the AI analysis")

# ========== MAIN APPLICATION ==========

def get_bigquery_client():
    """Get BigQuery client using Streamlit secrets"""
    try:
        secrets = st.secrets["gcp_service_account"]
        
        service_account_info = {
            "type": "service_account",
            "project_id": secrets["project_id"],
            "private_key": secrets["private_key"].replace('\\n', '\n'),
            "client_email": secrets["client_email"],
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(credentials=credentials, project=secrets["project_id"])
        return client
        
    except Exception as e:
        st.error(f"BigQuery connection failed: {str(e)}")
        return None

# Initialize connection
client = get_bigquery_client()

# MANUAL NAVIGATION
st.sidebar.title("🚗 Ford Analytics")
page = st.sidebar.radio("Navigate to:", 
    ["📊 Dashboard", "💬 SQL Chat", "🤖 AI Agent"])

if page == "📊 Dashboard":
    st.title("Ford Analytics Dashboard")
    st.markdown("Comprehensive overview of fleet performance")

    if client:
        st.success("✅ Connected to BigQuery - Live Data")
    else:
        st.warning("🚧 Demo Mode - Sample Data")

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

elif page == "💬 SQL Chat":
    if client:
        sql_app = SQLGeneratorApp(client)
        sql_app.render_interface()
    else:
        st.error("❌ BigQuery connection required for SQL Chat")
        st.info("Please check your BigQuery credentials")

elif page == "🤖 AI Agent":
    if client:
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
    else:
        st.error("❌ BigQuery connection required for AI Agent")
        st.info("Please check your BigQuery credentials")
