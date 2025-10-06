import streamlit as st
import pandas as pd
import numpy as np
import re
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from dotenv import load_dotenv

load_dotenv('key.env')

st.title("Intelligent SQL Generator")
st.markdown("**Natural Language to SQL** - Describe your analysis in plain English")

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
    
    def get_table_suggestions(self, keyword):
        keyword = keyword.lower()
        suggestions = []
        
        for table, info in self.tables.items():
            if keyword in table.lower() or keyword in info['description'].lower():
                suggestions.append(table)
            else:
                for column in info['columns']:
                    if keyword in column.lower():
                        suggestions.append(table)
                        break
        
        return list(set(suggestions))

class IntelligentSQLGenerator:
    def __init__(self, schema_manager):
        self.schema_manager = schema_manager
        
    def generate_sql(self, natural_language):
        """Generate proper SQL from natural language with correct syntax"""
        nl_lower = natural_language.lower()
        
        # Handle common patterns with proper SQL
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
            # Extract limit number
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
        
        # Fallback for simple queries
        elif 'show me' in nl_lower or 'list' in nl_lower:
            table = self._guess_table(nl_lower)
            if table == 'customer_profiles':
                table_ref = 'ford_credit_curated.customer_360_view'
            else:
                table_ref = f'ford_credit_raw.{table}'
            return f"""
            SELECT *
            FROM `ford-assessment-100425.{table_ref}`
            LIMIT 10
            """
        
        else:
            # Default fallback - try to guess the intent
            return self._generate_fallback_sql(nl_lower)
    
    def _guess_table(self, query):
        """Guess the most relevant table based on keywords"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['customer', 'client', 'user']):
            return 'customer_profiles'
        elif any(word in query_lower for word in ['sale', 'purchase', 'transaction']):
            return 'consumer_sales'
        elif any(word in query_lower for word in ['payment', 'billing']):
            return 'billing_payments'
        elif any(word in query_lower for word in ['loan', 'interest']):
            return 'loan_originations'
        elif any(word in query_lower for word in ['fleet', 'business']):
            return 'fleet_sales'
        elif any(word in query_lower for word in ['service', 'support', 'interaction']):
            return 'customer_service'
        elif any(word in query_lower for word in ['vehicle', 'telemetry', 'mileage']):
            return 'vehicle_telemetry'
        else:
            return 'consumer_sales'  # Default table
    
    def _generate_fallback_sql(self, query):
        """Generate a basic SQL query as fallback"""
        table = self._guess_table(query)
        
        # Check if it's likely an aggregate query
        if any(word in query for word in ['average', 'count', 'sum', 'total', 'number']):
            numeric_cols = self._get_numeric_columns(table)
            if numeric_cols:
                if table == 'customer_profiles':
                    table_ref = 'ford_credit_curated.customer_360_view'
                else:
                    table_ref = f'ford_credit_raw.{table}'
                return f"""
                SELECT COUNT(*) as record_count, AVG({numeric_cols[0]}) as avg_value
                FROM `ford-assessment-100425.{table_ref}`
                """
        
        # Simple select with limit
        if table == 'customer_profiles':
            table_ref = 'ford_credit_curated.customer_360_view'
        else:
            table_ref = f'ford_credit_raw.{table}'
        return f"""
        SELECT *
        FROM `ford-assessment-100425.{table_ref}`
        LIMIT 10
        """
    
    def _get_numeric_columns(self, table_name):
        """Get numeric columns for a table"""
        if table_name not in self.schema_manager.tables:
            return []
        
        numeric_cols = []
        table_info = self.schema_manager.tables[table_name]
        for col in table_info['columns']:
            # Common numeric column patterns
            if any(pattern in col.lower() for pattern in ['amount', 'price', 'value', 'count', 'total', 'rate', 'score', 'miles', 'mpg', 'size', 'duration', 'owned', 'loans', 'payments', 'income']):
                numeric_cols.append(col)
        return numeric_cols

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
    def __init__(self):
        self.setup_services()
        self.schema_manager = SchemaManager(self.client)
        self.sql_generator = IntelligentSQLGenerator(self.schema_manager)
        self.templates = QueryTemplates(self.schema_manager)
    
    def setup_services(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                'ford-credit-key.json',
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.client = bigquery.Client(
                credentials=credentials, 
                project="ford-assessment-100425"
            )
        except Exception as e:
            st.error(f"Database connection failed: {e}")
    
    def execute_query(self, query):
        try:
            return pd.read_gbq(query, project_id="ford-assessment-100425")
        except Exception as e:
            st.error(f"Query execution failed: {e}")
            return pd.DataFrame()
    
    def render_interface(self):
        st.sidebar.header("Quick Analysis Templates")
        
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
            if st.sidebar.button(display_name):
                sql = self.templates.get_template(template_key)
                st.session_state.generated_sql = sql
                st.session_state.last_query_type = "template"
                st.session_state.natural_language_query = display_name
        
        st.sidebar.markdown("---")
        st.sidebar.header("Available Data")
        
        with st.sidebar.expander("Tables"):
            for table, info in self.schema_manager.tables.items():
                st.write(f"**{table}**")
                st.caption(f"Columns: {', '.join(info['columns'][:3])}...")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Describe Your Analysis")
            natural_language = st.text_area(
                "Tell me what you want to analyze...",
                placeholder="e.g., 'Show me the top 5 customers by spending' or 'What's the average sale price?'",
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
                    st.info(f"**Your request:** '{st.session_state.natural_language_query}'")
                else:
                    st.info(f"**Template used:** {st.session_state.natural_language_query}")
            
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
                if st.button(f"\"{example}\"", key=f"example_{i}", use_container_width=True):
                    generated_sql = self.sql_generator.generate_sql(example)
                    st.session_state.generated_sql = generated_sql
                    st.session_state.last_query_type = "natural_language"
                    st.session_state.natural_language_query = example
                    st.rerun()

if __name__ == "__main__":
    app = SQLGeneratorApp()
    app.render_interface()
