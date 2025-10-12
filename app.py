import streamlit as st
import hmac
import pandas as pd
import numpy as np
import re
import json
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
import google.generativeai as genai

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", layout="wide")

# Hide ALL Streamlit default elements
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        .st-emotion-cache-1oe5cao { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none !important; }
        .sidebar .sidebar-content { display: block !important; }
        .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini
try:
    genai.configure(api_key=st.secrets["gemini_api_key"])
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
except Exception as e:
    st.warning(f"Gemini not configured: {e}")
    gemini_model = None

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

# ============================================================================
# GEMINI SQL GENERATOR WITH SCHEMA AWARENESS
# ============================================================================

class GeminiSQLGenerator:
    """Uses Gemini to generate SQL queries from natural language"""
    
    def __init__(self, client, gemini_model):
        self.client = client
        self.gemini_model = gemini_model
        self.schema_cache = None
    
    def get_database_schema(self):
        """Fetch actual schema from BigQuery"""
        if self.schema_cache is not None:
            return self.schema_cache
        
        try:
            query = """
            SELECT 
                table_name,
                STRING_AGG(CONCAT(column_name, ' (', data_type, ')'), ', ' ORDER BY ordinal_position) as columns
            FROM `ford-assessment-100425.ford_credit_raw.INFORMATION_SCHEMA.COLUMNS`
            GROUP BY table_name
            ORDER BY table_name
            """
            df = self.client.query(query).to_dataframe()
            
            schema_text = "DATABASE SCHEMA:\n\n"
            for _, row in df.iterrows():
                schema_text += f"Table: {row['table_name']}\n"
                schema_text += f"Columns: {row['columns']}\n\n"
            
            # Get sample vehicle models
            try:
                sample_query = """
                SELECT DISTINCT vehicle_model 
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                WHERE vehicle_model IS NOT NULL
                LIMIT 20
                """
                models_df = self.client.query(sample_query).to_dataframe()
                schema_text += f"Sample vehicle models: {', '.join(models_df['vehicle_model'].tolist())}\n"
            except:
                pass
            
            self.schema_cache = schema_text
            return schema_text
        except Exception as e:
            st.warning(f"Could not fetch schema: {e}")
            return self._get_default_schema()
    
    def _get_default_schema(self):
        return """
DATABASE SCHEMA:

Table: consumer_sales
Columns: vin, customer_id, dealer_id, sale_timestamp, vehicle_model, vehicle_year, trim_level, powertrain, sale_type, sale_price, dealer_state, warranty_type, purchase_financed

Table: customer_360_view
Columns: customer_id, first_name, last_name, credit_tier, household_income_range, state, vehicles_owned, total_loans, avg_loan_amount, total_payments, late_payment_rate, service_interactions

Table: loan_originations
Columns: contract_id, customer_id, vin, contract_type, origination_date, loan_amount, interest_rate_apr, term_months, monthly_payment, remaining_balance, risk_tier, loan_status

Table: billing_payments
Columns: payment_id, customer_id, payment_amount, payment_date, payment_status, due_date

Sample vehicle models: F-150, Mach-E, Explorer, Bronco, Escape, Mustang, Edge, Ranger
        """
    
    def generate_sql(self, natural_language):
        """Use Gemini to generate SQL from natural language"""
        if not self.gemini_model:
            return self._fallback_sql_generation(natural_language)
        
        try:
            schema = self.get_database_schema()
            
            prompt = f"""You are a SQL expert for Ford Analytics. Generate a valid BigQuery SQL query based on the user's request.

{schema}

USER REQUEST: {natural_language}

CRITICAL RULES:
1. Return ONLY the SQL query, no explanation, no markdown, no code blocks
2. Use backticks for table names: `ford-assessment-100425.ford_credit_raw.table_name`
3. If user mentions vehicle models (Mach-E, F-150, Explorer, etc), search vehicle_model column with LIKE '%model%'
4. Use proper BigQuery date functions: DATE_TRUNC, DATE_DIFF, EXTRACT, CURRENT_DATE()
5. Always add reasonable LIMIT (default 100 unless user specifies)
6. Handle NULL values with WHERE column IS NOT NULL
7. For date filters, ensure proper TIMESTAMP or DATE casting

Generate the SQL query now:
"""
            
            response = self.gemini_model.generate_content(prompt)
            sql = response.text.strip()
            
            # Clean up response
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql = "\n".join([line for line in lines if not line.strip().startswith("```")])
                sql = sql.strip()
            
            # Remove "sql" prefix if present
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()
            
            return sql
        except Exception as e:
            st.warning(f"Gemini SQL generation failed: {e}")
            return self._fallback_sql_generation(natural_language)
    
    def _fallback_sql_generation(self, natural_language):
        """Fallback to pattern matching if Gemini fails"""
        nl_lower = natural_language.lower()
        
        if 'mach-e' in nl_lower or 'mach e' in nl_lower or 'mache' in nl_lower:
            return """
SELECT customer_id, vehicle_model, sale_price, sale_timestamp, dealer_state
FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
WHERE LOWER(vehicle_model) LIKE '%mach%'
ORDER BY sale_timestamp DESC
LIMIT 100
            """
        elif 'f-150' in nl_lower or 'f150' in nl_lower:
            return """
SELECT customer_id, vehicle_model, sale_price, sale_timestamp, dealer_state
FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
WHERE LOWER(vehicle_model) LIKE '%f-150%' OR LOWER(vehicle_model) LIKE '%f150%'
ORDER BY sale_timestamp DESC
LIMIT 100
            """
        elif 'california' in nl_lower or 'ca' in nl_lower:
            return """
SELECT customer_id, vehicle_model, sale_price, sale_timestamp, dealer_state
FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
WHERE dealer_state = 'CA'
ORDER BY sale_timestamp DESC
LIMIT 100
            """
        elif any(word in nl_lower for word in ['average', 'avg']) and 'price' in nl_lower:
            return """
SELECT AVG(sale_price) as average_sale_price
FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
WHERE sale_price IS NOT NULL
            """
        else:
            return """
SELECT *
FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
LIMIT 10
            """

# ============================================================================
# GEMINI STRATEGY GENERATOR WITH DATA INSIGHTS
# ============================================================================

class GeminiStrategyManager:
    """Uses Gemini to generate data-driven business strategies"""
    
    def __init__(self, client, gemini_model):
        self.client = client
        self.gemini_model = gemini_model
    
    def get_data_insights(self):
        """Fetch comprehensive insights from BigQuery"""
        try:
            insights = []
            
            # Customer distribution
            query1 = """
            SELECT credit_tier, COUNT(*) as count, 
                   ROUND(AVG(total_loans), 2) as avg_loans,
                   ROUND(AVG(late_payment_rate) * 100, 2) as avg_late_rate_pct
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
            GROUP BY credit_tier
            ORDER BY count DESC
            """
            df1 = self.client.query(query1).to_dataframe()
            insights.append(f"CUSTOMER DISTRIBUTION BY CREDIT TIER:\n{df1.to_string(index=False)}")
            
            # Top vehicle models
            query2 = """
            SELECT vehicle_model, COUNT(*) as total_sales, 
                   ROUND(AVG(sale_price), 0) as avg_price
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE vehicle_model IS NOT NULL
            GROUP BY vehicle_model
            ORDER BY total_sales DESC
            LIMIT 10
            """
            df2 = self.client.query(query2).to_dataframe()
            insights.append(f"\nTOP 10 VEHICLE MODELS BY SALES:\n{df2.to_string(index=False)}")
            
            # Recent sales trends
            query3 = """
            SELECT 
                EXTRACT(YEAR FROM sale_timestamp) as year,
                EXTRACT(MONTH FROM sale_timestamp) as month,
                COUNT(*) as monthly_sales,
                ROUND(AVG(sale_price), 0) as avg_price
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE sale_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
            GROUP BY year, month
            ORDER BY year DESC, month DESC
            """
            df3 = self.client.query(query3).to_dataframe()
            insights.append(f"\nRECENT SALES TRENDS (LAST 6 MONTHS):\n{df3.to_string(index=False)}")
            
            # Payment behavior
            query4 = """
            SELECT payment_status, COUNT(*) as count,
                   ROUND(AVG(payment_amount), 0) as avg_amount
            FROM `ford-assessment-100425.ford_credit_raw.billing_payments`
            GROUP BY payment_status
            ORDER BY count DESC
            """
            df4 = self.client.query(query4).to_dataframe()
            insights.append(f"\nPAYMENT BEHAVIOR DISTRIBUTION:\n{df4.to_string(index=False)}")
            
            return "\n\n".join(insights)
        except Exception as e:
            st.warning(f"Could not fetch full insights: {e}")
            return "Limited data available - using baseline assumptions for strategy generation"
    
    def generate_strategies(self, insights):
        """Use Gemini to generate 4 core strategies"""
        if not self.gemini_model:
            return self._get_default_strategies()
        
        try:
            prompt = f"""You are a senior business strategy consultant for Ford Credit. Based on these REAL data insights from BigQuery, generate exactly 4 sophisticated, data-driven business strategies.

ACTUAL DATA FROM BIGQUERY:
{insights}

Generate ONE strategy for EACH of these 4 categories:

1. **CHURN REDUCTION** - Focus on retaining at-risk customers and reducing customer loss
2. **SALES FORECASTING** - Focus on revenue growth, demand prediction, and market expansion  
3. **CUSTOMER SEGMENTATION** - Focus on personalization, targeting, and value maximization
4. **PRICING ELASTICITY** - Focus on pricing optimization, APR adjustments, and revenue per customer

For EACH strategy, provide this EXACT JSON structure:
{{
  "type": "churn_reduction" | "sales_forecasting" | "customer_segmentation" | "pricing_elasticity",
  "name": "Specific actionable strategy name (10-15 words)",
  "description": "Detailed 2-3 sentence explanation of the strategy",
  "impact": "Quantitative expected impact (e.g., '8-12% churn reduction', '15-20% revenue increase')",
  "feasibility": <integer 1-10, where 10 is most feasible>,
  "rationale": "2-3 sentences explaining WHY this strategy makes sense given the data insights"
}}

IMPORTANT:
- Base your strategies on the ACTUAL data provided above
- Be specific with numbers from the data
- Each strategy must be in a DIFFERENT category
- Make feasibility realistic (consider implementation complexity)
- Return ONLY valid JSON, no other text

Return your response as a JSON object with a "strategies" array containing exactly 4 strategies:
{{
  "strategies": [
    {{strategy 1}},
    {{strategy 2}},
    {{strategy 3}},
    {{strategy 4}}
  ]
}}
"""
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                strategies = data.get('strategies', [])
                
                # Validate we have 4 strategies
                if len(strategies) == 4:
                    return strategies
                else:
                    st.warning(f"Gemini returned {len(strategies)} strategies instead of 4, using defaults")
                    return self._get_default_strategies()
            
            return self._get_default_strategies()
        except Exception as e:
            st.warning(f"Gemini strategy generation failed: {e}")
            return self._get_default_strategies()
    
    def _get_default_strategies(self):
        """Fallback strategies"""
        return [
            {
                "type": "churn_reduction",
                "name": "Proactive Retention Campaign for Inactive High-Value Customers",
                "description": "Identify customers with 120+ days of inactivity who have above-average loan balances. Launch personalized re-engagement campaigns with exclusive rate reductions and dedicated account management.",
                "impact": "10-15% churn reduction in high-value segment",
                "feasibility": 8,
                "rationale": "Data shows significant portion of high-value customers becoming inactive. Early intervention with targeted offers can prevent churn at lower cost than new acquisition."
            },
            {
                "type": "sales_forecasting",
                "name": "Strategic Q4 Push Leveraging Historical Seasonality Patterns",
                "description": "Launch aggressive promotional campaigns in Q4 aligned with historical sales peaks. Focus on top-performing vehicle models with targeted inventory positioning and dealer incentives.",
                "impact": "18-25% Q4 revenue increase",
                "feasibility": 7,
                "rationale": "Historical data shows consistent Q4 sales spikes. Capitalizing on this pattern with proactive inventory and marketing can maximize year-end performance."
            },
            {
                "type": "customer_segmentation",
                "name": "Premium Financial Products for Multi-Vehicle Owners",
                "description": "Create exclusive loan packages targeting customers who own 2+ vehicles. Offer bundled rates, loyalty rewards, and priority service to increase wallet share and lifetime value.",
                "impact": "12-18% revenue per customer increase",
                "feasibility": 9,
                "rationale": "Multi-vehicle owners demonstrate higher loyalty and spend. Capturing more of their business through tailored products is highly feasible and profitable."
            },
            {
                "type": "pricing_elasticity",
                "name": "Dynamic APR Optimization by Credit Tier and Market Conditions",
                "description": "Implement data-driven APR adjustments based on credit tier performance, competitive positioning, and demand elasticity. Test small increments (0.25-0.5%) to optimize revenue without impacting conversion.",
                "impact": "6-10% revenue optimization",
                "feasibility": 6,
                "rationale": "Credit tier data shows varying payment behaviors. Careful pricing adjustments can capture additional margin, though requires market testing and regulatory compliance."
            }
        ]

# ============================================================================
# GEMINI EXECUTIVE SUMMARIZER
# ============================================================================

class GeminiSummarizer:
    """Uses Gemini to create executive summaries of analysis results"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def summarize_analysis(self, strategy, analysis_results):
        """Generate executive summary"""
        if not self.gemini_model:
            return self._generate_basic_summary(strategy, analysis_results)
        
        try:
            # Prepare analysis data
            analysis_summary = f"STRATEGY: {strategy.get('name', 'Unknown')}\n"
            analysis_summary += f"DESCRIPTION: {strategy.get('description', 'N/A')}\n"
            analysis_summary += f"EXPECTED IMPACT: {strategy.get('impact', 'N/A')}\n\n"
            analysis_summary += "ANALYSIS RESULTS:\n\n"
            
            for analysis_type, result in analysis_results.items():
                analysis_summary += f"{analysis_type.upper().replace('_', ' ')}:\n"
                analysis_summary += f"  Summary: {result.get('executive_summary', 'N/A')}\n"
                if result.get('key_metrics'):
                    analysis_summary += f"  Key Metrics: {result['key_metrics']}\n"
                analysis_summary += "\n"
            
            prompt = f"""You are a business analyst creating an executive summary for Ford Credit leadership.

{analysis_summary}

Create a concise 3-4 sentence executive summary that:
1. States recommendation (RECOMMEND / CONSIDER / DO NOT RECOMMEND)
2. Highlights the single most important finding
3. Mentions the biggest risk OR opportunity
4. Provides one clear, actionable next step

Write professionally but directly. Use specific numbers. Keep it under 100 words.
"""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return self._generate_basic_summary(strategy, analysis_results)
    
    def _generate_basic_summary(self, strategy, analysis_results):
        feasibility = strategy.get('feasibility', 5)
        if feasibility >= 8:
            recommendation = "RECOMMEND"
        elif feasibility >= 6:
            recommendation = "CONSIDER"
        else:
            recommendation = "DO NOT RECOMMEND"
        
        return f"{recommendation}: Strategy '{strategy.get('name', 'Unknown')}' shows {strategy.get('impact', 'moderate')} potential impact with feasibility score of {feasibility}/10. Analysis completed across {len(analysis_results)} dimensions. Review detailed findings below for implementation guidance."

# ============================================================================
# AGENTIC DECISION SYSTEM
# ============================================================================

class StrategyAgent:
    """Agent decides which analyses to run based on strategy type"""
    
    @staticmethod
    def decide_analyses(strategy):
        """Intelligently decide analyses based on strategy type"""
        strategy_type = strategy.get('type', 'generic')
        
        analysis_map = {
            'churn_reduction': ['churn_prediction', 'customer_lifetime_value', 'sales_forecasting'],
            'sales_forecasting': ['sales_forecasting', 'revenue_impact', 'geographic_analysis'],
            'customer_segmentation': ['segmentation_analysis', 'customer_lifetime_value', 'pricing_elasticity'],
            'pricing_elasticity': ['pricing_elasticity', 'revenue_impact', 'churn_prediction']
        }
        
        return analysis_map.get(strategy_type, ['sales_forecasting', 'revenue_impact'])

# ============================================================================
# NAVIGATION
# ============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

with st.sidebar:
    st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent.png", width=150)
    st.markdown("---")
    
    st.title("Ford Analytics")
    
    if gemini_model:
        st.success("✨ Gemini Connected")
    else:
        st.error("✨ Gemini Not Connected")
    
    st.markdown("---")
    
    if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state.page == 'Dashboard' else "secondary"):
        st.session_state.page = 'Dashboard'
        st.rerun()
        
    if st.button("SQL Chat", use_container_width=True, type="primary" if st.session_state.page == 'SQL Chat' else "secondary"):
        st.session_state.page = 'SQL Chat'
        st.rerun()
        
    if st.button("Agentic AI System", use_container_width=True, type="primary" if st.session_state.page == 'AI Agent' else "secondary"):
        st.session_state.page = 'AI Agent'
        st.rerun()

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

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
            st.dataframe(data, use_container_width=True)
            st.success(f"Loaded {len(data)} rows from BigQuery")
        except Exception as e:
            st.error(f"Could not load data: {str(e)}")
    else:
        st.info("Connect to BigQuery to see live data")

# ============================================================================
# SQL CHAT PAGE
# ============================================================================

elif st.session_state.page == 'SQL Chat':
    client = get_bigquery_client()
    
    st.title("✨ Intelligent SQL Generator (Powered by Gemini)")
    st.markdown("Natural Language to SQL - Gemini understands your database schema and generates precise queries")
    
    if not gemini_model:
        st.error("⚠️ Gemini not configured. Add 'gemini_api_key' to Streamlit secrets.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Describe Your Analysis")
        natural_language = st.text_area(
            "Ask me anything about your Ford data...",
            placeholder="Examples:\n• Show me customers who purchased the Mach-E\n• Find F-150 sales in California from 2024\n• Which customers have late payments?\n• Average sale price by vehicle model",
            height=120,
            key="nl_input"
        )
        
        if st.button("🚀 Generate SQL with Gemini", type="primary") and natural_language:
            with st.spinner("Gemini is analyzing your request and database schema..."):
                sql_gen = GeminiSQLGenerator(client, gemini_model)
                generated_sql = sql_gen.generate_sql(natural_language)
                st.session_state.generated_sql = generated_sql
                st.session_state.natural_language_query = natural_language
    
    with col2:
        st.subheader("Options")
        auto_execute = st.checkbox("Auto-execute generated SQL", value=True)
        show_explanation = st.checkbox("Show query explanation", value=True)
    
    if hasattr(st.session_state, 'generated_sql'):
        st.markdown("---")
        st.subheader("Generated SQL")
        
        if show_explanation:
            st.info(f"📝 Your request: '{st.session_state.natural_language_query}'")
        
        st.code(st.session_state.generated_sql, language='sql')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Re-generate SQL"):
                with st.spinner("Re-generating with Gemini..."):
                    sql_gen = GeminiSQLGenerator(client, gemini_model)
                    generated_sql = sql_gen.generate_sql(st.session_state.natural_language_query)
                    st.session_state.generated_sql = generated_sql
                    st.rerun()
        
        with col2:
            if st.button("📋 Copy SQL"):
                st.success("SQL ready to copy!")
        
        if auto_execute or st.button("▶️ Execute Query"):
            with st.spinner("Executing query..."):
                try:
                    query_job = client.query(st.session_state.generated_sql)
                    results = query_job.to_dataframe()
                    
                    if not results.empty:
                        st.subheader("📊 Results")
                        st.dataframe(results, use_container_width=True)
                        
                        st.success(f"✅ Returned {len(results)} rows")
                        
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No results returned from the query.")
                except Exception as e:
                    st.error(f"❌ Query execution failed: {e}")
                    st.info("💡 Try regenerating the SQL or modify your question")

# ============================================================================
# AGENTIC AI SYSTEM PAGE
# ============================================================================

elif st.session_state.page == 'AI Agent':
    client = get_bigquery_client()
    
    st.title("🤖 Agentic AI Strategy Testing System")
    st.markdown("**Gemini analyzes data → Generates strategies → Agent decides tests → System executes → Gemini summarizes**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Data Analyzer")
        st.markdown("""
        **Gemini fetches insights**
        - Customer distribution
        - Sales trends
        - Payment behavior
        - Vehicle performance
        """)
        
    with col2:
        st.subheader("🎯 Strategy Generator")
        st.markdown("""
        **Gemini creates strategies**
        - 4 core types
        - Data-driven
        - Feasibility scored
        - Impact quantified
        """)
    
    with col3:
        st.subheader("🔬 Agentic Analyst")
        st.markdown("""
        **Autonomous testing**
        - Decides analyses
        - Runs models
        - Creates visuals
        - Gemini summarizes
        """)
    
    st.markdown("---")
    
    if not client:
        st.error("❌ BigQuery connection required")
        st.stop()
    
    if not gemini_model:
        st.error("❌ Gemini not configured")
        st.stop()
    
    # Initialize session state
    if 'strategies_generated' not in st.session_state:
        st.session_state.strategies_generated = []
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = None
    
    # Generate strategies button
    if st.button("🚀 Generate AI Strategies with Gemini", type="primary", use_container_width=True):
        with st.spinner("🔄 Gemini is analyzing your BigQuery data..."):
            strategy_manager = GeminiStrategyManager(client, gemini_model)
            
            with st.status("Fetching data insights...", expanded=True) as status:
                st.write("📊 Querying BigQuery for customer insights...")
                insights = strategy_manager.get_data_insights()
                st.write("✅ Data insights collected")
                
                st.write("🤖 Gemini generating strategies...")
                strategies = strategy_manager.generate_strategies(insights)
                st.write(f"✅ Generated {len(strategies)} strategies")
                
                status.update(label="✅ Strategy generation complete!", state="complete")
            
            st.session_state.strategies_generated = strategies
            if strategies:
                st.session_state.current_strategy = strategies[0]
            st.success(f"✅ Successfully generated {len(strategies)} data-driven strategies!")
            st.rerun()
