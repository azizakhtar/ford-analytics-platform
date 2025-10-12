"""
SETUP INSTRUCTIONS:
1. Get your Anthropic API key from: https://console.anthropic.com/
2. Add to Streamlit secrets:
   anthropic_api_key = "sk-ant-..."
3. Install: pip install anthropic
"""

import streamlit as st
import hmac
import pandas as pd
import numpy as np
import re
import json
import requests
from google.cloud import bigquery
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import scipy.stats as stats
import google.generativeai as genai
from anthropic import Anthropic

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

# Initialize Claude
try:
    claude_client = Anthropic(api_key=st.secrets["anthropic_api_key"])
    claude_available = True
except Exception as e:
    st.warning(f"Claude not configured: {e}")
    claude_client = None
    claude_available = False

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
# DUAL AI SYSTEM: Claude for Strategy, Gemini for SQL
# ============================================================================

class ClaudeSQLGenerator:
    """Uses Claude for SQL generation - better reasoning than Gemini"""
    
    def __init__(self, client, claude_client):
        self.client = client
        self.claude_client = claude_client
        self.schema_cache = None
    
    def get_database_schema(self):
        """Fetch actual schema from BigQuery"""
        if self.schema_cache is not None:
            return self.schema_cache
        
        try:
            query = """
            SELECT 
                table_name,
                STRING_AGG(CONCAT(column_name, ' (', data_type, ')'), ', ') as columns
            FROM `ford-assessment-100425.ford_credit_raw.INFORMATION_SCHEMA.COLUMNS`
            GROUP BY table_name
            """
            df = self.client.query(query).to_dataframe()
            
            schema_text = "Available tables:\n"
            for _, row in df.iterrows():
                schema_text += f"\n{row['table_name']}:\n  {row['columns']}\n"
            
            # Also get sample values for vehicle_model
            try:
                sample_query = """
                SELECT DISTINCT vehicle_model 
                FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
                WHERE vehicle_model IS NOT NULL
                LIMIT 20
                """
                models_df = self.client.query(sample_query).to_dataframe()
                schema_text += f"\nSample vehicle models: {', '.join(models_df['vehicle_model'].tolist())}\n"
            except:
                pass
            
            self.schema_cache = schema_text
            return schema_text
        except Exception as e:
            st.warning(f"Could not fetch schema: {e}")
            return self._get_default_schema()
    
    def _get_default_schema(self):
        return """
Available tables:
consumer_sales: vin, customer_id, dealer_id, sale_timestamp, vehicle_model (F-150, Mach-E, Explorer, Bronco, etc), 
                vehicle_year, trim_level, powertrain, sale_type, sale_price, dealer_state, warranty_type, purchase_financed
customer_360_view: customer_id, first_name, last_name, credit_tier, household_income_range, state, vehicles_owned
loan_originations: contract_id, customer_id, vin, loan_amount, interest_rate_apr, term_months, loan_status
billing_payments: payment_id, customer_id, payment_amount, payment_date, payment_status
        """
    
    def generate_sql(self, natural_language):
        """Use Claude to generate SQL from natural language"""
        if not self.claude_client:
            return self._fallback_sql_generation(natural_language)
        
        try:
            schema = self.get_database_schema()
            
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a SQL expert for Ford Analytics. Generate a valid BigQuery SQL query.

DATABASE SCHEMA:
{schema}

USER REQUEST: {natural_language}

CRITICAL RULES:
- Return ONLY the SQL query, no explanation, no markdown
- Use backticks for table names: `ford-assessment-100425.ford_credit_raw.table_name`
- If user mentions a vehicle model (Mach-E, F-150, Explorer, etc), search vehicle_model column with LIKE
- Use proper BigQuery date functions: DATE_TRUNC, DATE_DIFF, EXTRACT
- Always add LIMIT clause (default 100) unless user specifies otherwise
- Handle NULL values appropriately

Generate the SQL query now:"""
                    }
                ]
            )
            
            sql = message.content[0].text.strip()
            
            # Clean up if Claude added markdown
            if sql.startswith("```"):
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:]
                sql = sql.strip()
            
            return sql
        except Exception as e:
            st.warning(f"Claude SQL generation failed: {e}")
            return self._fallback_sql_generation(natural_language)
    
    def _fallback_sql_generation(self, natural_language):
        nl_lower = natural_language.lower()
        
        if 'mach-e' in nl_lower or 'mach e' in nl_lower or 'mache' in nl_lower:
            return """
            SELECT 
                customer_id, vehicle_model, sale_price, sale_timestamp, dealer_state
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE LOWER(vehicle_model) LIKE '%mach%'
            ORDER BY sale_timestamp DESC
            LIMIT 100
            """
        elif 'f-150' in nl_lower or 'f150' in nl_lower:
            return """
            SELECT 
                customer_id, vehicle_model, sale_price, sale_timestamp, dealer_state
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE LOWER(vehicle_model) LIKE '%f-150%' OR LOWER(vehicle_model) LIKE '%f150%'
            ORDER BY sale_timestamp DESC
            LIMIT 100
            """
        else:
            return """
            SELECT *
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            LIMIT 10
            """

class ClaudeStrategyManager:
    """Uses Claude to generate sophisticated business strategies"""
    
    def __init__(self, client, claude_client):
        self.client = client
        self.claude_client = claude_client
    
    def get_data_insights(self):
        """Fetch comprehensive insights from BigQuery"""
        try:
            insights = []
            
            # Customer distribution
            query1 = """
            SELECT credit_tier, COUNT(*) as count, AVG(total_loans) as avg_loans, AVG(late_payment_rate) as avg_late_rate
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
            GROUP BY credit_tier
            ORDER BY count DESC
            """
            df1 = self.client.query(query1).to_dataframe()
            insights.append(f"Customer Distribution:\n{df1.to_string()}")
            
            # Sales by vehicle model
            query2 = """
            SELECT vehicle_model, COUNT(*) as sales, AVG(sale_price) as avg_price
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE vehicle_model IS NOT NULL
            GROUP BY vehicle_model
            ORDER BY sales DESC
            LIMIT 10
            """
            df2 = self.client.query(query2).to_dataframe()
            insights.append(f"\nTop Vehicle Models:\n{df2.to_string()}")
            
            # Recent sales trends
            query3 = """
            SELECT 
                EXTRACT(YEAR FROM sale_timestamp) as year,
                EXTRACT(MONTH FROM sale_timestamp) as month,
                COUNT(*) as sales,
                AVG(sale_price) as avg_price
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE sale_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 MONTH)
            GROUP BY year, month
            ORDER BY year, month
            """
            df3 = self.client.query(query3).to_dataframe()
            insights.append(f"\nRecent Sales Trends (Last 6 Months):\n{df3.to_string()}")
            
            # Payment behavior
            query4 = """
            SELECT payment_status, COUNT(*) as count, AVG(payment_amount) as avg_amount
            FROM `ford-assessment-100425.ford_credit_raw.billing_payments`
            GROUP BY payment_status
            """
            df4 = self.client.query(query4).to_dataframe()
            insights.append(f"\nPayment Behavior:\n{df4.to_string()}")
            
            return "\n\n".join(insights)
        except Exception as e:
            st.warning(f"Could not fetch insights: {e}")
            return "Limited data available for analysis"
    
    def generate_strategies(self, insights):
        """Use Claude to generate sophisticated, data-driven strategies"""
        if not self.claude_client:
            return self._get_default_strategies()
        
        try:
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a senior business strategy consultant for Ford Credit. Based on these ACTUAL data insights from BigQuery, generate exactly 4 sophisticated business strategies.

REAL DATA INSIGHTS:
{insights}

Generate strategies in these 4 categories:

1. **CHURN REDUCTION** - Focus on customer retention and loyalty
2. **SALES FORECASTING** - Focus on revenue growth and market expansion  
3. **CUSTOMER SEGMENTATION** - Focus on targeting and personalization
4. **PRICING ELASTICITY** - Focus on pricing optimization and profitability

For EACH strategy, provide:
- **name**: Specific, actionable strategy name
- **description**: Detailed 2-3 sentence explanation
- **impact**: Expected quantitative impact (e.g., "8-12% churn reduction", "15-20% revenue increase")
- **feasibility**: Score 1-10 (10 = most feasible)
- **rationale**: Why this strategy makes sense given the data insights

FORMAT YOUR RESPONSE AS VALID JSON:
{{
  "strategies": [
    {{
      "type": "churn_reduction",
      "name": "...",
      "description": "...",
      "impact": "...",
      "feasibility": 8,
      "rationale": "..."
    }},
    {{
      "type": "sales_forecasting",
      "name": "...",
      "description": "...",
      "impact": "...",
      "feasibility": 7,
      "rationale": "..."
    }},
    {{
      "type": "customer_segmentation",
      "name": "...",
      "description": "...",
      "impact": "...",
      "feasibility": 9,
      "rationale": "..."
    }},
    {{
      "type": "pricing_elasticity",
      "name": "...",
      "description": "...",
      "impact": "...",
      "feasibility": 6,
      "rationale": "..."
    }}
  ]
}}

Generate the strategies now:"""
                    }
                ]
            )
            
            response_text = message.content[0].text.strip()
            
            # Extract JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                return data.get('strategies', self._get_default_strategies())
            
            return self._get_default_strategies()
        except Exception as e:
            st.warning(f"Claude strategy generation failed: {e}")
            return self._get_default_strategies()
    
    def _get_default_strategies(self):
        return [
            {
                "type": "churn_reduction",
                "name": "Proactive Retention Campaign for At-Risk Gold Tier Customers",
                "description": "Identify Gold tier customers with declining engagement and launch personalized outreach with exclusive offers and dedicated account management.",
                "impact": "10-15% churn reduction in high-value segment",
                "feasibility": 8,
                "rationale": "Gold tier customers represent highest lifetime value"
            },
            {
                "type": "sales_forecasting",
                "name": "Strategic Q4 Sales Push Leveraging Historical Peaks",
                "description": "Launch aggressive promotional campaigns in Q4 aligned with historical sales seasonality patterns to capture year-end demand.",
                "impact": "18-25% Q4 revenue increase",
                "feasibility": 7,
                "rationale": "Historical data shows strong Q4 performance potential"
            },
            {
                "type": "customer_segmentation",
                "name": "Premium Financial Products for High-Value Multi-Vehicle Owners",
                "description": "Create exclusive financial products targeting customers who own 2+ vehicles with premium rates and benefits.",
                "impact": "12-18% revenue per customer increase",
                "feasibility": 9,
                "rationale": "Multi-vehicle owners show higher loyalty and spend"
            },
            {
                "type": "pricing_elasticity",
                "name": "Dynamic APR Optimization Based on Credit Tier Performance",
                "description": "Implement data-driven APR adjustments by credit tier to maximize revenue while maintaining competitive positioning.",
                "impact": "6-10% revenue optimization",
                "feasibility": 6,
                "rationale": "Pricing changes require careful market testing"
            }
        ]

class ClaudeAnalystSummarizer:
    """Uses Claude to summarize analysis results intelligently"""
    
    def __init__(self, claude_client):
        self.claude_client = claude_client
    
    def summarize_analysis(self, strategy, analysis_results):
        """Use Claude to create executive summary of all analyses"""
        if not self.claude_client:
            return self._generate_basic_summary(strategy, analysis_results)
        
        try:
            # Prepare analysis data for Claude
            analysis_summary = ""
            for analysis_type, result in analysis_results.items():
                analysis_summary += f"\n\n{analysis_type.upper()}:\n"
                analysis_summary += f"Summary: {result.get('executive_summary', 'N/A')}\n"
                if result.get('key_metrics'):
                    analysis_summary += f"Key Metrics: {result['key_metrics']}\n"
            
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a business analyst creating an executive summary for Ford Credit leadership.

STRATEGY BEING TESTED:
{strategy.get('name', 'Unknown Strategy')}
Description: {strategy.get('description', 'N/A')}
Expected Impact: {strategy.get('impact', 'N/A')}

ANALYSIS RESULTS:
{analysis_summary}

Create a concise 3-4 sentence executive summary that:
1. States whether the strategy is recommended (yes/no/maybe)
2. Highlights the most important finding
3. Mentions the biggest risk or opportunity
4. Provides a clear next step

Write in a professional but direct tone. Be specific with numbers where available."""
                    }
                ]
            )
            
            return message.content[0].text.strip()
        except Exception as e:
            return self._generate_basic_summary(strategy, analysis_results)
    
    def _generate_basic_summary(self, strategy, analysis_results):
        return f"Strategy '{strategy.get('name', 'Unknown')}' shows {strategy.get('impact', 'moderate')} potential impact. Analysis completed across {len(analysis_results)} dimensions. Review detailed findings below for implementation recommendations."

# ============================================================================
# AGENTIC DECISION SYSTEM
# ============================================================================

class StrategyAgent:
    """Agent that decides which analyses to run based on strategy type"""
    
    @staticmethod
    def decide_analyses(strategy):
        """Decide which analyses to run based on strategy type"""
        strategy_type = strategy.get('type', 'generic')
        
        analysis_map = {
            'churn_reduction': ['churn_prediction', 'customer_lifetime_value', 'sales_forecasting'],
            'sales_forecasting': ['sales_forecasting', 'revenue_impact', 'geographic_analysis'],
            'customer_segmentation': ['segmentation_analysis', 'customer_lifetime_value', 'pricing_elasticity'],
            'pricing_elasticity': ['pricing_elasticity', 'revenue_impact', 'churn_prediction']
        }
        
        return analysis_map.get(strategy_type, ['sales_forecasting', 'revenue_impact'])

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

# Custom sidebar navigation
with st.sidebar:
    st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent.png", width=150)
    st.markdown("---")
    
    st.title("Ford Analytics")
    
    # AI Status indicators
    if claude_available:
        st.success("🤖 Claude Connected")
    else:
        st.error("🤖 Claude Not Connected")
    
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
    
    st.title("🤖 Intelligent SQL Generator (Powered by Claude)")
    st.markdown("Natural Language to SQL - Claude understands context and generates precise queries")
    
    if not claude_available:
        st.warning("⚠️ Claude not configured. Add 'anthropic_api_key' to Streamlit secrets to enable.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Describe Your Analysis")
        natural_language = st.text_area(
            "Tell me what you want to analyze...",
            placeholder="Examples:\n• Show me customers who purchased the Mach-E\n• Find F-150 sales in California from 2024\n• Which customers have late payments?",
            height=120,
            key="nl_input"
        )
        
        if st.button("Generate SQL with Claude", type="primary") and natural_language:
            with st.spinner("Claude is analyzing your request..."):
                sql_gen = ClaudeSQLGenerator(client, claude_client)
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
            st.info(f"Your request: '{st.session_state.natural_language_query}'")
        
        st.code(st.session_state.generated_sql, language='sql')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Re-generate SQL"):
                with st.spinner("Re-generating with Claude..."):
                    sql_gen = ClaudeSQLGenerator(client, claude_client)
                    generated_sql = sql_gen.generate_sql(st.session_state.natural_language_query)
                    st.session_state.generated_sql = generated_sql
                    st.rerun()
        
        with col2:
            if st.button("Copy SQL"):
                st.success("SQL ready to copy!")
        
        if auto_execute or st.button("Execute Query"):
            with st.spinner("Executing query..."):
                try:
                    query_job = client.query(st.session_state.generated_sql)
                    results = query_job.to_dataframe()
                    
                    if not results.empty:
                        st.subheader("Results")
                        st.dataframe(results, use_container_width=True)
                        
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No results returned from the query.")
                except Exception as e:
                    st.error(f"Query execution failed: {e}")

elif st.session_state.page == 'AI Agent':
    client = get_bigquery_client()
    
    st.title("🤖 Agentic AI Strategy System")
    st.markdown("**Claude generates strategies • Agent decides analyses • System executes & summarizes**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Claude Strategy Generator")
        st.markdown("""
        **Analyzes real data**
        - Fetches BigQuery insights
        - Generates 4 core strategies
        - Scores feasibility
        - Provides rationale
        """)
        
    with col2:
        st.subheader("Agentic Analyst")
        st.markdown("""
        **Autonomous execution**
        - Decides which tests to run
        - Builds statistical models
        - Creates visualizations
        - Runs analyses
        """)
    
    with col3:
        st.subheader("Claude Summarizer")
        st.markdown("""
        **Executive insights**
        - Synthesizes all findings
        - Highlights key risks
        - Clear recommendations
        - Actionable next steps
        """)
    
    st.markdown("---")
    
    if not client:
        st.error("BigQuery connection required")
        st.stop()
    
    if not claude_available:
        st.warning("⚠️ Claude not configured. Add 'anthropic_api_key' to Streamlit secrets for full functionality.")
    
    # Initialize session state
    if 'strategies_generated' not in st.session_state:
        st.session_state.strategies_generated = []
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = None
    
    # Generate strategies button
    if st.button("🚀 Generate AI Strategies with Claude", type="primary"):
        with st.spinner("Claude is analyzing your data and generating strategies..."):
            strategy_manager = ClaudeStrategyManager(client, claude_client)
            insights = strategy_manager.get_data_insights()
            strategies = strategy_manager.generate_strategies(insights)
            st.session_state.strategies_generated = strategies
            if strategies:
                st.session_state.current_strategy = strategies[0]
            st.success(f"✅ Generated {len(strategies)} data-driven strategies")
            st.rerun()
    
    # Display strategies
    if st.session_state.strategies_generated:
        st.subheader("Generated Strategies")
        
        for strategy in st.session_state.strategies_generated:
            strategy_name = strategy.get('name', 'Unknown')
            feasibility = strategy.get('feasibility', 0)
            
            with st.expander(f"⭐ {feasibility}/10 - {strategy_name}", expanded=(strategy == st.session_state.current_strategy)):
                st.write(f"**Type:** {strategy.get('type', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Description:** {strategy.get('description', 'N/A')}")
                st.write(f"**Expected Impact:** {strategy.get('impact', 'N/A')}")
                if strategy.get('rationale'):
                    st.write(f"**Rationale:** {strategy.get('rationale', 'N/A')}")
                
                if st.button(f"🔬 Test This Strategy", key=f"test_{strategy_name}"):
                    st.session_state.current_strategy = strategy
                    st.rerun()
    
    # Test strategy
    if st.session_state.current_strategy:
        strategy = st.session_state.current_strategy
        strategy_name = strategy.get('name', 'Unknown')
        
        st.markdown("---")
        st.header(f"Testing: {strategy_name}")
        
        if strategy_name not in st.session_state.test_results:
            if st.button("▶️ Run Agentic Analysis", type="primary"):
                with st.spinner("Agent deciding which analyses to run..."):
                    # Agent decides analyses
                    required_analyses = StrategyAgent.decide_analyses(strategy)
                    st.info(f"Agent decided to run: {', '.join(required_analyses)}")
                    
                    # This would connect to your existing BusinessAnalyst class
                    # For now, showing the framework
                    st.success("✅ Analysis complete! (Full integration with your existing analysts would go here)")
                    
                    # Placeholder for demonstration
                    test_results = {
                        "strategy": strategy,
                        "analyses_run": required_analyses,
                        "summary": "Analysis framework ready for integration"
                    }
                    
                    st.session_state.test_results[strategy_name] = test_results
                    st.rerun()
        else:
            test_results = st.session_state.test_results[strategy_name]
            st.success(f"✅ Completed {len(test_results.get('analyses_run', []))} analyses")
            
            # Claude summary would go here
            if claude_available:
                st.subheader("Claude Executive Summary")
                with st.spinner("Claude is synthesizing findings..."):
                    summarizer
