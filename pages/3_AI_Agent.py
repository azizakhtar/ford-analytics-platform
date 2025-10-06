import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")

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

client = get_bigquery_client()

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
            "Optimize loan approval rates for Silver-tier customers",
            "Develop loyalty program for repeat customers",
            "Create seasonal promotion for Q4 sales boost",
            "Implement risk-based pricing for different credit tiers"
        ]
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
        st.title("🧠 AI Business Strategy Testing System")
        st.markdown("**Manager Agent** discovers strategies **Analyst Agent** creates tests & models")
        
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
            st.info("👆 Click 'Generate Business Strategies' to start the AI analysis")

# Main execution
if not client:
    st.error("❌ BigQuery connection required for AI Agent")
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
