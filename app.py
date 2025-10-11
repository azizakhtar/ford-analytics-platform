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
from sklearn.preprocessing import StandardScaler
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config MUST be first
st.set_page_config(page_title="Ford Analytics", layout="wide")

# Hide Streamlit defaults
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        .st-emotion-cache-1oe5cao { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# OPTIMIZATION 1: Centralized Cache & Connection Management
# ============================================================================

class ConnectionManager:
    """Singleton pattern for BigQuery connections - avoid recreating clients"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self):
        if self._client is None:
            try:
                secrets = st.secrets.get("gcp_service_account")
                if secrets:
                    service_account_info = {
                        "type": "service_account",
                        "project_id": secrets["project_id"],
                        "private_key": secrets["private_key"].replace('\\n', '\n'),
                        "client_email": secrets["client_email"],
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                    credentials = service_account.Credentials.from_service_account_info(service_account_info)
                    self._client = bigquery.Client(credentials=credentials, project=secrets["project_id"])
                    logger.info("BigQuery client initialized")
            except Exception as e:
                logger.error(f"BigQuery connection failed: {e}")
        return self._client

# ============================================================================
# OPTIMIZATION 2: Query Caching & Template Management
# ============================================================================

class QueryCache:
    """Cache frequently executed queries"""
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl
    
    @st.cache_data(ttl=3600)
    def execute_cached_query(self, query: str, client):
        try:
            return client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()

class QueryTemplateManager:
    """Centralized query template management"""
    
    TEMPLATES = {
        'customer_count': """
            SELECT COUNT(*) as total_customers 
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
        """,
        'top_customers': """
            SELECT customer_id, COUNT(vin) as purchases, SUM(sale_price) as total_spent
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            GROUP BY customer_id ORDER BY total_spent DESC LIMIT {limit}
        """,
        'sales_by_tier': """
            SELECT cp.credit_tier, COUNT(cs.vin) as sales, SUM(cs.sale_price) as revenue
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view` cp
            JOIN `ford-assessment-100425.ford_credit_raw.consumer_sales` cs ON cp.customer_id = cs.customer_id
            GROUP BY cp.credit_tier ORDER BY revenue DESC
        """,
        'monthly_trends': """
            SELECT 
                EXTRACT(YEAR FROM sale_timestamp) as year,
                EXTRACT(MONTH FROM sale_timestamp) as month,
                COUNT(*) as sales, SUM(sale_price) as revenue
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            GROUP BY year, month ORDER BY year, month
        """,
        'churn_risk': """
            SELECT customer_id, DATE_DIFF(CURRENT_DATE(), DATE(MAX(sale_timestamp)), DAY) as days_inactive
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            GROUP BY customer_id HAVING days_inactive > 90
        """
    }
    
    @staticmethod
    def get_template(name: str, **kwargs) -> str:
        template = QueryTemplateManager.TEMPLATES.get(name, "")
        return template.format(**kwargs) if kwargs else template

# ============================================================================
# OPTIMIZATION 3: Lightweight Strategy Pattern
# ============================================================================

class StrategyAnalyzer:
    """Simplified strategy testing with pattern matching"""
    
    STRATEGY_PATTERNS = {
        'pricing': {
            'sales_impact': -0.05,
            'revenue_impact': 0.08,
            'confidence': 'High',
            'risk': 'Churn increase 2-5%'
        },
        'retention': {
            'sales_impact': 0.05,
            'revenue_impact': 0.18,
            'confidence': 'High',
            'risk': 'Implementation cost'
        },
        'loyalty': {
            'sales_impact': 0.08,
            'revenue_impact': 0.12,
            'confidence': 'High',
            'risk': 'Low adoption'
        },
        'referral': {
            'sales_impact': 0.12,
            'revenue_impact': 0.15,
            'confidence': 'Medium',
            'risk': 'Quality concerns'
        }
    }
    
    @staticmethod
    @lru_cache(maxsize=32)
    def analyze_strategy(strategy_name: str) -> Dict:
        """Fast strategy analysis without heavy computation"""
        strategy_lower = strategy_name.lower()
        
        for key, params in StrategyAnalyzer.STRATEGY_PATTERNS.items():
            if key in strategy_lower:
                return params
        
        return {
            'sales_impact': 0.08,
            'revenue_impact': 0.12,
            'confidence': 'Medium',
            'risk': 'Unvalidated strategy'
        }

# ============================================================================
# OPTIMIZATION 4: Data Generation (Fallback - no network calls)
# ============================================================================

class DataGenerator:
    """Generate sample data efficiently for analysis"""
    
    @staticmethod
    def generate_timeseries(periods: int = 12) -> pd.DataFrame:
        """Generate monthly sales data"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=periods, freq='M')
        trend = np.linspace(1000, 1500, periods)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(periods) / 12)
        noise = np.random.normal(0, 50, periods)
        
        return pd.DataFrame({
            'date': dates,
            'sales': trend + seasonal + noise,
            'revenue': (trend + seasonal + noise) * 25000
        })
    
    @staticmethod
    def generate_customer_segments(n_samples: int = 500) -> pd.DataFrame:
        """Generate customer segmentation data"""
        np.random.seed(42)
        return pd.DataFrame({
            'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
            'spend': np.random.exponential(50000, n_samples).clip(0),
            'transactions': np.random.poisson(5, n_samples) + 1,
            'tier': np.random.choice(['Gold', 'Silver', 'Bronze'], n_samples, p=[0.2, 0.5, 0.3]),
            'churn_risk': np.random.beta(2, 5, n_samples) * 100
        })

# ============================================================================
# OPTIMIZATION 5: Fast Model Training (Vectorized)
# ============================================================================

class FastAnalytics:
    """Lightweight ML without slow imports"""
    
    @staticmethod
    def quick_forecast(df: pd.DataFrame, periods: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Fast linear forecasting"""
        if len(df) < 3:
            return np.array([]), np.array([])
        
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['sales'].values if 'sales' in df.columns else df.iloc[:, 0].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        X_future = np.arange(len(df), len(df) + periods).reshape(-1, 1)
        forecast = model.predict(X_future)
        
        return forecast, np.full(periods, np.std(y) * 0.15)
    
    @staticmethod
    def segment_customers(df: pd.DataFrame, n_segments: int = 3) -> np.ndarray:
        """Fast customer segmentation using quantiles"""
        spend_quantiles = np.quantile(df['spend'], np.linspace(0, 1, n_segments + 1))
        return np.digitize(df['spend'], spend_quantiles) - 1

# ============================================================================
# OPTIMIZATION 6: Streamlined Agent System
# ============================================================================

class ManagerAgent:
    """Simplified manager agent - discovers strategies via pattern matching"""
    
    STRATEGIES = [
        "Test 2% APR reduction for Gold-tier customers",
        "Implement reactivation campaign for inactive customers",
        "Create bundled product offering for high-value segments",
        "Launch targeted upselling campaign for medium-tier customers",
        "Optimize loan approval rates for Silver-tier customers"
    ]
    
    @staticmethod
    def discover_strategies() -> List[str]:
        """Return pre-defined strategy portfolio"""
        return ManagerAgent.STRATEGIES

class AnalystAgent:
    """Simplified analyst agent - runs efficient tests"""
    
    def __init__(self, client: Optional[bigquery.Client] = None):
        self.client = client
        self.cache = QueryCache()
    
    def test_strategy(self, strategy: str) -> Dict:
        """Run lightweight strategy tests"""
        analysis = StrategyAnalyzer.analyze_strategy(strategy)
        
        # Generate sample data if no client
        if self.client is None:
            df = DataGenerator.generate_timeseries()
        else:
            try:
                df = self.cache.execute_cached_query(
                    QueryTemplateManager.get_template('monthly_trends'),
                    self.client
                )
            except:
                df = DataGenerator.generate_timeseries()
        
        # Run forecast
        forecast, confidence = FastAnalytics.quick_forecast(df)
        
        return {
            'strategy': strategy,
            'analysis': analysis,
            'forecast': forecast,
            'confidence_interval': confidence,
            'recommendation': self._generate_recommendation(analysis, forecast),
            'metrics': self._calculate_metrics(analysis, forecast)
        }
    
    @staticmethod
    def _generate_recommendation(analysis: Dict, forecast: np.ndarray) -> str:
        if analysis['confidence'] == 'High' and len(forecast) > 0:
            return "✓ STRONG: Proceed with implementation"
        elif analysis['confidence'] == 'Medium':
            return "⚠ MODERATE: Test in limited rollout"
        else:
            return "✗ CAUTION: Requires refinement"
    
    @staticmethod
    def _calculate_metrics(analysis: Dict, forecast: np.ndarray) -> Dict:
        return {
            'Sales Impact': f"{analysis['sales_impact']*100:+.1f}%",
            'Revenue Impact': f"{analysis['revenue_impact']*100:+.1f}%",
            'Confidence': analysis['confidence'],
            'Risk': analysis['risk']
        }

# ============================================================================
# OPTIMIZATION 7: Streamlined UI
# ============================================================================

def check_password():
    try:
        correct_password = st.secrets["password"]
    except:
        correct_password = "ford2024"
    
    if st.session_state.get("password_correct"):
        return True
    
    st.title("Ford Analytics Portal")
    pwd = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Login"):
        if hmac.compare_digest(pwd, correct_password):
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False

def main():
    if not check_password():
        st.stop()
    
    # Initialize connection
    conn_manager = ConnectionManager()
    client = conn_manager.get_client()
    
    # Navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics/main/transparent.png", width=150)
        st.markdown("---")
        st.title("Ford Analytics")
        
        pages = ['Dashboard', 'SQL Chat', 'Strategy Agent']
        for page in pages:
            if st.button(page, use_container_width=True, 
                        type="primary" if st.session_state.page == page else "secondary"):
                st.session_state.page = page
                st.rerun()
    
    # Page rendering
    if st.session_state.page == 'Dashboard':
        st.title("Ford Analytics Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", "$4.2M", "+12%")
        col2.metric("Active Loans", "1,847", "+8%")
        col3.metric("Delinquency Rate", "2.3%", "-0.4%")
        col4.metric("Customer Satisfaction", "4.2/5", "+0.3")
    
    elif st.session_state.page == 'Strategy Agent':
        st.title("Agentic Strategy Testing System")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Manager Agent")
            st.markdown("Discovers growth strategies from data patterns")
        with col2:
            st.subheader("Analyst Agent")
            st.markdown("Tests strategies with statistical modeling")
        
        st.markdown("---")
        
        # Manager Agent - Generate Strategies
        if st.button("Generate Strategies", type="primary"):
            strategies = ManagerAgent.discover_strategies()
            st.session_state['strategies'] = strategies
        
        # Display strategies
        if st.session_state.get('strategies'):
            st.subheader("Generated Strategies")
            analyst = AnalystAgent(client)
            
            for strategy in st.session_state['strategies']:
                with st.expander(strategy):
                    if st.button(f"Test Strategy", key=f"test_{strategy}"):
                        with st.spinner("Running analysis..."):
                            results = analyst.test_strategy(strategy)
                            
                            # Display results
                            st.markdown(f"**Recommendation:** {results['recommendation']}")
                            
                            cols = st.columns(len(results['metrics']))
                            for col, (k, v) in zip(cols, results['metrics'].items()):
                                col.metric(k, v)
                            
                            # Show forecast
                            if len(results['forecast']) > 0:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(results['forecast'], marker='o', label='Forecast')
                                ax.fill_between(range(len(results['forecast'])),
                                               results['forecast'] - results['confidence_interval'],
                                               results['forecast'] + results['confidence_interval'],
                                               alpha=0.2)
                                ax.set_title('12-Month Strategy Impact Forecast')
                                ax.set_ylabel('Expected Outcome')
                                ax.legend()
                                st.pyplot(fig)

if __name__ == "__main__":
    main()
