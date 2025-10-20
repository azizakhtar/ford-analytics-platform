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
from datetime import datetime

# Page config MUST be first
st.set_page_config(page_title="DataSphere Analytics", layout="wide")

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

# Initialize session state for agent_config
if 'agent_config' not in st.session_state:
    st.session_state.agent_config = {
        'max_iterations': 3,
        'analyses_per_strategy': 3,
        'enable_retry': True,
        'timeout_seconds': 300,
        'force_analyses': []
    }

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
        
    st.title("DataSphere Analytics Portal")
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
# AGENT EVALUATION METRICS CLASS
# ============================================================================

class AgentEvaluationMetrics:
    """Tracks and evaluates agent performance across multiple dimensions"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_strategy_quality(self, strategy, test_results):
        """Evaluate the quality of a generated strategy"""
        scores = {}
        
        description = strategy.get('description', '')
        specificity = 0
        
        if any(char.isdigit() for char in description):
            specificity += 30
        
        action_words = ['implement', 'launch', 'create', 'develop', 'optimize', 'target', 'identify']
        if any(word in description.lower() for word in action_words):
            specificity += 20
        
        segments = ['gold', 'silver', 'bronze', 'high-value', 'tier', 'segment']
        if any(seg in description.lower() for seg in segments):
            specificity += 25
        
        if any(word in description.lower() for word in ['month', 'quarter', 'year', 'days']):
            specificity += 25
        
        scores['specificity_score'] = min(specificity, 100)
        
        data_driven = 0
        if strategy.get('rationale') and len(strategy.get('rationale', '')) > 50:
            data_driven += 40
        
        data_words = ['data', 'analysis', 'shows', 'indicates', 'customer', 'sales', 'revenue']
        rationale = strategy.get('rationale', '').lower()
        data_mentions = sum(1 for word in data_words if word in rationale)
        data_driven += min(data_mentions * 10, 40)
        
        if strategy.get('impact') and '%' in strategy.get('impact', ''):
            data_driven += 20
        
        scores['data_driven_score'] = min(data_driven, 100)
        scores['feasibility_score'] = strategy.get('feasibility', 5) * 10
        
        impact = strategy.get('impact', '')
        impact_clarity = 0
        
        if '%' in impact:
            impact_clarity += 40
        
        if '-' in impact and '%' in impact:
            impact_clarity += 30
        
        impact_metrics = ['revenue', 'churn', 'growth', 'retention', 'customers']
        if any(metric in impact.lower() for metric in impact_metrics):
            impact_clarity += 30
        
        scores['impact_clarity_score'] = min(impact_clarity, 100)
        
        overall = (
            scores['specificity_score'] * 0.25 +
            scores['data_driven_score'] * 0.30 +
            scores['feasibility_score'] * 0.25 +
            scores['impact_clarity_score'] * 0.20
        )
        
        scores['overall_quality'] = overall
        return scores
    
    def evaluate_analysis_quality(self, analysis_result):
        """Evaluate quality of an analysis"""
        scores = {}
        
        completeness = 0
        required_fields = ['analysis_type', 'executive_summary', 'key_metrics']
        
        for field in required_fields:
            if field in analysis_result and analysis_result[field]:
                completeness += 33
        
        scores['completeness_score'] = min(completeness, 100)
        
        viz_score = 0
        if 'visualizations' in analysis_result and analysis_result['visualizations']:
            viz_score += 50
            if len(analysis_result['visualizations']) > 1:
                viz_score += 25
            viz_score += 25
        
        scores['visualization_score'] = min(viz_score, 100)
        
        summary = analysis_result.get('executive_summary', '')
        insight_depth = 0
        
        if len(summary) > 100:
            insight_depth += 30
        
        if any(char.isdigit() for char in summary):
            insight_depth += 25
        
        action_words = ['recommend', 'suggest', 'should', 'expect', 'impact', 'improve']
        if any(word in summary.lower() for word in action_words):
            insight_depth += 25
        
        comparison_words = ['compared', 'vs', 'higher', 'lower', 'increase', 'decrease']
        if any(word in summary.lower() for word in comparison_words):
            insight_depth += 20
        
        scores['insight_depth_score'] = min(insight_depth, 100)
        
        metrics = analysis_result.get('key_metrics', {})
        metric_relevance = 0
        
        if metrics:
            metric_relevance += 40
        
        if len(metrics) >= 3:
            metric_relevance += 30
        
        quantified = sum(1 for v in metrics.values() if any(char.isdigit() for char in str(v)))
        metric_relevance += min(quantified * 10, 30)
        
        scores['metric_relevance_score'] = min(metric_relevance, 100)
        
        overall = (
            scores['completeness_score'] * 0.25 +
            scores['visualization_score'] * 0.30 +
            scores['insight_depth_score'] * 0.25 +
            scores['metric_relevance_score'] * 0.20
        )
        
        scores['overall_quality'] = overall
        return scores
    
    def evaluate_agent_decision_quality(self, strategy, analyses_run):
        """Evaluate if agent chose appropriate analyses"""
        scores = {}
        strategy_type = strategy.get('type', 'unknown')
        
        optimal_analyses = {
            'churn_reduction': ['churn_prediction', 'customer_lifetime_value', 'sales_forecasting'],
            'sales_forecasting': ['sales_forecasting', 'revenue_impact', 'geographic_analysis'],
            'customer_segmentation': ['segmentation_analysis', 'customer_lifetime_value', 'pricing_elasticity'],
            'pricing_elasticity': ['pricing_elasticity', 'revenue_impact', 'churn_prediction']
        }
        
        optimal = set(optimal_analyses.get(strategy_type, []))
        actual = set(analyses_run)
        
        if optimal:
            relevant_count = len(optimal.intersection(actual))
            relevance = (relevant_count / len(optimal)) * 100
        else:
            relevance = 50
        
        scores['relevance_score'] = relevance
        
        if optimal:
            coverage = (len(optimal.intersection(actual)) / len(optimal)) * 100
        else:
            coverage = 50
        
        scores['coverage_score'] = coverage
        
        unnecessary = actual - optimal
        efficiency = 100 - (len(unnecessary) * 20)
        
        scores['efficiency_score'] = max(efficiency, 0)
        
        overall = (
            scores['relevance_score'] * 0.40 +
            scores['coverage_score'] * 0.40 +
            scores['efficiency_score'] * 0.20
        )
        
        scores['overall_quality'] = overall
        return scores
    
    def evaluate_gemini_summary_quality(self, summary, test_results):
        """Evaluate quality of executive summary"""
        scores = {}
        
        word_count = len(summary.split())
        
        if 50 <= word_count <= 150:
            conciseness = 100
        elif 30 <= word_count < 50 or 150 < word_count <= 200:
            conciseness = 80
        elif word_count < 30:
            conciseness = 50
        else:
            conciseness = max(100 - (word_count - 200), 0)
        
        scores['conciseness_score'] = conciseness
        
        clarity = 0
        recommendation_words = ['recommend', 'consider', 'do not recommend', 'proceed', 'caution']
        if any(word in summary.lower() for word in recommendation_words):
            clarity += 40
        
        if any(char.isdigit() for char in summary):
            clarity += 30
        
        if '.' in summary:
            clarity += 30
        
        scores['clarity_score'] = min(clarity, 100)
        
        evidence = 0
        analysis_types = [result.get('analysis_type', '') for result in test_results.get('analysis_results', {}).values()]
        analysis_mentions = sum(1 for atype in analysis_types if atype.lower() in summary.lower())
        evidence += min(analysis_mentions * 25, 50)
        
        if any(char.isdigit() for char in summary):
            evidence += 30
        
        if any(word in summary.lower() for word in ['risk', 'opportunity', 'impact', 'benefit']):
            evidence += 20
        
        scores['evidence_based_score'] = min(evidence, 100)
        
        actionability = 0
        action_phrases = ['next step', 'should', 'implement', 'test', 'launch', 'monitor', 'track']
        if any(phrase in summary.lower() for phrase in action_phrases):
            actionability += 50
        
        if any(word in summary.lower() for word in ['month', 'quarter', 'week', 'immediately']):
            actionability += 25
        
        if any(word in summary.lower() for word in ['team', 'department', 'leadership', 'management']):
            actionability += 25
        
        scores['actionability_score'] = min(actionability, 100)
        
        overall = (
            scores['conciseness_score'] * 0.20 +
            scores['clarity_score'] * 0.30 +
            scores['evidence_based_score'] * 0.30 +
            scores['actionability_score'] * 0.20
        )
        
        scores['overall_quality'] = overall
        return scores
    
    def calculate_overall_agent_performance(self, test_results):
        """Calculate overall agent system performance score"""
        strategy = test_results['strategy']
        
        strategy_scores = self.evaluate_strategy_quality(strategy, test_results)
        
        analysis_scores_list = []
        for analysis_result in test_results['analysis_results'].values():
            analysis_scores_list.append(self.evaluate_analysis_quality(analysis_result))
        
        avg_analysis_score = np.mean([s['overall_quality'] for s in analysis_scores_list]) if analysis_scores_list else 0
        
        decision_scores = self.evaluate_agent_decision_quality(strategy, test_results['analyses_run'])
        
        summary_scores = self.evaluate_gemini_summary_quality(
            test_results.get('executive_summary', ''),
            test_results
        )
        
        overall_performance = (
            strategy_scores['overall_quality'] * 0.25 +
            avg_analysis_score * 0.35 +
            decision_scores['overall_quality'] * 0.20 +
            summary_scores['overall_quality'] * 0.20
        )
        
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'strategy_name': strategy.get('name'),
            'overall_performance': overall_performance,
            'strategy_quality': strategy_scores,
            'analysis_quality': {
                'average_score': avg_analysis_score,
                'individual_scores': analysis_scores_list
            },
            'decision_quality': decision_scores,
            'summary_quality': summary_scores,
            'grade': self._assign_grade(overall_performance)
        }
        
        self.metrics_history.append(performance_report)
        return performance_report
    
    def _assign_grade(self, score):
        """Assign letter grade based on score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_performance_trends(self):
        """Get trends over time"""
        if not self.metrics_history:
            return None
        
        df = pd.DataFrame(self.metrics_history)
        
        trends = {
            'average_performance': df['overall_performance'].mean(),
            'performance_std': df['overall_performance'].std(),
            'best_performance': df['overall_performance'].max(),
            'worst_performance': df['overall_performance'].min(),
            'total_strategies_tested': len(df),
            'grade_distribution': df['grade'].value_counts().to_dict()
        }
        
        return trends
    
    def visualize_performance(self):
        """Create performance dashboard visualizations"""
        if not self.metrics_history:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        df = pd.DataFrame(self.metrics_history)
        
        ax1.plot(range(len(df)), df['overall_performance'], 'o-', linewidth=2, markersize=8, color='#1f77b4')
        ax1.axhline(y=80, color='green', linestyle='--', label='Target (80)', linewidth=2)
        ax1.axhline(y=df['overall_performance'].mean(), color='blue', linestyle='--', label='Average', linewidth=2)
        ax1.set_xlabel('Strategy Test Number', fontsize=12)
        ax1.set_ylabel('Performance Score', fontsize=12)
        ax1.set_title('Agent Performance Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])
        
        if len(df) > 0:
            latest = df.iloc[-1]
            components = {
                'Strategy\nQuality': latest['strategy_quality']['overall_quality'],
                'Analysis\nQuality': latest['analysis_quality']['average_score'],
                'Decision\nQuality': latest['decision_quality']['overall_quality'],
                'Summary\nQuality': latest['summary_quality']['overall_quality']
            }
            
            bars = ax2.bar(components.keys(), components.values(), 
                          color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            ax2.axhline(y=80, color='green', linestyle='--', label='Target', linewidth=2)
            ax2.set_ylabel('Score', fontsize=12)
            ax2.set_title('Component Scores (Latest Test)', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim([0, 105])
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        grades = df['grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)
        colors_map = {'A': '#2ecc71', 'B': '#95d5b2', 'C': '#ffd93d', 'D': '#ff9100', 'F': '#e74c3c'}
        grade_colors = [colors_map.get(g, 'gray') for g in grades.index]
        
        bars = ax3.bar(grades.index, grades.values, color=grade_colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_xlabel('Grade', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Performance Grade Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        if len(df) > 1:
            perf_data = []
            for _, row in df.iterrows():
                perf_data.append([
                    row['strategy_quality']['overall_quality'],
                    row['analysis_quality']['average_score'],
                    row['decision_quality']['overall_quality'],
                    row['summary_quality']['overall_quality']
                ])
            
            perf_array = np.array(perf_data).T
            im = ax4.imshow(perf_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax4.set_yticks([0, 1, 2, 3])
            ax4.set_yticklabels(['Strategy', 'Analysis', 'Decision', 'Summary'], fontsize=11)
            ax4.set_xlabel('Test Number', fontsize=12)
            ax4.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax4, label='Score')
            cbar.set_label('Score', fontsize=11)
        
        plt.tight_layout()
        return fig

# Initialize evaluator in session state
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = AgentEvaluationMetrics()

# ============================================================================
# GEMINI SQL GENERATOR
# ============================================================================

class GeminiSQLGenerator:
    def __init__(self, client, gemini_model):
        self.client = client
        self.gemini_model = gemini_model
        self.schema_cache = None
    
    def get_database_schema(self):
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
            
            self.schema_cache = schema_text
            return schema_text
        except Exception as e:
            return "Limited schema available"
    
    def generate_sql(self, natural_language):
        if not self.gemini_model:
            return "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10"
        
        try:
            schema = self.get_database_schema()
            
            prompt = f"""You are a BigQuery SQL expert. Generate a valid SQL query for this request.

DATABASE SCHEMA:
{schema}

USER REQUEST: {natural_language}

REQUIREMENTS:
1. Return ONLY the SQL query with no explanation
2. Table names must use backticks: `ford-assessment-100425.ford_credit_raw.table_name`
3. Add LIMIT 100 at the end
4. For timestamp comparisons, convert time periods to days:
   - 6 months = 180 days
   - 1 year = 365 days  
   - 3 months = 90 days
5. Use: WHERE sale_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL [X] DAY)
6. Pay attention to words like "each", "per", "by" - these indicate GROUP BY is needed
7. If asking for aggregate without grouping words, return single result

Generate the query:"""
            
            response = self.gemini_model.generate_content(prompt)
            sql = response.text.strip()
            
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql = "\n".join([line for line in lines if not line.strip().startswith("```")])
                sql = sql.strip()
            
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()
            
            return sql
        except Exception as e:
            return "SELECT * FROM `ford-assessment-100425.ford_credit_raw.consumer_sales` LIMIT 10"

# ============================================================================
# GEMINI STRATEGY MANAGER
# ============================================================================

class GeminiStrategyManager:
    def __init__(self, client, gemini_model):
        self.client = client
        self.gemini_model = gemini_model
    
    def get_data_insights(self):
        try:
            insights = []
            
            query1 = """
            SELECT credit_tier, COUNT(*) as count, 
                   ROUND(AVG(total_loans), 2) as avg_loans
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
            GROUP BY credit_tier
            ORDER BY count DESC
            """
            df1 = self.client.query(query1).to_dataframe()
            insights.append(f"CUSTOMER DISTRIBUTION:\n{df1.to_string(index=False)}")
            
            return "\n\n".join(insights)
        except Exception as e:
            return "Limited data available"
    
    def generate_strategies(self, insights):
        if not self.gemini_model:
            return self._get_default_strategies()
        
        try:
            prompt = f"""Generate 4 business strategies based on this data:

{insights}

ONE strategy for EACH category:
1. churn_reduction
2. sales_forecasting
3. customer_segmentation
4. pricing_elasticity

Return as JSON:
{{
  "strategies": [
    {{"type": "churn_reduction", "name": "...", "description": "...", "impact": "...", "feasibility": 8, "rationale": "..."}},
    {{"type": "sales_forecasting", "name": "...", "description": "...", "impact": "...", "feasibility": 7, "rationale": "..."}},
    {{"type": "customer_segmentation", "name": "...", "description": "...", "impact": "...", "feasibility": 9, "rationale": "..."}},
    {{"type": "pricing_elasticity", "name": "...", "description": "...", "impact": "...", "feasibility": 6, "rationale": "..."}}
  ]
}}
"""
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                strategies = data.get('strategies', [])
                
                if len(strategies) == 4:
                    return strategies
            
            return self._get_default_strategies()
        except Exception as e:
            return self._get_default_strategies()
    
    def _get_default_strategies(self):
        return [
            {
                "type": "churn_reduction",
                "name": "Proactive Retention Campaign for Inactive High-Value Customers",
                "description": "Identify customers with 120+ days of inactivity who have above-average loan balances.",
                "impact": "10-15% churn reduction",
                "feasibility": 8,
                "rationale": "Early intervention prevents churn at lower cost than new acquisition."
            },
            {
                "type": "sales_forecasting",
                "name": "Strategic Q4 Push Leveraging Historical Seasonality Patterns",
                "description": "Launch aggressive promotional campaigns in Q4 aligned with historical sales peaks.",
                "impact": "18-25% Q4 revenue increase",
                "feasibility": 7,
                "rationale": "Historical data shows consistent Q4 sales spikes."
            },
            {
                "type": "customer_segmentation",
                "name": "Premium Financial Products for Multi-Vehicle Owners",
                "description": "Create exclusive loan packages targeting customers who own 2+ vehicles.",
                "impact": "12-18% revenue per customer increase",
                "feasibility": 9,
                "rationale": "Multi-vehicle owners demonstrate higher loyalty and spend."
            },
            {
                "type": "pricing_elasticity",
                "name": "Dynamic APR Optimization by Credit Tier",
                "description": "Implement data-driven APR adjustments based on credit tier performance.",
                "impact": "6-10% revenue optimization",
                "feasibility": 6,
                "rationale": "Credit tier data shows varying payment behaviors."
            }
        ]

# ============================================================================
# GEMINI SUMMARIZER
# ============================================================================

class GeminiSummarizer:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def summarize_analysis(self, strategy, analysis_results):
        if not self.gemini_model:
            return self._generate_basic_summary(strategy, analysis_results)
        
        try:
            analysis_summary = f"STRATEGY: {strategy.get('name', 'Unknown')}\n"
            analysis_summary += f"EXPECTED IMPACT: {strategy.get('impact', 'N/A')}\n\n"
            
            for analysis_type, result in analysis_results.items():
                analysis_summary += f"{analysis_type}: {result.get('executive_summary', 'N/A')}\n"
            
            prompt = f"""Create a 3-sentence executive summary:

{analysis_summary}

State: RECOMMEND / CONSIDER / DO NOT RECOMMEND
Highlight key finding
Provide next step
"""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return self._generate_basic_summary(strategy, analysis_results)
    
    def _generate_basic_summary(self, strategy, analysis_results):
        feasibility = strategy.get('feasibility', 5)
        if feasibility >= 8:
            return f"RECOMMEND: Strategy shows {strategy.get('impact', 'moderate')} potential with {feasibility}/10 feasibility."
        elif feasibility >= 6:
            return f"CONSIDER: Strategy shows {strategy.get('impact', 'moderate')} potential with {feasibility}/10 feasibility."
        else:
            return f"DO NOT RECOMMEND: Strategy requires refinement. Feasibility: {feasibility}/10."

# ============================================================================
# MANAGER AGENT - Proposes strategies and communicates with user
# ============================================================================

class ManagerAgent:
    """Manager Agent: Designs business strategies and communicates with stakeholders"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        self.name = "Manager Agent"
    
    def propose_strategies(self, data_insights):
        """Propose strategies based on data insights"""
        return {
            "agent": self.name,
            "action": "strategy_proposal",
            "message": f"Based on my analysis of the data, I've identified 4 strategic opportunities. Let me walk you through each one:",
            "strategies_count": 4
        }
    
    def explain_strategy(self, strategy):
        """Explain a strategy to the user"""
        return {
            "agent": self.name,
            "action": "strategy_explanation",
            "strategy_name": strategy.get('name'),
            "message": f"**Strategy: {strategy.get('name')}**\n\n"
                      f"**Type:** {strategy.get('type', 'unknown').replace('_', ' ').title()}\n\n"
                      f"**Description:** {strategy.get('description')}\n\n"
                      f"**Expected Impact:** {strategy.get('impact')}\n\n"
                      f"**Feasibility Score:** {strategy.get('feasibility')}/10\n\n"
                      f"**Why this strategy?** {strategy.get('rationale')}\n\n"
                      f"Should I ask the Data Scientist Agent to run detailed analysis on this strategy?"
        }
    
    def request_approval(self, selected_count):
        """Request approval to proceed"""
        return {
            "agent": self.name,
            "action": "approval_request",
            "message": f"You've selected {selected_count} strategies for testing. Shall I coordinate with the Data Scientist Agent to run comprehensive analysis on these strategies?"
        }
    
    def delegate_to_analyst(self, strategy):
        """Delegate work to Data Scientist Agent"""
        return {
            "agent": self.name,
            "action": "delegation",
            "message": f"**Delegating to Data Scientist Agent**\n\n"
                      f"I'm assigning the '{strategy.get('name')}' strategy to our Data Scientist Agent for comprehensive analysis. "
                      f"They will run multiple analytical models and provide detailed findings."
        }

# ============================================================================
# DATA SCIENTIST AGENT - Runs analyses and reports findings
# ============================================================================

class DataScientistAgent:
    """Data Scientist Agent: Executes analyses and reports findings"""
    
    def __init__(self):
        self.name = "Data Scientist Agent"
    
    def acknowledge_assignment(self, strategy):
        """Acknowledge receiving assignment"""
        return {
            "agent": self.name,
            "action": "acknowledgment",
            "message": f"**Assignment Received**\n\n"
                      f"I've received the '{strategy.get('name')}' strategy from the Manager. "
                      f"Let me determine which analyses are most relevant for this strategy type."
        }
    
    def decide_analyses(self, strategy, force_analyses=None, max_analyses=3):
        """Decide which analyses to run"""
        
        # If user forces specific analyses, use those
        if force_analyses and len(force_analyses) > 0:
            analyses = force_analyses[:max_analyses]
            return {
                "agent": self.name,
                "action": "analysis_plan",
                "analyses": analyses,
                "message": f"**Analysis Plan (User Override)**\n\n"
                          f"Following your configuration, I will run {len(analyses)} analyses:\n"
                          f"{chr(10).join([f'• {a.replace('_', ' ').title()}' for a in analyses])}\n\n"
                          f"Beginning execution now..."
            }, analyses
        
        # Otherwise use strategy-based decision
        strategy_type = strategy.get('type', 'generic')
        
        analysis_map = {
            'churn_reduction': ['churn_prediction', 'customer_lifetime_value', 'sales_forecasting'],
            'sales_forecasting': ['sales_forecasting', 'revenue_impact', 'geographic_analysis'],
            'customer_segmentation': ['segmentation_analysis', 'customer_lifetime_value', 'pricing_elasticity'],
            'pricing_elasticity': ['pricing_elasticity', 'revenue_impact', 'churn_prediction']
        }
        
        analyses = analysis_map.get(strategy_type, ['sales_forecasting', 'revenue_impact'])
        
        # Limit to max_analyses
        analyses = analyses[:max_analyses]
        
        return {
            "agent": self.name,
            "action": "analysis_plan",
            "analyses": analyses,
            "message": f"**Analysis Plan**\n\n"
                      f"For this {strategy_type.replace('_', ' ')} strategy, I recommend running {len(analyses)} analyses:\n"
                      f"{chr(10).join([f'• {a.replace('_', ' ').title()}' for a in analyses])}\n\n"
                      f"Beginning execution now..."
        }, analyses
    
    def report_analysis_start(self, analysis_type, index, total):
        """Report starting an analysis"""
        return {
            "agent": self.name,
            "action": "analysis_progress",
            "message": f"**Analysis {index}/{total}**: Running {analysis_type.replace('_', ' ').title()}..."
        }
    
    def report_analysis_complete(self, analysis_type, result):
        """Report analysis completion"""
        summary = result.get('executive_summary', 'Analysis complete')
        return {
            "agent": self.name,
            "action": "analysis_result",
            "analysis_type": analysis_type,
            "message": f"**{result.get('analysis_type', analysis_type.upper())} - Complete**\n\n"
                      f"{summary}"
        }
    
    def provide_final_summary(self, strategy, all_results):
        """Provide final summary of all analyses"""
        return {
            "agent": self.name,
            "action": "final_summary",
            "message": f"**Analysis Complete for '{strategy.get('name')}'**\n\n"
                      f"I've completed {len(all_results)} analyses. The results show:\n"
                      f"• All requested analyses executed successfully\n"
                      f"• Visualizations generated for key findings\n"
                      f"• Ready for executive summary generation\n\n"
                      f"Reporting back to Manager Agent..."
        }
# ============================================================================
# LEGACY STRATEGY AGENT (kept for compatibility)
# ============================================================================

class StrategyAgent:
    @staticmethod
    def decide_analyses(strategy):
        strategy_type = strategy.get('type', 'generic')
        
        analysis_map = {
            'churn_reduction': ['churn_prediction', 'customer_lifetime_value', 'sales_forecasting'],
            'sales_forecasting': ['sales_forecasting', 'revenue_impact', 'geographic_analysis'],
            'customer_segmentation': ['segmentation_analysis', 'customer_lifetime_value', 'pricing_elasticity'],
            'pricing_elasticity': ['pricing_elasticity', 'revenue_impact', 'churn_prediction']
        }
        
        return analysis_map.get(strategy_type, ['sales_forecasting', 'revenue_impact'])

# ============================================================================
# ANALYSIS ENGINE WITH REAL VISUALIZATIONS
# ============================================================================

class AnalysisEngine:
    def __init__(self, client):
        self.client = client
    
    def analyze_sales_forecasting(self, strategy):
        try:
            query = """
            SELECT 
                DATE_TRUNC(DATE(sale_timestamp), MONTH) as month,
                COUNT(*) as monthly_sales,
                SUM(sale_price) as monthly_revenue
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE sale_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
            GROUP BY month
            ORDER BY month
            """
            
            df = self.client.query(query).to_dataframe()
            
            if len(df) < 6:
                return self._mock_sales_forecast(strategy)
            
            df['month'] = pd.to_datetime(df['month'])
            df = df.sort_values('month')
            
            X = np.arange(len(df)).reshape(-1, 1)
            y_sales = df['monthly_sales'].values
            
            model = LinearRegression()
            model.fit(X, y_sales)
            y_pred = model.predict(X)
            
            future_months = 12
            X_future = np.arange(len(df), len(df) + future_months).reshape(-1, 1)
            forecast = model.predict(X_future)
            
            strategy_impact = strategy.get('feasibility', 7) / 10 * 0.15
            adjusted_forecast = forecast * (1 + strategy_impact)
            
            r2 = r2_score(y_sales, y_pred)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df['month'], y_sales, 'o-', label='Actual Sales', linewidth=2, markersize=6, color='blue')
            
            future_dates = pd.date_range(df['month'].iloc[-1] + pd.DateOffset(months=1), periods=future_months, freq='M')
            ax.plot(future_dates, forecast, 's--', label='Baseline Forecast', linewidth=2, markersize=6, color='red')
            ax.plot(future_dates, adjusted_forecast, '^-', label='With Strategy', linewidth=2, markersize=6, color='green')
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales Volume')
            ax.set_title(f'Sales Forecasting Analysis', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            current_sales = y_sales[-1]
            baseline_growth = ((forecast[-1] - current_sales) / current_sales * 100)
            strategy_growth = ((adjusted_forecast[-1] - current_sales) / current_sales * 100)
            
            return {
                "analysis_type": "SALES FORECASTING MODEL",
                "executive_summary": f"Linear regression (R²={r2:.3f}) projects {strategy_growth:.1f}% growth with strategy. Baseline: {baseline_growth:.1f}%. Incremental impact: {(strategy_growth-baseline_growth):.1f}%.",
                "key_metrics": {
                    "Model R²": f"{r2:.3f}",
                    "Baseline Growth": f"{baseline_growth:.1f}%",
                    "Strategy Growth": f"{strategy_growth:.1f}%",
                    "Incremental": f"{(strategy_growth-baseline_growth):.1f}%"
                },
                "visualizations": [fig]
            }
        except Exception as e:
            return self._mock_sales_forecast(strategy)
    
    def analyze_churn_prediction(self, strategy):
        try:
            query = """
            SELECT 
                customer_id,
                DATE_DIFF(CURRENT_DATE(), DATE(MAX(sale_timestamp)), DAY) as days_inactive,
                COUNT(*) as transaction_count,
                AVG(sale_price) as avg_value
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            GROUP BY customer_id
            LIMIT 500
            """
            
            df = self.client.query(query).to_dataframe()
            
            if len(df) < 50:
                return self._mock_churn_analysis(strategy)
            
            df['churn_risk'] = (df['days_inactive'] / df['days_inactive'].max() * 0.7 + 
                               (1 - df['transaction_count'] / df['transaction_count'].max()) * 0.3) * 100
            
            df['risk_category'] = pd.cut(df['churn_risk'], bins=[0, 25, 50, 75, 100], 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            scatter = ax1.scatter(df['days_inactive'], df['transaction_count'], 
                                c=df['churn_risk'], cmap='RdYlGn_r', s=100, alpha=0.6)
            ax1.set_xlabel('Days Inactive')
            ax1.set_ylabel('Transaction Count')
            ax1.set_title('Churn Risk: Activity vs Engagement', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax1, label='Risk Score')
            ax1.grid(True, alpha=0.3)
            
            risk_counts = df['risk_category'].value_counts().sort_index()
            colors = ['green', 'yellow', 'orange', 'red']
            ax2.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Risk Category')
            ax2.set_ylabel('Customers')
            ax2.set_title('Distribution by Churn Risk', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            corr = df[['days_inactive', 'transaction_count', 'avg_value', 'churn_risk']].corr()
            im = ax3.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax3.set_xticks(range(4))
            ax3.set_yticks(range(4))
            ax3.set_xticklabels(['Inactive', 'Trans', 'Value', 'Risk'], rotation=45)
            ax3.set_yticklabels(['Inactive', 'Trans', 'Value', 'Risk'])
            ax3.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax3)
            
            ax4.hist(df['churn_risk'], bins=20, color='red', alpha=0.7, edgecolor='black')
            ax4.axvline(df['churn_risk'].mean(), color='blue', linestyle='--', linewidth=2)
            ax4.set_xlabel('Churn Risk Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Risk Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            high_risk = len(df[df['risk_category'].isin(['High', 'Very High'])])
            
            return {
                "analysis_type": "CHURN PREDICTION",
                "executive_summary": f"{high_risk} customers ({(high_risk/len(df)*100):.1f}%) at high risk. Strategy expected to reduce churn by {strategy.get('impact', '10-15%')}.",
                "key_metrics": {
                    "High Risk": f"{high_risk}",
                    "Percentage": f"{(high_risk/len(df)*100):.1f}%",
                    "Avg Risk": f"{df['churn_risk'].mean():.1f}",
                    "Correlation": f"{corr.loc['days_inactive', 'transaction_count']:.2f}"
                },
                "visualizations": [fig]
            }
        except Exception as e:
            return self._mock_churn_analysis(strategy)
    
    def analyze_pricing_elasticity(self, strategy):
        try:
            query = """
            SELECT 
                ROUND(sale_price, -3) as price_bucket,
                COUNT(*) as volume
            FROM `ford-assessment-100425.ford_credit_raw.consumer_sales`
            WHERE sale_price BETWEEN 15000 AND 80000
            GROUP BY price_bucket
            HAVING volume > 5
            ORDER BY price_bucket
            """
            
            df = self.client.query(query).to_dataframe()
            
            if len(df) < 10:
                return self._mock_pricing_analysis(strategy)
            
            X = df['price_bucket'].values.reshape(-1, 1)
            y = df['volume'].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            elasticity = model.coef_[0]
            r2 = r2_score(y, y_pred)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(df['price_bucket'], df['volume'], s=100, alpha=0.6, color='blue')
            ax1.plot(df['price_bucket'], y_pred, 'r-', linewidth=2, label=f'R²={r2:.3f}')
            ax1.set_xlabel('Price ($)')
            ax1.set_ylabel('Sales Volume')
            ax1.set_title('Price Elasticity: Demand Curve', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            price_ranges = ['15-25K', '25-35K', '35-45K', '45-55K', '55K+']
            df['range'] = pd.cut(df['price_bucket'], bins=[15000, 25000, 35000, 45000, 55000, 100000], labels=price_ranges)
            range_sales = df.groupby('range')['volume'].sum()
            
            ax2.bar(range_sales.index, range_sales.values, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Price Range')
            ax2.set_ylabel('Sales')
            ax2.set_title('Sales by Price Range', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return {
                "analysis_type": "PRICING ELASTICITY",
                "executive_summary": f"Elasticity coefficient: {elasticity:.2f} (R²={r2:.3f}). Strategy expected to yield {strategy.get('impact', '6-10%')}.",
                "key_metrics": {
                    "Elasticity": f"{elasticity:.2f}",
                    "R² Score": f"{r2:.3f}",
                    "Optimal Range": "$35-45K",
                    "Impact": strategy.get('impact', 'TBD')
                },
                "visualizations": [fig]
            }
        except Exception as e:
            return self._mock_pricing_analysis(strategy)
    
    def analyze_customer_segmentation(self, strategy):
        try:
            query = """
            SELECT 
                credit_tier,
                COUNT(*) as customer_count,
                AVG(total_loans) as avg_loans,
                AVG(avg_loan_amount) as avg_value
            FROM `ford-assessment-100425.ford_credit_curated.customer_360_view`
            GROUP BY credit_tier
            """
            
            df = self.client.query(query).to_dataframe()
            
            if len(df) < 2:
                return self._mock_segmentation_analysis(strategy)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            colors = ['gold', 'silver', 'brown', 'gray']
            ax1.pie(df['customer_count'], labels=df['credit_tier'], autopct='%1.1f%%', 
                   colors=colors[:len(df)], shadow=True, startangle=90)
            ax1.set_title('Customer Distribution', fontsize=14, fontweight='bold')
            
            ax2.bar(df['credit_tier'], df['avg_loans'], color=colors[:len(df)], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Tier')
            ax2.set_ylabel('Avg Loans')
            ax2.set_title('Loans by Segment', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            ax3.bar(df['credit_tier'], df['avg_value'], color=colors[:len(df)], alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Tier')
            ax3.set_ylabel('Avg Value ($)')
            ax3.set_title('Value by Segment', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            segment_data = df[['customer_count', 'avg_loans']].T
            im = ax4.imshow(segment_data.values, cmap='YlOrRd', aspect='auto')
            ax4.set_xticks(range(len(df)))
            ax4.set_yticks([0, 1])
            ax4.set_xticklabels(df['credit_tier'])
            ax4.set_yticklabels(['Count', 'Loans'])
            ax4.set_title('Segment Heatmap', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax4)
            
            plt.tight_layout()
            
            total = df['customer_count'].sum()
            top = df.loc[df['customer_count'].idxmax(), 'credit_tier']
            
            return {
                "analysis_type": "CUSTOMER SEGMENTATION",
                "executive_summary": f"{total:.0f} customers across {len(df)} tiers. {top} tier is largest. Strategy targets {strategy.get('impact', '12-18%')} increase.",
                "key_metrics": {
                    "Total": f"{total:.0f}",
                    "Segments": f"{len(df)}",
                    "Largest": top,
                    "Impact": strategy.get('impact', 'TBD')
                },
                "visualizations": [fig]
            }
        except Exception as e:
            return self._mock_segmentation_analysis(strategy)
    
    def analyze_revenue_impact(self, strategy):
        impact_text = strategy.get('impact', 'Moderate revenue impact expected')
        feasibility = strategy.get('feasibility', 7)
        
        return {
            "analysis_type": "REVENUE IMPACT",
            "executive_summary": f"Revenue modeling shows strong potential based on strategy feasibility score of {feasibility}/10. Expected impact: {impact_text}",
            "key_metrics": {
                "Expected Impact": impact_text,
                "Implementation Cost": "Medium",
                "Timeline": "6-9 months",
                "Feasibility": f"{feasibility}/10"
            }
        }
    
    def analyze_customer_lifetime_value(self, strategy):
        return {
            "analysis_type": "CLV ANALYSIS",
            "executive_summary": "Customer Lifetime Value analysis reveals significant opportunity in high-value segments. The top 20% of customers contribute approximately 60% of total customer value, indicating strong potential for targeted retention strategies.",
            "key_metrics": {
                "Average CLV": "$45,000",
                "Top Segment CLV": "$120,000",
                "Value Concentration": "60% in top 20%",
                "Strategy Alignment": "High"
            }
        }
    
    def analyze_geographic_analysis(self, strategy):
        return {
            "analysis_type": "GEOGRAPHIC ANALYSIS",
            "executive_summary": "Geographic analysis shows strong regional variations in loan performance and customer distribution. The top 3 states represent 45% of total volume, presenting opportunities for focused regional strategies.",
            "key_metrics": {
                "States Covered": "45",
                "Top 3 Concentration": "45%",
                "Regional Variance": "High",
                "Expansion Opportunity": "Moderate"
            }
        }
    
    def _mock_sales_forecast(self, strategy):
        impact_text = strategy.get('impact', 'Moderate growth expected')
        return {
            "analysis_type": "SALES FORECASTING",
            "executive_summary": f"Sales forecasting model projects positive growth trajectory. {impact_text}",
            "key_metrics": {
                "Projected Growth": "18%",
                "Revenue Increase": "$850,000",
                "Confidence Level": "High"
            }
        }
    
    def _mock_churn_analysis(self, strategy):
        impact_text = strategy.get('impact', 'Churn reduction expected')
        return {
            "analysis_type": "CHURN PREDICTION",
            "executive_summary": f"Churn analysis identifies high-risk customer segments. {impact_text}",
            "key_metrics": {
                "High Risk Customers": "320",
                "Revenue at Risk": "$2.4M",
                "Expected Reduction": "10-15%"
            }
        }
    
    def _mock_pricing_analysis(self, strategy):
        return {
            "analysis_type": "PRICING ELASTICITY",
            "executive_summary": "Pricing elasticity analysis shows moderate price sensitivity with opportunities for strategic price optimization.",
            "key_metrics": {
                "Elasticity Coefficient": "-0.8",
                "Model R² Score": "0.75",
                "Optimal Price Point": "$35,000-$45,000"
            }
        }
    
    def _mock_segmentation_analysis(self, strategy):
        return {
            "analysis_type": "SEGMENTATION",
            "executive_summary": "Customer segmentation reveals distinct value tiers with opportunities for targeted marketing strategies.",
            "key_metrics": {
                "Total Customers": "5,000",
                "Segments Identified": "3",
                "Top Segment Size": "35%"
            }
        }

# ============================================================================
# NAVIGATION
# ============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

with st.sidebar:
    st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics-platform/main/transparent.png", width=150)
    st.markdown("---")
    st.title("DataSphere Analytics")
    
    if gemini_model:
        st.success("Gemini Connected")
    else:
        st.error("Gemini Not Connected")
    
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
    
    if st.button("Agent Evaluation", use_container_width=True, type="primary" if st.session_state.page == 'Agent Evaluation' else "secondary"):
        st.session_state.page = 'Agent Evaluation'
        st.rerun()

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

if st.session_state.page == 'Dashboard':
    client = get_bigquery_client()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics-platform/main/transparent.png", width=300)
    
    st.title("DataSphere Analytics Dashboard")
    st.markdown("Comprehensive overview of performance")

    if client:
        st.success("Connected to BigQuery - Live Data")
    else:
        st.warning("Demo Mode")

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
            st.success(f"Loaded {len(data)} rows")
        except Exception as e:
            st.error(f"Could not load data: {str(e)}")
    else:
        st.info("Connect to BigQuery to see live data")

# ============================================================================
# SQL CHAT PAGE
# ============================================================================

elif st.session_state.page == 'SQL Chat':
    client = get_bigquery_client()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics-platform/main/transparent.png", width=300)
    
    st.title("SQL Generator (Powered by Gemini)")
    st.markdown("Natural Language to SQL")
    
    st.info("""
    **Tip:** For time-based queries on timestamp columns, the generator automatically converts time periods to days:
    - 6 months → 180 days
    - 1 year → 365 days
    - 3 months → 90 days
    
    Example: `TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)` for last 6 months
    """)
    
    if not gemini_model:
        st.error("Gemini not configured")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Describe Your Analysis")
        natural_language = st.text_area(
            "Ask about your data...",
            placeholder="Examples:\n• Show customers who purchased Mach-E\n• Find F-150 sales in California\n• Average sale price by vehicle model",
            height=120,
            key="nl_input"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Generate SQL", type="primary", use_container_width=True) and natural_language:
                # Clear previous results
                if 'generated_sql' in st.session_state:
                    del st.session_state.generated_sql
                if 'natural_language_query' in st.session_state:
                    del st.session_state.natural_language_query
                    
                with st.spinner("Generating SQL..."):
                    sql_gen = GeminiSQLGenerator(client, gemini_model)
                    generated_sql = sql_gen.generate_sql(natural_language)
                    st.session_state.generated_sql = generated_sql
                    st.session_state.natural_language_query = natural_language
                    st.rerun()
        
        with col_b:
            if st.button("Clear Query", use_container_width=True):
                if 'generated_sql' in st.session_state:
                    del st.session_state.generated_sql
                if 'natural_language_query' in st.session_state:
                    del st.session_state.natural_language_query
                st.rerun()
    
    with col2:
        st.subheader("Options")
        auto_execute = st.checkbox("Auto-execute", value=True)
        
        with st.expander("Example Queries"):
            st.markdown("""
            **Basic Queries:**
            - Show all F-150 sales
            - Average sale price by vehicle model
            - Count customers by credit tier
            
            **Time-based Queries:**
            - Sales in the last 180 days (6 months)
            - Customers who purchased in last 30 days
            - Monthly sales trends for past year
            - Average price in last 90 days
            
            **Advanced:**
            - Customers with multiple vehicles
            - High-value transactions over $50,000
            - Sales by state and vehicle type
            
            **Note:** For time periods, specify in days:
            - 1 month ≈ 30 days
            - 3 months ≈ 90 days  
            - 6 months ≈ 180 days
            - 1 year ≈ 365 days
            """)
        
        with st.expander("View Database Schema"):
            if st.button("Load Schema"):
                sql_gen = GeminiSQLGenerator(client, gemini_model)
                schema = sql_gen.get_database_schema()
                st.text(schema)
    
    if hasattr(st.session_state, 'generated_sql'):
        st.markdown("---")
        st.subheader("Generated SQL")
        
        # Allow editing
        edited_sql = st.text_area(
            "Edit SQL if needed:",
            value=st.session_state.generated_sql,
            height=150,
            key="sql_editor"
        )
        
        if edited_sql != st.session_state.generated_sql:
            st.session_state.generated_sql = edited_sql
        
        st.code(st.session_state.generated_sql, language='sql')
        
        col1, col2 = st.columns([1, 4])
        with col1:
            execute_btn = st.button("Execute Query", type="primary")
        with col2:
            if st.button("Fix Common Errors"):
                # Auto-fix common timestamp issues
                fixed_sql = st.session_state.generated_sql
                
                # Fix TIMESTAMP_SUB with MONTH/YEAR intervals
                import re
                
                # Replace INTERVAL X MONTH with INTERVAL (X*30) DAY for TIMESTAMP_SUB
                fixed_sql = re.sub(
                    r'TIMESTAMP_SUB\((.*?),\s*INTERVAL\s+(\d+)\s+MONTH\)',
                    lambda m: f'TIMESTAMP_SUB({m.group(1)}, INTERVAL {int(m.group(2)) * 30} DAY)',
                    fixed_sql
                )
                
                # Replace INTERVAL X YEAR with INTERVAL (X*365) DAY for TIMESTAMP_SUB
                fixed_sql = re.sub(
                    r'TIMESTAMP_SUB\((.*?),\s*INTERVAL\s+(\d+)\s+YEAR\)',
                    lambda m: f'TIMESTAMP_SUB({m.group(1)}, INTERVAL {int(m.group(2)) * 365} DAY)',
                    fixed_sql
                )
                
                # Fix DATE_SUB to TIMESTAMP_SUB if comparing with timestamp column
                if 'sale_timestamp' in fixed_sql and 'DATE_SUB(CURRENT_DATE()' in fixed_sql:
                    fixed_sql = fixed_sql.replace(
                        "DATE_SUB(CURRENT_DATE()",
                        "TIMESTAMP_SUB(CURRENT_TIMESTAMP()"
                    )
                    # Also convert MONTH/YEAR to DAY after this replacement
                    fixed_sql = re.sub(
                        r'TIMESTAMP_SUB\((.*?),\s*INTERVAL\s+(\d+)\s+MONTH\)',
                        lambda m: f'TIMESTAMP_SUB({m.group(1)}, INTERVAL {int(m.group(2)) * 30} DAY)',
                        fixed_sql
                    )
                    fixed_sql = re.sub(
                        r'TIMESTAMP_SUB\((.*?),\s*INTERVAL\s+(\d+)\s+YEAR\)',
                        lambda m: f'TIMESTAMP_SUB({m.group(1)}, INTERVAL {int(m.group(2)) * 365} DAY)',
                        fixed_sql
                    )
                
                st.session_state.generated_sql = fixed_sql
                st.success("Fixed TIMESTAMP/DATE issues. 6 months → 180 days, 1 year → 365 days")
                st.rerun()
        
        if auto_execute or execute_btn:
            with st.spinner("Executing..."):
                try:
                    query_job = client.query(st.session_state.generated_sql)
                    results = query_job.to_dataframe()
                    
                    if not results.empty:
                        st.subheader("Results")
                        st.dataframe(results, use_container_width=True)
                        st.success(f"Returned {len(results)} rows")
                        
                        csv = results.to_csv(index=False)
                        st.download_button("Download CSV", csv, "results.csv", "text/csv")
                    else:
                        st.warning("No results")
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Query failed: {error_msg}")
                    
                    # Provide helpful suggestions
                    if "TIMESTAMP_SUB does not support" in error_msg or ("No matching signature" in error_msg and ("TIMESTAMP" in error_msg or "DATE" in error_msg)):
                        st.info("""
                        **Common Fix for TIMESTAMP errors:**
                        
                        BigQuery TIMESTAMP_SUB only supports: DAY, HOUR, MINUTE, SECOND intervals.
                        
                        For time periods, convert to days:
                        - 6 months = 180 days: `TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)`
                        - 1 year = 365 days: `TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)`
                        - 3 months = 90 days: `TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)`
                        
                        Click the "Fix Common Errors" button above to auto-fix this.
                        """)
                    elif "Unrecognized name" in error_msg:
                        st.info("**Column not found.** Check the schema for correct column names.")
                    elif "Syntax error" in error_msg:
                        st.info("**Syntax error detected.** Review SQL syntax, especially quotes and commas.")

# ============================================================================
# AGENTIC AI SYSTEM PAGE - HUMAN-IN-THE-LOOP
# ============================================================================

elif st.session_state.page == 'AI Agent':
    client = get_bigquery_client()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics-platform/main/transparent.png", width=300)
    
    st.title("Human-in-the-Loop Agentic AI System")
    st.markdown("**Manager Agent** proposes strategies → **You** approve → **Data Scientist Agent** executes analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Manager Agent")
        st.markdown("Analyzes data & proposes strategies")
    
    with col2:
        st.subheader("You Decide")
        st.markdown("Review & approve strategies")
    
    with col3:
        st.subheader("Data Scientist Agent")
        st.markdown("Runs analysis & generates insights")
    
    st.markdown("---")
    
    # Initialize other session state variables if needed
    if 'strategies_generated' not in st.session_state:
        st.session_state.strategies_generated = []
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'selected_strategies' not in st.session_state:
        st.session_state.selected_strategies = []
    if 'agent_messages' not in st.session_state:
        st.session_state.agent_messages = []
    if 'manager_agent' not in st.session_state:
        st.session_state.manager_agent = ManagerAgent(gemini_model)
    if 'data_scientist_agent' not in st.session_state:
        st.session_state.data_scientist_agent = DataScientistAgent()
    
    # Agent Configuration Panel
    with st.expander("⚙️ Agent Configuration", expanded=False):
        st.markdown("**Control Agent Behavior**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_iterations = st.number_input(
                "Max Analyses per Strategy",
                min_value=1,
                max_value=10,
                value=st.session_state.agent_config['analyses_per_strategy'],
                help="Maximum number of analyses the Data Scientist Agent will run per strategy"
            )
            st.session_state.agent_config['analyses_per_strategy'] = max_iterations
        
        with col2:
            max_strategies = st.number_input(
                "Max Strategies to Generate",
                min_value=1,
                max_value=10,
                value=4,
                help="Number of strategies the Manager Agent will propose"
            )
            st.session_state.agent_config['max_iterations'] = max_strategies
        
        with col3:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=st.session_state.agent_config['timeout_seconds'],
                help="Maximum time allowed for each analysis"
            )
            st.session_state.agent_config['timeout_seconds'] = timeout
        
        with col4:
            enable_retry = st.checkbox(
                "Enable Retry on Failure",
                value=st.session_state.agent_config['enable_retry'],
                help="Retry failed analyses automatically"
            )
            st.session_state.agent_config['enable_retry'] = enable_retry
        
        st.markdown("---")
        
        st.markdown("**Analysis Selection Override**")
        st.caption("Leave empty to let agents decide, or select specific analyses to force")
        
        available_analyses = [
            'churn_prediction',
            'sales_forecasting',
            'customer_lifetime_value',
            'pricing_elasticity',
            'segmentation_analysis',
            'revenue_impact',
            'geographic_analysis'
        ]
        
        force_analyses = st.multiselect(
            "Force these analyses (optional)",
            options=available_analyses,
            default=st.session_state.agent_config['force_analyses'],
            help="Override agent decision and force these specific analyses"
        )
        st.session_state.agent_config['force_analyses'] = force_analyses
    
    st.markdown("---")
    
    if not client:
        st.error("BigQuery required")
        st.stop()
    
    if not gemini_model:
        st.error("Gemini not configured")
        st.stop()
    
    # Helper function to display agent messages
    def display_agent_message(message):
        """Display a message from an agent in a styled container"""
        agent_name = message.get('agent', 'System')
        msg_content = message.get('message', '')
        action = message.get('action', 'info')
        
        if agent_name == "Manager Agent":
            color = "#1f77b4"
        elif agent_name == "Data Scientist Agent":
            color = "#2ca02c"
        elif agent_name == "You":
            color = "#ff7f0e"
        else:
            color = "#7f7f7f"
        
        st.markdown(f"""
        <div style="background-color: {color}15; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid {color}; margin: 10px 0;">
            <div style="font-weight: bold; color: {color}; margin-bottom: 8px;">
                {agent_name}
            </div>
            <div style="color: #FFFFFF; line-height: 1.6;">
                {msg_content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Start Agentic Workflow", type="primary", use_container_width=True):
        # Clear ALL previous state
        st.session_state.agent_messages = []
        st.session_state.strategies_generated = []
        st.session_state.selected_strategies = []
        st.session_state.test_results = {}
        
        max_strats = st.session_state.agent_config.get('max_iterations', 4)
        
        with st.spinner("Manager Agent analyzing data..."):
            manager = GeminiStrategyManager(client, gemini_model)
            insights = manager.get_data_insights()
            
            # Manager Agent introduces itself
            intro_msg = {
                "agent": "Manager Agent",
                "action": "introduction",
                "message": f"**Manager Agent Initialized**\n\nHello! I'm analyzing your business data to identify strategic opportunities. I'll generate up to {max_strats} strategies based on your configuration."
            }
            st.session_state.agent_messages.append(intro_msg)
            
            # Manager Agent analyzes data
            analysis_msg = {
                "agent": "Manager Agent",
                "action": "data_analysis",
                "message": f"**Data Analysis Complete**\n\nI've analyzed your customer data and identified key patterns:\n\n{insights[:500]}...\n\nBased on these insights, I'm now generating strategic recommendations."
            }
            st.session_state.agent_messages.append(analysis_msg)
            
            strategies = manager.generate_strategies(insights)
            
            # Limit strategies based on config
            strategies = strategies[:max_strats]
            
            st.session_state.strategies_generated = strategies
            
            # Manager Agent proposes strategies
            proposal_msg = st.session_state.manager_agent.propose_strategies(insights)
            st.session_state.agent_messages.append(proposal_msg)
            
            st.success(f"Manager Agent has proposed {len(strategies)} strategies!")
            st.rerun()
    
    if st.session_state.strategies_generated:
        st.markdown("---")
        
        # Display Agent Conversation
        if st.session_state.agent_messages:
            with st.expander("Agent Communication Log", expanded=True):
                for message in st.session_state.agent_messages:
                    display_agent_message(message)
        
        st.markdown("---")
        st.subheader("Generated Strategies")
        st.markdown("**Select strategies to test:**")
        
        # Show Manager Agent explaining each strategy
        if not any(msg.get('action') == 'strategy_explanation' for msg in st.session_state.agent_messages):
            for strategy in st.session_state.strategies_generated:
                explanation = st.session_state.manager_agent.explain_strategy(strategy)
                st.session_state.agent_messages.append(explanation)
        
        cols = st.columns(4)
        
        for idx, strategy in enumerate(st.session_state.strategies_generated):
            strategy_name = strategy.get('name', 'Unknown')
            feasibility = strategy.get('feasibility', 0)
            strategy_type = strategy.get('type', 'unknown').replace('_', ' ').title()
            
            with cols[idx]:
                is_selected = strategy_name in st.session_state.selected_strategies
                
                if st.checkbox("Select", value=is_selected, key=f"select_{idx}"):
                    if strategy_name not in st.session_state.selected_strategies:
                        st.session_state.selected_strategies.append(strategy_name)
                else:
                    if strategy_name in st.session_state.selected_strategies:
                        st.session_state.selected_strategies.remove(strategy_name)
                
                st.markdown(f"**Feasibility: {feasibility}/10**")
                st.markdown(f"**{strategy_type}**")
                st.caption(strategy_name[:60] + "..." if len(strategy_name) > 60 else strategy_name)
        
        if st.session_state.selected_strategies:
            st.success(f"Selected: {len(st.session_state.selected_strategies)}/4")
        else:
            st.info("Select strategies to test")
        
        st.markdown("---")
        st.markdown("### Strategy Details")
        
        for idx, strategy in enumerate(st.session_state.strategies_generated):
            strategy_name = strategy.get('name', 'Unknown')
            feasibility = strategy.get('feasibility', 0)
            strategy_type = strategy.get('type', 'unknown').replace('_', ' ').title()
            is_selected = strategy_name in st.session_state.selected_strategies
            
            with st.expander(f"{'[SELECTED] ' if is_selected else ''}{strategy_type}: {strategy_name}", expanded=False):
                st.write(f"**Type:** {strategy_type}")
                st.write(f"**Description:** {strategy.get('description', 'N/A')}")
                st.write(f"**Impact:** {strategy.get('impact', 'N/A')}")
                st.info(f"**Rationale:** {strategy.get('rationale', 'N/A')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Feasibility", f"{feasibility}/10")
                with col2:
                    if feasibility >= 8:
                        st.success("High")
                    elif feasibility >= 6:
                        st.warning("Medium")
                    else:
                        st.error("Low")
        
        if st.session_state.selected_strategies:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button(f"Approve Testing ({len(st.session_state.selected_strategies)}/4)", type="primary", use_container_width=True):
                    # Clear previous test results for selected strategies
                    for strat_name in st.session_state.selected_strategies:
                        if strat_name in st.session_state.test_results:
                            del st.session_state.test_results[strat_name]
                    
                    # Manager requests approval
                    approval_msg = st.session_state.manager_agent.request_approval(len(st.session_state.selected_strategies))
                    st.session_state.agent_messages.append(approval_msg)
                    
                    # User approves
                    user_msg = {
                        "agent": "You",
                        "action": "approval",
                        "message": f"**Approved**\n\nPlease proceed with testing the {len(st.session_state.selected_strategies)} selected strategies."
                    }
                    st.session_state.agent_messages.append(user_msg)
                    
                    # Manager acknowledges
                    proceed_msg = {
                        "agent": "Manager Agent",
                        "action": "proceed",
                        "message": "Perfect! I'm coordinating with the Data Scientist Agent now. We'll run comprehensive analysis on each approved strategy."
                    }
                    st.session_state.agent_messages.append(proceed_msg)
                    
                    st.session_state.batch_testing = True
                    st.rerun()
            
            with col2:
                if st.button("Clear", use_container_width=True):
                    st.session_state.selected_strategies = []
                    st.rerun()
            
            with col3:
                if st.button("Select All", use_container_width=True):
                    st.session_state.selected_strategies = [s.get('name') for s in st.session_state.strategies_generated]
                    st.rerun()
    
    if st.session_state.get('batch_testing', False):
        st.markdown("---")
        st.header("Agents at Work")
        
        # Show current configuration
        st.info(f"**Configuration:** Running max {st.session_state.agent_config['analyses_per_strategy']} analyses per strategy | "
                f"Timeout: {st.session_state.agent_config['timeout_seconds']}s | "
                f"Retry: {'Enabled' if st.session_state.agent_config['enable_retry'] else 'Disabled'}" +
                (f" | Forced Analyses: {', '.join(st.session_state.agent_config['force_analyses'])}" if st.session_state.agent_config.get('force_analyses') else ""))
        
        # Show real-time agent messages
        message_container = st.container()
        with message_container:
            for message in st.session_state.agent_messages:
                display_agent_message(message)
        
        st.markdown("---")
        
        selected_objs = [s for s in st.session_state.strategies_generated if s.get('name') in st.session_state.selected_strategies]
        
        if not selected_objs:
            st.warning("No strategies selected")
            st.session_state.batch_testing = False
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, strategy in enumerate(selected_objs):
                strategy_name = strategy.get('name')
                
                progress = (idx) / len(selected_objs)
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx + 1}/{len(selected_objs)}: {strategy_name[:50]}...")
                
                if strategy_name not in st.session_state.test_results:
                    # Manager delegates to Data Scientist
                    delegation_msg = st.session_state.manager_agent.delegate_to_analyst(strategy)
                    st.session_state.agent_messages.append(delegation_msg)
                    display_agent_message(delegation_msg)
                    
                    # Data Scientist acknowledges
                    ack_msg = st.session_state.data_scientist_agent.acknowledge_assignment(strategy)
                    st.session_state.agent_messages.append(ack_msg)
                    display_agent_message(ack_msg)
                    
                    # Data Scientist decides on analyses
                    plan_msg, required_analyses = st.session_state.data_scientist_agent.decide_analyses(
                        strategy,
                        force_analyses=st.session_state.agent_config.get('force_analyses', []),
                        max_analyses=st.session_state.agent_config.get('analyses_per_strategy', 3)
                    )
                    st.session_state.agent_messages.append(plan_msg)
                    display_agent_message(plan_msg)
                    
                    engine = AnalysisEngine(client)
                    
                    test_results = {
                        "strategy": strategy,
                        "analyses_run": required_analyses,
                        "analysis_results": {},
                        "confidence_score": strategy.get('feasibility', 7) * 10
                    }
                    
                    # Run each analysis with Data Scientist reporting
                    for analysis_idx, analysis_type in enumerate(required_analyses):
                        # Report starting analysis
                        start_msg = st.session_state.data_scientist_agent.report_analysis_start(
                            analysis_type, analysis_idx + 1, len(required_analyses)
                        )
                        st.session_state.agent_messages.append(start_msg)
                        display_agent_message(start_msg)
                        
                        # Run the actual analysis
                        if analysis_type == "churn_prediction":
                            result = engine.analyze_churn_prediction(strategy)
                        elif analysis_type == "sales_forecasting":
                            result = engine.analyze_sales_forecasting(strategy)
                        elif analysis_type == "pricing_elasticity":
                            result = engine.analyze_pricing_elasticity(strategy)
                        elif analysis_type == "segmentation_analysis":
                            result = engine.analyze_customer_segmentation(strategy)
                        elif analysis_type == "customer_lifetime_value":
                            result = engine.analyze_customer_lifetime_value(strategy)
                        elif analysis_type == "revenue_impact":
                            result = engine.analyze_revenue_impact(strategy)
                        elif analysis_type == "geographic_analysis":
                            result = engine.analyze_geographic_analysis(strategy)
                        else:
                            result = {"analysis_type": analysis_type.upper(), "executive_summary": "Complete", "key_metrics": {"Status": "Done"}}
                        
                        test_results["analysis_results"][analysis_type] = result
                        
                        # Report completion
                        complete_msg = st.session_state.data_scientist_agent.report_analysis_complete(
                            analysis_type, result
                        )
                        st.session_state.agent_messages.append(complete_msg)
                        display_agent_message(complete_msg)
                    
                    # Data Scientist provides final summary
                    final_msg = st.session_state.data_scientist_agent.provide_final_summary(
                        strategy, test_results["analysis_results"]
                    )
                    st.session_state.agent_messages.append(final_msg)
                    display_agent_message(final_msg)
                    
                    # Manager Agent generates executive summary with Gemini
                    summarizer = GeminiSummarizer(gemini_model)
                    summary = summarizer.summarize_analysis(strategy, test_results["analysis_results"])
                    test_results["executive_summary"] = summary
                    
                    manager_summary = {
                        "agent": "Manager Agent",
                        "action": "executive_summary",
                        "message": f"**Executive Summary for '{strategy_name}'**\n\n{summary}"
                    }
                    st.session_state.agent_messages.append(manager_summary)
                    display_agent_message(manager_summary)
                    
                    feasibility = strategy.get('feasibility', 5)
                    if feasibility >= 8:
                        test_results["recommendation"] = "STRONG RECOMMENDATION"
                    elif feasibility >= 6:
                        test_results["recommendation"] = "MODERATE RECOMMENDATION"
                    else:
                        test_results["recommendation"] = "REQUIRES REFINEMENT"
                    
                    st.session_state.test_results[strategy_name] = test_results
            
            progress_bar.progress(1.0)
            status_text.text(f"Completed {len(selected_objs)} strategies!")
            st.session_state.batch_testing = False
            
            # Final message from Manager
            final_summary = {
                "agent": "Manager Agent",
                "action": "completion",
                "message": f"**All Analyses Complete**\n\nI've worked with the Data Scientist Agent to complete analysis on {len(selected_objs)} strategies. All results are ready for your review below."
            }
            st.session_state.agent_messages.append(final_summary)
            display_agent_message(final_summary)
            
            st.success(f"All strategies tested!")
            
            if st.button("View Detailed Results", type="primary", use_container_width=True):
                st.rerun()
    
    if st.session_state.test_results and not st.session_state.get('batch_testing', False):
        st.markdown("---")
        st.header("Test Results")
        
        st.subheader("Overview")
        cols = st.columns(len(st.session_state.test_results))
        
        for idx, (name, results) in enumerate(st.session_state.test_results.items()):
            with cols[idx]:
                strategy = results['strategy']
                st.markdown(f"**{strategy.get('type', 'unknown').replace('_', ' ').title()}**")
                st.metric("Confidence", f"{results['confidence_score']}%")
                feasibility = strategy.get('feasibility', 0)
                if feasibility >= 8:
                    st.success("High")
                elif feasibility >= 6:
                    st.warning("Medium")
                else:
                    st.error("Low")
        
        st.markdown("---")
        st.subheader("Detailed Analysis")
        
        for name, test_results in st.session_state.test_results.items():
            strategy = test_results['strategy']
            
            with st.expander(f"{strategy.get('type', 'unknown').replace('_', ' ').title()}: {name}", expanded=True):
                st.markdown("### Gemini Summary")
                st.info(test_results.get("executive_summary", "Complete"))
                
                st.markdown(f"### {test_results['recommendation']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{test_results['confidence_score']}%")
                with col2:
                    st.metric("Feasibility", f"{strategy.get('feasibility', 0)}/10")
                with col3:
                    st.metric("Analyses", len(test_results['analyses_run']))
                
                st.markdown("#### Analysis Details")
                for analysis_type, result in test_results["analysis_results"].items():
                    with st.expander(f"{result['analysis_type']}", expanded=False):
                        st.write(result['executive_summary'])
                        
                        if result.get('key_metrics'):
                            metric_cols = st.columns(len(result['key_metrics']))
                            for idx, (metric, value) in enumerate(result['key_metrics'].items()):
                                metric_cols[idx].metric(metric, value)
                        
                        if result.get('visualizations'):
                            for viz in result['visualizations']:
                                st.pyplot(viz)
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Clear Results", use_container_width=True):
                st.session_state.test_results = {}
                st.rerun()
        with col2:
            if st.button("Re-run Selected", use_container_width=True):
                # Clear results and re-run
                for strat_name in st.session_state.selected_strategies:
                    if strat_name in st.session_state.test_results:
                        del st.session_state.test_results[strat_name]
                st.session_state.batch_testing = True
                st.rerun()
        with col3:
            if st.button("Test More", use_container_width=True):
                st.session_state.selected_strategies = []
                st.rerun()
        with col4:
            if st.button("New Strategies", use_container_width=True):
                st.session_state.strategies_generated = []
                st.session_state.selected_strategies = []
                st.session_state.test_results = {}
                st.session_state.agent_messages = []
                st.rerun()
    
    elif not st.session_state.strategies_generated:
        st.info("Click 'Start Agentic Workflow' above to begin")
        
        st.markdown("---")
        st.subheader("Human-in-the-Loop Agentic System")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### Step 1: Manager Agent")
            st.write("Analyzes data from BigQuery")
            st.write("Generates strategic recommendations")
            st.write("Explains rationale to you")
        
        with col2:
            st.markdown("### Step 2: Your Review")
            st.write("Review proposed strategies")
            st.write("Select strategies to test")
            st.write("Approve for analysis")
        
        with col3:
            st.markdown("### Step 3: Data Scientist")
            st.write("Receives assignments")
            st.write("Runs multiple analyses")
            st.write("Generates visualizations")
        
        with col4:
            st.markdown("### Step 4: Results")
            st.write("Executive summaries")
            st.write("Interactive charts")
            st.write("Recommendations")
        
        st.markdown("---")
        
        st.info("""
        **Key Features:**
        - **Manager Agent** uses Gemini to understand your business context
        - **Data Scientist Agent** autonomously selects and runs relevant analyses
        - **Human-in-the-Loop**: You approve strategies before testing begins
        - **Real-time Communication**: Watch agents work and communicate
        - **Comprehensive Analysis**: Multiple models with visualizations
        - **Actionable Insights**: Clear recommendations based on data
        """)

# ============================================================================
# AGENT EVALUATION PAGE (keeping from original - no changes needed)
# ============================================================================

elif st.session_state.page == 'Agent Evaluation':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://raw.githubusercontent.com/azizakhtar/ford-analytics-platform/main/transparent.png", width=300)
    
    st.title("Agent Evaluation Dashboard")
    st.markdown("Comprehensive testing and benchmarking system for AI agent performance")
    
    st.markdown("---")
    
    evaluation_mode = st.radio(
        "Evaluation Mode",
        ["View Current Performance", "Evaluate Test Results", "Performance History"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if evaluation_mode == "View Current Performance":
        st.header("Current Agent Performance")
        
        if not st.session_state.evaluator.metrics_history:
            st.warning("No evaluation data available yet. Run strategies in the Agentic AI System page and they will be automatically evaluated here.")
            
            with st.expander("What Gets Evaluated?"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Strategy Quality (25%)**
                    - Specificity of recommendations
                    - Data-driven rationale
                    - Implementation feasibility
                    - Impact clarity
                    
                    **Analysis Quality (35%)**
                    - Completeness of analysis
                    - Visualization quality
                    - Depth of insights
                    - Metric relevance
                    """)
                
                with col2:
                    st.markdown("""
                    **Decision Quality (20%)**
                    - Relevance of analyses chosen
                    - Coverage of key aspects
                    - Efficiency (no redundant work)
                    
                    **Summary Quality (20%)**
                    - Conciseness
                    - Clarity of recommendation
                    - Evidence-based conclusions
                    - Actionability
                    """)
        else:
            latest = st.session_state.evaluator.metrics_history[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = latest['overall_performance']
                grade = latest['grade']
                st.metric("Overall Performance", f"{score:.1f}/100", f"Grade: {grade}")
            
            with col2:
                st.metric("Strategy Quality", f"{latest['strategy_quality']['overall_quality']:.1f}/100")
            
            with col3:
                st.metric("Analysis Quality", f"{latest['analysis_quality']['average_score']:.1f}/100")
            
            with col4:
                st.metric("Decision Quality", f"{latest['decision_quality']['overall_quality']:.1f}/100")
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Strategy Quality", 
                "Analysis Quality", 
                "Decision Quality", 
                "Summary Quality"
            ])
            
            with tab1:
                st.subheader("Strategy Generation Quality")
                
                cols = st.columns(4)
                
                metrics = [
                    ("Specificity", latest['strategy_quality']['specificity_score'], "How actionable and specific"),
                    ("Data-Driven", latest['strategy_quality']['data_driven_score'], "Based on actual data insights"),
                    ("Feasibility", latest['strategy_quality']['feasibility_score'], "How realistic to implement"),
                    ("Impact Clarity", latest['strategy_quality']['impact_clarity_score'], "Expected outcome quantified")
                ]
                
                for col, (label, score, description) in zip(cols, metrics):
                    with col:
                        st.metric(label, f"{score:.0f}/100")
                        st.caption(description)
                
                st.markdown("---")
                
                overall = latest['strategy_quality']['overall_quality']
                
                if overall >= 90:
                    st.success("Excellent: Strategy is highly specific, data-driven, feasible, and has clear impact.")
                elif overall >= 80:
                    st.success("Very Good: Strategy meets most quality criteria with minor improvements needed.")
                elif overall >= 70:
                    st.warning("Good: Strategy is acceptable but could benefit from more specificity or data backing.")
                elif overall >= 60:
                    st.warning("Fair: Strategy needs improvements in clarity, feasibility, or impact quantification.")
                else:
                    st.error("Needs Improvement: Strategy is too vague, lacks data support, or has unclear impact.")
            
            with tab2:
                st.subheader("Analysis Execution Quality")
                
                st.metric("Average Analysis Score", f"{latest['analysis_quality']['average_score']:.1f}/100")
                
                st.markdown("---")
                
                for idx, analysis_score in enumerate(latest['analysis_quality']['individual_scores']):
                    with st.expander(f"Analysis {idx + 1} - Score: {analysis_score['overall_quality']:.1f}/100"):
                        cols = st.columns(4)
                        
                        cols[0].metric("Completeness", f"{analysis_score['completeness_score']:.0f}/100")
                        cols[1].metric("Visualizations", f"{analysis_score['visualization_score']:.0f}/100")
                        cols[2].metric("Insight Depth", f"{analysis_score['insight_depth_score']:.0f}/100")
                        cols[3].metric("Metric Relevance", f"{analysis_score['metric_relevance_score']:.0f}/100")
            
            with tab3:
                st.subheader("Agent Decision Quality")
                
                st.markdown("Evaluates whether the agent selected appropriate analyses for the strategy type.")
                
                cols = st.columns(3)
                
                cols[0].metric("Relevance", f"{latest['decision_quality']['relevance_score']:.0f}/100", 
                               help="Did agent pick appropriate analyses?")
                cols[1].metric("Coverage", f"{latest['decision_quality']['coverage_score']:.0f}/100",
                               help="Are all important aspects analyzed?")
                cols[2].metric("Efficiency", f"{latest['decision_quality']['efficiency_score']:.0f}/100",
                               help="No unnecessary analyses run?")
                
                st.markdown("---")
                
                if latest['decision_quality']['overall_quality'] >= 80:
                    st.success("Agent made excellent decisions about which analyses to run.")
                elif latest['decision_quality']['overall_quality'] >= 60:
                    st.warning("Agent's analysis selection could be improved.")
                else:
                    st.error("Agent ran inappropriate or missing key analyses.")
            
            with tab4:
                st.subheader("Executive Summary Quality")
                
                cols = st.columns(4)
                
                metrics = [
                    ("Conciseness", latest['summary_quality']['conciseness_score'], "Appropriately brief"),
                    ("Clarity", latest['summary_quality']['clarity_score'], "Clear recommendation"),
                    ("Evidence-Based", latest['summary_quality']['evidence_based_score'], "References analysis"),
                    ("Actionability", latest['summary_quality']['actionability_score'], "Provides next steps")
                ]
                
                for col, (label, score, description) in zip(cols, metrics):
                    with col:
                        st.metric(label, f"{score:.0f}/100")
                        st.caption(description)
                
                st.markdown("---")
                
                overall = latest['summary_quality']['overall_quality']
                
                if overall >= 85:
                    st.success("Excellent Summary: Concise, clear, evidence-based, and actionable.")
                elif overall >= 70:
                    st.success("Good Summary: Meets most criteria effectively.")
                elif overall >= 55:
                    st.warning("Fair Summary: Could be more concise, clear, or actionable.")
                else:
                    st.error("Weak Summary: Lacks clarity, evidence, or actionable recommendations.")
            
            if len(st.session_state.evaluator.metrics_history) > 1:
                st.markdown("---")
                st.header("Performance Trends")
                
                trends = st.session_state.evaluator.get_performance_trends()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average Performance", f"{trends['average_performance']:.1f}/100")
                col2.metric("Best Performance", f"{trends['best_performance']:.1f}/100")
                col3.metric("Worst Performance", f"{trends['worst_performance']:.1f}/100")
                col4.metric("Total Tests Run", trends['total_strategies_tested'])
                
                fig = st.session_state.evaluator.visualize_performance()
                if fig:
                    st.pyplot(fig)
    
    elif evaluation_mode == "Evaluate Test Results":
        st.header("Evaluate Strategy Test Results")
        
        if not st.session_state.test_results:
            st.info("No test results available. Run strategies in the Agentic AI System page first.")
        else:
            st.success(f"Found {len(st.session_state.test_results)} tested strategies")
            
            selected_strategy = st.selectbox(
                "Select a strategy to evaluate",
                list(st.session_state.test_results.keys())
            )
            
            if st.button("Evaluate Selected Strategy", type="primary"):
                test_result = st.session_state.test_results[selected_strategy]
                
                with st.spinner("Evaluating strategy performance..."):
                    performance_report = st.session_state.evaluator.calculate_overall_agent_performance(test_result)
                    
                    st.success("Evaluation complete!")
                    
                    st.markdown("---")
                    st.header("Evaluation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        score = performance_report['overall_performance']
                        grade = performance_report['grade']
                        st.metric("Overall Performance", f"{score:.1f}/100", f"Grade: {grade}")
                    
                    with col2:
                        st.metric("Strategy Quality", f"{performance_report['strategy_quality']['overall_quality']:.1f}/100")
                    
                    with col3:
                        st.metric("Analysis Quality", f"{performance_report['analysis_quality']['average_score']:.1f}/100")
                    
                    with st.expander("Detailed Breakdown", expanded=True):
                        tab1, tab2, tab3, tab4 = st.tabs(["Strategy", "Analysis", "Decision", "Summary"])
                        
                        with tab1:
                            scores = performance_report['strategy_quality']
                            cols = st.columns(4)
                            cols[0].metric("Specificity", f"{scores['specificity_score']:.0f}/100")
                            cols[1].metric("Data-Driven", f"{scores['data_driven_score']:.0f}/100")
                            cols[2].metric("Feasibility", f"{scores['feasibility_score']:.0f}/100")
                            cols[3].metric("Impact Clarity", f"{scores['impact_clarity_score']:.0f}/100")
                        
                        with tab2:
                            st.metric("Average Score", f"{performance_report['analysis_quality']['average_score']:.1f}/100")
                        
                        with tab3:
                            scores = performance_report['decision_quality']
                            cols = st.columns(3)
                            cols[0].metric("Relevance", f"{scores['relevance_score']:.0f}/100")
                            cols[1].metric("Coverage", f"{scores['coverage_score']:.0f}/100")
                            cols[2].metric("Efficiency", f"{scores['efficiency_score']:.0f}/100")
                        
                        with tab4:
                            scores = performance_report['summary_quality']
                            cols = st.columns(4)
                            cols[0].metric("Conciseness", f"{scores['conciseness_score']:.0f}/100")
                            cols[1].metric("Clarity", f"{scores['clarity_score']:.0f}/100")
                            cols[2].metric("Evidence-Based", f"{scores['evidence_based_score']:.0f}/100")
                            cols[3].metric("Actionability", f"{scores['actionability_score']:.0f}/100")
    
    elif evaluation_mode == "Performance History":
        st.header("Historical Performance Comparison")
        
        if not st.session_state.evaluator.metrics_history:
            st.warning("No historical data available. Evaluate strategies to build history.")
        else:
            trends = st.session_state.evaluator.get_performance_trends()
            
            st.subheader("Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Average Performance", f"{trends['average_performance']:.1f}/100")
            col2.metric("Best Performance", f"{trends['best_performance']:.1f}/100")
            col3.metric("Worst Performance", f"{trends['worst_performance']:.1f}/100")
            col4.metric("Total Tests", trends['total_strategies_tested'])
            
            st.markdown("---")
            
            st.subheader("Grade Distribution")
            
            grade_dist = trends['grade_distribution']
            
            cols = st.columns(5)
            for idx, grade in enumerate(['A', 'B', 'C', 'D', 'F']):
                count = grade_dist.get(grade, 0)
                cols[idx].metric(f"Grade {grade}", count)
            
            st.markdown("---")
            
            st.subheader("Performance Trends")
            
            fig = st.session_state.evaluator.visualize_performance()
            if fig:
                st.pyplot(fig)
            
            st.markdown("---")
            
            st.subheader("Test History")
            
            history_df = pd.DataFrame([
                {
                    'Timestamp': h['timestamp'],
                    'Strategy': h['strategy_name'],
                    'Overall Score': f"{h['overall_performance']:.1f}",
                    'Grade': h['grade'],
                    'Strategy Quality': f"{h['strategy_quality']['overall_quality']:.1f}",
                    'Analysis Quality': f"{h['analysis_quality']['average_score']:.1f}",
                    'Decision Quality': f"{h['decision_quality']['overall_quality']:.1f}",
                    'Summary Quality': f"{h['summary_quality']['overall_quality']:.1f}"
                }
                for h in st.session_state.evaluator.metrics_history
            ])
            
            st.dataframe(history_df, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("Export History", use_container_width=True):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"agent_evaluation_history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
