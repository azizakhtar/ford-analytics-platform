"""
DATASPHERE ANALYTICS - COMPLETE APP
Single app.py file with all pages including Agent Evaluation Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DataSphere Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PASSWORD PROTECTION
# ============================================================================

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "datasphere2024":  # Change this password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please enter your password to access DataSphere Analytics*")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

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
        
        # 1. Specificity Score
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
        
        # 2. Data-Driven Score
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
        
        # 3. Feasibility Score
        scores['feasibility_score'] = strategy.get('feasibility', 5) * 10
        
        # 4. Impact Clarity
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
        
        # Overall Strategy Quality Score
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
        
        # 1. Completeness
        completeness = 0
        required_fields = ['analysis_type', 'executive_summary', 'key_metrics']
        
        for field in required_fields:
            if field in analysis_result and analysis_result[field]:
                completeness += 33
        
        scores['completeness_score'] = min(completeness, 100)
        
        # 2. Visualization Quality
        viz_score = 0
        
        if 'visualizations' in analysis_result and analysis_result['visualizations']:
            viz_score += 50
            
            if len(analysis_result['visualizations']) > 1:
                viz_score += 25
            
            viz_score += 25
        
        scores['visualization_score'] = min(viz_score, 100)
        
        # 3. Insight Depth
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
        
        # 4. Metric Relevance
        metrics = analysis_result.get('key_metrics', {})
        metric_relevance = 0
        
        if metrics:
            metric_relevance += 40
        
        if len(metrics) >= 3:
            metric_relevance += 30
        
        quantified = sum(1 for v in metrics.values() if any(char.isdigit() for char in str(v)))
        metric_relevance += min(quantified * 10, 30)
        
        scores['metric_relevance_score'] = min(metric_relevance, 100)
        
        # Overall Analysis Quality
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
        
        # 1. Relevance Score
        if optimal:
            relevant_count = len(optimal.intersection(actual))
            relevance = (relevant_count / len(optimal)) * 100
        else:
            relevance = 50
        
        scores['relevance_score'] = relevance
        
        # 2. Coverage Score
        if optimal:
            coverage = (len(optimal.intersection(actual)) / len(optimal)) * 100
        else:
            coverage = 50
        
        scores['coverage_score'] = coverage
        
        # 3. Efficiency Score
        unnecessary = actual - optimal
        efficiency = 100 - (len(unnecessary) * 20)
        
        scores['efficiency_score'] = max(efficiency, 0)
        
        # Overall Decision Quality
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
        
        # 1. Conciseness
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
        
        # 2. Clarity
        clarity = 0
        
        recommendation_words = ['recommend', 'consider', 'do not recommend', 'proceed', 'caution']
        if any(word in summary.lower() for word in recommendation_words):
            clarity += 40
        
        if any(char.isdigit() for char in summary):
            clarity += 30
        
        if '.' in summary:
            clarity += 30
        
        scores['clarity_score'] = min(clarity, 100)
        
        # 3. Evidence-Based
        evidence = 0
        
        analysis_types = [result.get('analysis_type', '') for result in test_results.get('analysis_results', {}).values()]
        analysis_mentions = sum(1 for atype in analysis_types if atype.lower() in summary.lower())
        evidence += min(analysis_mentions * 25, 50)
        
        if any(char.isdigit() for char in summary):
            evidence += 30
        
        if any(word in summary.lower() for word in ['risk', 'opportunity', 'impact', 'benefit']):
            evidence += 20
        
        scores['evidence_based_score'] = min(evidence, 100)
        
        # 4. Actionability
        actionability = 0
        
        action_phrases = ['next step', 'should', 'implement', 'test', 'launch', 'monitor', 'track']
        if any(phrase in summary.lower() for phrase in action_phrases):
            actionability += 50
        
        if any(word in summary.lower() for word in ['month', 'quarter', 'week', 'immediately']):
            actionability += 25
        
        if any(word in summary.lower() for word in ['team', 'department', 'leadership', 'management']):
            actionability += 25
        
        scores['actionability_score'] = min(actionability, 100)
        
        # Overall Summary Quality
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
        
        # 1. Strategy Quality
        strategy_scores = self.evaluate_strategy_quality(strategy, test_results)
        
        # 2. Analysis Quality
        analysis_scores_list = []
        for analysis_result in test_results['analysis_results'].values():
            analysis_scores_list.append(self.evaluate_analysis_quality(analysis_result))
        
        avg_analysis_score = np.mean([s['overall_quality'] for s in analysis_scores_list]) if analysis_scores_list else 0
        
        # 3. Decision Quality
        decision_scores = self.evaluate_agent_decision_quality(strategy, test_results['analyses_run'])
        
        # 4. Summary Quality
        summary_scores = self.evaluate_gemini_summary_quality(
            test_results.get('executive_summary', ''),
            test_results
        )
        
        # Overall System Performance
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
        
        # 1. Overall Performance Trend
        ax1.plot(range(len(df)), df['overall_performance'], 'o-', linewidth=2, markersize=8, color='#1f77b4')
        ax1.axhline(y=80, color='green', linestyle='--', label='Target (80)', linewidth=2)
        ax1.axhline(y=df['overall_performance'].mean(), color='blue', linestyle='--', label='Average', linewidth=2)
        ax1.set_xlabel('Strategy Test Number', fontsize=12)
        ax1.set_ylabel('Performance Score', fontsize=12)
        ax1.set_title('Agent Performance Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # 2. Component Breakdown
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
        
        # 3. Grade Distribution
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
        
        # 4. Performance Heatmap
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

# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

BENCHMARK_STRATEGIES = [
    {
        'name': 'High-Value Customer Retention Program',
        'type': 'churn_reduction',
        'description': 'Target Gold and Silver tier customers with personalized retention offers including 15% loyalty discount and dedicated account manager. Implement within 60 days.',
        'rationale': 'Customer analysis shows that Gold tier customers generate 45% of total revenue but represent only 12% of customer base. Churn analysis indicates 23% higher retention among customers with dedicated support.',
        'impact': 'Expected 18-22% reduction in churn among high-value segments, translating to $2.4M annual revenue protection',
        'feasibility': 8,
        'expected_analyses': ['churn_prediction', 'customer_lifetime_value', 'segmentation_analysis']
    },
    {
        'name': 'Q4 Holiday Sales Acceleration',
        'type': 'sales_forecasting',
        'description': 'Launch targeted marketing campaign for top 3 product categories during Nov-Dec period with 25% discount promotion. Focus on high-conversion customer segments identified through predictive modeling.',
        'rationale': 'Historical sales data shows 340% increase in electronics and home goods during Q4. Customer segmentation reveals untapped opportunity in Silver tier segment with high purchase intent scores.',
        'impact': 'Projected 35-40% increase in Q4 revenue, adding $8.5M in incremental sales',
        'feasibility': 9,
        'expected_analyses': ['sales_forecasting', 'customer_segmentation', 'revenue_impact']
    },
    {
        'name': 'Dynamic Pricing Optimization',
        'type': 'pricing_elasticity',
        'description': 'Implement AI-driven dynamic pricing for top 50 SKUs based on demand elasticity, competitive positioning, and customer tier. Adjust prices weekly.',
        'rationale': 'Pricing analysis reveals significant elasticity variance across product categories. Premium customers show -0.3 price elasticity while budget segment shows -1.8 elasticity.',
        'impact': '12-15% margin improvement without volume loss, estimated $3.2M annual profit increase',
        'feasibility': 6,
        'expected_analyses': ['pricing_elasticity', 'revenue_impact', 'customer_segmentation']
    }
]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = AgentEvaluationMetrics()
    
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def render_home_page():
    """Render home/dashboard page"""
    st.title("🏠 DataSphere Analytics")
    st.markdown("### AI-Powered Business Intelligence Platform")
    
    st.markdown("---")
    
    overall = scores['overall_quality']
    
    if overall >= 85:
        st.success("**Excellent Summary**: Concise, clear, evidence-based, and actionable.")
    elif overall >= 70:
        st.success("**Good Summary**: Meets most criteria effectively.")
    elif overall >= 55:
        st.warning("**Fair Summary**: Could be more concise, clear, or actionable.")
    else:
        st.error("**Weak Summary**: Lacks clarity, evidence, or actionable recommendations.")


def render_benchmark_testing():
    """Run benchmark tests on predefined strategies"""
    st.header("Benchmark Testing")
    
    st.markdown("""
    Run standardized tests against benchmark strategies to evaluate agent performance.
    These tests use predefined strategies with known expected outcomes.
    """)
    
    # Display benchmark strategies
    st.subheader("Available Benchmark Strategies")
    
    for idx, strategy in enumerate(BENCHMARK_STRATEGIES):
        with st.expander(f"{strategy['name']}", expanded=(idx == 0)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Type:** {strategy['type']}")
                st.markdown(f"**Description:** {strategy['description']}")
                st.markdown(f"**Rationale:** {strategy['rationale']}")
                st.markdown(f"**Expected Impact:** {strategy['impact']}")
            
            with col2:
                st.metric("Feasibility", f"{strategy['feasibility']}/10")
                st.markdown("**Expected Analyses:**")
                for analysis in strategy['expected_analyses']:
                    st.markdown(f"- {analysis}")
    
    st.markdown("---")
    
    # Test controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_strategy = st.selectbox(
            "Select Strategy to Test",
            [s['name'] for s in BENCHMARK_STRATEGIES]
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_test = st.button("Run Benchmark Test", type="primary", use_container_width=True)
    
    if run_test:
        st.info("**Note**: This is a demonstration. In the actual app, this would trigger the full agent workflow and evaluate results.")
        
        # Find selected strategy
        strategy = next(s for s in BENCHMARK_STRATEGIES if s['name'] == selected_strategy)
        
        with st.spinner("Running benchmark test..."):
            # Simulate test results
            test_results = simulate_benchmark_test(strategy)
            
            # Store results
            st.session_state.test_results[strategy['name']] = test_results
            
            # Calculate performance
            performance_report = st.session_state.evaluator.calculate_overall_agent_performance(test_results)
            
            st.success("Benchmark test completed!")
            
            # Display results
            st.markdown("---")
            st.header("Test Results")
            
            # Overall score
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = performance_report['overall_performance']
                grade = performance_report['grade']
                st.metric("Overall Performance", f"{score:.1f}/100", f"Grade: {grade}")
            
            with col2:
                st.metric("Strategy Quality", f"{performance_report['strategy_quality']['overall_quality']:.1f}/100")
            
            with col3:
                st.metric("Analysis Quality", f"{performance_report['analysis_quality']['average_score']:.1f}/100")
            
            # Detailed results
            with st.expander("Detailed Results", expanded=True):
                tab1, tab2, tab3, tab4 = st.tabs(["Strategy", "Analysis", "Decision", "Summary"])
                
                with tab1:
                    render_strategy_quality_tab(performance_report['strategy_quality'])
                
                with tab2:
                    render_analysis_quality_tab(performance_report['analysis_quality'])
                
                with tab3:
                    render_decision_quality_tab(performance_report['decision_quality'])
                
                with tab4:
                    render_summary_quality_tab(performance_report['summary_quality'])


def simulate_benchmark_test(strategy):
    """
    Simulate benchmark test results
    In real app, this would call the actual agent system
    """
    
    # Create mock test results based on strategy
    test_results = {
        'strategy': strategy,
        'analyses_run': strategy['expected_analyses'],
        'analysis_results': {},
        'executive_summary': f"Based on comprehensive analysis, I recommend proceeding with {strategy['name']}. "
                           f"Our analysis shows {strategy['impact']}. Implementation should begin within 30 days "
                           f"with close monitoring of key metrics. Risk level is moderate with high upside potential.",
        'timestamp': datetime.now().isoformat()
    }
    
    # Generate mock analysis results
    for analysis_type in strategy['expected_analyses']:
        test_results['analysis_results'][analysis_type] = {
            'analysis_type': analysis_type,
            'executive_summary': f"Analysis of {analysis_type} shows significant opportunity. "
                               f"Key metrics indicate positive trends with 85% confidence level. "
                               f"Recommend immediate action to capitalize on findings.",
            'key_metrics': {
                'Primary Metric': '18.5%',
                'Secondary Metric': '$2.4M',
                'Confidence Level': '85%'
            },
            'visualizations': ['chart1.png', 'chart2.png'],
            'recommendations': ['Implement strategy', 'Monitor metrics', 'Adjust as needed']
        }
    
    return test_results


def render_historical_comparison():
    """Compare performance across historical runs"""
    st.header("Historical Performance Comparison")
    
    if not st.session_state.evaluator.metrics_history:
        st.warning("No historical data available. Run benchmark tests to build history.")
        return
    
    # Performance trends
    trends = st.session_state.evaluator.get_performance_trends()
    
    st.subheader("Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Average Performance", f"{trends['average_performance']:.1f}/100")
    col2.metric("Best Performance", f"{trends['best_performance']:.1f}/100")
    col3.metric("Worst Performance", f"{trends['worst_performance']:.1f}/100")
    col4.metric("Total Tests", trends['total_strategies_tested'])
    
    st.markdown("---")
    
    # Grade distribution
    st.subheader("Grade Distribution")
    
    grade_dist = trends['grade_distribution']
    
    cols = st.columns(5)
    for idx, grade in enumerate(['A', 'B', 'C', 'D', 'F']):
        count = grade_dist.get(grade, 0)
        cols[idx].metric(f"Grade {grade}", count)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("Performance Trends")
    
    fig = st.session_state.evaluator.visualize_performance()
    if fig:
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Detailed history table
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
    
    # Export option
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


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Check password first
    if not check_password():
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("DataSphere Analytics")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Home", "Strategy Generator", "Analytics Dashboard", "Agent Evaluation"],
            key="navigation"
        )
        
        st.markdown("---")
        
        st.markdown("### About")
        st.markdown("""
        DataSphere Analytics is an AI-powered business intelligence platform 
        that combines advanced analytics with intelligent agents to deliver 
        actionable insights.
        """)
        
        st.markdown("---")
        st.caption("Version 1.0.0")
    
    # Render selected page
    if page == "Home":
        render_home_page()
    elif page == "Strategy Generator":
        render_strategy_page()
    elif page == "Analytics Dashboard":
        render_analytics_page()
    elif page == "Agent Evaluation":
        render_evaluation_page()


if __name__ == "__main__":
    main()")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Analytics")
        st.markdown("""
        - Customer Segmentation
        - Sales Forecasting
        - Churn Prediction
        - Revenue Analysis
        """)
    
    with col2:
        st.markdown("### 🤖 AI Agents")
        st.markdown("""
        - Strategy Generator
        - Data Analyst
        - Executive Summarizer
        - Performance Evaluator
        """)
    
    with col3:
        st.markdown("### 📈 Insights")
        st.markdown("""
        - Real-time Dashboards
        - Predictive Models
        - Actionable Recommendations
        - Performance Metrics
        """)
    
    st.markdown("---")
    
    st.info("👈 **Select a page from the sidebar to get started!**")


def render_strategy_page():
    """Render strategy generation page"""
    st.title("💡 Strategy Generator")
    st.markdown("Generate data-driven business strategies")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Churn Reduction", "Sales Growth", "Customer Segmentation", "Pricing Optimization"]
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🚀 Generate Strategy", type="primary", use_container_width=True)
    
    if generate_btn:
        with st.spinner("Generating strategy..."):
            st.success("✅ Strategy generated successfully!")
            
            st.markdown("### Generated Strategy")
            st.markdown("**Strategy Name:** High-Value Customer Retention Program")
            st.markdown("**Description:** Implement targeted retention campaigns for high-value customer segments")
            st.markdown("**Expected Impact:** 15-20% reduction in churn rate")
            st.markdown("**Feasibility:** 8/10")


def render_analytics_page():
    """Render analytics page"""
    st.title("📊 Analytics Dashboard")
    st.markdown("Comprehensive data analysis and insights")
    
    st.markdown("---")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Revenue", "$2.4M", "+12%")
    col2.metric("Active Customers", "1,234", "+5%")
    col3.metric("Churn Rate", "8.2%", "-2.1%")
    col4.metric("Avg Order Value", "$195", "+8%")
    
    st.markdown("---")
    
    # Sample chart
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Revenue', 'Customers', 'Orders']
    )
    
    st.line_chart(chart_data)


def render_evaluation_page():
    """Render agent evaluation dashboard page"""
    st.title("📊 Agent Evaluation Dashboard")
    st.markdown("**Comprehensive testing and benchmarking system for AI agent performance**")
    
    st.markdown("---")
    
    # Evaluation mode selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        evaluation_mode = st.radio(
            "Evaluation Mode",
            ["View Current Performance", "Run Benchmark Tests", "Compare Historical Runs"],
            horizontal=True
        )
    
    st.markdown("---")
    
    if evaluation_mode == "View Current Performance":
        render_current_performance()
    elif evaluation_mode == "Run Benchmark Tests":
        render_benchmark_testing()
    else:
        render_historical_comparison()


def render_current_performance():
    """Show current performance metrics"""
    st.header("Current Agent Performance")
    
    if not st.session_state.evaluator.metrics_history:
        st.warning("⚠️ No evaluation data available yet. Run benchmark tests to generate performance metrics.")
        
        with st.expander("📚 What Gets Evaluated?"):
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
        return
    
    # Get latest performance report
    latest = st.session_state.evaluator.metrics_history[-1]
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = latest['overall_performance']
        grade = latest['grade']
        
        if grade == 'A':
            color = '🟢'
        elif grade == 'B':
            color = '🟡'
        else:
            color = '🔴'
        
        st.metric("Overall Performance", f"{score:.1f}/100", f"{color} Grade: {grade}")
    
    with col2:
        st.metric("Strategy Quality", f"{latest['strategy_quality']['overall_quality']:.1f}/100")
    
    with col3:
        st.metric("Analysis Quality", f"{latest['analysis_quality']['average_score']:.1f}/100")
    
    with col4:
        st.metric("Decision Quality", f"{latest['decision_quality']['overall_quality']:.1f}/100")
    
    st.markdown("---")
    
    # Detailed breakdown tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Strategy Quality", 
        "📊 Analysis Quality", 
        "🎯 Decision Quality", 
        "📄 Summary Quality"
    ])
    
    with tab1:
        render_strategy_quality_tab(latest['strategy_quality'])
    
    with tab2:
        render_analysis_quality_tab(latest['analysis_quality'])
    
    with tab3:
        render_decision_quality_tab(latest['decision_quality'])
    
    with tab4:
        render_summary_quality_tab(latest['summary_quality'])
    
    # Performance trends
    if len(st.session_state.evaluator.metrics_history) > 1:
        st.markdown("---")
        st.header("📈 Performance Trends")
        
        trends = st.session_state.evaluator.get_performance_trends()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Performance", f"{trends['average_performance']:.1f}/100")
        col2.metric("Best Performance", f"{trends['best_performance']:.1f}/100")
        col3.metric("Worst Performance", f"{trends['worst_performance']:.1f}/100")
        col4.metric("Total Tests Run", trends['total_strategies_tested'])
        
        fig = st.session_state.evaluator.visualize_performance()
        if fig:
            st.pyplot(fig)


def render_strategy_quality_tab(scores):
    """Render strategy quality metrics"""
    st.subheader("Strategy Generation Quality")
    
    cols = st.columns(4)
    
    metrics = [
        ("Specificity", scores['specificity_score'], "How actionable and specific"),
        ("Data-Driven", scores['data_driven_score'], "Based on actual data insights"),
        ("Feasibility", scores['feasibility_score'], "How realistic to implement"),
        ("Impact Clarity", scores['impact_clarity_score'], "Expected outcome quantified")
    ]
    
    for col, (label, score, description) in zip(cols, metrics):
        with col:
            st.metric(label, f"{score:.0f}/100")
            st.caption(description)
    
    st.markdown("---")
    st.markdown("**Score Interpretation:**")
    
    overall = scores['overall_quality']
    
    if overall >= 90:
        st.success("🌟 **Excellent**: Strategy is highly specific, data-driven, feasible, and has clear impact.")
    elif overall >= 80:
        st.success("✅ **Very Good**: Strategy meets most quality criteria with minor improvements needed.")
    elif overall >= 70:
        st.warning("⚠️ **Good**: Strategy is acceptable but could benefit from more specificity or data backing.")
    elif overall >= 60:
        st.warning("⚠️ **Fair**: Strategy needs significant improvements in clarity, feasibility, or impact quantification.")
    else:
        st.error("❌ **Needs Improvement**: Strategy is too vague, lacks data support, or has unclear impact.")


def render_analysis_quality_tab(quality):
    """Render analysis quality metrics"""
    st.subheader("Analysis Execution Quality")
    
    st.metric("Average Analysis Score", f"{quality['average_score']:.1f}/100")
    
    st.markdown("---")
    
    # Individual analysis scores
    for idx, analysis_score in enumerate(quality['individual_scores']):
        with st.expander(f"Analysis {idx + 1} - Score: {analysis_score['overall_quality']:.1f}/100"):
            cols = st.columns(4)
            
            cols[0].metric("Completeness", f"{analysis_score['completeness_score']:.0f}/100")
            cols[1].metric("Visualizations", f"{analysis_score['visualization_score']:.0f}/100")
            cols[2].metric("Insight Depth", f"{analysis_score['insight_depth_score']:.0f}/100")
            cols[3].metric("Metric Relevance", f"{analysis_score['metric_relevance_score']:.0f}/100")
    
    st.markdown("---")
    st.info("""
    **What This Measures:**
    - **Completeness**: Has all required components (summary, metrics, visualizations)
    - **Visualizations**: Quality and informativeness of charts
    - **Insight Depth**: Quality and actionability of findings
    - **Metric Relevance**: Are metrics meaningful and quantified?
    """)


def render_decision_quality_tab(scores):
    """Render decision quality metrics"""
    st.subheader("Agent Decision Quality")
    
    st.markdown("Evaluates whether the agent selected appropriate analyses for the strategy type.")
    
    cols = st.columns(3)
    
    cols[0].metric("Relevance", f"{scores['relevance_score']:.0f}/100", 
                   help="Did agent pick appropriate analyses?")
    cols[1].metric("Coverage", f"{scores['coverage_score']:.0f}/100",
                   help="Are all important aspects analyzed?")
    cols[2].metric("Efficiency", f"{scores['efficiency_score']:.0f}/100",
                   help="No unnecessary analyses run?")
    
    st.markdown("---")
    
    if scores['overall_quality'] >= 80:
        st.success("✅ Agent made excellent decisions about which analyses to run.")
    elif scores['overall_quality'] >= 60:
        st.warning("⚠️ Agent's analysis selection could be improved.")
    else:
        st.error("❌ Agent ran inappropriate or missing key analyses.")


def render_summary_quality_tab(scores):
    """Render summary quality metrics"""
    st.subheader("Executive Summary Quality")
    
    cols = st.columns(4)
    
    metrics = [
        ("Conciseness", scores['conciseness_score'], "Appropriately brief"),
        ("Clarity", scores['clarity_score'], "Clear recommendation"),
        ("Evidence-Based", scores['evidence_based_score'], "References analysis"),
        ("Actionability", scores['actionability_score'], "Provides next steps")
    ]
    
    for col, (label, score, description) in zip(cols, metrics):
        with col:
            st.metric(label, f"{score:.0f}/100")
            st.caption(description)
    
    st.markdown("
