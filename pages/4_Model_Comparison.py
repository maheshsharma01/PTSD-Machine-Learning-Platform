import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer
from database.database_manager import DatabaseManager

# 1. Page Configuration
st.set_page_config(
    page_title="Model Comparison | Neurolink",
    page_icon="⚖️",
    layout="wide"
)

# 2. Enhanced UI Styling (Glassmorphism & Medical Aesthetic)
def load_enhanced_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        .stApp {
            background: radial-gradient(circle at top right, #0f172a, #020617);
            font-family: 'Inter', sans-serif;
        }
        
        /* High-Tech Card Style */
        .card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        h1, h2, h3, h4 {
            background: linear-gradient(90deg, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }

        /* Metrics Table Polishing */
        [data-testid="stDataFrame"] {
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: rgba(30, 41, 59, 0.5);
            border-radius: 10px;
            color: #94a3b8;
            padding: 0px 20px;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.2) !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

load_enhanced_css()

def main():
    # Header Section
    st.markdown("""
        <div style="margin-bottom: 30px; border-left: 4px solid #facc15; padding-left: 20px;">
            <span style="color: #facc15; font-weight: 800; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;">Performance Analytics</span>
            <h1 style="margin: 0; font-size: 2.5rem;">⚖️ Model Comparison & Analysis</h1>
            <p style="color: #94a3b8; margin-top: 5px;">Benchmarking neural architectures against clinical validation metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize components
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    # Data Availability Check
    if 'model_results' not in st.session_state or not st.session_state.model_results:
        st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; padding: 25px; border-radius: 15px; text-align: center;">
                <h3 style="color: #f59e0b; margin:0;">⚠️ Intelligence Deficit</h3>
                <p style="color: #fbbf24; margin: 10px 0;">No diagnostic models found in the registry. Please complete the Training phase.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Historical Registry
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📚 Historical Neural Registry")
    try:
        historical_results = st.session_state.db_manager.get_model_results()
        if not historical_results.empty:
            st.dataframe(historical_results, use_container_width=True)
        else:
            st.info("No historical benchmarks recorded in the database.")
    except Exception as e:
        st.error(f"Registry Access Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    results = st.session_state.model_results

    # 3. Model Selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔍 Comparative Scope")
    available_models = list(results.keys())
    col_sel, col_met = st.columns(2)
    with col_sel:
        selected_models = st.multiselect("Neural Architectures:",
                                         available_models,
                                         default=available_models)
    with col_met:
        comparison_metric = st.selectbox("Primary Clinical Metric:",
                                         ["Accuracy", "F1-Score", "AUC", "Precision", "Recall"])
    st.markdown('</div>', unsafe_allow_html=True)

    if not selected_models:
        st.warning("Please select at least one architecture for comparison.")
        return

    # Filtered model results
    filtered_results = {name: results[name] for name in selected_models}

    # 4. Comprehensive Comparison Table
    st.header("📊 Competitive Benchmarking")
    comparison_data = []
    for model_name, result in filtered_results.items():
        row = {
            'Model': model_name,
            'Accuracy': float(result.get('accuracy', 0)),
            'Precision': float(result.get('precision', 0)),
            'Recall': float(result.get('recall', 0)),
            'F1-Score': float(result.get('f1_score', 0)),
            'AUC': float(result.get('auc', 0)) if result.get('auc') else 0.0,
            'CV Mean': float(result.get('cv_mean', 0)),
            'CV Std': float(result.get('cv_std', 0))
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(comparison_metric, ascending=False)
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))

    st.dataframe(comparison_df.style.background_gradient(cmap="Blues", subset=[comparison_metric]), 
                 use_container_width=True, hide_index=True)

    # Winner Highlight
    best_model = comparison_df.iloc[0]['Model']
    best_val = comparison_df.iloc[0][comparison_metric]
    st.success(f"🏆 **Top Performing Architecture:** {best_model} | {comparison_metric}: {best_val:.4f}")

    # 5. Visualizations
    st.header("📈 Diagnostic Analytics")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕸️ Radar Chart", "📊 Bar metrics", "🎯 ROC Curve",
        "⚖️ PR Curve", "🔄 CV Stability"
    ])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        fig_radar = go.Figure()
        for m, result in filtered_results.items():
            values = [result.get(metric, 0) for metric in metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=[metric.replace('_', ' ').title() for metric in metrics],
                fill='toself', name=m
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b"),
                       bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        plot_data = []
        for m, result in filtered_results.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                plot_data.append({"Model": m, "Metric": metric.title(), "Score": result.get(metric, 0)})
        fig_bar = px.bar(pd.DataFrame(plot_data), x='Model', y='Score',
                         color='Metric', barmode='group', range_y=[0, 1],
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig_roc = go.Figure()
        for m, result in filtered_results.items():
            if result.get('probabilities') is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(result['actual'], result['probabilities'])
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{m}'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(template="plotly_dark", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig_pr = go.Figure()
        for m, result in filtered_results.items():
            if result.get('probabilities') is not None:
                from sklearn.metrics import precision_recall_curve
                precision, recall, _ = precision_recall_curve(result['actual'], result['probabilities'])
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{m}'))
        fig_pr.update_layout(template="plotly_dark", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cv_data = []
        for m, result in filtered_results.items():
            cv_data.append({"Model": m, "CV Mean": result.get('cv_mean', 0), "CV Std": result.get('cv_std', 0)})
        fig_cv = px.bar(pd.DataFrame(cv_data), x='Model', y='CV Mean', error_y='CV Std', range_y=[0, 1], template="plotly_dark")
        st.plotly_chart(fig_cv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()