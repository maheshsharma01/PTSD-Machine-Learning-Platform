import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer
from database.database_manager import DatabaseManager

# Configure page
st.set_page_config(
    page_title="Model Comparison - PTSD ML Platform",
    page_icon="⚖️",
    layout="wide"
)

def main():
    st.title("⚖️ Model Comparison & Analysis")
    st.markdown("Compare different machine learning models and analyze their performance for PTSD prediction.")

    # Initialize components in session state
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    # No trained models check
    if 'model_results' not in st.session_state or not st.session_state.model_results:
        st.warning("⚠️ No trained models available. Please train models first in the Model Training page.")
        return

    # Historical results from DB
    st.subheader("📚 Historical Model Results")
    try:
        historical_results = st.session_state.db_manager.get_model_results()
        if not historical_results.empty:
            st.dataframe(historical_results, use_container_width=True)
        else:
            st.info("No historical model results found in database.")
    except Exception as e:
        st.error(f"Error loading historical results: {e}")

    results = st.session_state.model_results

    # Model selection for comparison
    st.header("🔍 Model Selection")
    available_models = list(results.keys())
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect("Select models to compare:",
                                         available_models,
                                         default=available_models,
                                         help="Choose which models to include in the comparison")
    with col2:
        comparison_metric = st.selectbox("Primary comparison metric:",
                                         ["Accuracy", "F1-Score", "AUC", "Precision", "Recall"],
                                         help="Metric to use for ranking models")

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    # Filtered model results
    filtered_results = {name: results[name] for name in selected_models}

    # Build comparison table
    st.header("📊 Comprehensive Model Comparison")
    comparison_data = []
    for model_name, result in filtered_results.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{result.get('accuracy', 0):.4f}",
            'Precision': f"{result.get('precision', 0):.4f}",
            'Recall': f"{result.get('recall', 0):.4f}",
            'F1-Score': f"{result.get('f1_score', 0):.4f}",
            'AUC': f"{result.get('auc', 0):.4f}" if result.get('auc') else 'N/A',
            'CV Mean': f"{result.get('cv_mean', 0):.4f}",
            'CV Std': f"{result.get('cv_std', 0):.4f}",
            'Training Time': 'N/A'
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by selected metric
    sort_column = comparison_metric
    if sort_column in comparison_df.columns:
        comparison_df['sort_val'] = pd.to_numeric(comparison_df[sort_column], errors='coerce')
        comparison_df = comparison_df.sort_values('sort_val', ascending=False).drop('sort_val', axis=1)

    # Ranking
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Best model
    best_model = comparison_df.iloc[0]['Model']
    best_metric_value = comparison_df.iloc[0][sort_column]
    st.success(f"🏆 **Best Model by {comparison_metric}:** {best_model} ({best_metric_value})")

    # ===== Performance Visualizations =====
    st.header("📈 Performance Visualizations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics Radar", "📈 Performance Trends", "🎯 ROC Analysis",
        "⚖️ Precision-Recall", "🔄 Cross-Validation"
    ])

    with tab1:
        st.subheader("Multi-Metric Radar Chart")
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if any(res.get('auc') for res in filtered_results.values()):
            metrics.append('auc')
        fig_radar = go.Figure()
        for m, result in filtered_results.items():
            values = [result.get(metric, 0) for metric in metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=[metric.replace('_', ' ').title() for metric in metrics],
                fill='toself',
                name=m
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader("Performance Metrics Comparison")
        plot_data = []
        for m, result in filtered_results.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                plot_data.append({"Model": m, "Metric": metric.title(), "Score": result.get(metric, 0)})
        fig_bar = px.bar(pd.DataFrame(plot_data), x='Model', y='Score',
                         color='Metric', barmode='group', range_y=[0, 1])
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("ROC Curves Comparison")
        fig_roc = go.Figure()
        for m, result in filtered_results.items():
            if result.get('probabilities') is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(result['actual'], result['probabilities'])
                auc_score = result.get('auc', 0)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{m} (AUC = {auc_score:.3f})'
                ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     line=dict(dash='dash', color='gray')))
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab4:
        st.subheader("Precision-Recall Curves")
        fig_pr = go.Figure()
        for m, result in filtered_results.items():
            if result.get('probabilities') is not None:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision, recall, _ = precision_recall_curve(result['actual'], result['probabilities'])
                ap = average_precision_score(result['actual'], result['probabilities'])
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                            name=f'{m} (AP = {ap:.3f})'))
        st.plotly_chart(fig_pr, use_container_width=True)

    with tab5:
        st.subheader("Cross-Validation Performance")
        cv_data = []
        for m, result in filtered_results.items():
            cv_data.append({
                "Model": m,
                "CV Mean": result.get('cv_mean', 0),
                "CV Std": result.get('cv_std', 0)
            })
        fig_cv = px.bar(pd.DataFrame(cv_data), x='Model', y='CV Mean',
                        error_y='CV Std', range_y=[0, 1])
        fig_cv.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cv, use_container_width=True)

if __name__ == "__main__":
    main()
