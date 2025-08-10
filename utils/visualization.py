"""
Visualization Utilities for PTSD ML Platform
Provides comprehensive visualization for data analysis and model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization utilities for PTSD ML platform
    """
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf'
        }

    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str] = None) -> go.Figure:
        if labels is None:
            labels = ['No PTSD', 'PTSD']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=[[f'{cm[i][j]} ({cm_norm[i][j]:.2%})'
                   for j in range(len(labels))] for i in range(len(labels))],
            texttemplate='%{text}',
            textfont={'size': 12},
            hoverongaps=False
        ))
        fig.update_layout(title='Confusion Matrix',
                          xaxis_title='Predicted',
                          yaxis_title='Actual',
                          width=500,
                          height=400)
        return fig

    def plot_roc_curves(self, roc_data: Dict[str, Dict]) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                 mode='lines',
                                 line=dict(dash='dash', color='gray'),
                                 name='Random Classifier'))
        for model_name, data in roc_data.items():
            if 'roc_curve' in data:
                fpr, tpr = data['roc_curve']['fpr'], data['roc_curve']['tpr']
                auc = data.get('auc_roc', 0)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{model_name} (AUC = {auc:.3f})',
                    line=dict(width=2)
                ))
        fig.update_layout(title='ROC Curves Comparison',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          width=600,
                          height=500)
        return fig

    def plot_precision_recall_curves(self, pr_data: Dict[str, Dict]) -> go.Figure:
        fig = go.Figure()
        for model_name, data in pr_data.items():
            if 'pr_curve' in data:
                precision, recall = data['pr_curve']['precision'], data['pr_curve']['recall']
                auc_pr = data.get('auc_pr', 0)
                fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                         name=f'{model_name} (AUC-PR = {auc_pr:.3f})'))
        fig.update_layout(title='Precision-Recall Curves Comparison',
                          xaxis_title='Recall',
                          yaxis_title='Precision',
                          width=600,
                          height=500)
        return fig

    def plot_feature_importance(self, importance_scores: np.ndarray,
                                feature_names: List[str], top_n: int = 15) -> go.Figure:
        indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        fig = go.Figure(data=go.Bar(
            x=top_scores,
            y=top_features,
            orientation='h',
            marker_color=self.colors['primary']
        ))
        fig.update_layout(title=f'Top {top_n} Feature Importances',
                          xaxis_title='Importance Score',
                          yaxis_title='Features',
                          height=500,
                          width=700)
        return fig

    # ... (you can re-add other methods following the same indentation/syntax style) ...


# Make an instance available globally
visualizer = Visualizer()
