"""
Visualization Utilities for PTSD ML Platform
Provides comprehensive visualization for data analysis and model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PTSDVisualizer:
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
        """
        Create interactive confusion matrix plot
        """
        if labels is None:
            labels = ['No PTSD', 'PTSD']
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=[[f'{cm[i][j]}<br>({cm_norm[i][j]:.2%})' for j in range(len(labels))] for i in range(len(labels))],
            texttemplate='%{text}',
            textfont={'size': 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=400
        )
        
        return fig
    
    def plot_roc_curves(self, roc_data: Dict[str, Dict]) -> go.Figure:
        """
        Plot ROC curves for multiple models
        """
        fig = go.Figure()
        
        # Plot diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=True
        ))
        
        for model_name, data in roc_data.items():
            if 'roc_curve' in data:
                fpr = data['roc_curve']['fpr']
                tpr = data['roc_curve']['tpr']
                auc = data.get('auc_roc', 0)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc:.3f})',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=500
        )
        
        return fig
    
    def plot_precision_recall_curves(self, pr_data: Dict[str, Dict]) -> go.Figure:
        """
        Plot Precision-Recall curves for multiple models
        """
        fig = go.Figure()
        
        for model_name, data in pr_data.items():
            if 'pr_curve' in data:
                precision = data['pr_curve']['precision']
                recall = data['pr_curve']['recall']
                auc_pr = data.get('auc_pr', 0)
                
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{model_name} (AUC-PR = {auc_pr:.3f})',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=500
        )
        
        return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Create model comparison radar chart
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity']
        
        fig = go.Figure()
        
        for idx, row in comparison_df.iterrows():
            model_name = row['Model']
            values = [row[metric] for metric in metrics if metric in row.index]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison",
            width=600,
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, 
                              importance_scores: np.ndarray, 
                              feature_names: List[str], 
                              top_n: int = 15) -> go.Figure:
        """
        Plot feature importance
        """
        # Get top N features
        indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        fig = go.Figure(data=go.Bar(
            x=top_scores,
            y=top_features,
            orientation='h',
            marker_color=self.colors['primary']
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importances',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=500,
            width=700
        )
        
        return fig
    
    def plot_data_distribution(self, df: pd.DataFrame, columns: List[str]) -> go.Figure:
        """
        Plot distribution of selected columns
        """
        n_cols = len(columns)
        
        if n_cols == 1:
            fig = px.histogram(df, x=columns[0], title=f'Distribution of {columns[0]}')
        else:
            # Create subplots
            rows = (n_cols + 1) // 2
            fig = make_subplots(
                rows=rows, cols=2,
                subplot_titles=columns,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(columns):
                row = (i // 2) + 1
                col_pos = (i % 2) + 1
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
        
        fig.update_layout(
            title='Data Distributions',
            height=300 * rows,
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                numeric_columns: List[str]) -> go.Figure:
        """
        Create correlation heatmap
        """
        corr_matrix = df[numeric_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={'size': 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=700,
            height=600
        )
        
        return fig
    
    def plot_pcl5_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Create PCL-5 subscale analysis
        """
        pcl5_subscales = ['pcl5_intrusive', 'pcl5_avoidance', 'pcl5_negative_mood', 'pcl5_hyperarousal']
        available_subscales = [col for col in pcl5_subscales if col in df.columns]
        
        if not available_subscales:
            return go.Figure().add_annotation(text="No PCL-5 data available", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create box plots for each subscale
        fig = go.Figure()
        
        for subscale in available_subscales:
            fig.add_trace(go.Box(
                y=df[subscale],
                name=subscale.replace('pcl5_', '').title(),
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title='PCL-5 Subscale Score Distribution',
            yaxis_title='Score',
            xaxis_title='Subscale',
            height=500,
            width=700
        )
        
        return fig
    
    def plot_class_distribution(self, y: np.ndarray, labels: List[str] = None) -> go.Figure:
        """
        Plot class distribution
        """
        if labels is None:
            labels = ['No PTSD', 'PTSD']
        
        unique, counts = np.unique(y, return_counts=True)
        percentages = counts / len(y) * 100
        
        fig = go.Figure(data=go.Bar(
            x=[labels[i] for i in unique],
            y=counts,
            text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(counts, percentages)],
            textposition='auto',
            marker_color=[self.colors['success'], self.colors['danger']]
        ))
        
        fig.update_layout(
            title='Class Distribution',
            xaxis_title='Class',
            yaxis_title='Count',
            height=400,
            width=500
        )
        
        return fig
    
    def plot_learning_curves(self, train_scores: np.ndarray, 
                           val_scores: np.ndarray, 
                           train_sizes: np.ndarray) -> go.Figure:
        """
        Plot learning curves
        """
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            error_y=dict(array=np.std(train_scores, axis=1)),
            line=dict(color=self.colors['primary'])
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            error_y=dict(array=np.std(val_scores, axis=1)),
            line=dict(color=self.colors['secondary'])
        ))
        
        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            height=500,
            width=700
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions: np.ndarray, 
                                 probabilities: np.ndarray) -> go.Figure:
        """
        Plot prediction confidence distribution
        """
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=['Prediction Distribution', 'Confidence Distribution'])
        
        # Prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        fig.add_trace(go.Bar(
            x=['No PTSD', 'PTSD'],
            y=counts,
            name='Predictions',
            showlegend=False
        ), row=1, col=1)
        
        # Confidence distribution
        fig.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=20,
            name='Confidence',
            showlegend=False
        ), row=1, col=2)
        
        fig.update_layout(
            title='Prediction Analysis',
            height=400,
            width=800
        )
        
        return fig
    
    def create_clinical_dashboard(self, model_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create comprehensive clinical dashboard
        """
        dashboard = {}
        
        # Model performance overview
        metrics_data = {
            'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV'],
            'Score': [
                model_results.get('accuracy', 0),
                model_results.get('sensitivity', 0),
                model_results.get('specificity', 0),
                model_results.get('ppv', 0),
                model_results.get('npv', 0)
            ]
        }
        
        dashboard['performance_overview'] = go.Figure(data=go.Bar(
            x=metrics_data['Metric'],
            y=metrics_data['Score'],
            marker_color=[self.colors['primary'] if score >= 0.8 else self.colors['warning'] 
                         for score in metrics_data['Score']]
        ))
        
        dashboard['performance_overview'].update_layout(
            title='Clinical Performance Metrics',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        # Confusion matrix
        if 'confusion_matrix' in model_results:
            dashboard['confusion_matrix'] = self.plot_confusion_matrix(
                model_results['confusion_matrix']
            )
        
        return dashboard

# Global visualizer instance
visualizer = PTSDVisualizer()