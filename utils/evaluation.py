"""
Model Evaluation Module for PTSD ML Platform
Provides comprehensive evaluation metrics and visualization
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report, average_precision_score)
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
from datetime import datetime


class ModelEvaluator:
    """Comprehensive model evaluation with clinical metrics"""
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_data = pd.DataFrame()

    def evaluate_model(self, model_name: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results = {
            'model_name': model_name,
            'evaluation_date': datetime.utcnow(),
            'sample_size': len(y_true)
        }
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm

        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            results['lr_positive'] = results['sensitivity'] / (1 - results['specificity']) if results['specificity'] < 1 else float('inf')
            results['lr_negative'] = (1 - results['sensitivity']) / results['specificity'] if results['specificity'] > 0 else float('inf')

        if y_pred_proba is not None:
            try:
                results['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                results['auc_pr'] = average_precision_score(y_true, y_pred_proba)
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                results['pr_curve'] = {'precision': precision_curve, 'recall': recall_curve}
            except Exception as e:
                st.warning(f"Could not calculate probability-based metrics: {str(e)}")

        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        self.evaluation_results[model_name] = results
        return results

    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        rows = []
        for model_name, results in models_results.items():
            rows.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'Sensitivity': results.get('sensitivity', 0),
                'Specificity': results.get('specificity', 0),
                'PPV': results.get('ppv', 0),
                'NPV': results.get('npv', 0),
                'AUC-ROC': results.get('auc_roc', 0),
                'AUC-PR': results.get('auc_pr', 0)
            })
        df = pd.DataFrame(rows)
        self.comparison_data = df
        return df


# Global evaluator instance
evaluator = ModelEvaluator()
