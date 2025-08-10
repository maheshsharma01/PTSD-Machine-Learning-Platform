"""
Model Evaluation Module for PTSD ML Platform
Provides comprehensive evaluation metrics and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, average_precision_score
)
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
from datetime import datetime

class ModelEvaluator:
    """
    Comprehensive model evaluation with clinical metrics
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_data = {}
    
    def evaluate_model(
        self, 
        model_name: str,
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with clinical metrics
        """
        
        results = {
            'model_name': model_name,
            'evaluation_date': datetime.utcnow(),
            'sample_size': len(y_true)
        }
        
        # Basic classification metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Clinical metrics (for binary classification)
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity (Recall for positive class)
            results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Specificity
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Positive Predictive Value (Precision for positive class)
            results['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Negative Predictive Value
            results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Likelihood ratios
            results['lr_positive'] = results['sensitivity'] / (1 - results['specificity']) if results['specificity'] < 1 else float('inf')
            results['lr_negative'] = (1 - results['sensitivity']) / results['specificity'] if results['specificity'] > 0 else float('inf')
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                results['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                results['auc_pr'] = average_precision_score(y_true, y_pred_proba)
                
                # Store curves for plotting
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                results['pr_curve'] = {'precision': precision_curve, 'recall': recall_curve}
                
            except Exception as e:
                st.warning(f"Could not calculate probability-based metrics: {str(e)}")
        
        # Classification report
        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create comparison table for multiple models
        """
        
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {
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
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        self.comparison_data = df
        
        return df
    
    def get_clinical_interpretation(self, model_name: str) -> str:
        """
        Provide clinical interpretation of model performance
        """
        
        if model_name not in self.evaluation_results:
            return "Model evaluation results not found."
        
        results = self.evaluation_results[model_name]
        
        interpretation = f"Clinical Performance Analysis for {model_name}:\n\n"
        
        # Overall performance
        accuracy = results.get('accuracy', 0)
        if accuracy >= 0.9:
            interpretation += "🟢 Excellent overall accuracy (≥90%) - Highly reliable for clinical use\n"
        elif accuracy >= 0.8:
            interpretation += "🟡 Good overall accuracy (80-89%) - Suitable for clinical screening\n"
        elif accuracy >= 0.7:
            interpretation += "🟠 Moderate accuracy (70-79%) - Requires caution in clinical interpretation\n"
        else:
            interpretation += "🔴 Low accuracy (<70%) - Not recommended for clinical use\n"
        
        # Sensitivity analysis
        sensitivity = results.get('sensitivity', 0)
        if sensitivity >= 0.9:
            interpretation += "🟢 Excellent sensitivity (≥90%) - Very few PTSD cases will be missed\n"
        elif sensitivity >= 0.8:
            interpretation += "🟡 Good sensitivity (80-89%) - Acceptable for screening purposes\n"
        else:
            interpretation += "🔴 Low sensitivity (<80%) - Risk of missing PTSD cases\n"
        
        # Specificity analysis
        specificity = results.get('specificity', 0)
        if specificity >= 0.9:
            interpretation += "🟢 Excellent specificity (≥90%) - Very few false positives\n"
        elif specificity >= 0.8:
            interpretation += "🟡 Good specificity (80-89%) - Acceptable false positive rate\n"
        else:
            interpretation += "🔴 Low specificity (<80%) - High false positive rate\n"
        
        # Clinical utility
        ppv = results.get('ppv', 0)
        npv = results.get('npv', 0)
        
        interpretation += f"\nClinical Utility:\n"
        interpretation += f"• Positive Predictive Value: {ppv:.3f} - {ppv*100:.1f}% of positive predictions are correct\n"
        interpretation += f"• Negative Predictive Value: {npv:.3f} - {npv*100:.1f}% of negative predictions are correct\n"
        
        # Recommendations
        interpretation += "\nClinical Recommendations:\n"
        
        if accuracy >= 0.85 and sensitivity >= 0.8 and specificity >= 0.8:
            interpretation += "✅ Recommended for clinical screening and assessment support\n"
        elif accuracy >= 0.75 and sensitivity >= 0.8:
            interpretation += "⚠️ Suitable for initial screening, but requires clinical confirmation\n"
        else:
            interpretation += "❌ Not recommended for clinical use - requires model improvement\n"
        
        return interpretation
    
    def get_model_ranking(self, metric: str = 'f1_score') -> List[Tuple[str, float]]:
        """
        Rank models by specified metric
        """
        
        if not self.evaluation_results:
            return []
        
        rankings = []
        for model_name, results in self.evaluation_results.items():
            score = results.get(metric, 0)
            rankings.append((model_name, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def generate_evaluation_report(self, model_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        """
        
        if model_name not in self.evaluation_results:
            return {}
        
        results = self.evaluation_results[model_name]
        
        report = {
            'model_name': model_name,
            'evaluation_summary': {
                'sample_size': results.get('sample_size', 0),
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0)
            },
            'clinical_metrics': {
                'sensitivity': results.get('sensitivity', 0),
                'specificity': results.get('specificity', 0),
                'ppv': results.get('ppv', 0),
                'npv': results.get('npv', 0),
                'lr_positive': results.get('lr_positive', 0),
                'lr_negative': results.get('lr_negative', 0)
            },
            'probabilistic_metrics': {
                'auc_roc': results.get('auc_roc', 0),
                'auc_pr': results.get('auc_pr', 0)
            },
            'confusion_matrix': results.get('confusion_matrix', np.array([])),
            'clinical_interpretation': self.get_clinical_interpretation(model_name),
            'classification_report': results.get('classification_report', {}),
            'evaluation_date': results.get('evaluation_date', datetime.utcnow())
        }
        
        return report
    
    def calculate_clinical_impact(self, model_name: str, prevalence: float = 0.1) -> Dict[str, float]:
        """
        Calculate clinical impact metrics given population prevalence
        """
        
        if model_name not in self.evaluation_results:
            return {}
        
        results = self.evaluation_results[model_name]
        
        sensitivity = results.get('sensitivity', 0)
        specificity = results.get('specificity', 0)
        
        # Calculate predictive values adjusted for prevalence
        ppv_adjusted = (sensitivity * prevalence) / (
            (sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence))
        )
        
        npv_adjusted = (specificity * (1 - prevalence)) / (
            (specificity * (1 - prevalence)) + ((1 - sensitivity) * prevalence)
        )
        
        # Number needed to screen (for every true positive found)
        nns = 1 / (sensitivity * prevalence) if (sensitivity * prevalence) > 0 else float('inf')
        
        return {
            'ppv_adjusted': ppv_adjusted,
            'npv_adjusted': npv_adjusted,
            'number_needed_to_screen': nns,
            'false_positive_rate': 1 - specificity,
            'false_negative_rate': 1 - sensitivity
        }
    
    def export_results(self, format_type: str = 'csv') -> str:
        """
        Export evaluation results in specified format
        """
        
        if not self.comparison_data.empty:
            if format_type == 'csv':
                return self.comparison_data.to_csv(index=False)
            elif format_type == 'json':
                return self.comparison_data.to_json(orient='records', indent=2)
        
        return ""

# Global evaluator instance
evaluator = ModelEvaluator()