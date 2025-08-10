"""
Enhanced ML Models Module for PTSD ML Platform
Implements multiple algorithms with advanced features and error handling
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional, Tuple, List
import joblib
import streamlit as st
from datetime import datetime

class PTSDMLModels:
    """
    Enhanced Machine Learning models for PTSD prediction
    Implements multiple algorithms with hyperparameter optimization and ensemble methods
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.hyperparameters = {}
        
        # Initialize models with optimized parameters from research
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with research-based optimal hyperparameters"""
        
        # Support Vector Machine - Best performer in neuroimaging studies
        self.models['SVM'] = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Artificial Neural Network - Best for PCL-5 data
        self.models['ANN'] = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
        
        # Decision Tree - Interpretable model
        self.models['Decision_Tree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Gaussian Naive Bayes - Fast and effective baseline
        self.models['Gaussian_NB'] = Pipeline([
            ('scaler', StandardScaler()),
            ('gnb', GaussianNB())
        ])
        
        # Random Forest - Strong ensemble method
        self.models['Random_Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Gradient Boosting - Advanced ensemble
        self.models['Gradient_Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Store hyperparameters for reference
        self.hyperparameters = {
            name: model.get_params() if hasattr(model, 'get_params') else {}
            for name, model in self.models.items()
        }
    
    def train_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        cv_folds: int = 5,
        selected_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train all selected models with comprehensive evaluation
        """
        
        # Select models to train
        models_to_train = selected_models or list(self.models.keys())
        models_subset = {name: self.models[name] for name in models_to_train if name in self.models}
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (model_name, model) in enumerate(models_subset.items()):
            status_text.text(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except:
                    # Some models might not support predict_proba
                    pass
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Calculate AUC if probabilities available
                auc = None
                if y_pred_proba is not None:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        pass
                
                # Cross-validation scores with error handling
                try:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy',
                        error_score='raise'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as e:
                    st.warning(f"Cross-validation failed for {model_name}: {str(e)}")
                    cv_mean = accuracy
                    cv_std = 0
                
                # Store comprehensive results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'predictions': y_pred,
                    'actual': y_test,
                    'probabilities': y_pred_proba,
                    'training_time': datetime.utcnow(),
                    'hyperparameters': self.hyperparameters.get(model_name, {}),
                    'training_data_size': X_train.shape[0],
                    'feature_count': X_train.shape[1]
                }
                
                # Update best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model_name
                
                # Update progress
                progress_bar.progress((i + 1) / len(models_subset))
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                continue
        
        status_text.text("Training completed!")
        progress_bar.empty()
        
        # Store results
        self.results = results
        self.trained_models = {name: result['model'] for name, result in results.items()}
        
        return results
    
    def predict(
        self, 
        X: np.ndarray, 
        model_name: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using trained model
        """
        
        if model_name is None:
            model_name = self.best_model
        
        if not model_name or model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        predictions = model.predict(X)
        
        probabilities = None
        try:
            probabilities = model.predict_proba(X)[:, 1]
        except:
            pass
        
        return predictions, probabilities
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'AUC': f"{result['auc']:.4f}" if result.get('auc') else 'N/A',
                'CV Mean': f"{result['cv_mean']:.4f}" if 'cv_mean' in result else 'N/A',
                'CV Std': f"{result['cv_std']:.4f}" if 'cv_std' in result else 'N/A'
            })
        
        return pd.DataFrame(summary_data)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.results:
            return {}
        
        return {
            'total_models_trained': len(self.results),
            'best_model': self.best_model,
            'best_score': self.best_score,
            'models_with_probabilities': sum(
                1 for result in self.results.values() 
                if result.get('probabilities') is not None
            ),
            'average_accuracy': np.mean([
                result['accuracy'] for result in self.results.values()
            ]),
            'training_data_size': list(self.results.values())[0].get('training_data_size', 0) if self.results else 0,
            'feature_count': list(self.results.values())[0].get('feature_count', 0) if self.results else 0
        }