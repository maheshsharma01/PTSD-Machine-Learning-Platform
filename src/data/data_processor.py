"""
Enhanced Data Processing Module for PTSD ML Platform
Handles various data types including PCL-5, biomarkers, and clinical data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Tuple, List, Dict, Any
import streamlit as st

class DataProcessor:
    """
    Enhanced data processing utilities for PTSD prediction datasets
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_selector = None
        self.processed_features = None
        self.target_column = None
        self.feature_names = []
        
    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file with enhanced error handling"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None
            
            # Basic validation
            if df.empty:
                st.error("The uploaded file is empty.")
                return None
                
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data analysis with enhanced insights"""
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'target_candidates': []
        }
        
        # Identify potential target columns
        potential_targets = ['ptsd', 'diagnosis', 'ptsd_diagnosis', 'label', 'target', 'outcome']
        for col in df.columns:
            if any(target in col.lower() for target in potential_targets):
                analysis['target_candidates'].append(col)
        
        # Check for PCL-5 scale columns
        pcl5_columns = [col for col in df.columns if 'pcl5' in col.lower() or 'pcl-5' in col.lower()]
        analysis['pcl5_columns'] = pcl5_columns
        
        # Check for biomarker columns
        biomarker_keywords = ['cortisol', 'hormone', 'biomarker', 'blood', 'serum']
        biomarker_columns = [col for col in df.columns
                            if any(keyword in col.lower() for keyword in biomarker_keywords)]
        analysis['biomarker_columns'] = biomarker_columns
        
        # Check for neuroimaging columns
        neuroimaging_keywords = ['meg', 'mri', 'eeg', 'brain', 'neural', 'connectivity']
        neuroimaging_columns = [col for col in df.columns
                               if any(keyword in col.lower() for keyword in neuroimaging_keywords)]
        analysis['neuroimaging_columns'] = neuroimaging_columns
        
        # Data quality metrics
        analysis['data_quality'] = {
            'completeness': (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1]
        }
        
        return analysis
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        handle_missing: str = 'mean',
        encode_categorical: bool = True,
        scale_features: bool = True,
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Comprehensive data preprocessing pipeline with advanced options
        """
        
        df_processed = df.copy()
        
        # Validate target column
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = [col for col in df_processed.columns if col != target_column]
        
        # Extract target
        y = df_processed[target_column].values
        X = df_processed[feature_columns].copy()
        
        # Handle missing values in target
        valid_mask = pd.notnull(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Handle missing values in features
        X = self._handle_missing_values(X, handle_missing)
        
        # Encode categorical variables
        if encode_categorical:
            X = self._encode_categorical_variables(X)
        
        # Convert to numpy arrays
        X_array = X.values.astype(float)
        
        # Feature selection
        if feature_selection and n_features:
            X_array, selected_indices = self._perform_feature_selection(
                X_array, y, method=feature_selection, k=n_features
            )
            self.feature_names = [X.columns[i] for i in selected_indices]
        else:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        if scale_features:
            X_array = self.scaler.fit_transform(X_array)
        
        # Store processed information
        self.processed_features = feature_columns
        self.target_column = target_column
        
        return X_array, y
    
    def _handle_missing_values(self, X: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values with various strategies"""
        
        if strategy == 'drop':
            return X.dropna()
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        if len(numeric_columns) > 0:
            if strategy in ['mean', 'median']:
                numeric_imputer = SimpleImputer(strategy=strategy)
                X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
                self.imputers['numeric'] = numeric_imputer
            elif strategy == 'knn':
                knn_imputer = KNNImputer(n_neighbors=5)
                X[numeric_columns] = knn_imputer.fit_transform(X[numeric_columns])
                self.imputers['numeric'] = knn_imputer
        
        # Handle categorical columns
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
            self.imputers['categorical'] = categorical_imputer
        
        return X
    
    def _encode_categorical_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables with proper handling"""
        
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Handle common categorical mappings
            if col.lower() in ['gender', 'sex']:
                # Binary encoding for gender
                X[col] = X[col].map({'M': 1, 'Male': 1, 'F': 0, 'Female': 0, 'Other': 2})
            elif col.lower() in ['employment_status', 'employed']:
                # Binary encoding for employment
                X[col] = X[col].map({'Employed': 1, 'Unemployed': 0, 'Student': 0.5, 'Retired': 0.5})
            else:
                # Label encoding for other categorical variables
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        return X
    
    def _perform_feature_selection(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        method: str, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature selection with multiple methods"""
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
        
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
        
        elif method == 'importance':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            X_selected = X[:, indices]
            selected_indices = indices
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.feature_selector = selector if method in ['univariate', 'rfe'] else None
        return X_selected, selected_indices
    
    def create_pcl5_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced PCL-5 derived features based on clinical research"""
        
        df_enhanced = df.copy()
        
        # PCL-5 subscale patterns based on DSM-5
        pcl5_subscales = {
            'intrusive': ['pcl5_1', 'pcl5_2', 'pcl5_3', 'pcl5_4', 'pcl5_5'],
            'avoidance': ['pcl5_6', 'pcl5_7'],
            'negative_mood': ['pcl5_8', 'pcl5_9', 'pcl5_10', 'pcl5_11', 'pcl5_12', 'pcl5_13', 'pcl5_14'],
            'hyperarousal': ['pcl5_15', 'pcl5_16', 'pcl5_17', 'pcl5_18', 'pcl5_19', 'pcl5_20']
        }
        
        # Create subscale scores
        for subscale, items in pcl5_subscales.items():
            available_items = [item for item in items if item in df.columns]
            if available_items:
                df_enhanced[f'pcl5_{subscale}_score'] = df_enhanced[available_items].mean(axis=1)
                df_enhanced[f'pcl5_{subscale}_max'] = df_enhanced[available_items].max(axis=1)
                df_enhanced[f'pcl5_{subscale}_variability'] = df_enhanced[available_items].std(axis=1)
        
        # Create total PCL-5 score if individual items exist
        pcl5_items = [col for col in df.columns if col.startswith('pcl5_') and col.split('_')[-1].isdigit()]
        if pcl5_items:
            df_enhanced['pcl5_total_score'] = df_enhanced[pcl5_items].sum(axis=1)
            df_enhanced['pcl5_mean_score'] = df_enhanced[pcl5_items].mean(axis=1)
            df_enhanced['pcl5_median_score'] = df_enhanced[pcl5_items].median(axis=1)
            
            # Create severity categories based on research thresholds
            df_enhanced['pcl5_severity'] = pd.cut(
                df_enhanced['pcl5_total_score'],
                bins=[0, 33, 45, 57, 80],
                labels=['Minimal', 'Mild', 'Moderate', 'Severe']
            )
        
        return df_enhanced
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation with actionable insights"""
        
        validation = {
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'quality_score': 100.0
        }
        
        # Check for excessive missing values
        missing_percent = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percent[missing_percent > 20]
        if len(high_missing) > 0:
            validation['warnings'].append(f"Columns with >20% missing values: {high_missing.index.tolist()}")
            validation['quality_score'] -= len(high_missing) * 5
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation['issues'].append(f"Found {duplicates} duplicate rows")
            validation['quality_score'] -= min(duplicates * 2, 20)
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation['issues'].append(f"Constant columns found: {constant_cols}")
            validation['quality_score'] -= len(constant_cols) * 10
        
        # Check sample size
        if len(df) < 100:
            validation['warnings'].append("Small sample size (<100) may affect model performance")
            validation['quality_score'] -= 20
        elif len(df) < 30:
            validation['issues'].append("Very small sample size (<30) - insufficient for reliable modeling")
            validation['quality_score'] -= 50
        
        # Generate recommendations
        if validation['quality_score'] >= 90:
            validation['recommendations'].append("✅ Data quality is excellent!")
        elif validation['quality_score'] >= 70:
            validation['recommendations'].append("⚠️ Data quality is good but could be improved")
        else:
            validation['recommendations'].append("❌ Data quality needs significant improvement")
        
        return validation
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        
        if self.processed_features is None:
            raise ValueError("Preprocessors not fitted. Run preprocess_data first.")
        
        # Select same features used in training
        X = df[self.processed_features].copy()
        
        # Handle missing values using fitted imputers
        if 'numeric' in self.imputers:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                X[numeric_columns] = self.imputers['numeric'].transform(X[numeric_columns])
        
        # Encode categorical variables using fitted encoders
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except ValueError:
                    # Handle unseen labels
                    X[col] = X[col].astype(str).map(
                        {label: code for code, label in enumerate(encoder.classes_)}
                    ).fillna(0)
        
        # Convert to numpy array and scale
        X_array = X.values.astype(float)
        X_scaled = self.scaler.transform(X_array)
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return X_scaled
    
    def get_feature_names(self) -> List[str]:
        """Get names of processed features"""
        return self.feature_names if self.feature_names else []
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied"""
        return {
            'target_column': self.target_column,
            'feature_count': len(self.processed_features) if self.processed_features else 0,
            'selected_features': len(self.feature_names),
            'scalers_fitted': hasattr(self.scaler, 'mean_'),
            'imputers_fitted': len(self.imputers),
            'encoders_fitted': len(self.label_encoders),
            'feature_selector_fitted': self.feature_selector is not None
        }