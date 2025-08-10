"""
Enhanced Database Manager with Connection Pooling for PTSD ML Platform
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import streamlit as st

try:
    from src.core.config_manager import get_database_config
    from src.core.error_handler import DatabaseError, handle_errors, ErrorSeverity
except ImportError:
    # Fallback if modules not available
    class DatabaseError(Exception):
        pass
    def handle_errors(**kwargs):
        def decorator(func):
            return func
        return decorator

Base = declarative_base()

class PatientData(Base):
    """Enhanced Patient Data model"""
    __tablename__ = 'patient_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    trauma_type = Column(String(100), nullable=True)
    trauma_severity = Column(Float, nullable=True)
    
    # PCL-5 Scale Scores
    pcl5_intrusive = Column(Float, nullable=True)
    pcl5_avoidance = Column(Float, nullable=True)
    pcl5_negative_mood = Column(Float, nullable=True)
    pcl5_hyperarousal = Column(Float, nullable=True)
    pcl5_total_score = Column(Float, nullable=True)
    
    # Biomarkers and Clinical Data
    cortisol_level = Column(Float, nullable=True)
    time_since_trauma = Column(Integer, nullable=True)  # in months
    social_support_score = Column(Float, nullable=True)
    education_years = Column(Integer, nullable=True)
    employment_status = Column(Integer, nullable=True)  # 1=employed, 0=unemployed
    
    # Medical History
    previous_trauma = Column(Boolean, nullable=True)
    family_history_mental_health = Column(Boolean, nullable=True)
    
    # Outcome
    ptsd_diagnosis = Column(Integer, nullable=True)  # 1=PTSD, 0=No PTSD
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class ModelResults(Base):
    """Enhanced Model Results with better tracking"""
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    
    # Performance Metrics
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_score = Column(Float, nullable=True)
    
    # Cross-Validation Results
    cv_mean = Column(Float, nullable=True)
    cv_std = Column(Float, nullable=True)
    
    # Model Artifacts
    confusion_matrix = Column(Text, nullable=True)  # JSON string
    hyperparameters = Column(Text, nullable=True)  # JSON string
    
    # Training Information
    training_data_size = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    
    # Metadata
    training_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)

class PredictionHistory(Base):
    """Enhanced Prediction History with better tracking"""
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    
    # Prediction Results
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)  # low, medium, high
    
    # Input Data
    feature_values = Column(Text, nullable=True)  # JSON string
    
    # Metadata
    prediction_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)

class DatabaseManager:
    """
    Enhanced Database Manager with connection pooling and error handling
    """
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection with proper configuration"""
        try:
            # Get database URL
            database_url = self._get_database_url()
            
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Test connection
            self._test_connection()
            
            self.is_connected = True
            self.logger.info("Database connection initialized successfully")
            
        except Exception as e:
            self.is_connected = False
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            # Don't raise here to prevent app from crashing
    
    def _get_database_url(self) -> str:
        """Get database URL from configuration or environment"""
        # Check if DATABASE_URL environment variable exists
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Handle Heroku-style postgres:// URLs
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            return database_url
        
        # Build URL from environment variables
        host = os.getenv('DATABASE_HOST', 'localhost')
        port = os.getenv('DATABASE_PORT', '5432')
        database = os.getenv('DATABASE_NAME', 'ptsd_ml')
        username = os.getenv('DATABASE_USER', 'ptsd_user')
        password = os.getenv('DATABASE_PASSWORD', '')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _test_connection(self):
        """Test database connection"""
        with self.engine.connect() as conn:
            conn.execute("SELECT 1")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with proper cleanup"""
        if not self.is_connected:
            raise DatabaseError("Database not connected")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            session.close()
    
    @handle_errors(show_user_message=True)
    def save_patient_data(self, patient_data_dict: Dict[str, Any]) -> bool:
        """Save patient data to database with validation"""
        if not self.is_connected:
            st.error("Database not connected")
            return False
            
        with self.get_session() as session:
            try:
                # Validate required fields
                if 'patient_id' not in patient_data_dict:
                    st.error("Patient ID is required")
                    return False
                
                # Check if patient exists
                existing_patient = session.query(PatientData).filter_by(
                    patient_id=patient_data_dict['patient_id']
                ).first()
                
                if existing_patient:
                    # Update existing patient
                    for key, value in patient_data_dict.items():
                        if hasattr(existing_patient, key) and key != 'id':
                            setattr(existing_patient, key, value)
                    
                    existing_patient.updated_at = datetime.utcnow()
                else:
                    # Create new patient record
                    patient = PatientData(**patient_data_dict)
                    session.add(patient)
                
                return True
                
            except IntegrityError as e:
                st.error(f"Data integrity error: {str(e)}")
                return False
            except Exception as e:
                st.error(f"Error saving patient data: {str(e)}")
                return False
    
    @handle_errors(show_user_message=True)
    def get_patient_data(self, patient_id: str = None, limit: int = 100) -> pd.DataFrame:
        """Retrieve patient data with optional filtering"""
        if not self.is_connected:
            st.error("Database not connected")
            return pd.DataFrame()
            
        with self.get_session() as session:
            try:
                query = session.query(PatientData)
                
                if patient_id:
                    query = query.filter(PatientData.patient_id == patient_id)
                else:
                    query = query.order_by(PatientData.created_at.desc()).limit(limit)
                
                patients = query.all()
                
                if patients:
                    # Convert to DataFrame
                    data = []
                    for patient in patients:
                        patient_dict = {
                            column.name: getattr(patient, column.name)
                            for column in patient.__table__.columns
                        }
                        data.append(patient_dict)
                    
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                st.error(f"Error retrieving patient data: {str(e)}")
                return pd.DataFrame()
    
    @handle_errors(show_user_message=True)
    def save_model_results(self, model_name: str, results_dict: Dict[str, Any]) -> bool:
        """Save model training results"""
        if not self.is_connected:
            st.error("Database not connected")
            return False
            
        with self.get_session() as session:
            try:
                # Prepare model results data
                model_data = {
                    'model_name': model_name,
                    'accuracy': results_dict.get('accuracy'),
                    'precision_score': results_dict.get('precision'),
                    'recall': results_dict.get('recall'),
                    'f1_score': results_dict.get('f1_score'),
                    'auc_score': results_dict.get('auc'),
                    'cv_mean': results_dict.get('cv_mean'),
                    'cv_std': results_dict.get('cv_std'),
                    'training_data_size': results_dict.get('training_data_size'),
                    'feature_count': results_dict.get('feature_count')
                }
                
                # Convert complex objects to JSON
                if 'confusion_matrix' in results_dict:
                    cm = results_dict['confusion_matrix']
                    if isinstance(cm, np.ndarray):
                        model_data['confusion_matrix'] = json.dumps(cm.tolist())
                    else:
                        model_data['confusion_matrix'] = json.dumps(cm)
                
                if 'hyperparameters' in results_dict:
                    model_data['hyperparameters'] = json.dumps(results_dict['hyperparameters'])
                
                # Create model result record
                model_result = ModelResults(**model_data)
                session.add(model_result)
                
                return True
                
            except Exception as e:
                st.error(f"Error saving model results: {str(e)}")
                return False
    
    @handle_errors(show_user_message=True)
    def save_prediction(
        self, 
        patient_id: str, 
        model_name: str, 
        prediction: int, 
        probability: float,
        feature_values: Dict[str, Any], 
        risk_level: str = None, 
        notes: str = None
    ) -> bool:
        """Save prediction result to database"""
        if not self.is_connected:
            st.error("Database not connected")
            return False
            
        with self.get_session() as session:
            try:
                prediction_record = PredictionHistory(
                    patient_id=patient_id,
                    model_name=model_name,
                    prediction=prediction,
                    probability=probability,
                    feature_values=json.dumps(feature_values),
                    risk_level=risk_level,
                    notes=notes
                )
                
                session.add(prediction_record)
                return True
                
            except Exception as e:
                st.error(f"Error saving prediction: {str(e)}")
                return False
    
    @handle_errors(show_user_message=True)
    def get_model_results(self, model_name: str = None) -> pd.DataFrame:
        """Retrieve model results from database"""
        if not self.is_connected:
            return pd.DataFrame()
            
        with self.get_session() as session:
            try:
                query = session.query(ModelResults)
                if model_name:
                    query = query.filter(ModelResults.model_name == model_name)
                
                results = query.order_by(ModelResults.training_date.desc()).all()
                
                if results:
                    data = []
                    for result in results:
                        result_dict = {
                            column.name: getattr(result, column.name)
                            for column in result.__table__.columns
                        }
                        data.append(result_dict)
                    
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                st.error(f"Error retrieving model results: {str(e)}")
                return pd.DataFrame()
    
    @handle_errors(show_user_message=True)
    def get_prediction_history(self, patient_id: str = None, model_name: str = None, limit: int = 100) -> pd.DataFrame:
        """Retrieve prediction history from database"""
        if not self.is_connected:
            return pd.DataFrame()
            
        with self.get_session() as session:
            try:
                query = session.query(PredictionHistory)
                
                if patient_id:
                    query = query.filter(PredictionHistory.patient_id == patient_id)
                if model_name:
                    query = query.filter(PredictionHistory.model_name == model_name)
                
                predictions = query.order_by(PredictionHistory.prediction_date.desc()).limit(limit).all()
                
                if predictions:
                    data = []
                    for prediction in predictions:
                        pred_dict = {
                            column.name: getattr(prediction, column.name)
                            for column in prediction.__table__.columns
                        }
                        data.append(pred_dict)
                    
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                st.error(f"Error retrieving prediction history: {str(e)}")
                return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_connected:
            return {}
            
        try:
            with self.get_session() as session:
                stats = {
                    'total_patients': session.query(PatientData).count(),
                    'total_predictions': session.query(PredictionHistory).count(),
                    'total_models': session.query(ModelResults).count(),
                    'ptsd_positive_cases': session.query(PatientData).filter_by(ptsd_diagnosis=1).count(),
                    'ptsd_negative_cases': session.query(PatientData).filter_by(ptsd_diagnosis=0).count()
                }
                
                return stats
        except Exception as e:
            self.logger.error(f"Error retrieving database statistics: {str(e)}")
            return {}
    
    def bulk_insert_patient_data(self, df: pd.DataFrame) -> bool:
        """Bulk insert patient data from DataFrame"""
        if not self.is_connected:
            return False
            
        with self.get_session() as session:
            try:
                # Convert DataFrame to list of dictionaries
                records = df.to_dict('records')
                
                # Create PatientData objects
                patient_objects = []
                for record in records:
                    # Filter only columns that exist in the PatientData model
                    filtered_record = {
                        key: value for key, value in record.items()
                        if hasattr(PatientData, key) and value is not None
                    }
                    patient_objects.append(PatientData(**filtered_record))
                
                # Bulk insert
                session.bulk_save_objects(patient_objects)
                return True
                
            except Exception as e:
                st.error(f"Error bulk inserting patient data: {str(e)}")
                return False
    
    def close_connections(self):
        """Close all database connections"""
        try:
            if self.engine:
                self.engine.dispose()
            self.is_connected = False
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {str(e)}")

# Global database manager instance
db_manager = None

def get_db_manager():
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager