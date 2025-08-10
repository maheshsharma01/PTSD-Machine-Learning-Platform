"""
Unit tests for the PTSD ML Platform core modules
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Test imports with fallbacks
try:
    from src.core.config_manager import ConfigManager, get_config
    from src.core.error_handler import ErrorHandler, PTSDError
    from src.core.session_state_manager import SessionStateManager
    from src.data.data_processor import DataProcessor
    from src.models.ml_models import PTSDMLModels
except ImportError:
    pytest.skip("Source modules not available", allow_module_level=True)

class TestConfigManager:
    """Test cases for ConfigManager"""
    
    def test_config_manager_initialization(self):
        """Test config manager initializes correctly"""
        config_manager = ConfigManager()
        assert config_manager.env in ['development', 'production', 'testing']
        assert hasattr(config_manager, '_config_cache')
    
    def test_get_config_value(self):
        """Test getting configuration values"""
        config_manager = ConfigManager()
        
        # Test getting existing value
        app_name = config_manager.get('app.name', 'Default Name')
        assert isinstance(app_name, str)
        
        # Test getting non-existent value with default
        non_existent = config_manager.get('non.existent.key', 'default')
        assert non_existent == 'default'
    
    def test_database_config(self):
        """Test database configuration retrieval"""
        config_manager = ConfigManager()
        db_config = config_manager.get_database_config()
        
        assert hasattr(db_config, 'host')
        assert hasattr(db_config, 'port')
        assert hasattr(db_config, 'database')
        assert isinstance(db_config.port, int)

class TestErrorHandler:
    """Test cases for ErrorHandler"""
    
    def test_error_handler_initialization(self):
        """Test error handler initializes correctly"""
        error_handler = ErrorHandler()
        assert hasattr(error_handler, 'error_stats')
        assert error_handler.error_stats['total_errors'] == 0
    
    def test_ptsd_error_creation(self):
        """Test PTSD error creation"""
        error = PTSDError("Test error message")
        
        assert error.message == "Test error message"
        assert hasattr(error, 'error_id')
        assert hasattr(error, 'timestamp')
        assert error.user_message is not None
    
    def test_error_handling(self):
        """Test error handling functionality"""
        error_handler = ErrorHandler()
        
        # Create a test error
        test_error = Exception("Test exception")
        
        # Handle the error
        error_id = error_handler.handle_error(test_error, show_user_message=False)
        
        assert error_id is not None
        assert error_handler.error_stats['total_errors'] == 1

class TestDataProcessor:
    """Test cases for DataProcessor"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'age': [25, 34, 45],
            'gender': ['F', 'M', 'F'],
            'pcl5_intrusive': [3.2, 2.8, 4.1],
            'pcl5_avoidance': [2.9, 3.5, 3.8],
            'cortisol_level': [15.2, 12.8, 18.5],
            'ptsd_diagnosis': [1, 0, 1]
        })
    
    def test_data_processor_initialization(self):
        """Test data processor initializes correctly"""
        processor = DataProcessor()
        
        assert hasattr(processor, 'scaler')
        assert hasattr(processor, 'label_encoders')
        assert hasattr(processor, 'imputers')
    
    def test_data_analysis(self, sample_dataframe):
        """Test data analysis functionality"""
        processor = DataProcessor()
        analysis = processor.analyze_data(sample_dataframe)
        
        assert 'shape' in analysis
        assert 'columns' in analysis
        assert 'dtypes' in analysis
        assert 'missing_values' in analysis
        assert 'numeric_columns' in analysis
        assert 'categorical_columns' in analysis
        
        assert analysis['shape'] == (3, 7)
        assert len(analysis['columns']) == 7
    
    def test_data_preprocessing(self, sample_dataframe):
        """Test data preprocessing pipeline"""
        processor = DataProcessor()
        
        X, y = processor.preprocess_data(
            sample_dataframe,
            target_column='ptsd_diagnosis',
            handle_missing='mean',
            encode_categorical=True,
            scale_features=True
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert len(y) == 3
    
    def test_pcl5_feature_creation(self, sample_dataframe):
        """Test PCL-5 feature engineering"""
        processor = DataProcessor()
        
        # Add some PCL-5 individual items for testing
        sample_dataframe['pcl5_1'] = [4, 3, 5]
        sample_dataframe['pcl5_2'] = [3, 2, 4]
        
        enhanced_df = processor.create_pcl5_features(sample_dataframe)
        
        assert len(enhanced_df.columns) > len(sample_dataframe.columns)
    
    def test_data_quality_validation(self, sample_dataframe):
        """Test data quality validation"""
        processor = DataProcessor()
        validation = processor.validate_data_quality(sample_dataframe)
        
        assert 'issues' in validation
        assert 'warnings' in validation
        assert 'recommendations' in validation
        assert 'quality_score' in validation
        
        assert isinstance(validation['quality_score'], float)
        assert 0 <= validation['quality_score'] <= 100

class TestMLModels:
    """Test cases for PTSDMLModels"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for ML testing"""
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_ml_models_initialization(self):
        """Test ML models initialization"""
        ml_models = PTSDMLModels()
        
        assert hasattr(ml_models, 'models')
        assert hasattr(ml_models, 'trained_models')
        assert hasattr(ml_models, 'results')
        assert len(ml_models.models) > 0
    
    @pytest.mark.slow
    def test_model_training(self, sample_data):
        """Test model training functionality"""
        ml_models = PTSDMLModels()
        X, y = sample_data
        
        # Train a subset of models for faster testing
        results = ml_models.train_models(
            X, y, 
            selected_models=['Decision_Tree', 'Gaussian_NB'],
            test_size=0.3,
            cv_folds=3
        )
        
        assert len(results) == 2
        assert 'Decision_Tree' in results
        assert 'Gaussian_NB' in results
        
        for model_name, result in results.items():
            assert 'accuracy' in result
            assert 'precision' in result
            assert 'recall' in result
            assert 'f1_score' in result
            assert 0 <= result['accuracy'] <= 1
    
    def test_model_prediction(self, sample_data):
        """Test model prediction functionality"""
        ml_models = PTSDMLModels()
        X, y = sample_data
        
        # Train a simple model first
        results = ml_models.train_models(
            X, y, 
            selected_models=['Decision_Tree'],
            test_size=0.3
        )
        
        # Test prediction
        X_test = X[:10]  # Use first 10 samples for prediction
        predictions, probabilities = ml_models.predict(X_test, 'Decision_Tree')
        
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_summary(self, sample_data):
        """Test model summary generation"""
        ml_models = PTSDMLModels()
        X, y = sample_data
        
        # Train models first
        results = ml_models.train_models(
            X, y, 
            selected_models=['Decision_Tree'],
            test_size=0.3
        )
        
        summary_df = ml_models.get_model_summary()
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 1
        assert 'Model' in summary_df.columns
        assert 'Accuracy' in summary_df.columns

class TestIntegration:
    """Integration tests for the full pipeline"""
    
    @pytest.fixture
    def full_pipeline_data(self):
        """Create data for full pipeline testing"""
        return pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(1, 51)],
            'age': np.random.randint(18, 80, 50),
            'gender': np.random.choice(['M', 'F'], 50),
            'pcl5_intrusive': np.random.uniform(0, 4, 50),
            'pcl5_avoidance': np.random.uniform(0, 4, 50),
            'pcl5_negative_mood': np.random.uniform(0, 4, 50),
            'pcl5_hyperarousal': np.random.uniform(0, 4, 50),
            'cortisol_level': np.random.uniform(10, 25, 50),
            'ptsd_diagnosis': np.random.randint(0, 2, 50)
        })
    
    @pytest.mark.integration
    def test_full_pipeline(self, full_pipeline_data):
        """Test the complete data processing and ML pipeline"""
        # Step 1: Data Processing
        processor = DataProcessor()
        analysis = processor.analyze_data(full_pipeline_data)
        
        assert analysis['shape'][0] == 50
        
        # Step 2: Preprocessing
        X, y = processor.preprocess_data(
            full_pipeline_data,
            target_column='ptsd_diagnosis',
            handle_missing='mean',
            encode_categorical=True,
            scale_features=True
        )
        
        assert X.shape[0] == 50
        assert len(y) == 50
        
        # Step 3: Model Training
        ml_models = PTSDMLModels()
        results = ml_models.train_models(
            X, y,
            selected_models=['Decision_Tree'],
            test_size=0.3,
            cv_folds=3
        )
        
        assert len(results) == 1
        assert 'Decision_Tree' in results
        
        # Step 4: Prediction
        predictions, probabilities = ml_models.predict(X[:10], 'Decision_Tree')
        
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

# Test fixtures
@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing"""
    with patch('streamlit.error'), \
         patch('streamlit.warning'), \
         patch('streamlit.info'), \
         patch('streamlit.success'):
        yield

# Test utilities
def create_mock_csv_file(filename, data):
    """Create a mock CSV file for testing"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def assert_dataframe_equal(df1, df2, check_dtype=True):
    """Assert that two DataFrames are equal"""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)