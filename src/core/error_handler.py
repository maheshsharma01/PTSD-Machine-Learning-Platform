"""
Comprehensive Error Handling System for PTSD ML Platform
Provides centralized error handling, logging, and user-friendly error messages
"""

import traceback
import logging
import sys
from typing import Optional, Dict, Any, Callable
from functools import wraps
from enum import Enum
import streamlit as st
from datetime import datetime
import uuid

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NETWORK_ERROR = "network_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"

class PTSDError(Exception):
    """Base exception class for PTSD ML Platform"""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: str = None,
        details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.user_message = user_message or self._get_user_friendly_message()
        self.details = details or {}
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
    
    def _get_user_friendly_message(self) -> str:
        """Generate user-friendly error message"""
        if self.category == ErrorCategory.DATA_ERROR:
            return "There was an issue processing your data. Please check the data format and try again."
        elif self.category == ErrorCategory.MODEL_ERROR:
            return "The model encountered an error. Please try again or contact support."
        elif self.category == ErrorCategory.DATABASE_ERROR:
            return "Database connection issue. Please try again in a moment."
        elif self.category == ErrorCategory.VALIDATION_ERROR:
            return "The data provided doesn't meet the required format. Please check and try again."
        elif self.category == ErrorCategory.AUTHENTICATION_ERROR:
            return "Authentication failed. Please check your credentials."
        elif self.category == ErrorCategory.PERMISSION_ERROR:
            return "You don't have permission to perform this action."
        elif self.category == ErrorCategory.NETWORK_ERROR:
            return "Network connection issue. Please check your internet connection."
        else:
            return "An unexpected error occurred. Please try again or contact support."

class DataProcessingError(PTSDError):
    """Error in data processing operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA_ERROR, **kwargs)

class ModelError(PTSDError):
    """Error in model operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL_ERROR, **kwargs)

class DatabaseError(PTSDError):
    """Error in database operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE_ERROR, **kwargs)

class ValidationError(PTSDError):
    """Error in data validation"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION_ERROR, **kwargs)

class ErrorHandler:
    """
    Centralized error handling system with logging and user feedback
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or self._setup_logger()
        self.error_stats = {
            "total_errors": 0,
            "by_category": {},
            "by_severity": {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger('ptsd_ml_platform')
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def handle_error(
        self, 
        error: Exception, 
        show_user_message: bool = True,
        additional_context: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Handle errors with logging and user feedback
        
        Args:
            error: The exception that occurred
            show_user_message: Whether to show user-friendly message in Streamlit
            additional_context: Additional context for logging
            
        Returns:
            Error ID for tracking
        """
        # Convert to PTSDError if not already
        if not isinstance(error, PTSDError):
            ptsd_error = PTSDError(
                message=str(error),
                category=self._categorize_error(error),
                severity=self._determine_severity(error),
                details={"original_exception": type(error).__name__}
            )
        else:
            ptsd_error = error
        
        # Log the error
        self._log_error(ptsd_error, additional_context)
        
        # Update statistics
        self._update_error_stats(ptsd_error)
        
        # Show user message if requested
        if show_user_message:
            self._show_user_message(ptsd_error)
        
        return ptsd_error.error_id
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Database related errors
        if any(keyword in error_message for keyword in ['database', 'connection', 'sql', 'postgresql']):
            return ErrorCategory.DATABASE_ERROR
        
        # Data processing errors
        if any(keyword in error_message for keyword in ['data', 'pandas', 'numpy', 'csv', 'excel']):
            return ErrorCategory.DATA_ERROR
        
        # Model errors
        if any(keyword in error_message for keyword in ['model', 'sklearn', 'prediction', 'training']):
            return ErrorCategory.MODEL_ERROR
        
        # Validation errors
        if any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION_ERROR
        
        # Network errors
        if any(keyword in error_message for keyword in ['network', 'timeout', 'connection refused']):
            return ErrorCategory.NETWORK_ERROR
        
        return ErrorCategory.SYSTEM_ERROR
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on type and impact"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors that break the application
        if any(keyword in error_message for keyword in ['fatal', 'critical', 'system']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_message for keyword in ['security', 'permission', 'authentication']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors (default)
        if any(keyword in error_message for keyword in ['database', 'model', 'network']):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if any(keyword in error_message for keyword in ['validation', 'format', 'input']):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _log_error(self, error: PTSDError, additional_context: Dict[str, Any] = None):
        """Log error with structured information"""
        log_data = {
            "error_id": error.error_id,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "timestamp": error.timestamp.isoformat(),
            "details": error.details
        }
        
        if additional_context:
            log_data["context"] = additional_context
        
        # Log at appropriate level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical Error: {log_data}")
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High Severity Error: {log_data}")
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium Severity Error: {log_data}")
        else:
            self.logger.info(f"Low Severity Error: {log_data}")
    
    def _update_error_stats(self, error: PTSDError):
        """Update error statistics"""
        self.error_stats["total_errors"] += 1
        
        category = error.category.value
        severity = error.severity.value
        
        if category not in self.error_stats["by_category"]:
            self.error_stats["by_category"][category] = 0
        self.error_stats["by_category"][category] += 1
        
        if severity not in self.error_stats["by_severity"]:
            self.error_stats["by_severity"][severity] = 0
        self.error_stats["by_severity"][severity] += 1
    
    def _show_user_message(self, error: PTSDError):
        """Show user-friendly error message in Streamlit"""
        if error.severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 Critical Error: {error.user_message}")
            st.error(f"Error ID: {error.error_id} - Please contact support")
        elif error.severity == ErrorSeverity.HIGH:
            st.error(f"❌ Error: {error.user_message}")
            st.info(f"Error ID: {error.error_id}")
        elif error.severity == ErrorSeverity.MEDIUM:
            st.warning(f"⚠️ Warning: {error.user_message}")
        else:
            st.info(f"ℹ️ Notice: {error.user_message}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return self.error_stats.copy()

# Decorators for error handling
def handle_errors(
    error_handler: ErrorHandler = None,
    show_user_message: bool = True,
    reraise: bool = False
):
    """
    Decorator for automatic error handling
    
    Args:
        error_handler: Custom error handler instance
        show_user_message: Whether to show user-friendly messages
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = handler.handle_error(e, show_user_message)
                if reraise:
                    raise
                return None
        return wrapper
    return decorator

def handle_streamlit_errors(func: Callable) -> Callable:
    """
    Streamlit-specific error handling decorator
    Shows errors in Streamlit UI and provides user feedback
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PTSDError as e:
            error_handler = ErrorHandler()
            error_handler.handle_error(e)
            return None
        except Exception as e:
            error_handler = ErrorHandler()
            error_id = error_handler.handle_error(e)
            
            # Additional Streamlit-specific handling
            with st.expander("🔧 Technical Details", expanded=False):
                st.code(f"Error ID: {error_id}")
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.code(traceback.format_exc())
            
            return None
    return wrapper

# Context managers for error handling
class ErrorContext:
    """Context manager for handling errors in code blocks"""
    
    def __init__(
        self, 
        error_handler: ErrorHandler = None,
        show_user_message: bool = True,
        operation_name: str = "operation"
    ):
        self.error_handler = error_handler or ErrorHandler()
        self.show_user_message = show_user_message
        self.operation_name = operation_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_error(
                exc_val, 
                show_user_message=self.show_user_message,
                additional_context={"operation": self.operation_name}
            )
            return True  # Suppress the exception
        return False

# Global error handler instance
error_handler = ErrorHandler()

# Utility functions
def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Optional[str]]:
    """
    Safely execute a function and return result with error ID if failed
    
    Returns:
        Tuple of (result, error_id). error_id is None if successful.
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_id = error_handler.handle_error(e)
        return None, error_id

def create_error_report() -> Dict[str, Any]:
    """Create comprehensive error report for debugging"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "error_stats": error_handler.get_error_stats(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }