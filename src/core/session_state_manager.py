"""
Enhanced Session State Manager for PTSD ML Platform
Provides centralized state management with persistence, validation, and cleanup
"""

import json
import uuid
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import streamlit as st
import hashlib

class StateType(Enum):
    """Types of state data for better organization"""
    USER_DATA = "user_data"
    MODEL_DATA = "model_data"
    APP_STATE = "app_state"
    CACHE_DATA = "cache_data"
    TEMP_DATA = "temp_data"

@dataclass
class StateMetadata:
    """Metadata for state entries"""
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    state_type: StateType = StateType.APP_STATE
    description: str = ""
    is_sensitive: bool = False

class SessionStateManager:
    """
    Enhanced session state management with validation, persistence, and cleanup
    """
    
    def __init__(self):
        self.session_id = self._get_or_create_session_id()
        self.metadata_key = "_session_metadata"
        self.cleanup_interval = timedelta(hours=1)
        self.default_expiry = timedelta(hours=24)
        
        # Initialize metadata if not exists
        if self.metadata_key not in st.session_state:
            st.session_state[self.metadata_key] = {}
        
        # Perform cleanup on initialization
        self._cleanup_expired_state()
    
    def _get_or_create_session_id(self) -> str:
        """Get or create a unique session ID"""
        session_id_key = "_session_id"
        
        if session_id_key not in st.session_state:
            st.session_state[session_id_key] = str(uuid.uuid4())
        
        return st.session_state[session_id_key]
    
    def _get_metadata(self, key: str) -> Optional[StateMetadata]:
        """Get metadata for a state key"""
        metadata_dict = st.session_state.get(self.metadata_key, {}).get(key)
        if metadata_dict:
            return StateMetadata(**metadata_dict)
        return None
    
    def _set_metadata(self, key: str, metadata: StateMetadata):
        """Set metadata for a state key"""
        if self.metadata_key not in st.session_state:
            st.session_state[self.metadata_key] = {}
        
        st.session_state[self.metadata_key][key] = asdict(metadata)
    
    def _cleanup_expired_state(self):
        """Clean up expired state entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        metadata = st.session_state.get(self.metadata_key, {})
        
        for key, meta_dict in metadata.items():
            if 'expires_at' in meta_dict and meta_dict['expires_at']:
                try:
                    expires_at = datetime.fromisoformat(meta_dict['expires_at'])
                    if expires_at < current_time:
                        expired_keys.append(key)
                except:
                    pass
        
        # Remove expired keys
        for key in expired_keys:
            self.remove(key)
    
    def set(
        self,
        key: str,
        value: Any,
        state_type: StateType = StateType.APP_STATE,
        description: str = "",
        expires_in: Optional[timedelta] = None,
        is_sensitive: bool = False,
        validate_func: Optional[Callable[[Any], bool]] = None
    ) -> bool:
        """
        Set a value in session state with metadata
        """
        try:
            # Validate value if validator provided
            if validate_func and not validate_func(value):
                return False
            
            # Calculate expiration time
            expires_at = None
            if expires_in:
                expires_at = datetime.utcnow() + expires_in
            elif state_type == StateType.TEMP_DATA:
                expires_at = datetime.utcnow() + timedelta(minutes=30)
            elif state_type == StateType.CACHE_DATA:
                expires_at = datetime.utcnow() + self.default_expiry
            
            # Create metadata
            now = datetime.utcnow()
            metadata = StateMetadata(
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                state_type=state_type,
                description=description,
                is_sensitive=is_sensitive
            )
            
            # Store value and metadata
            st.session_state[key] = value
            self._set_metadata(key, metadata)
            
            return True
            
        except Exception:
            return False
    
    def get(self, key: str, default: Any = None, check_expiry: bool = True) -> Any:
        """Get a value from session state with expiry checking"""
        if key not in st.session_state:
            return default
        
        if check_expiry:
            metadata = self._get_metadata(key)
            if metadata and metadata.expires_at:
                if datetime.utcnow() > metadata.expires_at:
                    self.remove(key)
                    return default
        
        return st.session_state.get(key, default)
    
    def remove(self, key: str) -> bool:
        """Remove a key from session state"""
        try:
            # Remove value
            if key in st.session_state:
                del st.session_state[key]
            
            # Remove metadata
            if self.metadata_key in st.session_state and key in st.session_state[self.metadata_key]:
                del st.session_state[self.metadata_key][key]
            
            return True
            
        except Exception:
            return False
    
    def exists(self, key: str, check_expiry: bool = True) -> bool:
        """Check if a key exists in session state"""
        if key not in st.session_state:
            return False
        
        if check_expiry:
            metadata = self._get_metadata(key)
            if metadata and metadata.expires_at:
                if datetime.utcnow() > metadata.expires_at:
                    self.remove(key)
                    return False
        
        return True
    
    def get_all_keys(self, state_type: Optional[StateType] = None) -> List[str]:
        """Get all keys of a specific state type"""
        if state_type is None:
            return [key for key in st.session_state.keys() if not key.startswith('_')]
        
        keys = []
        metadata = st.session_state.get(self.metadata_key, {})
        
        for key, meta_dict in metadata.items():
            if meta_dict.get('state_type') == state_type.value:
                if key in st.session_state:
                    keys.append(key)
        
        return keys
    
    def clear_by_type(self, state_type: StateType) -> int:
        """Clear all state entries of a specific type"""
        keys_to_remove = self.get_all_keys(state_type)
        
        for key in keys_to_remove:
            self.remove(key)
        
        return len(keys_to_remove)
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about current session state"""
        metadata = st.session_state.get(self.metadata_key, {})
        
        # Count by type
        type_counts = {}
        expired_count = 0
        sensitive_count = 0
        current_time = datetime.utcnow()
        
        for key, meta_dict in metadata.items():
            state_type = meta_dict.get('state_type', 'unknown')
            type_counts[state_type] = type_counts.get(state_type, 0) + 1
            
            if meta_dict.get('is_sensitive'):
                sensitive_count += 1
            
            if meta_dict.get('expires_at'):
                try:
                    expires_at = datetime.fromisoformat(meta_dict['expires_at'])
                    if expires_at < current_time:
                        expired_count += 1
                except:
                    pass
        
        return {
            "session_id": self.session_id,
            "total_keys": len([k for k in st.session_state.keys() if not k.startswith('_')]),
            "type_counts": type_counts,
            "expired_count": expired_count,
            "sensitive_count": sensitive_count,
            "metadata_size": len(metadata)
        }

# Global instances
session_manager = SessionStateManager()

# Utility functions for backward compatibility
def initialize_session_state():
    """Initialize default session state values"""
    defaults = {
        'app_initialized': (True, StateType.APP_STATE),
        'debug_mode': (False, StateType.USER_DATA),
        'current_page': ('home', StateType.APP_STATE),
        'user_id': (str(uuid.uuid4()), StateType.USER_DATA),
        'data_loaded': (False, StateType.APP_STATE),
        'models_trained': (False, StateType.APP_STATE),
    }
    
    for key, (default_value, state_type) in defaults.items():
        if not session_manager.exists(key):
            session_manager.set(key, default_value, state_type=state_type)

def get_session_info():
    """Get session information for debugging"""
    return session_manager.get_state_info()

def cleanup_session():
    """Manual cleanup of expired session state"""
    session_manager._cleanup_expired_state()

# Initialize session state on import
initialize_session_state()