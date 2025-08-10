"""
Enhanced Configuration Management System for PTSD ML Platform
Provides centralized configuration with environment-specific settings
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class MLConfig:
    """Machine learning configuration settings"""
    model_storage_path: str
    default_test_size: float = 0.2
    default_cv_folds: int = 5
    default_random_state: int = 42
    max_features: int = 1000
    cache_ttl: int = 3600

@dataclass
class AppConfig:
    """Application configuration settings"""
    app_name: str = "PTSD ML Platform"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    max_upload_size: int = 200  # MB
    session_timeout: int = 3600  # seconds
    enable_analytics: bool = True

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str
    enable_csrf: bool = True
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    password_min_length: int = 8
    session_cookie_secure: bool = True

class ConfigManager:
    """
    Centralized configuration management system
    Supports multiple environments and configuration sources
    """
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path(__file__).parent.parent.parent / 'config'
        self._config_cache = {}
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from multiple sources"""
        try:
            # Load base configuration
            base_config = self._load_yaml_config('base.yaml')
            
            # Load environment-specific configuration
            env_config = self._load_yaml_config(f'{self.env}.yaml')
            
            # Merge configurations (env overrides base)
            self._config_cache = self._merge_configs(base_config, env_config)
            
            # Override with environment variables
            self._load_env_overrides()
            
            self.logger.info(f"Configuration loaded for environment: {self.env}")
            
        except Exception as e:
            # Fallback to default configuration
            self._config_cache = self._get_default_config()
            self.logger.warning(f"Using default config due to error: {e}")
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / filename
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                self.logger.error(f"Error parsing YAML file {filename}: {e}")
                return {}
        else:
            self.logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'DATABASE_HOST': ('database', 'host'),
            'DATABASE_PORT': ('database', 'port'),
            'DATABASE_NAME': ('database', 'database'),
            'DATABASE_USER': ('database', 'username'),
            'DATABASE_PASSWORD': ('database', 'password'),
            'DATABASE_URL': ('database', 'url'),
            'SECRET_KEY': ('security', 'secret_key'),
            'DEBUG': ('app', 'debug'),
            'LOG_LEVEL': ('app', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config_cache:
                    self._config_cache[section] = {}
                
                # Convert string values to appropriate types
                if key in ['port', 'pool_size', 'max_overflow', 'pool_timeout']:
                    try:
                        value = int(value)
                    except ValueError:
                        self.logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                elif key in ['debug']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                self._config_cache[section][key] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration as fallback"""
        return {
            'app': {
                'name': 'PTSD ML Platform',
                'version': '2.0.0',
                'debug': False,
                'log_level': 'INFO',
                'max_upload_size': 200,
                'session_timeout': 3600,
                'enable_analytics': True
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'ptsd_ml',
                'username': 'ptsd_user',
                'password': '',
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600
            },
            'ml': {
                'model_storage_path': './models',
                'default_test_size': 0.2,
                'default_cv_folds': 5,
                'default_random_state': 42,
                'max_features': 1000,
                'cache_ttl': 3600
            },
            'security': {
                'secret_key': 'change-me-in-production',
                'enable_csrf': True,
                'enable_rate_limiting': True,
                'rate_limit_per_minute': 100,
                'password_min_length': 8,
                'session_cookie_secure': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'database.host')"""
        keys = key.split('.')
        value = self._config_cache
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration as dataclass"""
        db_config = self._config_cache.get('database', {})
        return DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'ptsd_ml'),
            username=db_config.get('username', 'ptsd_user'),
            password=db_config.get('password', ''),
            pool_size=db_config.get('pool_size', 10),
            max_overflow=db_config.get('max_overflow', 20),
            pool_timeout=db_config.get('pool_timeout', 30),
            pool_recycle=db_config.get('pool_recycle', 3600)
        )
    
    def get_ml_config(self) -> MLConfig:
        """Get ML configuration as dataclass"""
        ml_config = self._config_cache.get('ml', {})
        return MLConfig(
            model_storage_path=ml_config.get('model_storage_path', './models'),
            default_test_size=ml_config.get('default_test_size', 0.2),
            default_cv_folds=ml_config.get('default_cv_folds', 5),
            default_random_state=ml_config.get('default_random_state', 42),
            max_features=ml_config.get('max_features', 1000),
            cache_ttl=ml_config.get('cache_ttl', 3600)
        )
    
    def get_app_config(self) -> AppConfig:
        """Get app configuration as dataclass"""
        app_config = self._config_cache.get('app', {})
        return AppConfig(
            app_name=app_config.get('name', 'PTSD ML Platform'),
            app_version=app_config.get('version', '2.0.0'),
            debug=app_config.get('debug', False),
            log_level=app_config.get('log_level', 'INFO'),
            max_upload_size=app_config.get('max_upload_size', 200),
            session_timeout=app_config.get('session_timeout', 3600),
            enable_analytics=app_config.get('enable_analytics', True)
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration as dataclass"""
        sec_config = self._config_cache.get('security', {})
        return SecurityConfig(
            secret_key=sec_config.get('secret_key', 'change-me-in-production'),
            enable_csrf=sec_config.get('enable_csrf', True),
            enable_rate_limiting=sec_config.get('enable_rate_limiting', True),
            rate_limit_per_minute=sec_config.get('rate_limit_per_minute', 100),
            password_min_length=sec_config.get('password_min_length', 8),
            session_cookie_secure=sec_config.get('session_cookie_secure', False)
        )
    
    def get_database_url(self) -> str:
        """Get database URL for SQLAlchemy"""
        # Check for full DATABASE_URL first
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Handle Heroku-style postgres:// URLs
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            return database_url
        
        # Build URL from configuration
        db_config = self.get_database_config()
        return f"postgresql://{db_config.username}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"

# Global configuration instance
config = ConfigManager()

# Helper functions for backward compatibility
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key, default)

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.get_database_config()

def get_ml_config() -> MLConfig:
    """Get ML configuration"""
    return config.get_ml_config()

def get_app_config() -> AppConfig:
    """Get app configuration"""
    return config.get_app_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config.get_security_config()