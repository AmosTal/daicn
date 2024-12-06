import os
import json
from typing import Dict, Any, Optional
from enum import Enum, auto

class ConfigurationMode(Enum):
    """Possible configuration modes for the application"""
    DEVELOPMENT = auto()
    TESTING = auto()
    PRODUCTION = auto()

class MinimalConfigurationManager:
    """
    Minimal configuration management system
    
    Key Design Principles:
    - Simple configuration loading
    - Environment-based configuration
    - Minimal external dependencies
    - Secure default settings
    """
    
    DEFAULT_CONFIG = {
        'application': {
            'name': 'DAICN',
            'version': '0.1.0',
            'mode': ConfigurationMode.DEVELOPMENT.name
        },
        'system': {
            'max_concurrent_tasks': 10,
            'task_timeout_minutes': 5,
            'log_level': 'INFO'
        },
        'security': {
            'password_min_length': 8,
            'max_login_attempts': 5,
            'token_expiry_hours': 24
        },
        'ml': {
            'complexity_prediction_threshold': 0.5,
            'performance_history_size': 1000
        },
        'resources': {
            'cpu_allocation_percentage': 60,
            'memory_allocation_percentage': 70
        }
    }
    
    def __init__(
        self, 
        config_file: Optional[str] = None,
        mode: Optional[ConfigurationMode] = None
    ):
        """
        Initialize configuration manager
        
        Args:
            config_file (Optional[str]): Path to custom configuration file
            mode (Optional[ConfigurationMode]): Deployment mode
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Determine configuration mode
        self.mode = mode or self._detect_mode()
        self.config['application']['mode'] = self.mode.name
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # Override with environment variables
        self._load_env_config()
    
    def _detect_mode(self) -> ConfigurationMode:
        """
        Detect deployment mode based on environment
        
        Returns:
            ConfigurationMode: Detected configuration mode
        """
        env_mode = os.environ.get('DAICN_MODE', 'DEVELOPMENT').upper()
        try:
            return ConfigurationMode[env_mode]
        except KeyError:
            return ConfigurationMode.DEVELOPMENT
    
    def _load_config_file(self, config_file: str):
        """
        Load configuration from a JSON file
        
        Args:
            config_file (str): Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._deep_update(self.config, file_config)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file. Using defaults. Error: {e}")
    
    def _load_env_config(self):
        """
        Override configuration with environment variables
        
        Supports nested configuration via __
        Example: DAICN_SYSTEM__MAX_CONCURRENT_TASKS=15
        """
        def _update_nested(config: Dict, env_key: str, value: str):
            keys = env_key.lower().split('__')
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = self._convert_type(value)
        
        for key, value in os.environ.items():
            if key.startswith('DAICN_'):
                config_key = key[6:]  # Remove 'DAICN_' prefix
                _update_nested(self.config, config_key, value)
    
    def _convert_type(self, value: str) -> Any:
        """
        Convert string to appropriate type
        
        Args:
            value (str): Input value
        
        Returns:
            Any: Converted value
        """
        value = value.lower()
        if value in ['true', 'yes', '1']:
            return True
        elif value in ['false', 'no', '0']:
            return False
        
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def _deep_update(self, original: Dict, update: Dict):
        """
        Recursively update nested dictionaries
        
        Args:
            original (Dict): Original configuration
            update (Dict): Configuration to update with
        """
        for key, value in update.items():
            if isinstance(value, dict):
                original[key] = self._deep_update(
                    original.get(key, {}), 
                    value
                )
            else:
                original[key] = value
        return original
    
    def get(
        self, 
        key: str, 
        default: Optional[Any] = None
    ) -> Any:
        """
        Retrieve configuration value
        
        Args:
            key (str): Dot-separated configuration key
            default (Optional[Any]): Default value if key not found
        
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def save_config(self, output_file: str):
        """
        Save current configuration to a JSON file
        
        Args:
            output_file (str): Path to save configuration
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except IOError as e:
            print(f"Error saving configuration: {e}")
    
    def __repr__(self) -> str:
        """
        String representation of configuration
        
        Returns:
            str: Configuration summary
        """
        return f"DAICN Configuration (Mode: {self.mode.name})"

def main():
    """
    Demonstration of Minimal Configuration Manager
    """
    # Basic usage
    config = MinimalConfigurationManager()
    
    # Retrieve configuration values
    print("Application Name:", config.get('application.name'))
    print("Max Concurrent Tasks:", config.get('system.max_concurrent_tasks'))
    
    # Save current configuration
    config.save_config('daicn_config.json')
    
    # Load from a specific file
    custom_config = MinimalConfigurationManager('daicn_config.json')
    print("Custom Config:", custom_config)

if __name__ == '__main__':
    main()
