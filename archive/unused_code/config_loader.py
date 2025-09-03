"""
Configuration Loader for Data Processing Pipeline
Loads and validates YAML configuration files for the romance novel data processing.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates configuration files for data processing."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        # Resolve config directory relative to project root
        if Path(config_dir).is_absolute():
            self.config_dir = Path(config_dir)
        else:
            # Find project root by looking for config directory
            current = Path.cwd()
            while current != current.parent:
                if (current / config_dir).exists():
                    self.config_dir = current / config_dir
                    break
                current = current.parent
            else:
                # Fallback to relative path
                self.config_dir = Path(config_dir)
        
        self.configs = {}
        
    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        Load a single configuration file.
        
        Args:
            filename: Name of the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration: {filename}")
                return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing {filename}: {e}")
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files.
        
        Returns:
            Dictionary containing all configurations
        """
        config_files = [
            "fields_required.yml",
            "sampling_policy.yml", 
            "variable_selection.yml",
            "csv_schema.yml",
            "data_processing_pipeline.yml"
        ]
        
        for filename in config_files:
            try:
                config_name = filename.replace('.yml', '').replace('.yaml', '')
                self.configs[config_name] = self.load_config(filename)
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                raise
                
        logger.info(f"Loaded {len(self.configs)} configuration files")
        return self.configs
    
    def get_fields_required(self) -> List[Dict[str, Any]]:
        """Get required fields configuration."""
        if 'fields_required' not in self.configs:
            self.configs['fields_required'] = self.load_config('fields_required.yml')
        # Return implemented_fields instead of fields
        return self.configs['fields_required']['implemented_fields']
    
    def get_sampling_policy(self) -> Dict[str, Any]:
        """Get sampling policy configuration."""
        if 'sampling_policy' not in self.configs:
            self.configs['sampling_policy'] = self.load_config('sampling_policy.yml')
        return self.configs['sampling_policy']
    
    def get_variable_selection(self) -> Dict[str, Any]:
        """Get variable selection configuration."""
        if 'variable_selection' not in self.configs:
            self.configs['variable_selection'] = self.load_config('variable_selection.yml')
        return self.configs['variable_selection']
    
    def get_csv_schema(self) -> Dict[str, Any]:
        """Get CSV schema configuration."""
        if 'csv_schema' not in self.configs:
            self.configs['csv_schema'] = self.load_config('csv_schema.yml')
        return self.configs['csv_schema']
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get data processing pipeline configuration."""
        if 'data_processing_pipeline' not in self.configs:
            self.configs['data_processing_pipeline'] = self.load_config('data_processing_pipeline.yml')
        return self.configs['data_processing_pipeline']
    
    def validate_configs(self) -> bool:
        """
        Validate that all required configurations are present and valid.
        
        Returns:
            True if all validations pass
        """
        required_configs = [
            'fields_required',
            'sampling_policy',
            'variable_selection', 
            'csv_schema',
            'data_processing_pipeline'
        ]
        
        for config_name in required_configs:
            if config_name not in self.configs:
                logger.error(f"Missing required configuration: {config_name}")
                return False
                
        # Validate specific configurations
        try:
            self._validate_fields_required()
            self._validate_sampling_policy()
            self._validate_pipeline_config()
            logger.info("All configuration validations passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_fields_required(self):
        """Validate fields_required configuration."""
        fields_config = self.configs['fields_required']
        if 'implemented_fields' not in fields_config:
            raise ValueError("fields_required.yml must contain 'implemented_fields' section")
            
        for field in fields_config['implemented_fields']:
            required_keys = ['name', 'dtype', 'required']
            for key in required_keys:
                if key not in field:
                    raise ValueError(f"Field missing required key '{key}': {field}")
    
    def _validate_sampling_policy(self):
        """Validate sampling_policy configuration."""
        policy = self.configs['sampling_policy']
        required_sections = ['quality_filters', 'author_limits', 'decade_stratification']
        
        for section in required_sections:
            if section not in policy:
                raise ValueError(f"sampling_policy.yml missing required section: {section}")
    
    def _validate_pipeline_config(self):
        """Validate pipeline configuration."""
        pipeline = self.configs['data_processing_pipeline']
        required_sections = ['steps', 'quality_thresholds', 'output_files']
        
        for section in required_sections:
            if section not in pipeline:
                raise ValueError(f"data_processing_pipeline.yml missing required section: {section}")


def load_configs(config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to load all configurations.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Dictionary containing all configurations
    """
    loader = ConfigLoader(config_dir)
    configs = loader.load_all_configs()
    
    if not loader.validate_configs():
        raise ValueError("Configuration validation failed")
        
    return configs
