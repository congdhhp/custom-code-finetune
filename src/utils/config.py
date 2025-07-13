"""Configuration utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
from omegaconf import OmegaConf


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = OmegaConf.create({})
    
    for config in configs:
        if config:
            merged = OmegaConf.merge(merged, OmegaConf.create(config))
    
    return OmegaConf.to_container(merged, resolve=True)


def save_config(config: Dict[str, Any], output_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that configuration contains required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys (supports nested keys with dots)
        
    Returns:
        True if all required keys are present
    """
    def get_nested_value(d: dict, key: str):
        """Get nested value using dot notation."""
        keys = key.split('.')
        value = d
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value
    
    for key in required_keys:
        if get_nested_value(config, key) is None:
            raise ValueError(f"Required configuration key missing: {key}")
    
    return True
