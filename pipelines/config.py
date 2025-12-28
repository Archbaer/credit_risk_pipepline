import yaml
from pathlib import Path
from typing import Union, Dict

def load_yaml(file_path: Union[str, Path]) -> Dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path (Union[str, Path]): The path to the YAML file.
    
    Returns:
        Dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    return config

def flatten_dict(config_dict: Dict) -> Dict[str, Union[str, int, float, bool]]:
    """
    Flatten a nested configuration dictionary.

    Args:
        config_dict (Dict): The nested configuration dictionary.
    
    Returns:
        Dict[str, Union[str, int, float, bool]]: The flattened configuration dictionary.
    """
    flat_config = {}
    for section, params in config_dict.items():
        for key, value in params.items():
            flat_key = f"{section}.{key}"
            flat_config[flat_key] = value
    return flat_config

def load_config(file_path: Union[str, Path]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Load and flatten a YAML configuration file.

    Args:
        file_path (Union[str, Path]): The path to the YAML file.

    Returns:
        Dict[str, Union[str, int, float, bool]]: The flattened configuration dictionary.
    """

    config = load_yaml(file_path)
    return flatten_dict(config)

def get_specific_configs(flat_config: Dict[str, Union[str, int, float, bool]], prefix: str) -> Dict[str, Union[str, int, float, bool]]:
    """
    Extract specific configurations from the flattened configuration dictionary based on a prefix.

    Args:
        flat_config (Dict[str, Union[str, int, float, bool]]): The flattened configuration dictionary.
        prefix (str): The prefix to filter configurations.

    Returns:
        Dict[str, Union[str, int, float, bool]]: The filtered configuration dictionary.
    """
    specific_configs = {}
    for k, v in flat_config.items():
        if k.startswith(f"{prefix}."):
            specific_configs[k.replace(f"{prefix}.", "")] = v
    return specific_configs