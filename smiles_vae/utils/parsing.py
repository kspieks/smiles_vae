import os
import yaml


def read_yaml_file(path):
    """
    Read an input YAML file and return the parameters as a dictionary.

    Args:
        path (str): The input YAML file path to read.

    Returns: dict
        The content read from the file.
    """
    if not isinstance(path, str):
        raise ValueError(f'yaml file path must be a string, got {path} which is a {type(path)}')
    
    if not os.path.isfile(path):
        raise ValueError(f'Could not find the YAML file {path}')
    
    with open(path, 'r') as f:
        content = yaml.safe_load(f)

    return content


def write_yaml_file(path, content):
    """
    Write a YAML file.

    Args:
        path (str): The YAML file path to write.
        content (dict): Information to write.
    """
    with open(path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)
