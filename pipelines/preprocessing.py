import pandas as pd 
import numpy as np
from typing import Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from __init__ import logger

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    logger.info(f"Loading data from {file_path}")
    
    return pd.read_csv(file_path)

def preprocess_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Preprocess the data in order to prepare it for model training or evaluation in further steps.

    Args:
        file_path (Union[str, Path]): The path to the CSV file containing the data.
    
    Returns:
        pd.DataFrame: The preprocessed data.
    """

    data = load_data(file_path)
    data = data.drop(columns=['ID'])

    data = data.dropna()
    data = data.drop_duplicates()

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data

def save_preprocessed_data(data: np.array, output_path: Union[str, Path]) -> None:
    """
    Save the preprocessed data to a CSV file.

    Args:
        data (pd.DataFrame): The preprocessed data.
        output_path (Union[str, Path]): The path to save the CSV file.
    """
    try:
        data = pd.DataFrame(data)
        data.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessed data to {output_path}: {e}")
        raise e