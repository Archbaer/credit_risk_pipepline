import pandas as pd 
from typing import Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Preprocess the data in order to prepare it for model training or evaluation in further steps.

    Args:
        file_path (Union[str, Path]): The path to the CSV file containing the data.
    
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # Load the data
    # data = load_data(file_path)
    
    # # Initialize the scaler
    # scaler = StandardScaler()
    
    
    # return data