import pandas as pd 
import numpy as np
import joblib
from typing import Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from pipelines.__init__ import logger

def create_dir(path: Union[str, Path]) -> None:
    """Ensure the directory at the given path exists. Accepts str or Path.
    
    Parameters:
    path (Union[str, Path]): The directory path to create. Accepts both str and Path types.
    """
    if not path:
        return
    logger.info(f"Creating directory at: {path}")
    Path(path).mkdir(parents=True, exist_ok=True)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    logger.info(f"Loading data from {file_path}")
    
    return pd.read_csv(file_path)

def preprocess_data(file_path: Union[str, Path], scaler_path: Union[str, Path], scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Preprocess the data in order to prepare it for model training or evaluation in further steps of the pipeline. Additionally, fits 
    and saves a StandardScaler for future use.

    Args:
        file_path (Union[str, Path]): The path to the CSV file containing the data.
        scaler_path (Union[str, Path]): The path to save the fitted StandardScaler.
    
    Returns:
        pd.DataFrame: The preprocessed data.
        StandardScaler: The fitted StandardScaler object.
    """
    try:
        data = load_data(file_path)
        logger.info(f"Preprocessing data at {file_path}")
        
        # Dropping duplicates and na
        data = data.dropna()
        data = data.drop_duplicates()

        data = data.drop(columns='ID')

        y = data['default.payment.next.month']
        x = data.drop(columns='default.payment.next.month')

        if scaler is None:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        else:
            x = scaler.transform(x)

        data = pd.DataFrame(x)
        data['target_y'] = y

        if scaler_path:
            create_dir(Path(scaler_path).parent)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved at {scaler_path}")

    except Exception as e:
        logger.error(f"Failed to preprocess data from {file_path}: {e}")
        raise e
    
    logger.info("Data preprocessing completed.")

    return data, scaler


def save_preprocessed_data(data: Union[np.array, pd.DataFrame], output_path: Union[str, Path]) -> None:
    """
    Save the preprocessed data to a CSV file.

    Args:
        data (pd.DataFrame): The preprocessed data.
        output_path (Union[str, Path]): The path to save the CSV file.
    """
    path = Path(output_path).parent
    create_dir(path)

    try:
        data = pd.DataFrame(data)
        data.to_csv(output_path, index=False, header=False)
        logger.info(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessed data to {output_path}: {e}")
        raise e