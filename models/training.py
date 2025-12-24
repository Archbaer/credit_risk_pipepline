from sklearn import train_test_split
from pipelines.__init__ import logger
from pathlib import Path
from typing import Union 
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_split(data_path: Union[str, Path], test_size=0.2, random_state=42):
    """
    Load data from a CSV file and split it into training and testing sets.

    Args:
        data_path (Union[str, Path]): The path to the CSV file containing the data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training and testing features and labels.
    """
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    x = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Data loaded and split from {data_path}")

    return X_train, X_test, y_train, y_test

def model_training(X_train: np.ndarray, y_train: np.ndarray):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path: Union[str, Path]) -> None:
    """Save the trained model to the specified path.

    Args:
        model: The trained model to be saved.
        model_path (Union[str, Path]): The path where the model will be saved.
    """
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model at {model_path}: {e}")
        raise e