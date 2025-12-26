from sklearn import train_test_split
from pipelines.__init__ import logger
from pipelines.preprocessing import create_dir
import mlflow
from pathlib import Path
from typing import Union 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
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

def model_training(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, estimators: int = 100) -> None:
    """
    Train a Random Forest Classifier and log the model and metrics to MLflow.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        estimators (int): Number of trees in the Random Forest.
    
    Returns:
        None
    """
    mlflow.set_experiment("Credit_Risk_Classification_Training")

    run_name = f"RF_{estimators}_trees"

    with mlflow.start_run(run_name=run_name) as run:
        # Train the model
        model = RandomForestClassifier(n_estimators=estimators, random_state=42)
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Log parameters, metrics, and model to MLflow
        mlflow.log_param("n_estimators", estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_model(model, "rf_model")
        logger.info("Model training completed and logged to MLflow")

        # Save the scaler used during preprocessing
        mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
    
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/rf_model"
        mlflow.register_model(model_uri, "random_forest_model_registry")

def save_model(model, model_path: Union[str, Path]) -> None:
    """Save the trained model locally to the specified path.

    Args:
        model: The trained model to be saved.
        model_path (Union[str, Path]): The path where the model will be saved.
    """
    try:
        create_dir(Path(model_path).parent)
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model at {model_path}: {e}")
        raise e