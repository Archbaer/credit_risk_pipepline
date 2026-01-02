import tempfile
from typing import Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipelines.__init__ import logger

def get_best_model(metric_name: str) -> RandomForestClassifier:
    """
    Retrieve the best model from an MLflow experiment based on a specified metric.

    Parameters:
    - metric_name (str): The metric to evaluate the models.

    Returns:
    - best_model (RandomForestClassifier): The best model based on the specified metric.
    - scaler: The associated scaler used during preprocessing.
    """
    client = mlflow.tracking.MlflowClient()
    all_versions = client.search_model_versions()

    if not all_versions:
        logger.warning(f"No runs found'.")
        raise ValueError(f"No runs found'.")

    logger.info(f"Fetching best model from experiment  based on metric: {metric_name}")

    max_metric = 0
    best_model_version = None

    for version in all_versions:
        metrics = client.get_run(version.run_id).data.metrics
        metric = metrics.get(metric_name, 0)
        if metric > max_metric:
            max_metric = metric
            best_model_version = version    

    with tempfile.TemporaryDirectory() as temp_dir:
        client.download_artifacts(best_model_version.run_id, "preprocessing/scaler.pkl", dst_path=temp_dir)
        scaler = joblib.load(f"{temp_dir}/scaler.pkl")
    
    best_model_uri = f"models:/random_forest_model_registry/{best_model_version.version}"
    best_model = mlflow.sklearn.load_model(model_uri=best_model_uri)
    
    return best_model, scaler

def make_prediction(model: RandomForestClassifier, scaler, input_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Make predictions using the provided model and input data.

    Parameters:
    - model (RandomForestClassifier): The trained model to use for predictions.
    - input_data: The input data for making predictions.

    Returns:
    - predictions: The predictions made by the model.
    """
    if not isinstance(input_data, (pd.DataFrame, np.ndarray)):
        logger.error("Input data must be a pandas DataFrame or a numpy ndarray.")
        raise ValueError("Input data must be a pandas DataFrame or a numpy ndarray.")
    elif not scaler:
        logger.error("Scaler must be provided for data preprocessing.")
        raise ValueError("Scaler must be provided for data preprocessing.")

    # Scale the input data with the same scaler used during training for consistency.
    scaled_data = scaler.transform(input_data)

    # Make predictions using the trained model.
    predictions = model.predict(scaled_data)

    return predictions