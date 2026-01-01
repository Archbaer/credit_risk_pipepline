import glob
from pathlib import Path
from typing import Union

from models.training import load_and_split, model_training
from pipelines.__init__ import logger
from pipelines.config import load_config, get_specific_configs
from pipelines.preprocessing import preprocess_data, save_preprocessed_data

def preprocess_all_batches(batch_dir: Union[str, Path], processed_dir: Union[str, Path]) -> None:
    """
    Preprocess all batch files in the specified directory and save the preprocessed data.

    Args:
        batch_dir (Union[str, Path]): The directory containing the batch CSV files.
        processed_dir (Union[str, Path]): The directory to save the preprocessed CSV files.

    Returns:
        None
    """

    file_list = glob.glob(f'{batch_dir}/*.csv')

    scaler = None
    count = 1 

    for file_path in file_list:
        if count == 1:
            data, scaler = preprocess_data(
                file_path=file_path,
                scaler_path='scaler.pkl',
                scaler=None
            )
        else:
            data, _ = preprocess_data(
                file_path=file_path,
                scaler_path='',  
                scaler=scaler
            )
        
        save_preprocessed_data(
            data=data,
            output_path=f"{processed_dir}/preprocessed_batch_{count}.csv"
        )
        count += 1

def training_step(config_path: Union[str, Path], data_path: Union[str, Path]) -> None:
    """
    Execute the training step of the pipeline using configurations from a YAML file.

    Args:
        config_path (Union[str, Path]): The path to the YAML configuration file.
        data_path (Union[str, Path]): The path to the preprocessed data CSV file.

    Returns:
        None
    """

    config = load_config(config_path)
    training_params = get_specific_configs(config, prefix='training')

    files = glob.glob(f'{data_path}/*.csv')

    for file in files: 
        X_train, X_test, y_train, y_test = load_and_split(file)

        model_training(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=training_params
        )