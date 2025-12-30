from pipelines.preprocessing import preprocess_data, save_preprocessed_data
from pathlib import Path
import glob

def preprocess_all_batches(batch_dir: str, processed_dir: str) -> None:
    """
    Preprocess all batch files in the specified directory and save the preprocessed data.

    Args:
        batch_dir (str): The directory containing the batch CSV files.
        processed_dir (str): The directory to save the preprocessed CSV files.

    Returns:
        None
    """

    file_list = glob.glob(f'{batch_dir}/*.csv')

    scaler = None
    count = 0

    for file_path in file_list:
        if count == 0:
            data, scaler = preprocess_data(
                file_path=file_path,
                scaler_path='scaler.pkl',
                scaler=None
            )
        else:
            data, scaler = preprocess_data(
                file_path=file_path,
                scaler_path='',  
                scaler=scaler
            )
        
        save_preprocessed_data(
            data=data,
            output_path=f"{processed_dir}/preprocessed_batch_{count}.csv"
        )
        count += 1

