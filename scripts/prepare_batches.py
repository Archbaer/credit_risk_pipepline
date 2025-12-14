import pandas as pd
import os
from pathlib import Path
from typing import Iterable, Union


def create_directories(paths: Iterable[Union[str, Path]]) -> None:
    """Ensure each path exists. Accepts str or Path."""
    if not paths:
        return
    
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def prepare_batches() -> None:
    """
    Splits the input DataFrame into 12 batches, which represents 12 months of the year and saves them as CSV files.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame containing the credit data.
    """
    data_path = os.path.join("..", "data", "raw", "credit_data.csv")
    df = pd.read_csv(data_path)

    create_directories([os.path.join("..", "data", "batches")])

    for i in range(1, 13):
        batch_path = os.path.join("..", "data", "batches", f"batch_2026_{i}.csv")
        temp_df = df[i-1::12].reset_index(drop=True)
        temp_df.to_csv(batch_path, index=False)
    

if __name__ == "__main__":
    prepare_batches()
    print("Batches prepared and saved successfully.")