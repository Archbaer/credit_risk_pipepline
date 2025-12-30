import pandas as pd
import os
from pathlib import Path
from typing import Iterable, Union

def create_directories(paths: Iterable[Union[str, Path]]) -> None:
    """Ensure each path exists. Accepts str or Path.
    
    
    
    Parameters:
    paths (Iterable[Union[str, Path]]): An iterable of directory paths to create. Accepts both str and Path types.
    """
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
    batch_dir = "data/batches"
    df = pd.read_csv("data/raw/credit_data.csv")

    create_directories([batch_dir])

    # 12 batches for 12 months
    for i in range(1, 13):
        batch_path = os.path.join(batch_dir, f"2026_{i}.csv")
        temp_df = df[i-1::12].reset_index(drop=True)
        temp_df.to_csv(batch_path, index=False)    
