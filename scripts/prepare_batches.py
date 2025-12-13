import pandas as pd
import os
from pathlib import Path

data_path = os.path.join("..", "data", "raw", "credit_data.csv")

def create_directories(paths: Iterable[Union[str, Path]]) -> None:
    """Ensure each path exists. Accepts str or Path."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created at: {Path(p)}")

def prepare_batches(data: pd.DataFrame) -> None:
    df = pd.read_csv(data_path)

    for i in range(1, 13):

    pass