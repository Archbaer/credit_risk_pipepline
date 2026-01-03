from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union, Any
from pathlib import Path
import pandas as pd
import numpy as np

from main import training_step, predictions

app = FastAPI(title="Credit Default Prediction API")