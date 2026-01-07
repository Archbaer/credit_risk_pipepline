from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from pipelines.__init__ import logger
import pandas as pd
import subprocess

from main import training_step, predictions

class PredictionRequest(BaseModel):
    data: Dict[str, Any]

def preprocessing_job():
    """
    Run the preprocessing script as a subprocess and log the output.

    Returns:
        subprocess.CompletedProcess or None: The result of the subprocess if successful, else None.
    """
    try: 
        logger.info("Starting preprocessing job...")
        result = subprocess.Popen(['bash', 'preprocess.sh'])
        
        if result.returncode != 0:
            logger.error(f"Preprocessing failed: {result.stderr}")
        logger.info("Preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise Exception

    return result if result.returncode == 0 else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions 
    app.state.script = preprocessing_job()
    training_step(
        config_path='config.yaml',
        data_path='data/preprocessed'
    )
    yield
    # Shutdown actions
    proc = app.state.script
    if proc:
        proc.terminate()
        try: 
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Preprocessing script did not terminate in time and was killed.")
            proc.kill()


app = FastAPI(title="Credit Default Prediction API")

@app.get("/predict")
def predict(request: PredictionRequest):
    try:
        
        input_data = pd.DataFrame([request.data])
        preds = predictions(input_data)
        return {"predictions": preds.tolist()}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))