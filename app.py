import subprocess
from contextlib import asynccontextmanager
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from main import predictions, training_step
from pipelines.__init__ import logger

class PredictionRequest(BaseModel):
    data: Dict[str, int | float]
    
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

@app.post("/predict")
def predict(request: PredictionRequest, response: Response):
    try:
        input_data = pd.DataFrame([request.model_dump()])
        preds = predictions(input_data)

        response.status_code = status.HTTP_200_OK
        return {"predictions": preds.tolist()}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/health")
def health_check(response: Response):
    try:
        response.status_code = status.HTTP_200_OK
        return {"status": "API is running"}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Service Unavailable")