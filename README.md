# Credit Risk Pipeline

Production-ready pipeline to predict credit default risk with Random Forests, MLflow tracking, and a FastAPI serving layer.

## Highlights
- Batch preparation splits raw credit data into 12 monthly CSVs.
- Preprocessing standardizes features, persists a shared scaler, and writes processed batches.
- Training logs metrics and artifacts to MLflow and registers the best model.
- FastAPI endpoint wraps model inference for online predictions.

## Project Structure
```
credit_risk_pipepline/
├── app.py                  # FastAPI app exposing /predict
├── main.py                 # Orchestrates preprocessing, training, inference helpers
├── preprocess.sh           # Looping batch + preprocess runner
├── config/config.yaml      # Training hyperparameters
├── data/                   # raw, batches, processed data
├── models/                 # training.py, evaluate.py (MLflow integration)
├── pipelines/              # prepare_batches, preprocessing, config helpers
├── logs/                   # runtime logs
├── mlruns/                 # MLflow tracking store
├── deployment/             # Docker/Docker Compose stubs
└── test.ipynb              # exploratory notebook
```

## Quick Start

```bash
git clone https://github.com/Archbaer/credit_risk_pipepline.git
cd credit_risk_pipepline

python -m venv venv
# Windows

venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Data Layout
- data/raw: Source dataset (credit_data.csv)
- data/batches: Monthly splits produced by the batch prep step
- data/processed: Standardized batches with target in the last column

## Pipeline Workflow
1) Split raw data into monthly batches
```bash
python -c "from pipelines.prepare_batches import prepare_batches; prepare_batches()"
```
2) Preprocess all batches (fits/saves scaler.pkl on first batch, reuses afterward)
```bash
python -c "from main import preprocess_all_batches; preprocess_all_batches('data/batches', 'data/processed')"
```
3) Train and log models (uses config/config.yaml for RF hyperparameters)
```bash
python -c "from main import training_step; training_step('config/config.yaml', 'data/processed')"
```
4) Predict with the best registered model (by metric)
```python
import pandas as pd
from main import predictions

sample = pd.read_csv('data/raw/credit_data.csv').drop(columns=['ID']).head(1)
preds = predictions('roc_auc', sample)
print(preds)
```

## FastAPI Inference
Run the API locally:
```bash
uvicorn app:app --reload
```
Endpoint: /predict expects JSON with feature keys matching the training data. Example payload:
```json
{
  "data": {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
  }
}
```

## Configuration
- Training hyperparameters live in config/config.yaml (n_estimators, max_depth, etc.).
- Logs are written to logs/running_logs.log.
- MLflow artifacts (models and scaler) are tracked under mlruns/ and the model registry.

## Automation
- preprocess.sh continuously prepares and preprocesses batches on a monthly interval (bash script; use WSL on Windows).

## CI/CD Pipeline
This project uses GitHub Actions to automatically build and push Docker images to Docker Hub on every push to the `main` branch or pull request.

### Docker Hub Deployment
- The workflow is defined in `.github/workflows/ci.yml`
- Triggers on push or pull request to the `main` branch
- Automatically builds the Docker image and pushes it to Docker Hub as `archbaer/credit_risk_pipeline:latest`