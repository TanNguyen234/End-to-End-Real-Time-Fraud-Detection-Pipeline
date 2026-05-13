# Fraud Classification System

A production-ready FastAPI service for detecting fraudulent credit card transactions based on an XGBoost model.

## Features
- **FastAPI Backend**: High-performance REST API for real-time inference.
- **XGBoost Classifier**: Trained to handle imbalanced datasets.
- **Robust Preprocessing**: Built-in scaling and feature engineering consistent with training.
- **Automated Tests**: Unit and integration tests for service reliability.

## Project Structure
- `app/`: FastAPI application source code.
- `scripts/`: Training and utility scripts.
- `models/`: Trained model artifacts and feature metadata.
- `tests/`: Pytest suite.
- `data/`: Local storage for training data.

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your `creditcard.csv` file in the `data/` directory.

3. **Train the Model**:
   ```bash
   python scripts/train.py
   ```
   This will generate `fraud_model.json`, `scaler.joblib`, and `features.json` in the `models/` directory.

## Running the API

Start the FastAPI server using `uvicorn`:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

### API Documentation
Once the server is running, you can access the interactive Swagger UI at:
`http://localhost:8000/docs`

## Usage Example

Send a POST request to `/predict`:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Time": 406.0,
  "V1": -2.31222654232745,
  "V2": 1.95199201064158,
  ...
  "Amount": 10.0
}'
```

## Running Tests

Run the test suite using `pytest`:
```bash
pytest tests/
```

## Environment Variables
Configurable via `.env` file:
- `MODEL_PATH`: Path to the XGBoost model.
- `SCALER_PATH`: Path to the RobustScaler artifact.
- `DATA_PATH`: Path to the CSV dataset for training.
