import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import logging
from sklearn.metrics import classification_report, average_precision_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Apply preprocessing steps from the notebook.
    """
    logger.info("Preprocessing data...")
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Time transformation: convert seconds to hour of day
    df['Time'] = (df['Time'] / 3600) % 24
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Amount scaling
    scaler = RobustScaler()
    X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    
    return X, y, scaler

def train_model(X_train, y_train):
    """
    Train XGBoost model with best parameters found in notebook.
    """
    logger.info("Training XGBoost model...")
    
    # Calculate scale_pos_weight for imbalance
    weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Best parameters from notebook
    params = {
        'subsample': 0.8,
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'scale_pos_weight': weight,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def main():
    data_path = os.getenv("DATA_PATH", "data/creditcard.csv")
    models_dir = "models"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please place creditcard.csv in the data/ directory.")
        return

    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    X, y, scaler = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate briefly
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info("\n" + classification_report(y_test, y_pred))
    pr_auc = average_precision_score(y_test, y_pred_proba)
    logger.info(f"PR-AUC Score: {pr_auc:.4f}")
    # Save artifacts
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, "fraud_model.json")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    
    logger.info(f"Saving model to {model_path}...")
    model.save_model(model_path)
    
    logger.info(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    # Save feature names for consistency
    feature_names_path = os.path.join(models_dir, "features.json")
    import json
    with open(feature_names_path, 'w') as f:
        json.dump(X.columns.tolist(), f)
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
