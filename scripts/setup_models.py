import os
import gdown
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    # Google Drive folder ID
    folder_id = '10gQT4JHAK0o7s_kVj2snaDKVMw3Sp0U2'
    output_dir = 'models'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    # Check if files already exist
    required_files = ['fraud_model.json', 'scaler.joblib', 'features.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
    
    if not missing_files:
        logger.info("All model files already exist. Skipping download.")
        return

    logger.info(f"Downloading models from Google Drive folder: {folder_id}")
    try:
        # Download the entire folder
        # gdown will handle merging into existing directory
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        # If folder download fails, try specific files if needed or raise
        raise

if __name__ == "__main__":
    download_models()
