import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'creditscope-secret-key-change-me')
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
    MODELS_UPLOAD_FOLDER = 'uploads/models'
    DATASETS_UPLOAD_FOLDER = 'uploads/datasets'
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_MODEL_EXTENSIONS = {'.h5', '.keras', '.joblib'}
    ALLOWED_DATASET_EXTENSIONS = {'.csv'}

# Helper to provide direct access if needed
config = Config()
