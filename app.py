import os
import uuid
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from ml_engine import ModelManager
from risk_analyzer import RiskAnalyzer
import re

app = Flask(__name__)
app.config.from_object(Config)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize components
model_manager = ModelManager(Config.MODELS_UPLOAD_FOLDER)
risk_analyzer = RiskAnalyzer()

# In-memory storage for current dataset
current_dataset = {
    'dataframe': None,
    'filename': None,
    'uploaded_at': None
}

# Simple User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

admin_user = User('1', Config.ADMIN_USERNAME)

@login_manager.user_loader
def load_user(user_id):
    if user_id == '1':
        return admin_user
    return None

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    return redirect(url_for('login'))


# --- Page Routes ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)


# --- API Routes ---

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'Missing JSON data'}), 400
        
        username = data.get('username')
        password = data.get('password')

        if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
            login_user(admin_user, remember=True)
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    try:
        logout_user()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/auth/status', methods=['GET'])
def api_auth_status():
    try:
        if current_user.is_authenticated:
            return jsonify({'authenticated': True, 'username': current_user.username})
        return jsonify({'authenticated': False, 'username': None})
    except Exception as e:
        return jsonify({'authenticated': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
@login_required
def api_get_models():
    try:
        models = model_manager.get_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/models/upload', methods=['POST'])
@login_required
def api_models_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})
        
        file = request.files['file']
        model_type = request.form.get('model_type')
        model_name = request.form.get('model_name')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
            
        if not model_type or model_type not in ['RNN', 'ANN', 'MLP']:
            return jsonify({'success': False, 'message': 'Invalid model type. Must be RNN, ANN, or MLP'})
            
        if not file.filename.endswith(('.h5', '.keras')):
            return jsonify({'success': False, 'message': 'Invalid file extension. Must be .h5 or .keras'})
            
        result = model_manager.upload_model(file, model_type, model_name)
        return jsonify({'success': True, 'model': result, 'message': 'Model uploaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/models/select', methods=['POST'])
@login_required
def api_models_select():
    try:
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        model_manager.select_models(model_ids)
        return jsonify({'success': True, 'message': 'Model selection updated'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
@login_required
def api_models_delete(model_id):
    try:
        success = model_manager.delete_model(model_id)
        if success:
            return jsonify({'success': True, 'message': 'Model deleted successfully'})
        return jsonify({'success': False, 'message': 'Model not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/dataset/upload', methods=['POST'])
@login_required
def api_dataset_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
            
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'message': 'Invalid file extension. Must be .csv'})
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.DATASETS_UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        if 'SK_ID_CURR' not in df.columns:
            os.remove(file_path)
            return jsonify({'success': False, 'message': 'CSV must contain SK_ID_CURR column'})
            
        current_dataset['dataframe'] = df
        current_dataset['filename'] = filename
        current_dataset['uploaded_at'] = pd.Timestamp.now().isoformat()
        
        return jsonify({
            'success': True,
            'info': {
                'filename': filename,
                'rows': len(df),
                'columns': df.columns.tolist()
            },
            'message': 'Dataset uploaded successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/dataset/info', methods=['GET'])
@login_required
def api_dataset_info():
    try:
        df = current_dataset.get('dataframe')
        if df is not None:
            sample_ids = df['SK_ID_CURR'].head(10).tolist()
            # convert any numpy ints to python ints
            sample_ids = [int(x) for x in sample_ids]
            return jsonify({
                'has_dataset': True,
                'info': {
                    'filename': current_dataset['filename'],
                    'rows': len(df),
                    'columns': len(df.columns),
                    'sample_ids': sample_ids
                }
            })
        return jsonify({'has_dataset': False})
    except Exception as e:
        return jsonify({'has_dataset': False, 'error': str(e)}), 500

@app.route('/api/dataset/applicant/<applicant_id>', methods=['GET'])
@login_required
def api_dataset_applicant(applicant_id):
    try:
        df = current_dataset.get('dataframe')
        if df is None:
            return jsonify({'success': False, 'message': 'No dataset loaded'})
            
        applicant_id = int(applicant_id)
        row = df[df['SK_ID_CURR'] == applicant_id]
        
        if row.empty:
            return jsonify({'success': False, 'message': 'Applicant not found'})
            
        row_dict = row.iloc[0].to_dict()
        # Handle NaN values
        clean_dict = {k: (v if pd.notna(v) else None) for k, v in row_dict.items()}
        
        return jsonify({'success': True, 'data': clean_dict})
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid applicant ID format'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        data = request.get_json()
        applicant_id = data.get('applicant_id')
        
        if not applicant_id:
            return jsonify({'success': False, 'message': 'Missing applicant_id'})
            
        df = current_dataset.get('dataframe')
        if df is None:
            return jsonify({'success': False, 'message': 'No dataset loaded'})
            
        applicant_id = int(applicant_id)
        row = df[df['SK_ID_CURR'] == applicant_id]
        
        if row.empty:
            return jsonify({'success': False, 'message': 'Applicant not found'})
            
        if not model_manager.get_selected_models():
            return jsonify({'success': False, 'message': 'No models selected for prediction'})
            
        # Extract features
        features_df = row.drop(columns=['SK_ID_CURR'])
        
        # Convert to numeric, coerce errors, fill NaN with 0
        features_numeric = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Convert to numpy array
        features_array = features_numeric.to_numpy()
        
        predictions = model_manager.predict_all_selected(features_array)
        
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'type': 'error', 'response': 'Empty message received'})
            
        # Parse intent
        id_match = re.search(r'\b\d+\b', message)
        message_lower = message.lower()
        
        if id_match:
            applicant_id_str = id_match.group(0)
            
            df = current_dataset.get('dataframe')
            if df is None:
                return jsonify({
                    'type': 'error',
                    'response': 'A dataset must be uploaded before making predictions.'
                })
                
            applicant_id = int(applicant_id_str)
            row = df[df['SK_ID_CURR'] == applicant_id]
            
            if row.empty:
                return jsonify({
                    'type': 'error',
                    'response': f'Applicant ID {applicant_id} not found in the current dataset.'
                })
                
            selected_models = model_manager.get_selected_models()
            if not selected_models:
                return jsonify({
                    'type': 'error',
                    'response': 'Please select at least one active model in the Models view.'
                })
                
            # Extract features and predict
            features_df = row.drop(columns=['SK_ID_CURR'])
            features_numeric = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            features_array = features_numeric.to_numpy()
            
            predictions = model_manager.predict_all_selected(features_array)
            
            if not predictions:
                return jsonify({
                    'type': 'error',
                    'response': 'Prediction failed or no predictions returned from models.'
                })
                
            # Calculate average probability
            avg_probability = sum(p['probability'] for p in predictions) / len(predictions)
            
            # Prepare clean applicant data dict
            row_dict = row.iloc[0].to_dict()
            clean_dict = {k: (v if pd.notna(v) else None) for k, v in row_dict.items()}
            
            # Analyze risk
            risk_factors = risk_analyzer.analyze_risk_factors(clean_dict)
            risk_level = risk_analyzer.get_risk_level(avg_probability)
            assessment_text = risk_analyzer.generate_assessment(
                clean_dict, predictions, risk_factors, risk_level,
                api_key=Config.GEMINI_API_KEY
            )
            
            return jsonify({
                'type': 'prediction',
                'response': assessment_text,
                'applicant_id': applicant_id,
                'applicant_data': clean_dict,
                'predictions': predictions,
                'avg_probability': avg_probability,
                'risk_level': risk_level,
                'risk_factors': risk_factors
            })
            
        else:
            # Info or general queries
            if 'hello' in message_lower or 'hi ' in message_lower or message_lower == 'hi':
                response = "Hello! I am the CreditScope assistant. Please enter an applicant ID (SK_ID_CURR) to assess their credit risk."
            elif 'help' in message_lower:
                response = "To use this chatbot, simply type or paste an applicant ID (e.g. 100002) from your uploaded dataset. I will run the active models and provide a risk assessment."
            elif 'dataset' in message_lower or 'model' in message_lower:
                models = model_manager.get_models()
                active = sum(1 for m in models if m.get('is_selected', False))
                has_data = current_dataset.get('dataframe') is not None
                data_status = f"Dataset is {'loaded' if has_data else 'NOT loaded'}."
                model_status = f"You have {active} active models out of {len(models)} uploaded."
                response = f"System status: {data_status} {model_status}"
            else:
                response = "I didn't detect an applicant ID in your message. Please provide a valid numerical applicant ID to perform a risk assessment."
                
            return jsonify({
                'type': 'info',
                'response': response,
                'applicant_id': None,
                'applicant_data': None,
                'predictions': None,
                'avg_probability': None,
                'risk_level': None,
                'risk_factors': None
            })
            
    except Exception as e:
        return jsonify({
            'type': 'error',
            'response': f'An error occurred: {str(e)}'
        }), 500


if __name__ == '__main__':
    os.makedirs(Config.MODELS_UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.DATASETS_UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
