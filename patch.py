import os
import re

# 1. Update config.py
with open('config.py', 'r') as f:
    config = f.read()
config = config.replace(
    "ALLOWED_MODEL_EXTENSIONS = {'.h5', '.keras'}",
    "ALLOWED_MODEL_EXTENSIONS = {'.h5', '.keras', '.joblib'}"
)
with open('config.py', 'w') as f:
    f.write(config)

# 2. Update app.py
with open('app.py', 'r') as f:
    app_code = f.read()

app_code = app_code.replace(
    "file = request.files['file']",
    "file = request.files['file']\n        metadata_file = request.files.get('metadata_file')"
)
app_code = app_code.replace(
    "if not file.filename.endswith(('.h5', '.keras')):",
    "if not file.filename.endswith(('.h5', '.keras', '.joblib')):"
)
app_code = app_code.replace(
    "Invalid file extension. Must be .h5 or .keras",
    "Invalid file extension. Must be .h5, .keras, or .joblib"
)
app_code = app_code.replace(
    "result = model_manager.upload_model(file, model_type, model_name)",
    "result = model_manager.upload_model(file, model_type, model_name, metadata_file)"
)

# Update predict calls in app.py to pass dataframe instead of numpy array
app_code = re.sub(
    r"# Convert to numeric.*?features_array = features_numeric\.to_numpy\(\)\s+predictions = model_manager\.predict_all_selected\(features_array\)",
    "predictions = model_manager.predict_all_selected(features_df)",
    app_code, flags=re.DOTALL
)

app_code = re.sub(
    r"features_numeric = features_df\.apply\(pd\.to_numeric, errors='coerce'\)\.fillna\(0\)\s+features_array = features_numeric\.to_numpy\(\)\s+predictions = model_manager\.predict_all_selected\(features_array\)",
    "predictions = model_manager.predict_all_selected(features_df)",
    app_code, flags=re.DOTALL
)

with open('app.py', 'w') as f:
    f.write(app_code)

print('app.py and config.py updated')
