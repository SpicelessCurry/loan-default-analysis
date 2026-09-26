import json
import re

# --- 1. Rewrite ml_engine.py ---
ml_engine_code = '''
import os
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir='uploads/models'):
        self.models_dir = models_dir
        self.models_registry = {}  
        self.loaded_models = {}  
        self.registry_path = os.path.join(models_dir, 'registry.json')
        os.makedirs(models_dir, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.models_registry = json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                self.models_registry = {}
    
    def _save_registry(self):
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.models_registry, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def upload_model(self, file, model_type, model_name, metadata_file=None, preprocessor_file=None):
        if model_type not in ['MLP', 'DCN']:
            raise ValueError("Invalid model type. Must be one of: MLP, DCN")
        
        model_id = str(uuid.uuid4())
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = '.pth' if model_type == 'DCN' else '.joblib'
            
        filename = f"{model_id}{ext}"
        filepath = os.path.join(self.models_dir, filename)
        
        file.save(filepath)
        
        metadata_path = None
        if model_type == 'MLP' and metadata_file and metadata_file.filename:
            metadata_filename = f"{model_id}_metadata.json"
            metadata_path = os.path.join(self.models_dir, metadata_filename)
            metadata_file.save(metadata_path)
            
        preprocessor_path = None
        if model_type == 'DCN' and preprocessor_file and preprocessor_file.filename:
            preprocessor_filename = f"{model_id}_preprocessor.joblib"
            preprocessor_path = os.path.join(self.models_dir, preprocessor_filename)
            preprocessor_file.save(preprocessor_path)
            
        model_info = {
            'id': model_id,
            'name': model_name,
            'type': model_type,
            'filename': filename,
            'path': filepath,
            'metadata_path': metadata_path,
            'preprocessor_path': preprocessor_path,
            'uploaded_at': datetime.now().isoformat(),
            'is_selected': False
        }
        
        self.models_registry[model_id] = model_info
        self._save_registry()
        return model_info
    
    def delete_model(self, model_id):
        if model_id in self.models_registry:
            model_info = self.models_registry[model_id]
            
            for path_key in ['path', 'metadata_path', 'preprocessor_path']:
                if model_info.get(path_key) and os.path.exists(model_info[path_key]):
                    try: os.remove(model_info[path_key])
                    except Exception: pass
            
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
            del self.models_registry[model_id]
            self._save_registry()
            return True
        return False
    
    def select_models(self, model_ids):
        for m_id, info in self.models_registry.items():
            info['is_selected'] = m_id in model_ids
        self._save_registry()
    
    def get_models(self):
        return list(self.models_registry.values())
    
    def get_selected_models(self):
        return [info for info in self.models_registry.values() if info.get('is_selected')]
    
    def load_model(self, model_id):
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
            
        if model_id not in self.models_registry:
            return None
            
        model_info = self.models_registry[model_id]
        model_type = model_info['type']
        
        try:
            if model_type == 'MLP':
                import joblib
                model = joblib.load(model_info['path'])
                metadata = None
                if model_info.get('metadata_path') and os.path.exists(model_info['metadata_path']):
                    with open(model_info['metadata_path'], 'r') as f:
                        metadata = json.load(f)
                self.loaded_models[model_id] = (model, metadata)
                return (model, metadata)
                
            elif model_type == 'DCN':
                import torch
                import joblib
                
                preprocessor = None
                if model_info.get('preprocessor_path') and os.path.exists(model_info['preprocessor_path']):
                    preprocessor = joblib.load(model_info['preprocessor_path'])
                
                # We attempt to load the torch model directly.
                # If they saved state_dict only, this will fail without the class, 
                # but we'll try torch.load anyway.
                model = torch.load(model_info['path'], map_location=torch.device('cpu'))
                if hasattr(model, 'eval'):
                    model.eval()
                    
                self.loaded_models[model_id] = (model, preprocessor)
                return (model, preprocessor)
        except Exception as e:
            print(f"Error loading {model_type} model {model_id}: {e}")
            return None
    
    def predict(self, model_id, features_df):
        loaded = self.load_model(model_id)
        if loaded is None:
            return None
            
        model, aux_data = loaded
        model_type = self.models_registry[model_id]['type']
        
        try:
            if model_type == 'MLP':
                metadata = aux_data
                features = metadata.get("feature_columns", []) if metadata else features_df.columns.tolist()
                threshold = metadata.get("recommended_threshold", 0.5) if metadata else 0.5
                
                # Align columns based on metadata, fill missing with 0
                X = features_df.reindex(columns=features, fill_value=0)
                prob = float(model.predict_proba(X)[:, 1][0])
                return (prob, threshold)
                
            elif model_type == 'DCN':
                import torch
                preprocessor = aux_data
                
                if preprocessor is not None:
                    try:
                        X = preprocessor.transform(features_df)
                    except Exception:
                        X = features_df.values
                else:
                    X = features_df.values
                
                # Support dense arrays from transform
                if hasattr(X, 'todense'):
                    X = X.todense()
                    
                tensor_X = torch.tensor(X, dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = model(tensor_X)
                    # Support logits or raw probabilities
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        prob = torch.softmax(outputs, dim=1)[0, 1].item()
                    else:
                        prob = torch.sigmoid(outputs).item() if outputs.min() < 0 else outputs.item()
                        
                return (prob, 0.5) # Default threshold 0.5 for DCN
                
        except Exception as e:
            print(f"Prediction error for {model_type} model {model_id}: {e}")
            return None
    
    def predict_all_selected(self, features_df):
        results = []
        selected_models = self.get_selected_models()
        
        for model_info in selected_models:
            model_id = model_info['id']
            res = self.predict(model_id, features_df)
            
            if res is not None:
                prob, threshold = res
                label = 'Default' if prob >= threshold else 'Non-Default'
                results.append({
                    'model_id': model_id,
                    'model_name': model_info['name'],
                    'model_type': model_info['type'],
                    'probability': prob,
                    'label': label
                })
        
        return results
'''
with open('ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(ml_engine_code)


# --- 2. Patch config.py ---
with open('config.py', 'r', encoding='utf-8') as f:
    config = f.read()

config = re.sub(
    r"ALLOWED_MODEL_EXTENSIONS = \{.*?\}",
    "ALLOWED_MODEL_EXTENSIONS = {'.joblib', '.pth'}",
    config
)
with open('config.py', 'w', encoding='utf-8') as f:
    f.write(config)


# --- 3. Patch app.py ---
with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

app_code = app_code.replace(
    "model_type = request.form.get('model_type')",
    "model_type = request.form.get('model_type')\n        preprocessor_file = request.files.get('preprocessor_file')"
)

app_code = re.sub(
    r"if not model_type or model_type not in \[.*?\]:",
    "if not model_type or model_type not in ['MLP', 'DCN']:",
    app_code
)

app_code = re.sub(
    r"Invalid model type\. Must be .*?\"",
    "Invalid model type. Must be MLP or DCN\"",
    app_code
)

app_code = re.sub(
    r"if not file\.filename\.endswith\(.*?\):",
    "if not file.filename.endswith(('.joblib', '.pth')):",
    app_code
)

app_code = re.sub(
    r"Invalid file extension\. Must be .*?\"",
    "Invalid file extension. Must be .joblib or .pth\"",
    app_code
)

app_code = app_code.replace(
    "result = model_manager.upload_model(file, model_type, model_name, metadata_file)",
    "result = model_manager.upload_model(file, model_type, model_name, metadata_file, preprocessor_file)"
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)


# --- 4. Patch dashboard.html ---
with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace select options
html = re.sub(
    r'<select class="input" id="model-type" required onchange="handleModelTypeChange\(\)">.*?</select>',
    '''<select class="input" id="model-type" required onchange="handleModelTypeChange()">
                            <option value="">Select Type</option>
                            <option value="MLP">MLP</option>
                            <option value="DCN">DCN</option>
                        </select>''',
    html, flags=re.DOTALL
)

# Add preprocessor group directly after metadata group
preprocessor_group = '''
                    <div class="form-group" id="preprocessor-group" style="display: none; margin-bottom: 16px;">
                        <label style="font-size: 14px; margin-bottom: 8px; display: block;">Preprocessor (Required for DCN .joblib)</label>
                        <div class="drop-zone" id="preprocessor-drop-zone" onclick="document.getElementById('preprocessor-file').click()" style="min-height: 80px; padding: 15px;">
                            <span style="font-size: 20px;">??</span>
                            <p>Drop preprocessor.joblib here or click</p>
                            <input type="file" id="preprocessor-file" accept=".joblib" style="display: none;">
                        </div>
                        <div id="preprocessor-file-name" style="margin-top: 8px; font-size: 14px; color: var(--accent);"></div>
                    </div>'''

if 'id="preprocessor-group"' not in html:
    html = html.replace(
        '<div id="metadata-file-name" style="margin-top: 8px; font-size: 14px; color: var(--accent);"></div>\n                    </div>',
        '<div id="metadata-file-name" style="margin-top: 8px; font-size: 14px; color: var(--accent);"></div>\n                    </div>' + preprocessor_group
    )

# Fix handleModelTypeChange
js_target = '''function handleModelTypeChange() {
            const type = document.getElementById('model-type').value;
            const metaGroup = document.getElementById('metadata-group');
            const modelFile = document.getElementById('model-file');
            
            if (type === 'MLP') {
                metaGroup.style.display = 'block';
                modelFile.accept = '.joblib,.pkl';
                document.getElementById('model-drop-zone').querySelector('p').textContent = 'Drag and drop .joblib file here, or click to browse';
            } else {
                metaGroup.style.display = 'none';
                modelFile.accept = '.h5,.keras';
                document.getElementById('model-drop-zone').querySelector('p').textContent = 'Drag and drop .h5 or .keras file here, or click to browse';
            }
        }'''

js_new = '''function handleModelTypeChange() {
            const type = document.getElementById('model-type').value;
            const metaGroup = document.getElementById('metadata-group');
            const prepGroup = document.getElementById('preprocessor-group');
            const modelFile = document.getElementById('model-file');
            const dropText = document.getElementById('model-drop-zone').querySelector('p');
            
            if (type === 'MLP') {
                metaGroup.style.display = 'block';
                if(prepGroup) prepGroup.style.display = 'none';
                modelFile.accept = '.joblib,.pkl';
                dropText.textContent = 'Drag and drop .joblib model here, or click to browse';
            } else if (type === 'DCN') {
                metaGroup.style.display = 'none';
                if(prepGroup) prepGroup.style.display = 'block';
                modelFile.accept = '.pth';
                dropText.textContent = 'Drag and drop .pth PyTorch weights here, or click to browse';
            } else {
                metaGroup.style.display = 'none';
                if(prepGroup) prepGroup.style.display = 'none';
                modelFile.accept = '';
                dropText.textContent = 'Please select a model type first';
            }
        }'''
html = html.replace(js_target, js_new)

# Fix uploadModel append logic
append_target = '''if (type === 'MLP') {
                const metadataFile = document.getElementById('metadata-file').files[0];
                if (metadataFile) {
                    formData.append('metadata_file', metadataFile);
                }
            }'''
append_new = '''if (type === 'MLP') {
                const metadataFile = document.getElementById('metadata-file').files[0];
                if (metadataFile) {
                    formData.append('metadata_file', metadataFile);
                }
            } else if (type === 'DCN') {
                const preprocessorFile = document.getElementById('preprocessor-file').files[0];
                if (preprocessorFile) {
                    formData.append('preprocessor_file', preprocessorFile);
                }
            }'''
html = html.replace(append_target, append_new)

# Fix drag drop setup
if "setupDragDrop('preprocessor-drop-zone'" not in html:
    html = html.replace(
        "setupDragDrop('metadata-drop-zone', 'metadata-file', 'metadata-file-name');",
        "setupDragDrop('metadata-drop-zone', 'metadata-file', 'metadata-file-name');\n            setupDragDrop('preprocessor-drop-zone', 'preprocessor-file', 'preprocessor-file-name');"
    )

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("All files patched!")
