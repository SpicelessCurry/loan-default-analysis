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
    
    def upload_model(self, file, model_type, model_name, metadata_file=None):
        if model_type not in ['RNN', 'ANN', 'MLP']:
            raise ValueError("Invalid model type. Must be one of: RNN, ANN, MLP")
        
        model_id = str(uuid.uuid4())
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = '.h5'
            
        filename = f"{model_id}{ext}"
        filepath = os.path.join(self.models_dir, filename)
        
        file.save(filepath)
        
        metadata_path = None
        if metadata_file and metadata_file.filename:
            metadata_filename = f"{model_id}_metadata.json"
            metadata_path = os.path.join(self.models_dir, metadata_filename)
            metadata_file.save(metadata_path)
            
        model_info = {
            'id': model_id,
            'name': model_name,
            'type': model_type,
            'filename': filename,
            'path': filepath,
            'metadata_path': metadata_path,
            'uploaded_at': datetime.now().isoformat(),
            'is_selected': False
        }
        
        self.models_registry[model_id] = model_info
        self._save_registry()
        return model_info
    
    def delete_model(self, model_id):
        if model_id in self.models_registry:
            model_info = self.models_registry[model_id]
            
            if os.path.exists(model_info['path']):
                try: os.remove(model_info['path'])
                except Exception: pass
                
            if model_info.get('metadata_path') and os.path.exists(model_info['metadata_path']):
                try: os.remove(model_info['metadata_path'])
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
        
        try:
            if model_info['path'].endswith('.joblib'):
                import joblib
                model = joblib.load(model_info['path'])
                metadata = None
                if model_info.get('metadata_path') and os.path.exists(model_info['metadata_path']):
                    with open(model_info['metadata_path'], 'r') as f:
                        metadata = json.load(f)
                self.loaded_models[model_id] = (model, metadata)
                return (model, metadata)
            else:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_info['path'])
                self.loaded_models[model_id] = (model, None)
                return (model, None)
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            return None
    
    def predict(self, model_id, features_df):
        loaded = self.load_model(model_id)
        if loaded is None:
            return None
            
        model, metadata = loaded
        model_type = self.models_registry[model_id]['type']
        is_joblib = self.models_registry[model_id]['path'].endswith('.joblib')
        
        try:
            if is_joblib:
                features = metadata.get("feature_columns", []) if metadata else features_df.columns.tolist()
                threshold = metadata.get("recommended_threshold", 0.5) if metadata else 0.5
                
                # Align columns based on metadata, fill missing with 0
                X = features_df.reindex(columns=features, fill_value=0)
                prob = float(model.predict_proba(X)[:, 1][0])
                return (prob, threshold)
            else:
                features_numeric = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                features_array = features_numeric.to_numpy()
                
                if model_type == 'RNN':
                    num_features = features_array.shape[1]
                    processed_features = features_array.reshape(1, num_features, 1)
                else:
                    processed_features = features_array
                    
                prediction = model.predict(processed_features, verbose=0)
                if hasattr(prediction, 'numpy'):
                    prediction = prediction.numpy()
                prob = float(np.squeeze(prediction))
                prob = max(0.0, min(1.0, prob))
                return (prob, 0.5)
        except Exception as e:
            print(f"Prediction error for model {model_id}: {e}")
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
