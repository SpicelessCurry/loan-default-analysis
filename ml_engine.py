import os
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir='uploads/models'):
        self.models_dir = models_dir
        self.models_registry = {}  # {id: {name, type, filename, path, uploaded_at, is_selected}}
        self.loaded_models = {}  # {id: keras_model}
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
    
    def upload_model(self, file, model_type, model_name):
        if model_type not in ['RNN', 'ANN', 'MLP']:
            raise ValueError("Invalid model type. Must be one of: RNN, ANN, MLP")
        
        model_id = str(uuid.uuid4())
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = '.h5'
            
        filename = f"{model_id}{ext}"
        filepath = os.path.join(self.models_dir, filename)
        
        file.save(filepath)
        
        model_info = {
            'id': model_id,
            'name': model_name,
            'type': model_type,
            'filename': filename,
            'path': filepath,
            'uploaded_at': datetime.now().isoformat(),
            'is_selected': False
        }
        
        self.models_registry[model_id] = model_info
        self._save_registry()
        return model_info
    
    def delete_model(self, model_id):
        if model_id in self.models_registry:
            model_info = self.models_registry[model_id]
            
            # Delete file
            if os.path.exists(model_info['path']):
                try:
                    os.remove(model_info['path'])
                except Exception as e:
                    print(f"Error deleting file {model_info['path']}: {e}")
            
            # Unload if loaded
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
            # Remove from registry
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
        return [info for info in self.models_registry.values() if info['is_selected']]
    
    def load_model(self, model_id):
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
            
        if model_id not in self.models_registry:
            return None
            
        import tensorflow as tf
        model_info = self.models_registry[model_id]
        
        try:
            model = tf.keras.models.load_model(model_info['path'])
            self.loaded_models[model_id] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            return None
    
    def predict(self, model_id, features):
        model = self.load_model(model_id)
        if model is None:
            return None
            
        if model_id not in self.models_registry:
            return None
            
        model_type = self.models_registry[model_id]['type']
        
        try:
            # Assumes features is already a 2D numpy array of shape (1, num_features)
            if model_type == 'RNN':
                # Reshape to (1, num_features, 1) for RNN
                num_features = features.shape[1]
                processed_features = features.reshape(1, num_features, 1)
            else:
                # For ANN and MLP, keep as (1, num_features)
                processed_features = features
                
            prediction = model.predict(processed_features, verbose=0)
            
            # Extract probability (handle different output shapes)
            if hasattr(prediction, 'numpy'):
                prediction = prediction.numpy()
                
            # Convert to scalar float
            prob = float(np.squeeze(prediction))
            
            # Bound between 0 and 1
            return max(0.0, min(1.0, prob))
            
        except Exception as e:
            print(f"Prediction error for model {model_id}: {e}")
            return None
    
    def predict_all_selected(self, features):
        results = []
        selected_models = self.get_selected_models()
        
        for model_info in selected_models:
            model_id = model_info['id']
            prob = self.predict(model_id, features)
            
            if prob is not None:
                label = 'Default' if prob > 0.5 else 'Non-Default'
                results.append({
                    'model_id': model_id,
                    'model_name': model_info['name'],
                    'model_type': model_info['type'],
                    'probability': prob,
                    'label': label
                })
        
        return results
