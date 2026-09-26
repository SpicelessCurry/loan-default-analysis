import os
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# DCN (Deep & Cross Network) — PyTorch model class
# Architecture must match the state_dict saved in the .pth checkpoint.
# Cross layers: element-wise cross with learned weight & bias per layer.
# Deep network: Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU → Linear
# Output: concat(cross_out, deep_out) → Linear(1) → sigmoid
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    """Single cross layer with learnable weight and bias vectors."""
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.randn(input_dim))

    def forward(self, x0, xl):
        # x_{l+1} = x0 * (xl · w) + b + xl
        return x0 * (xl * self.weight).sum(dim=1, keepdim=True) + self.bias + xl


class DCN(nn.Module):
    """Deep & Cross Network for binary classification."""

    def __init__(self, input_dim, num_cross_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_cross_layers = num_cross_layers

        # Cross layers — nn.ModuleList so state_dict keys are cross_layers.0.weight etc.
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])

        # Deep network
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 256),    # deep.0
            nn.ReLU(),                    # deep.1
            nn.BatchNorm1d(256),          # deep.2
            nn.Dropout(0.3),              # deep.3
            nn.Linear(256, 128),          # deep.4
            nn.ReLU(),                    # deep.5
            nn.BatchNorm1d(128),          # deep.6
            nn.Dropout(0.3),              # deep.7
            nn.Linear(128, 64),           # deep.8
            nn.ReLU(),                    # deep.9
        )

        # Output layer — takes concat of cross (input_dim) + deep (64)
        self.output = nn.Linear(input_dim + 64, 1)

    def forward(self, x):
        # --- Cross part ---
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        # --- Deep part ---
        deep_out = self.deep(x)

        # --- Combine ---
        combined = torch.cat([xl, deep_out], dim=1)
        out = self.output(combined)
        return out


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
                    try:
                        os.remove(model_info[path_key])
                    except Exception:
                        pass
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

    # ------------------------------------------------------------------
    # load_model — handles both MLP (.joblib pipeline) and DCN (.pth)
    # ------------------------------------------------------------------
    def load_model(self, model_id):
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        if model_id not in self.models_registry:
            return None

        model_info = self.models_registry[model_id]
        model_type = model_info['type']

        try:
            if model_type == 'MLP':
                import joblib as jl
                model = jl.load(model_info['path'])
                metadata = None
                if model_info.get('metadata_path') and os.path.exists(model_info['metadata_path']):
                    with open(model_info['metadata_path'], 'r') as f:
                        metadata = json.load(f)
                self.loaded_models[model_id] = (model, metadata)
                return (model, metadata)

            elif model_type == 'DCN':
                import joblib as jl

                # Load checkpoint dict
                checkpoint = torch.load(
                    model_info['path'],
                    map_location=torch.device('cpu'),
                    weights_only=False
                )

                input_dim = checkpoint.get('input_dim', 107)
                num_cross = checkpoint.get('cross_layers', 3)
                feature_columns = checkpoint.get('feature_columns', [])
                threshold = checkpoint.get('threshold', 0.5)

                # Rebuild the DCN model and load weights
                model = DCN(input_dim=input_dim, num_cross_layers=num_cross)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # Load preprocessor (dict with 'imputer' and 'scaler')
                preprocessor = None
                if model_info.get('preprocessor_path') and os.path.exists(model_info['preprocessor_path']):
                    preprocessor = jl.load(model_info['preprocessor_path'])

                # Bundle everything the predict step needs
                aux = {
                    'preprocessor': preprocessor,
                    'feature_columns': feature_columns,
                    'threshold': threshold,
                }

                self.loaded_models[model_id] = (model, aux)
                return (model, aux)

        except Exception as e:
            import traceback
            print(f"Error loading {model_type} model {model_id}:\n{traceback.format_exc()}")
            return None

    # ------------------------------------------------------------------
    # predict — single-row prediction
    # ------------------------------------------------------------------
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

                X = features_df.reindex(columns=features, fill_value=0)
                prob = float(model.predict_proba(X)[:, 1][0])
                return (prob, threshold)

            elif model_type == 'DCN':
                feature_columns = aux_data.get('feature_columns', [])
                threshold = aux_data.get('threshold', 0.5)
                preprocessor = aux_data.get('preprocessor')

                # 1. Align columns to what the model expects
                if feature_columns:
                    X_df = features_df.reindex(columns=feature_columns, fill_value=0)
                else:
                    X_df = features_df.copy()

                X = X_df.values.astype(np.float64)

                # 2. Apply imputer then scaler from the preprocessor dict
                if preprocessor is not None:
                    if isinstance(preprocessor, dict):
                        imputer = preprocessor.get('imputer')
                        scaler = preprocessor.get('scaler')
                        if imputer is not None:
                            X = imputer.transform(X)
                        if scaler is not None:
                            X = scaler.transform(X)
                    else:
                        # If it's a fitted sklearn Pipeline or ColumnTransformer
                        X = preprocessor.transform(X_df)

                # 3. Convert to tensor
                if hasattr(X, 'toarray'):
                    X = X.toarray()

                tensor_X = torch.tensor(X, dtype=torch.float32)

                # 4. Forward pass
                with torch.no_grad():
                    logits = model(tensor_X)
                    prob = torch.sigmoid(logits).item()

                return (prob, threshold)

        except Exception as e:
            import traceback
            print(f"Prediction error for {model_type} model {model_id}:\n{traceback.format_exc()}")
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
