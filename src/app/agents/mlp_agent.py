import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List
import logging

log = logging.getLogger(__name__)

class MLPNet(nn.Module):
    """
    The core Multi-Layer Perceptron model definition.
    Architecture adjusted to match the expected keys in the saved state_dict 
    (which includes Batch Normalization layers).
    """
    def __init__(self, input_dim: int = 7, output_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        
        # Structure inferred from previous error keys (fc1, bn1, etc.): 
        # Linear -> BN -> ReLU for hidden layers, followed by a final Linear layer.
        self.net = nn.Sequential(
            # Hidden Layer 1 (net.0, net.1, net.2)
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Hidden Layer 2 (net.3, net.4, net.5)
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Hidden Layer 3 (net.6, net.7, net.8)
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Output Layer (net.9)
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPAgent:
    """
    Agent for making air quality predictions using the trained MLP model.
    """
    def __init__(
        self,
        model_path: str = "src/app/models/mlp_pollutants_model.pth",
        x_scaler_path: str = "src/app/models/x_scaler.pkl",
        y_scaler_path: str = "src/app/models/y_scaler.pkl",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        
        try:
            # Load scalers
            self.x_scaler = joblib.load(Path(x_scaler_path))
            self.y_scaler = joblib.load(Path(y_scaler_path))

            # Load model checkpoint
            checkpoint = torch.load(Path(model_path), map_location=self.device)
        except Exception as e:
            log.error(f"Failed to load model or scalers. Check paths: {e}")
            raise

        # --- CRITICAL FIX: Explicit Key Remapping ---
        state_dict = {}
        
        # This mapping is based on the 'Unexpected key(s)' in your previous error 
        # and the new MLPNet Sequential indices (0, 1, 3, 4, 6, 7, 9)
        key_map = {
            'fc1': 'net.0', 'bn1': 'net.1', 
            'fc2': 'net.3', 'bn2': 'net.4', 
            'fc3': 'net.6', 'bn3': 'net.7', 
            'fc_out': 'net.9'
        }

        for old_key, value in checkpoint.items():
            parts = old_key.split('.')
            # The base name is the first part (e.g., 'fc1', 'bn1', etc.)
            base_name = parts[0]
            # The suffix is '.weight', '.bias', '.running_mean', etc.
            suffix = '.'.join(parts[1:])

            if base_name in key_map:
                # Construct the new Sequential key (e.g., 'fc1.weight' -> 'net.0.weight')
                new_key = f"{key_map[base_name]}.{suffix}"
                state_dict[new_key] = value
            else:
                # If the key doesn't match the expected structure, keep it (unlikely here)
                state_dict[old_key] = value
                
        # Initialize model with correct dimensions (7 inputs, 8 outputs, 128 hidden)
        self.model = MLPNet(input_dim=7, output_dim=8, hidden_dim=128)
        
        # Load the remapped weights
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        log.info("MLPAgent loaded successfully with state dict remapping.")

        # RESTORED: These features MUST match the training dataset column names, including units.
        self.features: List[str] = [
            "temperature_2m (°C)",
            "relative_humidity_2m (%)",
            "rain (mm)",
            "wind_speed_100m (km/h)",
            "wind_direction_100m (°)",
            "pressure_msl (hPa)",
            "surface_pressure (hPa)",
        ]

        self.pollutants: List[str] = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
        
        # Helper list of expected input keys (unit-less keys from IngestionAgent)
        self.input_keys = [self._get_unitless_key(f) for f in self.features]

    @staticmethod
    def _get_unitless_key(model_key: str) -> str:
        """Strips the unit part (e.g., ' (°C)') from the model feature name."""
        return model_key.split(' (', 1)[0].strip()

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Run inference on input weather features.
        """
        # --- Validate inputs ---
        missing = [f for f in self.input_keys if f not in features]
        if missing:
            log.error(f"Missing required input features: {missing}")
            # Raise an error to stop pipeline execution cleanly
            raise ValueError(f"Missing required input features: {missing}")

        # --- Prepare & scale input ---
        # Build X_raw by iterating over the model's feature names (with units) 
        # but retrieving the value using the unit-less key from the input dictionary.
        X_raw = np.array([
            [features.get(self._get_unitless_key(f_full), 0.0) for f_full in self.features]
        ], dtype=np.float32)
        
        # Check if the scaler is loaded before use
        if self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("Scalers were not loaded successfully. Cannot run prediction.")

        X_scaled = self.x_scaler.transform(X_raw)

        # --- Torch inference ---
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).cpu().numpy()

        # --- Inverse transform ---
        preds = self.y_scaler.inverse_transform(preds_scaled)[0]

        # --- Return as dict ---
        # Ensure predictions are positive before returning
        final_predictions = {
            pollutant: max(0.0, pred) 
            for pollutant, pred in zip(self.pollutants, preds.tolist())
        }
        
        return final_predictions
