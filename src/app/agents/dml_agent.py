import joblib
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

log = logging.getLogger(__name__)

class DMLAgent:
    """
    Agent for estimating the causal effect of weather variables
    on pollutant levels using a pre-trained Double Machine Learning (DML) model.
    """
    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        """
        Initialize the DMLAgent by loading a pre-trained DML model.

        Args:
            model_path (str, optional): Path to the pre-trained DML model.
                Defaults to 'src/app/models/dml_model.pkl' relative to the current file.
        """
        # --- Handle Path Initialization ---
        if model_path is None:
            # Use a robust, relative path by default
            model_path = Path(__file__).resolve().parent.parent / "models" / "dml_model.pkl"
        else:
            # Use the provided path, resolved
            model_path = Path(model_path).resolve()

        if not model_path.exists():
            log.error(f"DML model not found at: {model_path}")
            raise FileNotFoundError(f"DML model not found at: {model_path}")

        # Load the pre-trained model
        self.model = joblib.load(model_path)
        log.info(f"DMLAgent loaded successfully from {model_path}")

        # CRITICAL FIX: Treatments MUST match the training dataset column names, including units.
        self.treatments: List[str] = [
            "temperature_2m (°C)",
            "relative_humidity_2m (%)",
            "rain (mm)",
            "wind_speed_100m (km/h)",
            "wind_direction_100m (°)",
            "pressure_msl (hPa)",
            "surface_pressure (hPa)",
        ]

        # Outcomes (pollutants)
        self.outcomes: List[str] = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
        
        # Helper list of expected input keys (unit-less keys from IngestionAgent)
        self.input_keys = [self._get_unitless_key(t) for t in self.treatments]


    @staticmethod
    def _get_unitless_key(model_key: str) -> str:
        """Strips the unit part (e.g., ' (°C)') from the model feature name."""
        return model_key.split(' (', 1)[0].strip()


    def estimate_effects(self, features: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Estimate causal effects of weather features on pollutants.

        CRITICAL FIX: This function now uses the unit-less keys from the input 'features'
        dictionary (provided by IngestionAgent) to pull values, while maintaining the
        unit-full order defined in self.treatments for the model input.

        Args:
            features (Dict[str, Any]): Must include all unit-less keys expected 
                                        (i.e., "temperature_2m").

        Returns:
            Dict[str, Dict[str, float]]: Nested dict mapping treatment -> pollutant -> effect.

        Raises:
            ValueError: If required features are missing.
        """
        # --- Validate inputs (using the unit-less keys expected from the caller) ---
        missing = [t for t in self.input_keys if t not in features or features[t] is None]
        if missing:
            raise ValueError(f"Missing required input features: {missing}")

        # --- Prepare input for DML model ---
        # We must use the unit-less key (e.g., 'temperature_2m') to fetch the value
        # from the input `features`, but iterate in the order of the unit-full `self.treatments`.
        X = np.array([
            [features[self._get_unitless_key(t_full)] for t_full in self.treatments]
        ])

        # Get constant marginal effects (shape: n_treatments x n_outcomes)
        te = self.model.const_marginal_effect(X)[0]

        # Convert to nested dict
        results: Dict[str, Dict[str, float]] = {}
        # We map the results using the unit-full treatment name for clarity
        for i, treatment in enumerate(self.treatments):
            results[treatment] = dict(zip(self.outcomes, te[i]))

        return results
