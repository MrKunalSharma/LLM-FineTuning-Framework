"""Model registry for managing multiple models."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import random

@dataclass
class ModelVersion:
    """Model version information."""
    name: str
    version: str
    path: str
    metrics: Dict[str, float]
    loaded: bool = False
    traffic_percentage: float = 0.0
    created_at: datetime = None

class ModelRegistry:
    """Manage multiple model versions."""
    
    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.active_models: List[str] = []
    
    def register_model(
        self, 
        name: str, 
        version: str, 
        path: str,
        metrics: Dict[str, float]
    ):
        """Register a new model version."""
        model_id = f"{name}_v{version}"
        self.models[model_id] = ModelVersion(
            name=name,
            version=version,
            path=path,
            metrics=metrics,
            created_at=datetime.utcnow()
        )
        logger.info(f"Registered model: {model_id}")
    
    def set_traffic_split(self, traffic_config: Dict[str, float]):
        """Set traffic distribution for A/B testing."""
        total = sum(traffic_config.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError("Traffic percentages must sum to 100")
        
        for model_id, percentage in traffic_config.items():
            if model_id in self.models:
                self.models[model_id].traffic_percentage = percentage
        
        self.active_models = [
            mid for mid, pct in traffic_config.items() if pct > 0
        ]
    
    def select_model(self) -> str:
        """Select model based on traffic distribution."""
        if not self.active_models:
            return list(self.models.keys())[0]
        
        # Weighted random selection
        weights = [
            self.models[mid].traffic_percentage 
            for mid in self.active_models
        ]
        return random.choices(self.active_models, weights=weights)[0]
    
    def get_model_metrics(self) -> Dict[str, Dict]:
        """Get performance metrics for all models."""
        return {
            model_id: {
                "metrics": model.metrics,
                "traffic": model.traffic_percentage,
                "requests_served": 0  # Would track in production
            }
            for model_id, model in self.models.items()
        }
