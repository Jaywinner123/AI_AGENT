"""Simple configuration for the demo."""

import os
from typing import List

class SimpleConfig:
    """Simple configuration class for the demo."""
    
    def __init__(self):
        # Processing Settings
        self.whisper_model_size = "base"
        self.max_clip_duration = 60
        self.min_clip_duration = 15
        self.target_clips_per_video = 3
        self.scene_detection_threshold = 30.0
        self.audio_energy_threshold = 0.02
        
        # Storage Settings
        self.storage_path = "./storage"
        self.temp_path = "./temp"
        self.models_path = "./models"
        
        # Keywords for clip detection
        self.engagement_keywords = [
            "amazing", "incredible", "wow", "unbelievable", "shocking",
            "insane", "crazy", "mind-blowing", "epic", "legendary"
        ]
        
        # Logging
        self.log_level = "INFO"
        self.log_file = "clipper_agent.log"
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)