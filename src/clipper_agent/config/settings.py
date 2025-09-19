"""Configuration management for Clipper Agent."""

from pydantic import BaseSettings, Field
from typing import Optional, List, Dict, Any
import os
import yaml


class ClipperConfig(BaseSettings):
    """Main configuration class for Clipper Agent."""
    
    # API Keys
    youtube_api_key: Optional[str] = Field(None, env="YOUTUBE_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    instagram_access_token: Optional[str] = Field(None, env="INSTAGRAM_ACCESS_TOKEN")
    twitter_api_key: Optional[str] = Field(None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(None, env="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(None, env="TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: Optional[str] = Field(None, env="TWITTER_ACCESS_TOKEN_SECRET")
    
    # Processing Settings
    whisper_model_size: str = Field("base", env="WHISPER_MODEL_SIZE")
    max_clip_duration: int = Field(60, env="MAX_CLIP_DURATION")
    min_clip_duration: int = Field(15, env="MIN_CLIP_DURATION")
    target_clips_per_video: int = Field(3, env="TARGET_CLIPS_PER_VIDEO")
    scene_detection_threshold: float = Field(30.0, env="SCENE_DETECTION_THRESHOLD")
    audio_energy_threshold: float = Field(0.02, env="AUDIO_ENERGY_THRESHOLD")
    
    # Storage Settings
    storage_path: str = Field("./storage", env="STORAGE_PATH")
    temp_path: str = Field("./temp", env="TEMP_PATH")
    models_path: str = Field("./models", env="MODELS_PATH")
    
    # Keywords for clip detection
    engagement_keywords: List[str] = Field(
        ["amazing", "incredible", "wow", "unbelievable", "shocking"],
        env="ENGAGEMENT_KEYWORDS"
    )
    
    # Security settings
    enable_copyright_check: bool = Field(True, env="ENABLE_COPYRIGHT_CHECK")
    safe_transform_only: bool = Field(True, env="SAFE_TRANSFORM_ONLY")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("clipper_agent.log", env="LOG_FILE")
    
    # Database
    database_url: str = Field("sqlite:///clipper_agent.db", env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ClipperConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Extract clipper_agent section
        config_data = yaml_data.get('clipper_agent', {})
        
        # Flatten nested configuration
        flattened = cls._flatten_dict(config_data)
        
        return cls(**flattened)
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ClipperConfig._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific configuration."""
        platform_configs = {
            'youtube': {
                'api_key': self.youtube_api_key,
                'category_id': '22',
                'privacy_status': 'public',
                'max_title_length': 100,
                'max_description_length': 5000
            },
            'tiktok': {
                'max_caption_length': 150,
                'recommended_hashtags': 5
            },
            'instagram': {
                'access_token': self.instagram_access_token,
                'max_caption_length': 2200,
                'recommended_hashtags': 10
            },
            'twitter': {
                'api_key': self.twitter_api_key,
                'api_secret': self.twitter_api_secret,
                'access_token': self.twitter_access_token,
                'access_token_secret': self.twitter_access_token_secret,
                'max_caption_length': 280,
                'recommended_hashtags': 3
            }
        }
        
        return platform_configs.get(platform, {})