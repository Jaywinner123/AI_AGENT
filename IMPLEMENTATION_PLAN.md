# Clipper Agent Implementation Plan

## Component-by-Component Implementation

### 1. Content Ingestion Layer

#### 1.1 YouTube/Twitch Downloader
**Libraries:**
- `yt-dlp==2023.12.30` - Universal video downloader
- `requests==2.31.0` - HTTP requests
- `validators==0.22.0` - URL validation

**Commands:**
```bash
# Install yt-dlp
pip install yt-dlp

# Download video with metadata
yt-dlp --write-info-json --write-thumbnail --extract-flat --format "best[height<=1080]" <URL>

# Audio-only download for podcasts
yt-dlp --extract-audio --audio-format mp3 --audio-quality 0 <URL>
```

**Configuration:**
```python
YT_DLP_CONFIG = {
    'format': 'best[height<=1080]',
    'writeinfojson': True,
    'writethumbnail': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
    'outtmpl': '%(uploader)s/%(title)s.%(ext)s'
}
```

#### 1.2 Local File Handler
**Libraries:**
- `pathlib` - Path handling (built-in)
- `mimetypes` - File type detection (built-in)
- `ffprobe` - Media info extraction

**Commands:**
```bash
# Get video metadata
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# Validate video file
ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 input.mp4
```

#### 1.3 RSS/Feed Parser
**Libraries:**
- `feedparser==6.0.10` - RSS/Atom parsing
- `beautifulsoup4==4.12.2` - HTML parsing

**Commands:**
```bash
pip install feedparser beautifulsoup4
```

### 2. Clip Discovery Engine

#### 2.1 Audio Signal Analysis
**Libraries:**
- `librosa==0.10.1` - Audio analysis
- `numpy==1.24.3` - Numerical computing
- `scipy==1.11.4` - Scientific computing

**Commands:**
```bash
pip install librosa numpy scipy
```

**Implementation:**
```python
import librosa
import numpy as np

def detect_audio_energy(audio_file, window_size=30):
    """Detect high-energy audio segments"""
    y, sr = librosa.load(audio_file)
    # RMS energy calculation
    hop_length = sr * window_size
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return rms

def find_speech_segments(audio_file):
    """Find segments with speech activity"""
    y, sr = librosa.load(audio_file)
    # Voice activity detection using spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return spectral_centroids
```

#### 2.2 Speech-to-Text (Whisper)
**Libraries:**
- `openai-whisper==20231117` - OpenAI Whisper ASR
- `torch==2.1.0` - PyTorch backend

**Commands:**
```bash
pip install openai-whisper torch torchvision torchaudio

# Download Whisper models
whisper --model base --output_dir ./models dummy.wav
```

**Implementation:**
```python
import whisper

def transcribe_audio(audio_file, model_size="base"):
    """Transcribe audio using Whisper"""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file, word_timestamps=True)
    return result

def extract_keywords(transcript, keywords_list):
    """Extract keyword timestamps from transcript"""
    matches = []
    for segment in transcript['segments']:
        for word in segment.get('words', []):
            if word['word'].lower().strip() in keywords_list:
                matches.append({
                    'word': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word.get('probability', 0)
                })
    return matches
```

#### 2.3 Scene Detection
**Libraries:**
- `scenedetect==0.6.2` - Scene boundary detection
- `opencv-python==4.8.1.78` - Computer vision

**Commands:**
```bash
pip install scenedetect[opencv] opencv-python

# CLI scene detection
scenedetect -i input.mp4 detect-content list-scenes
```

**Implementation:**
```python
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_file, threshold=30.0):
    """Detect scene boundaries in video"""
    video_manager = VideoManager([video_file])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    
    return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
```

#### 2.4 LLM Ranking System
**Libraries:**
- `transformers==4.35.2` - Hugging Face transformers
- `torch==2.1.0` - PyTorch backend
- `sentence-transformers==2.2.2` - Sentence embeddings

**Commands:**
```bash
pip install transformers torch sentence-transformers

# Download local LLM model
python -c "from transformers import pipeline; pipeline('text-generation', model='microsoft/DialoGPT-medium')"
```

**Implementation:**
```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class ClipRanker:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def rank_clips(self, candidates, metadata):
        """Rank clip candidates by engagement potential"""
        scores = []
        for candidate in candidates:
            score = self._calculate_engagement_score(candidate, metadata)
            scores.append(score)
        
        # Sort by score descending
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked
    
    def _calculate_engagement_score(self, candidate, metadata):
        """Calculate engagement score for a clip candidate"""
        # Factors: transcript sentiment, audio energy, scene changes, keywords
        transcript_score = self._score_transcript(candidate.get('transcript', ''))
        energy_score = candidate.get('audio_energy', 0)
        scene_score = candidate.get('scene_changes', 0)
        keyword_score = candidate.get('keyword_matches', 0)
        
        return (transcript_score * 0.4 + 
                energy_score * 0.3 + 
                scene_score * 0.2 + 
                keyword_score * 0.1)
```

### 3. Video Editing Pipeline

#### 3.1 FFmpeg Integration
**Libraries:**
- `ffmpeg-python==0.2.0` - Python FFmpeg wrapper
- `subprocess` - System commands (built-in)

**Commands:**
```bash
# Install FFmpeg system dependency
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Python wrapper
pip install ffmpeg-python
```

**Implementation:**
```python
import ffmpeg

class VideoEditor:
    def __init__(self):
        self.temp_dir = "./temp"
    
    def extract_clip(self, input_file, start_time, duration, output_file):
        """Extract clip from video"""
        (
            ffmpeg
            .input(input_file, ss=start_time, t=duration)
            .output(output_file, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
    
    def convert_to_vertical(self, input_file, output_file):
        """Convert video to 9:16 aspect ratio"""
        (
            ffmpeg
            .input(input_file)
            .filter('scale', 1080, 1920, force_original_aspect_ratio='decrease')
            .filter('pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2', color='black')
            .output(output_file, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
    
    def add_subtitles(self, video_file, srt_file, output_file):
        """Burn subtitles into video"""
        (
            ffmpeg
            .input(video_file)
            .filter('subtitles', srt_file, force_style='FontSize=24,PrimaryColour=&Hffffff&,OutlineColour=&H000000&,Outline=2')
            .output(output_file, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
    
    def normalize_audio(self, input_file, output_file, target_lufs=-16):
        """Normalize audio loudness"""
        (
            ffmpeg
            .input(input_file)
            .filter('loudnorm', I=target_lufs, TP=-1.5, LRA=11)
            .output(output_file, vcodec='copy', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
```

#### 3.2 Subtitle Generation
**Libraries:**
- `pysrt==1.1.2` - SRT file handling
- `datetime` - Time formatting (built-in)

**Commands:**
```bash
pip install pysrt
```

**Implementation:**
```python
import pysrt
from datetime import timedelta

def create_srt_from_transcript(transcript, output_file):
    """Create SRT subtitle file from Whisper transcript"""
    subs = pysrt.SubRipFile()
    
    for i, segment in enumerate(transcript['segments']):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        text = segment['text'].strip()
        
        sub = pysrt.SubRipItem(
            index=i+1,
            start=start_time,
            end=end_time,
            text=text
        )
        subs.append(sub)
    
    subs.save(output_file, encoding='utf-8')
```

### 4. Content Optimization

#### 4.1 Caption and Hashtag Generation
**Libraries:**
- `transformers==4.35.2` - Text generation models
- `openai==1.3.5` - OpenAI API (optional)

**Commands:**
```bash
pip install transformers openai
```

**Implementation:**
```python
from transformers import pipeline

class ContentOptimizer:
    def __init__(self):
        self.text_generator = pipeline("text-generation", model="gpt2")
    
    def generate_caption(self, video_metadata, transcript_summary):
        """Generate engaging caption for social media"""
        prompt = f"""
        Create an engaging social media caption for a video clip about: {transcript_summary}
        Video title: {video_metadata.get('title', '')}
        Keep it under 150 characters, include relevant emojis, and make it clickable.
        Caption:
        """
        
        result = self.text_generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text'].split('Caption:')[-1].strip()
    
    def generate_hashtags(self, content_topic, platform="tiktok"):
        """Generate relevant hashtags for the content"""
        hashtag_sets = {
            "tiktok": ["#fyp", "#viral", "#trending", "#foryou"],
            "instagram": ["#reels", "#explore", "#viral", "#trending"],
            "youtube": ["#shorts", "#viral", "#trending"],
            "twitter": ["#viral", "#trending"]
        }
        
        base_tags = hashtag_sets.get(platform, [])
        # Add content-specific tags based on topic analysis
        content_tags = self._extract_topic_hashtags(content_topic)
        
        return base_tags + content_tags[:6]  # Limit to 10 total hashtags
```

#### 4.2 Thumbnail Generation
**Libraries:**
- `Pillow==10.1.0` - Image processing
- `opencv-python==4.8.1.78` - Video frame extraction

**Commands:**
```bash
pip install Pillow opencv-python
```

**Implementation:**
```python
import cv2
from PIL import Image, ImageDraw, ImageFont

def extract_thumbnail_candidates(video_file, num_frames=5):
    """Extract potential thumbnail frames from video"""
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    candidates = []
    for i in range(num_frames):
        frame_pos = int((i + 1) * total_frames / (num_frames + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            candidates.append(frame)
    
    cap.release()
    return candidates

def create_thumbnail_with_text(image, title_text, output_file):
    """Create thumbnail with overlay text"""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    # Add title text overlay
    font_size = 48
    font = ImageFont.load_default()  # Use system font in production
    
    # Add text with outline
    x, y = 50, img.height - 150
    draw.text((x-2, y-2), title_text, font=font, fill='black')
    draw.text((x+2, y+2), title_text, font=font, fill='black')
    draw.text((x, y), title_text, font=font, fill='white')
    
    img.save(output_file, 'JPEG', quality=95)
```

### 5. Publishing System

#### 5.1 Platform Adapters
**Libraries:**
- `requests==2.31.0` - HTTP requests
- `google-api-python-client==2.108.0` - YouTube API
- `facebook-sdk==3.1.0` - Instagram Graph API
- `tweepy==4.14.0` - Twitter API

**Commands:**
```bash
pip install requests google-api-python-client facebook-sdk tweepy
```

**Implementation:**
```python
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class PlatformPublisher:
    def __init__(self, config):
        self.config = config
    
    def upload_to_youtube_shorts(self, video_file, title, description, tags):
        """Upload video to YouTube Shorts"""
        youtube = build('youtube', 'v3', developerKey=self.config['youtube_api_key'])
        
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': '22'  # People & Blogs
            },
            'status': {
                'privacyStatus': 'public'
            }
        }
        
        media = MediaFileUpload(video_file, chunksize=-1, resumable=True)
        
        request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        response = request.execute()
        return response
    
    def prepare_tiktok_upload(self, video_file, caption, hashtags):
        """Prepare TikTok upload package (manual upload)"""
        upload_package = {
            'video_file': video_file,
            'caption': f"{caption} {' '.join(hashtags)}",
            'instructions': [
                "1. Open TikTok app or web interface",
                "2. Click '+' to create new video",
                "3. Upload the provided video file",
                "4. Copy and paste the provided caption",
                "5. Set privacy to Public",
                "6. Click Post"
            ]
        }
        return upload_package
```

#### 5.2 Upload Orchestration
**Libraries:**
- `celery==5.3.4` - Task queue
- `redis==5.0.1` - Message broker

**Commands:**
```bash
pip install celery redis

# Start Redis server
redis-server

# Start Celery worker
celery -A clipper_agent worker --loglevel=info
```

### 6. Analytics and Monitoring

#### 6.1 Metrics Collection
**Libraries:**
- `sqlite3` - Database (built-in)
- `pandas==2.1.3` - Data analysis
- `matplotlib==3.8.2` - Visualization

**Commands:**
```bash
pip install pandas matplotlib
```

**Implementation:**
```python
import sqlite3
import pandas as pd
from datetime import datetime

class AnalyticsTracker:
    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY,
                source_video TEXT,
                clip_file TEXT,
                platform TEXT,
                upload_date TIMESTAMP,
                title TEXT,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                engagement_rate REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def track_upload(self, clip_data):
        """Track new clip upload"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clips (source_video, clip_file, platform, upload_date, title)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            clip_data['source_video'],
            clip_data['clip_file'],
            clip_data['platform'],
            datetime.now(),
            clip_data['title']
        ))
        
        conn.commit()
        conn.close()
    
    def update_metrics(self, clip_id, metrics):
        """Update clip performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        engagement_rate = (metrics['likes'] + metrics['shares'] + metrics['comments']) / max(metrics['views'], 1)
        
        cursor.execute('''
            UPDATE clips 
            SET views=?, likes=?, shares=?, comments=?, engagement_rate=?
            WHERE id=?
        ''', (
            metrics['views'],
            metrics['likes'],
            metrics['shares'],
            metrics['comments'],
            engagement_rate,
            clip_id
        ))
        
        conn.commit()
        conn.close()
```

### 7. Configuration and Security

#### 7.1 Configuration Management
**Libraries:**
- `pydantic==2.5.0` - Data validation
- `python-dotenv==1.0.0` - Environment variables
- `pyyaml==6.0.1` - YAML parsing

**Commands:**
```bash
pip install pydantic python-dotenv pyyaml
```

**Implementation:**
```python
from pydantic import BaseSettings, Field
from typing import Optional, List
import os

class ClipperConfig(BaseSettings):
    # API Keys
    youtube_api_key: Optional[str] = Field(None, env="YOUTUBE_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    instagram_access_token: Optional[str] = Field(None, env="INSTAGRAM_ACCESS_TOKEN")
    twitter_api_key: Optional[str] = Field(None, env="TWITTER_API_KEY")
    
    # Processing Settings
    whisper_model_size: str = Field("base", env="WHISPER_MODEL_SIZE")
    max_clip_duration: int = Field(60, env="MAX_CLIP_DURATION")
    min_clip_duration: int = Field(15, env="MIN_CLIP_DURATION")
    target_clips_per_video: int = Field(3, env="TARGET_CLIPS_PER_VIDEO")
    
    # Storage Settings
    storage_path: str = Field("./storage", env="STORAGE_PATH")
    temp_path: str = Field("./temp", env="TEMP_PATH")
    
    # Keywords for clip detection
    engagement_keywords: List[str] = Field(
        ["amazing", "incredible", "wow", "unbelievable", "shocking"],
        env="ENGAGEMENT_KEYWORDS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
config = ClipperConfig()
```

#### 7.2 Security Implementation
**Libraries:**
- `cryptography==41.0.7` - Encryption
- `python-jose==3.3.0` - JWT tokens

**Commands:**
```bash
pip install cryptography python-jose
```

**Implementation:**
```python
from cryptography.fernet import Fernet
import os
import json

class SecurityManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self):
        """Get or create encryption key"""
        key_file = ".clipper_key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
    
    def encrypt_api_keys(self, api_keys_dict):
        """Encrypt API keys for storage"""
        json_data = json.dumps(api_keys_dict)
        encrypted_data = self.cipher.encrypt(json_data.encode())
        return encrypted_data
    
    def decrypt_api_keys(self, encrypted_data):
        """Decrypt API keys"""
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
```

## Installation Commands Summary

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg python3-pip python3-venv git

# macOS
brew install ffmpeg python git

# Windows (using Chocolatey)
choco install ffmpeg python git
```

### Python Environment Setup
```bash
# Create virtual environment
python -m venv clipper_env
source clipper_env/bin/activate  # Linux/macOS
# clipper_env\Scripts\activate  # Windows

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Model Downloads
```bash
# Download Whisper models
python -c "import whisper; whisper.load_model('base')"

# Download sentence transformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Development Tools
```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

This implementation plan provides exact libraries, commands, and code examples for each component of the Clipper Agent system.