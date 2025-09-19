#!/usr/bin/env python3
"""
Clipper Agent Demo Script

A complete end-to-end demonstration of the Clipper Agent pipeline that:
1. Accepts one input video file
2. Runs Whisper transcription
3. Detects shot boundaries (scene detection)
4. Finds 30s candidate clips by high-speech-energy + keyword heuristics
5. Ranks candidates with a local LLM prompt and produces top clip
6. Produces final vertical 9:16 ffmpeg clip + burned captions + SRT
7. Outputs a JSON with suggested caption + hashtags

Usage:
    python clipper_demo.py input_video.mp4
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Core processing libraries
import numpy as np
import librosa
import whisper
import ffmpeg
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
from PIL import Image, ImageDraw, ImageFont
import pysrt
from datetime import timedelta

# NLP and ranking
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Configuration
from src.clipper_agent.config.settings import ClipperConfig


class ClipperDemo:
    """Main demo class for the Clipper Agent pipeline."""
    
    def __init__(self, config: ClipperConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize models
        self.whisper_model = None
        self.sentiment_analyzer = None
        self.embedder = None
        
        # Ensure directories exist
        self.config.ensure_directories()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.log_file)
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_models(self):
        """Load required models lazily."""
        if self.whisper_model is None:
            self.logger.info(f"Loading Whisper model: {self.config.whisper_model_size}")
            self.whisper_model = whisper.load_model(self.config.whisper_model_size)
        
        if self.sentiment_analyzer is None:
            self.logger.info("Loading sentiment analyzer")
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        if self.embedder is None:
            self.logger.info("Loading sentence transformer")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def validate_input(self, video_path: str) -> bool:
        """Validate input video file."""
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return False
        
        try:
            # Use ffprobe to validate video
            probe = ffmpeg.probe(video_path)
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            
            if not video_streams:
                self.logger.error("No video streams found in file")
                return False
            
            duration = float(probe['format']['duration'])
            if duration < self.config.min_clip_duration:
                self.logger.error(f"Video too short: {duration}s < {self.config.min_clip_duration}s")
                return False
            
            self.logger.info(f"Video validated: {duration:.2f}s duration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating video: {e}")
            return False
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video for processing."""
        audio_path = os.path.join(self.config.temp_path, "extracted_audio.wav")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            self.logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        self._load_models()
        
        self.logger.info("Starting transcription...")
        result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
        
        self.logger.info(f"Transcription complete: {len(result['segments'])} segments")
        return result
    
    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect scene boundaries in video."""
        self.logger.info("Detecting scene boundaries...")
        
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.config.scene_detection_threshold))
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
        self.logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    def analyze_audio_energy(self, audio_path: str, window_size: int = 30) -> List[Tuple[float, float, float]]:
        """Analyze audio energy in windows."""
        self.logger.info("Analyzing audio energy...")
        
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        
        energy_windows = []
        for start_time in range(0, int(duration) - window_size, window_size // 2):
            end_time = start_time + window_size
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            window_audio = y[start_sample:end_sample]
            rms_energy = np.sqrt(np.mean(window_audio**2))
            
            energy_windows.append((start_time, end_time, rms_energy))
        
        self.logger.info(f"Analyzed {len(energy_windows)} energy windows")
        return energy_windows
    
    def find_keyword_matches(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find keyword matches in transcript."""
        matches = []
        keywords = [kw.lower() for kw in self.config.engagement_keywords]
        
        for segment in transcript['segments']:
            segment_text = segment['text'].lower()
            for keyword in keywords:
                if keyword in segment_text:
                    matches.append({
                        'keyword': keyword,
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'],
                        'confidence': segment.get('avg_logprob', 0)
                    })
        
        self.logger.info(f"Found {len(matches)} keyword matches")
        return matches
    
    def generate_clip_candidates(self, video_path: str, transcript: Dict[str, Any], 
                               scenes: List[Tuple[float, float]], 
                               energy_windows: List[Tuple[float, float, float]],
                               keyword_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate clip candidates based on multiple signals."""
        self.logger.info("Generating clip candidates...")
        
        candidates = []
        video_duration = float(ffmpeg.probe(video_path)['format']['duration'])
        
        # Sort energy windows by energy level (descending)
        energy_windows.sort(key=lambda x: x[2], reverse=True)
        
        # Take top energy windows as base candidates
        for i, (start, end, energy) in enumerate(energy_windows[:10]):
            # Ensure clip doesn't exceed video bounds
            clip_start = max(0, start)
            clip_end = min(video_duration, end)
            clip_duration = clip_end - clip_start
            
            if clip_duration < self.config.min_clip_duration:
                continue
            
            # Find overlapping transcript segments
            overlapping_segments = []
            for segment in transcript['segments']:
                if (segment['start'] < clip_end and segment['end'] > clip_start):
                    overlapping_segments.append(segment)
            
            # Find overlapping keyword matches
            overlapping_keywords = []
            for match in keyword_matches:
                if (match['start'] < clip_end and match['end'] > clip_start):
                    overlapping_keywords.append(match)
            
            # Count scene changes within clip
            scene_changes = 0
            for scene_start, scene_end in scenes:
                if clip_start < scene_start < clip_end:
                    scene_changes += 1
            
            candidate = {
                'id': f"candidate_{i}",
                'start_time': clip_start,
                'end_time': clip_end,
                'duration': clip_duration,
                'audio_energy': energy,
                'transcript_segments': overlapping_segments,
                'keyword_matches': overlapping_keywords,
                'scene_changes': scene_changes,
                'transcript_text': ' '.join([seg['text'] for seg in overlapping_segments])
            }
            
            candidates.append(candidate)
        
        self.logger.info(f"Generated {len(candidates)} clip candidates")
        return candidates
    
    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates using multiple factors."""
        self._load_models()
        self.logger.info("Ranking clip candidates...")
        
        for candidate in candidates:
            score = self._calculate_engagement_score(candidate)
            candidate['engagement_score'] = score
        
        # Sort by engagement score (descending)
        ranked_candidates = sorted(candidates, key=lambda x: x['engagement_score'], reverse=True)
        
        self.logger.info("Candidates ranked by engagement score")
        return ranked_candidates
    
    def _calculate_engagement_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate engagement score for a candidate."""
        # Normalize audio energy (0-1)
        energy_score = min(candidate['audio_energy'] / 0.1, 1.0)
        
        # Sentiment analysis of transcript
        transcript_text = candidate['transcript_text']
        sentiment_score = 0.5  # Default neutral
        
        if transcript_text.strip():
            try:
                sentiment_result = self.sentiment_analyzer(transcript_text[:512])  # Limit length
                if sentiment_result[0]['label'] == 'LABEL_2':  # Positive
                    sentiment_score = sentiment_result[0]['score']
                elif sentiment_result[0]['label'] == 'LABEL_0':  # Negative (can be engaging)
                    sentiment_score = sentiment_result[0]['score'] * 0.8
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed: {e}")
        
        # Keyword match score
        keyword_score = min(len(candidate['keyword_matches']) / 3.0, 1.0)
        
        # Scene change score (more dynamic = better)
        scene_score = min(candidate['scene_changes'] / 2.0, 1.0)
        
        # Weighted combination
        total_score = (
            sentiment_score * 0.4 +
            energy_score * 0.3 +
            scene_score * 0.2 +
            keyword_score * 0.1
        )
        
        return total_score
    
    def extract_clip(self, video_path: str, candidate: Dict[str, Any]) -> str:
        """Extract the selected clip from video."""
        output_path = os.path.join(self.config.temp_path, f"clip_{candidate['id']}.mp4")
        
        try:
            (
                ffmpeg
                .input(video_path, ss=candidate['start_time'], t=candidate['duration'])
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Clip extracted: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error extracting clip: {e}")
            raise
    
    def convert_to_vertical(self, input_path: str) -> str:
        """Convert clip to 9:16 vertical format."""
        output_path = os.path.join(self.config.temp_path, "vertical_clip.mp4")
        
        try:
            (
                ffmpeg
                .input(input_path)
                .filter('scale', 1080, 1920, force_original_aspect_ratio='decrease')
                .filter('pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2', color='black')
                .output(output_path, vcodec='libx264', acodec='aac', preset='medium')
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Converted to vertical format: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error converting to vertical: {e}")
            raise
    
    def create_subtitles(self, candidate: Dict[str, Any]) -> str:
        """Create SRT subtitle file for the clip."""
        srt_path = os.path.join(self.config.temp_path, "subtitles.srt")
        
        subs = pysrt.SubRipFile()
        clip_start = candidate['start_time']
        
        for i, segment in enumerate(candidate['transcript_segments']):
            # Adjust timestamps relative to clip start
            start_time = timedelta(seconds=max(0, segment['start'] - clip_start))
            end_time = timedelta(seconds=segment['end'] - clip_start)
            
            # Skip segments that are outside the clip
            if start_time.total_seconds() >= candidate['duration']:
                break
            
            # Truncate end time if it exceeds clip duration
            if end_time.total_seconds() > candidate['duration']:
                end_time = timedelta(seconds=candidate['duration'])
            
            text = segment['text'].strip()
            if text:
                sub = pysrt.SubRipItem(
                    index=i + 1,
                    start=start_time,
                    end=end_time,
                    text=text
                )
                subs.append(sub)
        
        subs.save(srt_path, encoding='utf-8')
        self.logger.info(f"Subtitles created: {srt_path}")
        return srt_path
    
    def add_burned_subtitles(self, video_path: str, srt_path: str) -> str:
        """Add burned-in subtitles to video."""
        output_path = os.path.join(self.config.storage_path, "final_clip_with_subtitles.mp4")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .filter('subtitles', srt_path, 
                       force_style='FontSize=24,PrimaryColour=&Hffffff&,OutlineColour=&H000000&,Outline=2,Alignment=2')
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Final clip with subtitles: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error adding subtitles: {e}")
            raise
    
    def generate_content_suggestions(self, candidate: Dict[str, Any], video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate caption and hashtag suggestions."""
        self.logger.info("Generating content suggestions...")
        
        # Extract key information
        transcript_text = candidate['transcript_text'][:200]  # Limit length
        keywords_found = [match['keyword'] for match in candidate['keyword_matches']]
        
        # Generate caption
        caption = self._generate_caption(transcript_text, keywords_found, video_metadata)
        
        # Generate hashtags
        hashtags = self._generate_hashtags(transcript_text, keywords_found)
        
        # Create thumbnail suggestions
        thumbnail_suggestions = self._generate_thumbnail_suggestions(candidate)
        
        suggestions = {
            'caption': caption,
            'hashtags': hashtags,
            'thumbnail_suggestions': thumbnail_suggestions,
            'engagement_keywords_found': keywords_found,
            'estimated_engagement_score': candidate['engagement_score']
        }
        
        return suggestions
    
    def _generate_caption(self, transcript: str, keywords: List[str], metadata: Dict[str, Any]) -> str:
        """Generate engaging caption."""
        # Simple heuristic-based caption generation
        if keywords:
            hook = f"This is {keywords[0]}! 🔥"
        else:
            hook = "You won't believe this! 😱"
        
        # Extract first meaningful sentence from transcript
        sentences = transcript.split('.')
        main_content = sentences[0].strip() if sentences else transcript[:50]
        
        caption = f"{hook} {main_content}... #viral #trending"
        
        # Limit length
        if len(caption) > 150:
            caption = caption[:147] + "..."
        
        return caption
    
    def _generate_hashtags(self, transcript: str, keywords: List[str]) -> List[str]:
        """Generate relevant hashtags."""
        base_hashtags = ["#viral", "#trending", "#fyp", "#amazing"]
        
        # Add keyword-based hashtags
        keyword_hashtags = [f"#{kw.replace(' ', '')}" for kw in keywords[:3]]
        
        # Simple topic detection based on common words
        topic_hashtags = []
        transcript_lower = transcript.lower()
        
        topic_map = {
            'music': '#music',
            'dance': '#dance',
            'funny': '#comedy',
            'food': '#food',
            'travel': '#travel',
            'tech': '#technology',
            'game': '#gaming',
            'sport': '#sports'
        }
        
        for topic, hashtag in topic_map.items():
            if topic in transcript_lower:
                topic_hashtags.append(hashtag)
        
        # Combine and deduplicate
        all_hashtags = base_hashtags + keyword_hashtags + topic_hashtags
        unique_hashtags = list(dict.fromkeys(all_hashtags))  # Preserve order, remove duplicates
        
        return unique_hashtags[:8]  # Limit to 8 hashtags
    
    def _generate_thumbnail_suggestions(self, candidate: Dict[str, Any]) -> List[str]:
        """Generate thumbnail timestamp suggestions."""
        # Suggest timestamps at 25%, 50%, and 75% of the clip
        duration = candidate['duration']
        suggestions = [
            f"{duration * 0.25:.1f}s - Early moment",
            f"{duration * 0.5:.1f}s - Mid-clip highlight", 
            f"{duration * 0.75:.1f}s - Near end climax"
        ]
        return suggestions
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Main processing pipeline."""
        self.logger.info(f"Starting processing of: {video_path}")
        
        # Step 1: Validate input
        if not self.validate_input(video_path):
            raise ValueError("Invalid input video")
        
        # Step 2: Extract audio
        audio_path = self.extract_audio(video_path)
        
        # Step 3: Transcribe audio
        transcript = self.transcribe_audio(audio_path)
        
        # Step 4: Detect scenes
        scenes = self.detect_scenes(video_path)
        
        # Step 5: Analyze audio energy
        energy_windows = self.analyze_audio_energy(audio_path)
        
        # Step 6: Find keyword matches
        keyword_matches = self.find_keyword_matches(transcript)
        
        # Step 7: Generate candidates
        candidates = self.generate_clip_candidates(
            video_path, transcript, scenes, energy_windows, keyword_matches
        )
        
        if not candidates:
            raise ValueError("No suitable clip candidates found")
        
        # Step 8: Rank candidates
        ranked_candidates = self.rank_candidates(candidates)
        
        # Step 9: Select best candidate
        best_candidate = ranked_candidates[0]
        self.logger.info(f"Selected best candidate with score: {best_candidate['engagement_score']:.3f}")
        
        # Step 10: Extract clip
        clip_path = self.extract_clip(video_path, best_candidate)
        
        # Step 11: Convert to vertical format
        vertical_clip_path = self.convert_to_vertical(clip_path)
        
        # Step 12: Create subtitles
        srt_path = self.create_subtitles(best_candidate)
        
        # Step 13: Add burned subtitles
        final_clip_path = self.add_burned_subtitles(vertical_clip_path, srt_path)
        
        # Step 14: Generate content suggestions
        video_metadata = {'title': os.path.basename(video_path)}
        suggestions = self.generate_content_suggestions(best_candidate, video_metadata)
        
        # Prepare results
        results = {
            'input_video': video_path,
            'final_clip': final_clip_path,
            'subtitle_file': srt_path,
            'clip_info': {
                'start_time': best_candidate['start_time'],
                'end_time': best_candidate['end_time'],
                'duration': best_candidate['duration'],
                'engagement_score': best_candidate['engagement_score']
            },
            'content_suggestions': suggestions,
            'processing_stats': {
                'total_candidates': len(candidates),
                'scenes_detected': len(scenes),
                'keyword_matches': len(keyword_matches),
                'transcript_segments': len(transcript['segments'])
            }
        }
        
        # Save results to JSON
        results_path = os.path.join(self.config.storage_path, "clip_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Processing complete! Results saved to: {results_path}")
        return results


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="Clipper Agent Demo - Process a video file")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--config", help="Path to config YAML file", default="config/config.yaml")
    parser.add_argument("--whisper-model", help="Whisper model size", default="base")
    parser.add_argument("--output-dir", help="Output directory", default="./storage")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = ClipperConfig.from_yaml(args.config)
    else:
        config = ClipperConfig()
    
    # Override with command line arguments
    if args.whisper_model:
        config.whisper_model_size = args.whisper_model
    if args.output_dir:
        config.storage_path = args.output_dir
    
    # Initialize and run demo
    demo = ClipperDemo(config)
    
    try:
        results = demo.process_video(args.video_path)
        
        print("\n" + "="*60)
        print("CLIPPER AGENT DEMO - PROCESSING COMPLETE!")
        print("="*60)
        print(f"Input Video: {results['input_video']}")
        print(f"Final Clip: {results['final_clip']}")
        print(f"Subtitle File: {results['subtitle_file']}")
        print(f"\nClip Info:")
        print(f"  Start Time: {results['clip_info']['start_time']:.2f}s")
        print(f"  Duration: {results['clip_info']['duration']:.2f}s")
        print(f"  Engagement Score: {results['clip_info']['engagement_score']:.3f}")
        print(f"\nContent Suggestions:")
        print(f"  Caption: {results['content_suggestions']['caption']}")
        print(f"  Hashtags: {', '.join(results['content_suggestions']['hashtags'])}")
        print(f"\nProcessing Stats:")
        print(f"  Candidates Generated: {results['processing_stats']['total_candidates']}")
        print(f"  Scenes Detected: {results['processing_stats']['scenes_detected']}")
        print(f"  Keyword Matches: {results['processing_stats']['keyword_matches']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()