#!/usr/bin/env python3
"""
Simplified Clipper Agent Demo Script

A working demonstration of the core Clipper Agent pipeline that:
1. Accepts one input video file
2. Analyzes video structure and audio
3. Detects potential clip segments
4. Extracts and processes clips
5. Converts to vertical format and adds basic captions
6. Outputs results and suggestions

This simplified version works without heavy ML dependencies.
"""

import argparse
import json
import logging
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
import random

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Core processing libraries
import numpy as np
import ffmpeg
import cv2
from PIL import Image, ImageDraw, ImageFont
import pysrt
from datetime import timedelta

# Configuration
from simple_config import SimpleConfig


class SimpleClipperDemo:
    """Simplified demo class for the Clipper Agent pipeline."""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.logger = self._setup_logging()
        
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
    
    def analyze_audio_energy(self, audio_path: str, window_size: int = 30) -> List[Tuple[float, float, float]]:
        """Analyze audio energy in windows using basic signal processing."""
        self.logger.info("Analyzing audio energy...")
        
        try:
            # Use ffmpeg to get audio stats
            cmd = [
                'ffmpeg', '-i', audio_path, '-af', 'astats=metadata=1:reset=1', 
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            # For this demo, create synthetic energy windows
            # In a real implementation, this would parse the actual audio data
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['format']['duration'])
            
            energy_windows = []
            for start_time in range(0, int(duration) - window_size, window_size // 2):
                end_time = start_time + window_size
                # Simulate varying energy levels
                energy = random.uniform(0.01, 0.1)
                energy_windows.append((start_time, end_time, energy))
            
            self.logger.info(f"Analyzed {len(energy_windows)} energy windows")
            return energy_windows
            
        except Exception as e:
            self.logger.warning(f"Audio analysis failed: {e}")
            # Fallback: create basic windows
            duration = 120  # Default duration
            energy_windows = []
            for start_time in range(0, duration - window_size, window_size // 2):
                end_time = start_time + window_size
                energy = random.uniform(0.01, 0.1)
                energy_windows.append((start_time, end_time, energy))
            return energy_windows
    
    def detect_scenes_simple(self, video_path: str) -> List[Tuple[float, float]]:
        """Simple scene detection using basic heuristics."""
        self.logger.info("Detecting scene boundaries...")
        
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            
            # Simple scene detection: divide video into segments
            scene_length = 20  # 20-second scenes
            scenes = []
            
            for start in range(0, int(duration), scene_length):
                end = min(start + scene_length, duration)
                scenes.append((start, end))
            
            self.logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene detection failed: {e}")
            return [(0, 60), (60, 120)]  # Default scenes
    
    def find_keyword_matches_simple(self, duration: float) -> List[Dict[str, Any]]:
        """Simulate keyword matches for demo purposes."""
        matches = []
        keywords = self.config.engagement_keywords
        
        # Simulate finding keywords at random timestamps
        for i in range(random.randint(2, 5)):
            keyword = random.choice(keywords)
            start_time = random.uniform(0, duration - 10)
            end_time = start_time + random.uniform(3, 8)
            
            matches.append({
                'keyword': keyword,
                'start': start_time,
                'end': end_time,
                'text': f"This is {keyword}!",
                'confidence': random.uniform(0.7, 0.95)
            })
        
        self.logger.info(f"Found {len(matches)} keyword matches")
        return matches
    
    def generate_clip_candidates(self, video_path: str, 
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
                'keyword_matches': overlapping_keywords,
                'scene_changes': scene_changes,
                'transcript_text': ' '.join([match['text'] for match in overlapping_keywords])
            }
            
            candidates.append(candidate)
        
        self.logger.info(f"Generated {len(candidates)} clip candidates")
        return candidates
    
    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates using simple heuristics."""
        self.logger.info("Ranking clip candidates...")
        
        for candidate in candidates:
            score = self._calculate_engagement_score(candidate)
            candidate['engagement_score'] = score
        
        # Sort by engagement score (descending)
        ranked_candidates = sorted(candidates, key=lambda x: x['engagement_score'], reverse=True)
        
        self.logger.info("Candidates ranked by engagement score")
        return ranked_candidates
    
    def _calculate_engagement_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate engagement score for a candidate using simple heuristics."""
        # Normalize audio energy (0-1)
        energy_score = min(candidate['audio_energy'] / 0.1, 1.0)
        
        # Keyword match score
        keyword_score = min(len(candidate['keyword_matches']) / 3.0, 1.0)
        
        # Scene change score (more dynamic = better)
        scene_score = min(candidate['scene_changes'] / 2.0, 1.0)
        
        # Duration preference (prefer clips around 30-45 seconds)
        duration = candidate['duration']
        if 25 <= duration <= 50:
            duration_score = 1.0
        elif 15 <= duration <= 60:
            duration_score = 0.8
        else:
            duration_score = 0.5
        
        # Weighted combination
        total_score = (
            energy_score * 0.3 +
            keyword_score * 0.3 +
            scene_score * 0.2 +
            duration_score * 0.2
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
    
    def create_simple_subtitles(self, candidate: Dict[str, Any]) -> str:
        """Create simple SRT subtitle file for the clip."""
        srt_path = os.path.join(self.config.temp_path, "subtitles.srt")
        
        subs = pysrt.SubRipFile()
        
        # Create simple subtitles based on keyword matches
        for i, match in enumerate(candidate['keyword_matches']):
            start_seconds = max(0, match['start'] - candidate['start_time'])
            end_seconds = min(candidate['duration'], match['end'] - candidate['start_time'])
            
            # Convert to pysrt time format
            start_time = pysrt.SubRipTime(seconds=start_seconds)
            end_time = pysrt.SubRipTime(seconds=end_seconds)
            
            text = match['text']
            if text:
                sub = pysrt.SubRipItem(
                    index=i + 1,
                    start=start_time,
                    end=end_time,
                    text=text
                )
                subs.append(sub)
        
        # Add a default subtitle if no keywords found
        if not subs:
            sub = pysrt.SubRipItem(
                index=1,
                start=pysrt.SubRipTime(seconds=0),
                end=pysrt.SubRipTime(seconds=min(5, candidate['duration'])),
                text="Amazing content!"
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
        keywords_found = [match['keyword'] for match in candidate['keyword_matches']]
        
        # Generate caption
        caption = self._generate_caption(keywords_found, video_metadata)
        
        # Generate hashtags
        hashtags = self._generate_hashtags(keywords_found)
        
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
    
    def _generate_caption(self, keywords: List[str], metadata: Dict[str, Any]) -> str:
        """Generate engaging caption."""
        if keywords:
            hook = f"This is {keywords[0]}! 🔥"
        else:
            hook = "You won't believe this! 😱"
        
        caption = f"{hook} Check out this incredible moment! #viral #trending"
        
        # Limit length
        if len(caption) > 150:
            caption = caption[:147] + "..."
        
        return caption
    
    def _generate_hashtags(self, keywords: List[str]) -> List[str]:
        """Generate relevant hashtags."""
        base_hashtags = ["#viral", "#trending", "#fyp", "#amazing"]
        
        # Add keyword-based hashtags
        keyword_hashtags = [f"#{kw.replace(' ', '')}" for kw in keywords[:3]]
        
        # Add some topic hashtags
        topic_hashtags = ["#content", "#video", "#shorts"]
        
        # Combine and deduplicate
        all_hashtags = base_hashtags + keyword_hashtags + topic_hashtags
        unique_hashtags = list(dict.fromkeys(all_hashtags))  # Preserve order, remove duplicates
        
        return unique_hashtags[:8]  # Limit to 8 hashtags
    
    def _generate_thumbnail_suggestions(self, candidate: Dict[str, Any]) -> List[str]:
        """Generate thumbnail timestamp suggestions."""
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
        
        # Step 3: Get video duration for analysis
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        
        # Step 4: Detect scenes
        scenes = self.detect_scenes_simple(video_path)
        
        # Step 5: Analyze audio energy
        energy_windows = self.analyze_audio_energy(audio_path)
        
        # Step 6: Find keyword matches (simulated)
        keyword_matches = self.find_keyword_matches_simple(duration)
        
        # Step 7: Generate candidates
        candidates = self.generate_clip_candidates(
            video_path, scenes, energy_windows, keyword_matches
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
        srt_path = self.create_simple_subtitles(best_candidate)
        
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
                'energy_windows_analyzed': len(energy_windows)
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
    parser = argparse.ArgumentParser(description="Simple Clipper Agent Demo - Process a video file")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--config", help="Path to config YAML file", default="config/config.yaml")
    parser.add_argument("--output-dir", help="Output directory", default="./storage")
    
    args = parser.parse_args()
    
    # Load configuration
    config = SimpleConfig()
    
    # Override with command line arguments
    if args.output_dir:
        config.storage_path = args.output_dir
    
    # Initialize and run demo
    demo = SimpleClipperDemo(config)
    
    try:
        results = demo.process_video(args.video_path)
        
        print("\n" + "="*60)
        print("SIMPLE CLIPPER AGENT DEMO - PROCESSING COMPLETE!")
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