#!/usr/bin/env python3
"""
Test suite for the Clipper Agent demo.

This module contains comprehensive tests for the Clipper Agent pipeline,
including unit tests, integration tests, and end-to-end validation.
"""

import unittest
import tempfile
import os
import json
import shutil
from pathlib import Path
import subprocess
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_config import SimpleConfig
from clipper_demo_simple import SimpleClipperDemo


class TestClipperDemo(unittest.TestCase):
    """Test cases for the Clipper Agent demo."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.config = SimpleConfig()
        cls.config.storage_path = os.path.join(cls.test_dir, "storage")
        cls.config.temp_path = os.path.join(cls.test_dir, "temp")
        cls.config.models_path = os.path.join(cls.test_dir, "models")
        
        # Create test video
        cls.test_video_path = os.path.join(cls.test_dir, "test_video.mp4")
        cls._create_test_video(cls.test_video_path)
        
        cls.demo = SimpleClipperDemo(cls.config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def _create_test_video(output_path: str):
        """Create a test video for testing."""
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'color=c=red:size=640x480:duration=60:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=60',
            '-c:v', 'libx264', '-c:a', 'aac', '-t', '60',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        self.assertIsInstance(self.config, SimpleConfig)
        self.assertEqual(self.config.min_clip_duration, 15)
        self.assertEqual(self.config.max_clip_duration, 60)
        self.assertIsInstance(self.config.engagement_keywords, list)
        self.assertGreater(len(self.config.engagement_keywords), 0)
    
    def test_demo_initialization(self):
        """Test demo class initialization."""
        self.assertIsInstance(self.demo, SimpleClipperDemo)
        self.assertEqual(self.demo.config, self.config)
        self.assertIsNotNone(self.demo.logger)
    
    def test_validate_input_valid_video(self):
        """Test input validation with valid video."""
        result = self.demo.validate_input(self.test_video_path)
        self.assertTrue(result)
    
    def test_validate_input_nonexistent_file(self):
        """Test input validation with nonexistent file."""
        result = self.demo.validate_input("nonexistent.mp4")
        self.assertFalse(result)
    
    def test_extract_audio(self):
        """Test audio extraction."""
        audio_path = self.demo.extract_audio(self.test_video_path)
        self.assertTrue(os.path.exists(audio_path))
        self.assertTrue(audio_path.endswith('.wav'))
        
        # Verify audio file has content
        self.assertGreater(os.path.getsize(audio_path), 1000)
    
    def test_analyze_audio_energy(self):
        """Test audio energy analysis."""
        audio_path = self.demo.extract_audio(self.test_video_path)
        energy_windows = self.demo.analyze_audio_energy(audio_path)
        
        self.assertIsInstance(energy_windows, list)
        self.assertGreater(len(energy_windows), 0)
        
        # Check window structure
        for window in energy_windows:
            self.assertEqual(len(window), 3)  # start, end, energy
            start, end, energy = window
            self.assertIsInstance(start, (int, float))
            self.assertIsInstance(end, (int, float))
            self.assertIsInstance(energy, (int, float))
            self.assertLess(start, end)
            self.assertGreaterEqual(energy, 0)
    
    def test_detect_scenes_simple(self):
        """Test simple scene detection."""
        scenes = self.demo.detect_scenes_simple(self.test_video_path)
        
        self.assertIsInstance(scenes, list)
        self.assertGreater(len(scenes), 0)
        
        # Check scene structure
        for scene in scenes:
            self.assertEqual(len(scene), 2)  # start, end
            start, end = scene
            self.assertIsInstance(start, (int, float))
            self.assertIsInstance(end, (int, float))
            self.assertLess(start, end)
    
    def test_find_keyword_matches_simple(self):
        """Test keyword matching simulation."""
        matches = self.demo.find_keyword_matches_simple(60.0)
        
        self.assertIsInstance(matches, list)
        
        # Check match structure
        for match in matches:
            self.assertIn('keyword', match)
            self.assertIn('start', match)
            self.assertIn('end', match)
            self.assertIn('text', match)
            self.assertIn('confidence', match)
            
            self.assertIn(match['keyword'], self.config.engagement_keywords)
            self.assertLess(match['start'], match['end'])
            self.assertGreaterEqual(match['confidence'], 0)
            self.assertLessEqual(match['confidence'], 1)
    
    def test_generate_clip_candidates(self):
        """Test clip candidate generation."""
        scenes = self.demo.detect_scenes_simple(self.test_video_path)
        audio_path = self.demo.extract_audio(self.test_video_path)
        energy_windows = self.demo.analyze_audio_energy(audio_path)
        keyword_matches = self.demo.find_keyword_matches_simple(60.0)
        
        candidates = self.demo.generate_clip_candidates(
            self.test_video_path, scenes, energy_windows, keyword_matches
        )
        
        self.assertIsInstance(candidates, list)
        
        # Check candidate structure
        for candidate in candidates:
            required_keys = [
                'id', 'start_time', 'end_time', 'duration',
                'audio_energy', 'keyword_matches', 'scene_changes',
                'transcript_text'
            ]
            for key in required_keys:
                self.assertIn(key, candidate)
            
            self.assertLess(candidate['start_time'], candidate['end_time'])
            self.assertGreaterEqual(candidate['duration'], self.config.min_clip_duration)
            self.assertLessEqual(candidate['duration'], self.config.max_clip_duration)
    
    def test_rank_candidates(self):
        """Test candidate ranking."""
        # Generate test candidates
        scenes = self.demo.detect_scenes_simple(self.test_video_path)
        audio_path = self.demo.extract_audio(self.test_video_path)
        energy_windows = self.demo.analyze_audio_energy(audio_path)
        keyword_matches = self.demo.find_keyword_matches_simple(60.0)
        
        candidates = self.demo.generate_clip_candidates(
            self.test_video_path, scenes, energy_windows, keyword_matches
        )
        
        ranked_candidates = self.demo.rank_candidates(candidates)
        
        self.assertIsInstance(ranked_candidates, list)
        self.assertEqual(len(ranked_candidates), len(candidates))
        
        # Check that all candidates have engagement scores
        for candidate in ranked_candidates:
            self.assertIn('engagement_score', candidate)
            self.assertIsInstance(candidate['engagement_score'], (int, float))
            self.assertGreaterEqual(candidate['engagement_score'], 0)
            self.assertLessEqual(candidate['engagement_score'], 1)
        
        # Check that candidates are sorted by score (descending)
        scores = [c['engagement_score'] for c in ranked_candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_extract_clip(self):
        """Test clip extraction."""
        # Create a test candidate
        candidate = {
            'id': 'test_candidate',
            'start_time': 10.0,
            'end_time': 25.0,
            'duration': 15.0
        }
        
        clip_path = self.demo.extract_clip(self.test_video_path, candidate)
        
        self.assertTrue(os.path.exists(clip_path))
        self.assertGreater(os.path.getsize(clip_path), 1000)
        
        # Verify clip duration using ffprobe
        import ffmpeg
        probe = ffmpeg.probe(clip_path)
        duration = float(probe['format']['duration'])
        self.assertAlmostEqual(duration, candidate['duration'], delta=1.0)
    
    def test_convert_to_vertical(self):
        """Test vertical format conversion."""
        # First extract a clip
        candidate = {
            'id': 'test_candidate',
            'start_time': 10.0,
            'end_time': 25.0,
            'duration': 15.0
        }
        
        clip_path = self.demo.extract_clip(self.test_video_path, candidate)
        vertical_path = self.demo.convert_to_vertical(clip_path)
        
        self.assertTrue(os.path.exists(vertical_path))
        self.assertGreater(os.path.getsize(vertical_path), 1000)
        
        # Verify aspect ratio using ffprobe
        import ffmpeg
        probe = ffmpeg.probe(vertical_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Should be 9:16 aspect ratio (1080x1920)
        self.assertEqual(width, 1080)
        self.assertEqual(height, 1920)
    
    def test_create_simple_subtitles(self):
        """Test subtitle creation."""
        candidate = {
            'start_time': 10.0,
            'duration': 15.0,
            'keyword_matches': [
                {
                    'keyword': 'amazing',
                    'start': 12.0,
                    'end': 15.0,
                    'text': 'This is amazing!'
                }
            ]
        }
        
        srt_path = self.demo.create_simple_subtitles(candidate)
        
        self.assertTrue(os.path.exists(srt_path))
        self.assertGreater(os.path.getsize(srt_path), 10)
        
        # Verify SRT content
        with open(srt_path, 'r') as f:
            content = f.read()
            self.assertIn('This is amazing!', content)
    
    def test_generate_content_suggestions(self):
        """Test content suggestion generation."""
        candidate = {
            'engagement_score': 0.8,
            'keyword_matches': [
                {'keyword': 'amazing', 'text': 'This is amazing!'}
            ]
        }
        
        video_metadata = {'title': 'Test Video'}
        
        suggestions = self.demo.generate_content_suggestions(candidate, video_metadata)
        
        required_keys = [
            'caption', 'hashtags', 'thumbnail_suggestions',
            'engagement_keywords_found', 'estimated_engagement_score'
        ]
        
        for key in required_keys:
            self.assertIn(key, suggestions)
        
        self.assertIsInstance(suggestions['caption'], str)
        self.assertIsInstance(suggestions['hashtags'], list)
        self.assertIsInstance(suggestions['thumbnail_suggestions'], list)
        self.assertGreater(len(suggestions['caption']), 0)
        self.assertGreater(len(suggestions['hashtags']), 0)


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.config = SimpleConfig()
        cls.config.storage_path = os.path.join(cls.test_dir, "storage")
        cls.config.temp_path = os.path.join(cls.test_dir, "temp")
        cls.config.models_path = os.path.join(cls.test_dir, "models")
        
        # Create test video
        cls.test_video_path = os.path.join(cls.test_dir, "test_video.mp4")
        cls._create_test_video(cls.test_video_path)
        
        cls.demo = SimpleClipperDemo(cls.config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def _create_test_video(output_path: str):
        """Create a test video for testing."""
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'color=c=blue:size=1280x720:duration=90:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=90',
            '-c:v', 'libx264', '-c:a', 'aac', '-t', '90',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def test_full_pipeline(self):
        """Test the complete processing pipeline."""
        results = self.demo.process_video(self.test_video_path)
        
        # Verify results structure
        required_keys = [
            'input_video', 'final_clip', 'subtitle_file',
            'clip_info', 'content_suggestions', 'processing_stats'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
        
        # Verify output files exist
        self.assertTrue(os.path.exists(results['final_clip']))
        self.assertTrue(os.path.exists(results['subtitle_file']))
        
        # Verify clip info
        clip_info = results['clip_info']
        self.assertIn('start_time', clip_info)
        self.assertIn('end_time', clip_info)
        self.assertIn('duration', clip_info)
        self.assertIn('engagement_score', clip_info)
        
        self.assertLess(clip_info['start_time'], clip_info['end_time'])
        self.assertGreaterEqual(clip_info['duration'], self.config.min_clip_duration)
        self.assertLessEqual(clip_info['duration'], self.config.max_clip_duration)
        
        # Verify content suggestions
        suggestions = results['content_suggestions']
        self.assertIn('caption', suggestions)
        self.assertIn('hashtags', suggestions)
        self.assertIsInstance(suggestions['hashtags'], list)
        self.assertGreater(len(suggestions['hashtags']), 0)
        
        # Verify processing stats
        stats = results['processing_stats']
        self.assertIn('total_candidates', stats)
        self.assertIn('scenes_detected', stats)
        self.assertGreater(stats['total_candidates'], 0)
        self.assertGreater(stats['scenes_detected'], 0)
    
    def test_output_file_quality(self):
        """Test the quality and format of output files."""
        results = self.demo.process_video(self.test_video_path)
        
        final_clip = results['final_clip']
        
        # Verify video properties using ffprobe
        import ffmpeg
        probe = ffmpeg.probe(final_clip)
        
        # Check video stream
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.assertEqual(int(video_stream['width']), 1080)
        self.assertEqual(int(video_stream['height']), 1920)
        self.assertEqual(video_stream['codec_name'], 'h264')
        
        # Check audio stream
        audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        self.assertEqual(audio_stream['codec_name'], 'aac')
        
        # Check duration
        duration = float(probe['format']['duration'])
        self.assertGreaterEqual(duration, self.config.min_clip_duration)
        self.assertLessEqual(duration, self.config.max_clip_duration)


class TestCommandLineInterface(unittest.TestCase):
    """Test the command-line interface."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_video_path = os.path.join(cls.test_dir, "test_video.mp4")
        cls._create_test_video(cls.test_video_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def _create_test_video(output_path: str):
        """Create a test video for testing."""
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'color=c=green:size=640x480:duration=30:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=30',
            '-c:v', 'libx264', '-c:a', 'aac', '-t', '30',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def test_cli_execution(self):
        """Test command-line execution."""
        output_dir = os.path.join(self.test_dir, "output")
        
        # Run the CLI script
        cmd = [
            sys.executable, 'clipper_demo_simple.py',
            self.test_video_path,
            '--output-dir', output_dir
        ]
        
        result = subprocess.run(
            cmd, 
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, 
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("PROCESSING COMPLETE", result.stdout)
        
        # Verify output files were created
        self.assertTrue(os.path.exists(output_dir))
        output_files = os.listdir(output_dir)
        self.assertIn("final_clip_with_subtitles.mp4", output_files)
        self.assertIn("clip_results.json", output_files)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestClipperDemo))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestCommandLineInterface))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)