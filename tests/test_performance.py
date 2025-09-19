#!/usr/bin/env python3
"""
Performance and load testing for Clipper Agent.

This module contains performance benchmarks and load tests to ensure
the system can handle various workloads efficiently.
"""

import unittest
import time
import tempfile
import os
import shutil
import subprocess
import sys
import psutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_config import SimpleConfig
from clipper_demo_simple import SimpleClipperDemo


class TestPerformance(unittest.TestCase):
    """Performance tests for the Clipper Agent."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.config = SimpleConfig()
        cls.config.storage_path = os.path.join(cls.test_dir, "storage")
        cls.config.temp_path = os.path.join(cls.test_dir, "temp")
        cls.config.models_path = os.path.join(cls.test_dir, "models")
        
        cls.demo = SimpleClipperDemo(cls.config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def _create_test_video(self, duration: int, resolution: str = "1280x720") -> str:
        """Create a test video with specified duration and resolution."""
        output_path = os.path.join(self.test_dir, f"test_video_{duration}s_{resolution}.mp4")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'color=c=blue:size={resolution}:duration={duration}:rate=30',
            '-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}',
            '-c:v', 'libx264', '-c:a', 'aac', '-t', str(duration),
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def test_processing_time_short_video(self):
        """Test processing time for short videos (30s)."""
        video_path = self._create_test_video(30)
        
        start_time = time.time()
        results = self.demo.process_video(video_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Short video should process quickly (under 30 seconds)
        self.assertLess(processing_time, 30.0)
        self.assertIsNotNone(results)
        
        print(f"Short video (30s) processing time: {processing_time:.2f}s")
    
    def test_processing_time_medium_video(self):
        """Test processing time for medium videos (2 minutes)."""
        video_path = self._create_test_video(120)
        
        start_time = time.time()
        results = self.demo.process_video(video_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Medium video should process in reasonable time (under 2 minutes)
        self.assertLess(processing_time, 120.0)
        self.assertIsNotNone(results)
        
        print(f"Medium video (2m) processing time: {processing_time:.2f}s")
    
    def test_processing_time_long_video(self):
        """Test processing time for longer videos (5 minutes)."""
        video_path = self._create_test_video(300)
        
        start_time = time.time()
        results = self.demo.process_video(video_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Long video should still process in reasonable time (under 5 minutes)
        self.assertLess(processing_time, 300.0)
        self.assertIsNotNone(results)
        
        print(f"Long video (5m) processing time: {processing_time:.2f}s")
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        video_path = self._create_test_video(120)
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process video
        results = self.demo.process_video(video_path)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory usage should be reasonable (under 500MB increase)
        self.assertLess(memory_increase, 500.0)
        self.assertIsNotNone(results)
        
        print(f"Memory usage increase: {memory_increase:.2f}MB")
    
    def test_disk_space_usage(self):
        """Test disk space usage during processing."""
        video_path = self._create_test_video(120)
        
        # Get initial disk usage
        initial_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(self.test_dir)
            for filename in filenames
        )
        
        # Process video
        results = self.demo.process_video(video_path)
        
        # Get final disk usage
        final_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(self.test_dir)
            for filename in filenames
        )
        
        disk_increase = (final_size - initial_size) / 1024 / 1024  # MB
        
        # Disk usage should be reasonable
        self.assertLess(disk_increase, 100.0)  # Under 100MB increase
        self.assertIsNotNone(results)
        
        print(f"Disk space increase: {disk_increase:.2f}MB")
    
    def test_concurrent_processing(self):
        """Test concurrent processing of multiple videos."""
        import threading
        import queue
        
        # Create multiple test videos
        video_paths = [
            self._create_test_video(30, "640x480"),
            self._create_test_video(45, "640x480"),
            self._create_test_video(60, "640x480")
        ]
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def process_video_thread(video_path):
            try:
                # Create separate demo instance for each thread
                config = SimpleConfig()
                config.storage_path = os.path.join(self.test_dir, f"storage_{threading.current_thread().ident}")
                config.temp_path = os.path.join(self.test_dir, f"temp_{threading.current_thread().ident}")
                config.models_path = self.config.models_path
                
                demo = SimpleClipperDemo(config)
                result = demo.process_video(video_path)
                results_queue.put(result)
            except Exception as e:
                errors_queue.put(e)
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for video_path in video_paths:
            thread = threading.Thread(target=process_video_thread, args=(video_path,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check results
        self.assertEqual(results_queue.qsize(), len(video_paths))
        self.assertEqual(errors_queue.qsize(), 0)
        
        # Concurrent processing should be faster than sequential
        # (though this is a simple test, real benefits depend on I/O vs CPU bound operations)
        print(f"Concurrent processing time for {len(video_paths)} videos: {total_time:.2f}s")
    
    def test_scalability_different_resolutions(self):
        """Test processing time with different video resolutions."""
        resolutions = ["640x480", "1280x720", "1920x1080"]
        processing_times = {}
        
        for resolution in resolutions:
            video_path = self._create_test_video(60, resolution)
            
            start_time = time.time()
            results = self.demo.process_video(video_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times[resolution] = processing_time
            
            self.assertIsNotNone(results)
            print(f"Processing time for {resolution}: {processing_time:.2f}s")
        
        # Higher resolution should take more time, but not exponentially more
        self.assertLess(processing_times["640x480"], processing_times["1920x1080"])
        
        # 1080p shouldn't take more than 3x the time of 480p
        ratio = processing_times["1920x1080"] / processing_times["640x480"]
        self.assertLess(ratio, 3.0)


class TestResourceLimits(unittest.TestCase):
    """Test system resource limits and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.config = SimpleConfig()
        cls.config.storage_path = os.path.join(cls.test_dir, "storage")
        cls.config.temp_path = os.path.join(cls.test_dir, "temp")
        cls.config.models_path = os.path.join(cls.test_dir, "models")
        
        cls.demo = SimpleClipperDemo(cls.config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_invalid_video_handling(self):
        """Test handling of invalid video files."""
        # Create an invalid video file (just text)
        invalid_video = os.path.join(self.test_dir, "invalid.mp4")
        with open(invalid_video, 'w') as f:
            f.write("This is not a video file")
        
        # Should handle gracefully
        result = self.demo.validate_input(invalid_video)
        self.assertFalse(result)
    
    def test_very_short_video_handling(self):
        """Test handling of very short videos."""
        # Create a 5-second video (below minimum)
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'color=c=red:size=640x480:duration=5:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=5',
            '-c:v', 'libx264', '-c:a', 'aac', '-t', '5',
            os.path.join(self.test_dir, "short.mp4")
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        short_video = os.path.join(self.test_dir, "short.mp4")
        
        # Should reject video that's too short
        result = self.demo.validate_input(short_video)
        self.assertFalse(result)
    
    def test_corrupted_video_handling(self):
        """Test handling of corrupted video files."""
        # Create a partially corrupted video by truncating a valid one
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'color=c=blue:size=640x480:duration=30:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=30',
            '-c:v', 'libx264', '-c:a', 'aac', '-t', '30',
            os.path.join(self.test_dir, "full.mp4")
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Truncate the file to simulate corruption
        full_video = os.path.join(self.test_dir, "full.mp4")
        corrupted_video = os.path.join(self.test_dir, "corrupted.mp4")
        
        with open(full_video, 'rb') as src, open(corrupted_video, 'wb') as dst:
            # Copy only first 1KB to simulate corruption
            dst.write(src.read(1024))
        
        # Should handle corrupted video gracefully
        result = self.demo.validate_input(corrupted_video)
        self.assertFalse(result)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceLimits))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)