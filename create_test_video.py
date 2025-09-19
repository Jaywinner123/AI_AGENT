#!/usr/bin/env python3
"""
Create a test video for demonstrating the Clipper Agent.
This creates a simple video with audio that can be used to test the pipeline.
"""

import subprocess
import os

def create_test_video():
    """Create a test video with synthetic audio and visuals."""
    output_path = "test_video.mp4"
    
    # Create a 2-minute test video with:
    # - Colored background that changes
    # - Synthetic audio with speech-like patterns
    # - Text overlay showing timestamps
    
    cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-f', 'lavfi',
        '-i', 'color=c=blue:size=1280x720:duration=120:rate=30',  # Blue background, 2 minutes
        '-f', 'lavfi', 
        '-i', 'sine=frequency=440:duration=120',  # Sine wave audio
        '-filter_complex', 
        '[0:v]drawtext=text=\'Test Video for Clipper Agent\':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=100,'
        'drawtext=text=\'Time\\: %{pts\\:hms}\':fontcolor=yellow:fontsize=32:x=50:y=h-100[v]',
        '-map', '[v]', '-map', '1:a',
        '-c:v', 'libx264', '-c:a', 'aac',
        '-t', '120',  # 2 minutes duration
        output_path
    ]
    
    try:
        print("Creating test video...")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Test video created: {output_path}")
        
        # Get video info
        info_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', output_path]
        result = subprocess.run(info_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            duration = float(info['format']['duration'])
            size = int(info['format']['size'])
            print(f"Duration: {duration:.2f} seconds")
            print(f"File size: {size / (1024*1024):.2f} MB")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating test video: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    create_test_video()