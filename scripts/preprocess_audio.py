#!/usr/bin/env python3
"""
Preprocess audio: Find all MP3/MP4/WAV files and convert to 16kHz Mono WAV.
"""

import os
import subprocess
from pathlib import Path
import multiprocessing

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_audio_files():
    """Find all audio/video files in RAW_DIR."""
    extensions = {".mp3", ".mp4", ".wav", ".m4a"}
    files = []
    for ext in extensions:
        files.extend(list(RAW_DIR.rglob(f"*{ext}")))
    return sorted(list(set(files)))

def convert_to_wav(input_path):
    """Convert single file to 16kHz mono WAV using ffmpeg."""
    try:
        # Create output filename (flatten directory structure to avoid mess)
        # e.g. "parliament/day1.mp4" -> "parliament_day1.wav"
        
        rel_path = input_path.relative_to(RAW_DIR)
        safe_name = str(rel_path).replace("/", "_").replace(" ", "_")
        filename = Path(safe_name).stem + ".wav"
        output_path = OUTPUT_DIR / filename
        
        if output_path.exists():
            print(f"Skipping {filename} (already exists)")
            return
            
        print(f"Converting {input_path.name} -> {filename}...")
        
        # ffmpeg command:
        # -i input
        # -ac 1 (mono)
        # -ar 16000 (16kHz)
        # -vn (no video)
        # -y (overwrite)
        # -loglevel error (quiet)
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            "-y",
            "-loglevel", "error",
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True)
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def main():
    print("="*60)
    print("Audio Preprocessing (16kHz Mono WAV)")
    print("="*60)
    
    files = get_audio_files()
    print(f"Found {len(files)} AV files to process.")
    
    # Process in parallel (ffmpeg is single-threaded usually, so parallel files works well)
    # But limit to modest number to avoid choking disk I/O with large files
    num_workers = min(multiprocessing.cpu_count(), 4)
    print(f"Starting conversion with {num_workers} workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(convert_to_wav, files)
        
    print("\nDone! Output files in:", OUTPUT_DIR)
    # List a few
    created = list(OUTPUT_DIR.glob("*.wav"))
    print(f"Total WAV files: {len(created)}")

if __name__ == "__main__":
    main()
