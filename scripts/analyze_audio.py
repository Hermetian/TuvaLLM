#!/usr/bin/env python3
"""
TuvaLLM Audio Analysis Script
Analyzes Tuvaluan audio files and attempts zero-shot transcription.
"""

import os
import sys
from pathlib import Path

# Add Python user bin to path for installed packages
sys.path.insert(0, '/Users/discordwell/Library/Python/3.9/lib/python/site-packages')

import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AudioInfo:
    """Audio file information"""
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    file_size_mb: float

def analyze_audio_file(filepath: str) -> AudioInfo:
    """Analyze a single audio file and return its properties."""
    path = Path(filepath)

    # Get file size
    file_size_mb = path.stat().st_size / (1024 * 1024)

    # Load audio info without loading full file
    info = sf.info(filepath)

    # Load a small sample to verify
    y, sr = librosa.load(filepath, sr=None, duration=10)

    return AudioInfo(
        filename=path.name,
        duration_sec=info.duration,
        sample_rate=info.samplerate,
        channels=info.channels,
        bit_depth=info.subtype_info if hasattr(info, 'subtype_info') else None,
        file_size_mb=file_size_mb
    )

def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def main():
    # Find audio files
    raw_dir = Path("/Users/discordwell/TuvaLLM/data/raw")
    audio_files = list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))

    if not audio_files:
        print("No audio files found in", raw_dir)
        return

    print("=" * 60)
    print("TuvaLLM Audio Analysis Report")
    print("=" * 60)
    print()

    total_duration = 0
    total_size = 0

    for filepath in sorted(audio_files):
        print(f"Analyzing: {filepath.name}")
        try:
            info = analyze_audio_file(str(filepath))
            total_duration += info.duration_sec
            total_size += info.file_size_mb

            print(f"  Duration:    {format_duration(info.duration_sec)} ({info.duration_sec:.1f}s)")
            print(f"  Sample Rate: {info.sample_rate} Hz")
            print(f"  Channels:    {info.channels}")
            print(f"  File Size:   {info.file_size_mb:.2f} MB")
            print()
        except Exception as e:
            print(f"  Error: {e}")
            print()

    print("-" * 60)
    print(f"Total Files:    {len(audio_files)}")
    print(f"Total Duration: {format_duration(total_duration)} ({total_duration:.1f}s)")
    print(f"Total Size:     {total_size:.2f} MB")
    print()

    # Quality assessment for ASR
    print("=" * 60)
    print("ASR Training Assessment")
    print("=" * 60)
    print()

    # Check sample rate
    sample_rates = set()
    for filepath in audio_files:
        info = sf.info(str(filepath))
        sample_rates.add(info.samplerate)

    if 16000 in sample_rates or any(sr >= 16000 for sr in sample_rates):
        print("[OK] Sample rate >= 16kHz (suitable for ASR)")
    else:
        print(f"[WARN] Sample rate < 16kHz ({sample_rates})")
        print("       May need upsampling for some ASR models")

    # Check total duration
    if total_duration >= 3600:  # 1 hour
        print(f"[OK] Total duration >= 1 hour ({total_duration/3600:.1f}h)")
    elif total_duration >= 1800:  # 30 min
        print(f"[WARN] Total duration ~{total_duration/60:.0f} min (minimal for fine-tuning)")
    else:
        print(f"[WARN] Total duration ~{total_duration/60:.0f} min (very limited)")

    print()
    print("Next Steps:")
    print("  1. Run zero-shot transcription with Whisper/MMS (Samoan proxy)")
    print("  2. Manually verify/correct transcriptions")
    print("  3. Segment audio into training chunks (5-30s)")
    print("  4. Fine-tune ASR model")

if __name__ == "__main__":
    main()
