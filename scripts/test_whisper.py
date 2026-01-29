#!/usr/bin/env python3
"""
Test Whisper transcription on a single file to verify quality and timestamps.
"""

import whisper
import torch
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
SAMPLE_FILE = AUDIO_DIR / "research_materials_Research_Materials-20260128T034747Z-3-001_Research_Materials_Parliament_Session_Recordings_Parliament_Session_Recording_Parliament_Session_December_2024_Day_1_Parliamnet_Afternoon_Session_Day_1_-_09-12-24.wav"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "whisper_test_sample.json"

def main():
    print("="*60)
    print("Whisper Transcription Test")
    print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print(f"File: {SAMPLE_FILE.name}")
    print("="*60)
    
    if not SAMPLE_FILE.exists():
        print(f"Error: {SAMPLE_FILE} not found!")
        return

    # Load model (use small for speed in test, large for real run)
    print("Loading Whisper model (large-v3)...")
    model_dir = PROJECT_ROOT / "data" / "models" / "whisper"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model("large-v3", device="cpu", download_root=str(model_dir))
    # if torch.backends.mps.is_available():
    #     print("Moving to MPS...")
    #     model = model.to("mps")
    
    print("Transcribing...")
    result = model.transcribe(
        str(SAMPLE_FILE),
        # language="tuvalu", # Invalid
        verbose=True
    )
    
    print("\nTranscription complete!")
    print(f"Detected language: {result.get('language')}")
    print(f"Segments: {len(result['segments'])}")
    
    # Save result
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"Saved to {OUTPUT_FILE}")
    
    # Print sample
    print("\nSample Segments:")
    for seg in result["segments"][:5]:
        print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")

if __name__ == "__main__":
    main()
