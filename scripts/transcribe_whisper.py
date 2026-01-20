#!/usr/bin/env python3
"""
TuvaLLM Zero-Shot Transcription Script
Uses OpenAI Whisper to transcribe Tuvaluan audio.
Tests both auto-detect and Samoan language proxy.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Ensure we can find installed packages
sys.path.insert(0, '/Users/discordwell/Library/Python/3.9/lib/python/site-packages')

import torch
import warnings
warnings.filterwarnings("ignore")

def transcribe_with_whisper(audio_path: str, language: str = None, model_size: str = "base"):
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_path: Path to audio file
        language: Language code (None for auto-detect, 'sm' for Samoan)
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        dict with transcription results
    """
    try:
        import whisper
    except ImportError:
        print("Installing openai-whisper...")
        os.system("pip3 install -q openai-whisper")
        import whisper

    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_path}")
    print(f"Language setting: {language if language else 'auto-detect'}")

    # Transcribe
    if language:
        result = model.transcribe(audio_path, language=language, verbose=False)
    else:
        result = model.transcribe(audio_path, verbose=False)

    return {
        "text": result["text"],
        "language": result.get("language", language),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for seg in result["segments"]
        ]
    }

def main():
    raw_dir = Path("/Users/discordwell/TuvaLLM/data/raw")
    transcript_dir = Path("/Users/discordwell/TuvaLLM/data/transcripts")
    transcript_dir.mkdir(exist_ok=True)

    audio_files = sorted(raw_dir.glob("grn_tuvalu_*.mp3"))

    if not audio_files:
        print("No audio files found!")
        return

    print("=" * 70)
    print("TuvaLLM Zero-Shot Transcription")
    print("=" * 70)
    print()

    # We'll test with the first few minutes of the first file
    # to save time, then do full transcription

    all_results = {}

    for audio_file in audio_files:
        print(f"\n{'='*70}")
        print(f"Processing: {audio_file.name}")
        print("=" * 70)

        # Test 1: Auto-detect language
        print("\n[Test 1] Auto-detect language...")
        try:
            result_auto = transcribe_with_whisper(
                str(audio_file),
                language=None,
                model_size="base"
            )
            print(f"  Detected language: {result_auto['language']}")
            print(f"  Segments: {len(result_auto['segments'])}")

            # Show first few segments
            print("\n  First 5 segments (auto-detect):")
            for seg in result_auto['segments'][:5]:
                print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

            all_results[f"{audio_file.stem}_auto"] = result_auto

        except Exception as e:
            print(f"  Error: {e}")
            result_auto = None

        # Test 2: Force Samoan language
        print("\n[Test 2] Force Samoan language (proxy)...")
        try:
            result_samoan = transcribe_with_whisper(
                str(audio_file),
                language="sm",  # Samoan
                model_size="base"
            )
            print(f"  Segments: {len(result_samoan['segments'])}")

            # Show first few segments
            print("\n  First 5 segments (Samoan proxy):")
            for seg in result_samoan['segments'][:5]:
                print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

            all_results[f"{audio_file.stem}_samoan"] = result_samoan

        except Exception as e:
            print(f"  Error: {e}")
            result_samoan = None

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = transcript_dir / f"whisper_transcripts_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    # Print full transcripts
    print("\n" + "=" * 70)
    print("FULL TRANSCRIPTS")
    print("=" * 70)

    for key, result in all_results.items():
        print(f"\n--- {key} ---")
        print(f"Detected/Set Language: {result['language']}")
        print(f"\nFull text:\n")
        print(result['text'][:3000])  # First 3000 chars
        if len(result['text']) > 3000:
            print(f"\n... [truncated, {len(result['text'])} chars total]")
        print()

if __name__ == "__main__":
    main()
