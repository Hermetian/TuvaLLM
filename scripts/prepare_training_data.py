#!/usr/bin/env python3
"""
TuvaLLM Data Preparation Script
Prepares corrected transcripts for MMS fine-tuning.

Output structure:
data/processed/
├── segments/          # Individual WAV files (16kHz mono)
├── train.json         # HuggingFace datasets format
├── validation.json
├── test.json
└── vocab.json         # Tuvaluan character vocabulary for CTC
"""

import os
import sys
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Any
from collections import Counter

sys.path.insert(0, '/Users/discordwell/Library/Python/3.9/lib/python/site-packages')

import soundfile as sf
import numpy as np

# Base paths
PROJECT_ROOT = Path("/Users/discordwell/TuvaLLM")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Output directories
SEGMENTS_DIR = PROCESSED_DIR / "segments"


def load_corrections() -> Dict[str, Any]:
    """Load corrected transcripts."""
    corrections_file = PROCESSED_DIR / "corrections" / "corrections.json"
    if not corrections_file.exists():
        print(f"Error: No corrections found at {corrections_file}")
        print("Run transcript_corrector.py first to create corrections.")
        return {}
    with open(corrections_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mms_transcript() -> Dict[str, Any]:
    """Load original MMS transcript as fallback."""
    mms_file = TRANSCRIPTS_DIR / "mms_samoan_transcript.json"
    if mms_file.exists():
        with open(mms_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze().numpy(), target_sr


def extract_segment(waveform: np.ndarray, sample_rate: int,
                   start_time: float, end_time: float) -> np.ndarray:
    """Extract audio segment from waveform."""
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return waveform[start_sample:end_sample]


def normalize_text(text: str) -> str:
    """Normalize transcript text for training."""
    # Convert to lowercase
    text = text.lower()

    # Keep Polynesian characters including macrons
    # Valid chars: a-z, āēīōū, space, apostrophe
    valid_chars = set("abcdefghijklmnopqrstuvwxyz āēīōū'")

    # Replace common variants
    text = text.replace("'", "'")  # Normalize apostrophes
    text = text.replace("'", "'")
    text = text.replace("`", "'")

    # Filter to valid characters
    text = "".join(c if c in valid_chars else " " for c in text)

    # Collapse multiple spaces
    text = " ".join(text.split())

    return text.strip()


def build_vocabulary(texts: List[str]) -> Dict[str, int]:
    """Build character vocabulary for CTC training."""
    # Count all characters
    char_counts = Counter()
    for text in texts:
        char_counts.update(text)

    # Build vocabulary with special tokens
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,  # Word boundary (CTC convention)
    }

    # Add characters sorted by frequency
    for char, _ in char_counts.most_common():
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


def prepare_segments(
    corrections: Dict[str, Any],
    mms_transcript: Dict[str, Any],
    min_quality: int = 3,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    use_uncorrected: bool = False
) -> List[Dict[str, Any]]:
    """
    Prepare training segments from corrections.

    Args:
        corrections: Corrected transcripts
        mms_transcript: Original MMS transcript (fallback)
        min_quality: Minimum quality score to include
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        use_uncorrected: Include uncorrected MMS segments with default quality

    Returns:
        List of segment dictionaries
    """
    segments = []
    corrected_segments = corrections.get("segments", {})

    # First, add corrected segments that meet quality threshold
    for seg_key, seg_data in corrected_segments.items():
        quality = seg_data.get("quality", 0)
        if quality < min_quality:
            continue

        text = seg_data.get("corrected_text", "")
        if not text.strip():
            continue

        start = seg_data.get("start", 0)
        end = seg_data.get("end", 0)
        duration = end - start

        if duration < min_duration or duration > max_duration:
            continue

        # Normalize text
        normalized = normalize_text(text)
        if not normalized:
            continue

        segments.append({
            "start": start,
            "end": end,
            "duration": duration,
            "text": normalized,
            "original_text": text,
            "quality": quality,
            "source": "corrected"
        })

    # Optionally add uncorrected MMS segments
    if use_uncorrected and mms_transcript:
        mms_segments = mms_transcript.get("segments", [])
        corrected_keys = set(corrected_segments.keys())

        for seg in mms_segments:
            seg_key = f"{seg['start']:.1f}-{seg['end']:.1f}"
            if seg_key in corrected_keys:
                continue  # Already handled

            text = seg.get("text", "")
            if not text.strip():
                continue

            start = seg["start"]
            end = seg["end"]
            duration = end - start

            if duration < min_duration or duration > max_duration:
                continue

            normalized = normalize_text(text)
            if not normalized:
                continue

            segments.append({
                "start": start,
                "end": end,
                "duration": duration,
                "text": normalized,
                "original_text": text,
                "quality": 2,  # Default quality for uncorrected
                "source": "mms_uncorrected"
            })

    return segments


def split_data(
    segments: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split segments into train/val/test sets."""
    random.seed(seed)
    random.shuffle(segments)

    n = len(segments)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = segments[:train_end]
    val = segments[train_end:val_end]
    test = segments[val_end:]

    return train, val, test


def save_audio_segments(
    segments: List[Dict],
    waveform: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    prefix: str = "seg"
) -> List[Dict]:
    """Extract and save audio segments as WAV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    for i, seg in enumerate(segments):
        # Extract audio
        audio = extract_segment(waveform, sample_rate, seg["start"], seg["end"])

        # Generate filename
        filename = f"{prefix}_{i:04d}_{seg['start']:.1f}_{seg['end']:.1f}.wav"
        filepath = output_dir / filename

        # Save as WAV
        sf.write(str(filepath), audio, sample_rate)

        # Update segment with audio path
        seg_copy = seg.copy()
        seg_copy["audio_path"] = str(filepath)
        seg_copy["audio_filename"] = filename
        processed.append(seg_copy)

    return processed


def create_huggingface_format(segments: List[Dict], output_file: Path):
    """Save segments in HuggingFace datasets JSON format."""
    # HuggingFace format expects: audio path, text, optional metadata
    hf_data = []
    for seg in segments:
        hf_data.append({
            "audio": seg["audio_path"],
            "sentence": seg["text"],
            "duration": seg["duration"],
            "quality": seg.get("quality", 0)
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hf_data, f, ensure_ascii=False, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data for TuvaLLM")
    parser.add_argument("--min-quality", type=int, default=3,
                       help="Minimum quality score to include (default: 3)")
    parser.add_argument("--use-uncorrected", action="store_true",
                       help="Include uncorrected MMS segments")
    parser.add_argument("--audio", type=str, default="grn_tuvalu_01.mp3",
                       help="Audio file to process")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for splitting")
    args = parser.parse_args()

    print("=" * 70)
    print("TuvaLLM Data Preparation")
    print("=" * 70)
    print()

    # Load data
    print("Loading corrections...")
    corrections = load_corrections()
    if not corrections:
        # Create sample corrections for testing if none exist
        print("\nNo corrections found. Creating from MMS transcript...")
        mms = load_mms_transcript()
        if mms:
            corrections = {
                "segments": {},
                "vocabulary": {},
                "metadata": {"audio_file": args.audio}
            }
            # Use MMS segments with default quality of 3
            for seg in mms.get("segments", []):
                seg_key = f"{seg['start']:.1f}-{seg['end']:.1f}"
                corrections["segments"][seg_key] = {
                    "original_mms": seg["text"],
                    "corrected_text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "quality": 3
                }
            print(f"Created {len(corrections['segments'])} segments from MMS transcript")
        else:
            print("Error: No MMS transcript found either!")
            return

    mms_transcript = load_mms_transcript()

    print(f"Loaded {len(corrections.get('segments', {}))} corrected segments")

    # Prepare segments
    print(f"\nPreparing segments (min_quality={args.min_quality})...")
    segments = prepare_segments(
        corrections,
        mms_transcript,
        min_quality=args.min_quality,
        use_uncorrected=args.use_uncorrected
    )
    print(f"Found {len(segments)} usable segments")

    if not segments:
        print("No segments found! Check corrections and quality threshold.")
        return

    # Load audio
    audio_path = RAW_DIR / args.audio
    print(f"\nLoading audio: {audio_path}")
    waveform, sample_rate = load_audio(str(audio_path))
    print(f"Audio: {len(waveform)/sample_rate:.1f}s at {sample_rate}Hz")

    # Split data
    print("\nSplitting data (80/10/10)...")
    train, val, test = split_data(segments, seed=args.seed)
    print(f"  Train: {len(train)} segments")
    print(f"  Validation: {len(val)} segments")
    print(f"  Test: {len(test)} segments")

    # Save audio segments
    print("\nExtracting audio segments...")
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    train = save_audio_segments(train, waveform, sample_rate, SEGMENTS_DIR / "train", "train")
    val = save_audio_segments(val, waveform, sample_rate, SEGMENTS_DIR / "val", "val")
    test = save_audio_segments(test, waveform, sample_rate, SEGMENTS_DIR / "test", "test")

    print(f"Saved audio segments to {SEGMENTS_DIR}")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    all_texts = [s["text"] for s in train + val + test]
    vocab = build_vocabulary(all_texts)
    print(f"Vocabulary size: {len(vocab)} characters")

    # Save vocabulary
    vocab_file = PROCESSED_DIR / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {vocab_file}")

    # Save HuggingFace format
    print("\nSaving HuggingFace datasets format...")
    create_huggingface_format(train, PROCESSED_DIR / "train.json")
    create_huggingface_format(val, PROCESSED_DIR / "validation.json")
    create_huggingface_format(test, PROCESSED_DIR / "test.json")

    # Summary
    print("\n" + "=" * 70)
    print("Data Preparation Complete!")
    print("=" * 70)

    total_duration = sum(s["duration"] for s in train + val + test)
    train_duration = sum(s["duration"] for s in train)

    print(f"\nTotal data: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Training data: {train_duration:.1f}s ({train_duration/60:.1f} min)")
    print(f"Vocabulary: {len(vocab)} characters")

    print(f"\nOutput files:")
    print(f"  {PROCESSED_DIR / 'train.json'}")
    print(f"  {PROCESSED_DIR / 'validation.json'}")
    print(f"  {PROCESSED_DIR / 'test.json'}")
    print(f"  {PROCESSED_DIR / 'vocab.json'}")
    print(f"  {SEGMENTS_DIR}/")

    # Show vocabulary
    print("\nVocabulary characters:")
    chars = [c for c in vocab.keys() if len(c) == 1]
    print("  " + " ".join(sorted(chars)))


if __name__ == "__main__":
    main()
