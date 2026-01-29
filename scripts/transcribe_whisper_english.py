#!/usr/bin/env python3
"""
Whisper English Transcription for TuvaLLM.

Runs Whisper large-v3 with language="en" on parliament audio to capture:
- English loanwords: "parliament", "minister", "budget", "climate"
- Proper names: "Feleti Teo", "Simon Kofe"
- Titles: "Honourable", "Prime Minister"

Output matches MMS JSON format for easy merging.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import whisper

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio_16k"
OUTPUT_DIR = PROCESSED_DIR / "whisper_english_transcripts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_audio(model, audio_path: Path, chunk_size: float = 10.0):
    """
    Transcribe audio with Whisper English, returning MMS-compatible format.

    Args:
        model: Loaded Whisper model
        audio_path: Path to audio file
        chunk_size: Target chunk duration in seconds for output format

    Returns:
        List of dicts with 'text', 'start', 'end', 'source' keys
    """
    result = model.transcribe(
        str(audio_path),
        language="en",
        verbose=False,
        word_timestamps=True,  # Get word-level timestamps if available
    )

    # Convert Whisper segments to MMS-compatible format
    # Whisper provides segment-level timestamps, we'll chunk them
    chunks = []
    current_chunk = {
        "text": "",
        "start": 0.0,
        "end": 0.0,
        "source": "whisper_en"
    }

    for segment in result.get("segments", []):
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_text = segment["text"].strip()

        # If this is the first segment or close to previous
        if not current_chunk["text"]:
            current_chunk["start"] = seg_start
            current_chunk["text"] = seg_text
            current_chunk["end"] = seg_end
        elif seg_start - current_chunk["start"] < chunk_size:
            # Append to current chunk
            current_chunk["text"] += " " + seg_text
            current_chunk["end"] = seg_end
        else:
            # Save current chunk and start new one
            if current_chunk["text"].strip():
                chunks.append(current_chunk.copy())
            current_chunk = {
                "text": seg_text,
                "start": seg_start,
                "end": seg_end,
                "source": "whisper_en"
            }

    # Don't forget the last chunk
    if current_chunk["text"].strip():
        chunks.append(current_chunk)

    # Normalize to consistent chunk boundaries (like MMS does with 10s chunks)
    normalized = []
    for chunk in chunks:
        # Round start to nearest chunk_size boundary for consistency
        chunk_start = int(chunk["start"] / chunk_size) * chunk_size
        chunk_end = chunk_start + chunk_size

        # Check if we need to merge with existing chunk at same boundary
        if normalized and normalized[-1]["start"] == chunk_start:
            normalized[-1]["text"] += " " + chunk["text"]
            normalized[-1]["end"] = max(normalized[-1]["end"], chunk["end"])
        else:
            normalized.append({
                "text": chunk["text"],
                "start": chunk_start,
                "end": max(chunk_end, chunk["end"]),
                "source": "whisper_en"
            })

    return normalized if normalized else chunks


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper English")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR, help="Input audio directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing transcripts")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("Whisper English Transcription")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Output dir: {args.output_dir}")

    # Load model
    print(f"\nLoading Whisper {args.model}...")
    model = whisper.load_model(args.model, device=device)
    print("Model loaded.")

    # Find audio files
    audio_files = sorted(args.audio_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files.")

    if not audio_files:
        print("No audio files found!")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    processed = 0
    skipped = 0

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        output_path = args.output_dir / (audio_path.stem + ".json")

        if output_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            chunks = transcribe_audio(model, audio_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            processed += 1

        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Processed: {processed}")
    print(f"Skipped (existing): {skipped}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
