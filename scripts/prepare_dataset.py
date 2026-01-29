#!/usr/bin/env python3
"""
Prepare Final Dataset (HuggingFace Format).

Updated to use confident segments from the improved alignment pipeline:
1. Load segments from segments_confident (high-confidence aligned data)
2. Filter by confidence threshold (default 0.6)
3. Slice audio files into clips (dataset/clips/)
4. Create train/dev/test splits (dataset/train.jsonl, etc)
"""

import json
import random
import argparse
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# New confident segments directory (from improved alignment pipeline)
SEGMENTS_CONFIDENT_DIR = PROCESSED_DIR / "segments_confident"
# Legacy directory (fallback)
SEGMENTS_ANCHORED_DIR = PROCESSED_DIR / "segments_anchored"

DATASET_DIR = PROCESSED_DIR / "dataset"
CLIPS_DIR = DATASET_DIR / "clips"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_segments(
    segments_dir: Path = SEGMENTS_CONFIDENT_DIR,
    min_confidence: float = 0.6
) -> list:
    """
    Load segments from JSON files in segments directory.

    Supports both old format (*_segments.json) and new format (confident_segments.json).
    Filters by minimum confidence threshold.
    """
    segments = []

    # Try new format first (single confident_segments.json)
    confident_path = segments_dir / "confident_segments.json"
    if confident_path.exists():
        print(f"Loading from {confident_path}")
        with open(confident_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for s in data:
                # Filter by confidence
                if s.get("confidence", 1.0) >= min_confidence:
                    # Infer session from audio_path if not present
                    if "session" not in s:
                        audio_path = s.get("audio_path", "")
                        if "june" in audio_path.lower() or "24-06-24" in audio_path:
                            s["session"] = "june"
                        elif "december" in audio_path.lower() or "12-24" in audio_path:
                            s["session"] = "december"
                        else:
                            s["session"] = "unknown"
                    segments.append(s)
        print(f"Loaded {len(segments)} segments (confidence >= {min_confidence})")
        return segments

    # Fallback to old format (*_segments.json)
    print(f"Looking for segment files in {segments_dir}")
    for f in sorted(segments_dir.glob("*_segments.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            session = f.stem.replace("_segments", "")
            for s in data:
                s["session"] = session
                # Old format doesn't have confidence, include all
                segments.append(s)

    # Also check anchored_segments.json
    anchored_path = segments_dir / "anchored_segments.json"
    if anchored_path.exists() and not segments:
        print(f"Loading from {anchored_path}")
        with open(anchored_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for s in data:
                if "session" not in s:
                    audio_path = s.get("audio_path", "")
                    if "june" in audio_path.lower() or "24-06-24" in audio_path:
                        s["session"] = "june"
                    elif "december" in audio_path.lower() or "12-24" in audio_path:
                        s["session"] = "december"
                    else:
                        s["session"] = "unknown"
                segments.append(s)

    print(f"Loaded {len(segments)} segments")
    return segments


def process_segments(segments: list, output_clips_dir: Path = CLIPS_DIR) -> list:
    """Slice audio and create dataset entries."""
    dataset_entries = []

    print(f"Processing {len(segments)} segments...")

    # Group by audio file to minimize file I/O
    by_audio = defaultdict(list)
    for i, s in enumerate(segments):
        by_audio[s.get("audio_path")].append((i, s))

    clip_count = 0
    total_duration = 0
    skipped_missing = 0
    skipped_duration = 0

    for audio_path_str, items in tqdm(by_audio.items(), desc="Slicing audio"):
        if not audio_path_str:
            continue

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            skipped_missing += len(items)
            continue

        try:
            # Load full audio
            audio, sr = sf.read(audio_path, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            for idx, seg in items:
                start_sec = seg["start"]
                end_sec = seg["end"]
                duration = end_sec - start_sec

                # Sanity check
                if duration < 0.5 or duration > 30.0:
                    skipped_duration += 1
                    continue

                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)

                # Boundary check
                if end_sample > len(audio):
                    end_sample = len(audio)

                if end_sample <= start_sample:
                    continue

                clip_audio = audio[start_sample:end_sample]

                # Save clip
                clip_name = f"{seg['session']}_{idx:05d}.wav"
                clip_path = output_clips_dir / clip_name
                sf.write(clip_path, clip_audio, sr)

                entry = {
                    "audio": str(clip_path.relative_to(PROJECT_ROOT)),
                    "file": str(clip_path.absolute()),
                    "text": seg["text"],
                    "duration": duration,
                    "session": seg["session"],
                }

                # Include confidence if available
                if "confidence" in seg:
                    entry["confidence"] = seg["confidence"]

                dataset_entries.append(entry)
                clip_count += 1
                total_duration += duration

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")

    print(f"\nCreated {clip_count} clips.")
    print(f"Total Duration: {total_duration/3600:.2f} hours")
    if skipped_missing:
        print(f"Skipped (missing audio): {skipped_missing}")
    if skipped_duration:
        print(f"Skipped (bad duration): {skipped_duration}")

    return dataset_entries


def create_splits(entries: list, output_dir: Path = DATASET_DIR) -> None:
    """Create Train/Dev/Test splits."""
    random.seed(42)
    random.shuffle(entries)

    total = len(entries)
    n_test = min(int(total * 0.05), 500)
    n_val = min(int(total * 0.05), 500)

    test_set = entries[:n_test]
    val_set = entries[n_test:n_test + n_val]
    train_set = entries[n_test + n_val:]

    print(f"Train: {len(train_set)}")
    print(f"Val:   {len(val_set)}")
    print(f"Test:  {len(test_set)}")

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_jsonl(train_set, output_dir / "train.jsonl")
    save_jsonl(val_set, output_dir / "valid.jsonl")
    save_jsonl(test_set, output_dir / "test.jsonl")

    # Save dataset metadata
    metadata = {
        "total_entries": total,
        "train_size": len(train_set),
        "valid_size": len(val_set),
        "test_size": len(test_set),
        "total_duration_hours": sum(e["duration"] for e in entries) / 3600,
    }

    if entries and "confidence" in entries[0]:
        metadata["avg_confidence"] = sum(e.get("confidence", 0) for e in entries) / total

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare TuvaLLM training dataset")
    parser.add_argument("--segments-dir", type=Path, default=SEGMENTS_CONFIDENT_DIR,
                        help="Directory containing segment JSON files")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Minimum confidence threshold (default: 0.6)")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR,
                        help="Output directory for dataset")
    parser.add_argument("--use-legacy", action="store_true",
                        help="Use legacy segments_anchored directory")
    args = parser.parse_args()

    if args.use_legacy:
        args.segments_dir = SEGMENTS_ANCHORED_DIR

    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)
    print(f"Segments directory: {args.segments_dir}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Output directory: {args.output_dir}")

    # Ensure output directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Load segments
    segments = load_all_segments(args.segments_dir, args.min_confidence)

    if not segments:
        print("\nNo segments found!")
        print("Run the alignment pipeline first:")
        print("  1. python scripts/transcribe_whisper_english.py")
        print("  2. python scripts/merge_anchor_sources.py")
        print("  3. python scripts/sequence_align.py")
        print("  4. python scripts/anchor_align.py")
        return

    # Process segments into clips
    entries = process_segments(segments, clips_dir)

    if not entries:
        print("No valid entries created!")
        return

    # Create splits
    create_splits(entries, args.output_dir)

    print(f"\nDataset ready in {args.output_dir}/")


if __name__ == "__main__":
    main()
