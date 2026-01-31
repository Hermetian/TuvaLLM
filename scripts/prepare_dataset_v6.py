#!/usr/bin/env python3
"""
Prepare Dataset V6 with Rough Patch Logic.

Tiered inclusion strategy:
- Tier 1: High confidence (always include)
- Tier 2: Good quality (include with WPS validation)
- Tier 3: Rough patches (include if neighbors are good)

This maximizes training data while maintaining quality.
"""

import json
import random
import argparse
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Input directories (in order of preference)
LLM_SCORED_DIR = PROCESSED_DIR / "llm_scored"
WORD_ALIGNED_DIR = PROCESSED_DIR / "segments_word_aligned"
CONFIDENT_SEGMENTS_DIR = PROCESSED_DIR / "segments_confident"

DATASET_DIR = PROCESSED_DIR / "dataset_v6"
CLIPS_DIR = DATASET_DIR / "clips"

# Tier thresholds
TIER1_LLM = 0.7
TIER1_ALIGN = 0.6
TIER2_LLM = 0.5
TIER2_ALIGN = 0.5
TIER3_LLM = 0.4
TIER3_ALIGN = 0.4
NEIGHBOR_THRESHOLD = 0.6  # Average neighbor score for rough patch inclusion

# WPS validation
MIN_WPS = 1.0
MAX_WPS = 4.5


def load_segments(
    llm_scored_dir: Path = LLM_SCORED_DIR,
    word_aligned_dir: Path = WORD_ALIGNED_DIR,
    confident_dir: Path = CONFIDENT_SEGMENTS_DIR
) -> List[Dict]:
    """
    Load segments from available sources in order of preference:
    1. LLM scored segments (best)
    2. Word-aligned segments
    3. Confident segments (fallback)
    """
    # Try LLM scored first
    scored_path = llm_scored_dir / "scored_segments.json"
    if scored_path.exists():
        print(f"Loading from {scored_path}")
        with open(scored_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Try word aligned
    aligned_path = word_aligned_dir / "candidate_segments.json"
    if aligned_path.exists():
        print(f"Loading from {aligned_path}")
        with open(aligned_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
            # Add default confidence if missing
            for seg in segments:
                if "confidence" not in seg:
                    seg["confidence"] = seg.get("align_score", 0.5)
                if "llm_score" not in seg:
                    seg["llm_score"] = seg.get("align_score", 0.5)
            return segments

    # Fallback to confident segments
    confident_path = confident_dir / "confident_segments.json"
    if confident_path.exists():
        print(f"Loading from {confident_path}")
        with open(confident_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
            # Adapt old format to new
            for seg in segments:
                if "align_score" not in seg:
                    seg["align_score"] = seg.get("confidence", 0.5)
                if "llm_score" not in seg:
                    seg["llm_score"] = seg.get("confidence", 0.5)
            return segments

    return []


def calculate_tier(segment: Dict) -> Tuple[int, str]:
    """
    Determine which tier a segment belongs to.

    Returns: (tier_number, tier_name)
    """
    llm = segment.get("llm_score", segment.get("confidence", 0.5))
    align = segment.get("align_score", segment.get("confidence", 0.5))
    wps = segment.get("wps", 2.0)

    # Tier 1: High confidence
    if llm >= TIER1_LLM and align >= TIER1_ALIGN:
        return 1, "high_confidence"

    # Tier 2: Good quality
    if llm >= TIER2_LLM and align >= TIER2_ALIGN:
        if MIN_WPS <= wps <= MAX_WPS:
            return 2, "good_quality"

    # Tier 3: Potential rough patch
    if llm >= TIER3_LLM and align >= TIER3_ALIGN:
        return 3, "rough_patch"

    # Below threshold
    return 0, "rejected"


def evaluate_neighbors(
    segments: List[Dict],
    idx: int,
    window: int = 2
) -> float:
    """
    Calculate average score of neighboring segments.

    Used to determine if a rough patch should be included.
    """
    scores = []

    for offset in range(-window, window + 1):
        if offset == 0:
            continue

        neighbor_idx = idx + offset
        if 0 <= neighbor_idx < len(segments):
            neighbor = segments[neighbor_idx]
            # Only count if same audio file
            if neighbor.get("audio_path") == segments[idx].get("audio_path"):
                score = neighbor.get("confidence", 0.5)
                scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def assign_tiers(segments: List[Dict]) -> List[Dict]:
    """
    Assign tier labels to all segments and mark inclusion.

    Rough patches are included if they bridge good segments.
    """
    # Sort by audio file and start time for neighbor evaluation
    segments_sorted = sorted(
        segments,
        key=lambda s: (s.get("audio_path", ""), s.get("start", 0))
    )

    # Assign base tiers
    for seg in segments_sorted:
        tier_num, tier_name = calculate_tier(seg)
        seg["tier"] = tier_name
        seg["tier_num"] = tier_num

    # Evaluate rough patches
    for i, seg in enumerate(segments_sorted):
        if seg["tier_num"] == 3:
            neighbor_avg = evaluate_neighbors(segments_sorted, i)
            seg["neighbor_score"] = neighbor_avg

            if neighbor_avg >= NEIGHBOR_THRESHOLD:
                seg["tier"] = "bridge_segment"
                seg["include"] = True
            else:
                seg["include"] = False
        elif seg["tier_num"] > 0:
            seg["include"] = True
        else:
            seg["include"] = False

    return segments_sorted


def slice_audio_clip(
    audio_path: Path,
    start: float,
    end: float,
    output_path: Path
) -> bool:
    """Slice a clip from audio file."""
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        if end_sample > len(audio):
            end_sample = len(audio)

        if end_sample <= start_sample:
            return False

        clip = audio[start_sample:end_sample]
        sf.write(output_path, clip, sr)
        return True

    except Exception as e:
        print(f"Error slicing {audio_path}: {e}")
        return False


def process_segments(
    segments: List[Dict],
    clips_dir: Path
) -> List[Dict]:
    """Slice audio and create dataset entries for included segments."""
    dataset_entries = []

    # Filter to included segments
    included = [s for s in segments if s.get("include", False)]

    print(f"Processing {len(included)} included segments...")

    # Group by audio file
    by_audio = defaultdict(list)
    for i, seg in enumerate(included):
        by_audio[seg.get("audio_path")].append((i, seg))

    clip_count = 0
    total_duration = 0
    skipped = 0

    for audio_path_str, items in tqdm(by_audio.items(), desc="Slicing audio"):
        if not audio_path_str:
            continue

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            skipped += len(items)
            continue

        for idx, seg in items:
            start = seg["start"]
            end = seg["end"]
            duration = end - start

            # Generate clip name
            session = seg.get("session", "unknown")
            seg_id = seg.get("id", f"{session}_{idx:05d}")
            clip_name = f"{seg_id}.wav"
            clip_path = clips_dir / clip_name

            if slice_audio_clip(audio_path, start, end, clip_path):
                entry = {
                    "audio": str(clip_path.relative_to(PROJECT_ROOT)),
                    "file": str(clip_path.absolute()),
                    "text": seg["text"],
                    "duration": duration,
                    "session": session,
                    "confidence": seg.get("confidence", 0.5),
                    "llm_score": seg.get("llm_score", 0.5),
                    "align_score": seg.get("align_score", 0.5),
                    "tier": seg.get("tier", "unknown"),
                    "wps": seg.get("wps", 0),
                }
                dataset_entries.append(entry)
                clip_count += 1
                total_duration += duration

    print(f"\nCreated {clip_count} clips")
    print(f"Total duration: {total_duration / 3600:.2f} hours")
    if skipped:
        print(f"Skipped (missing audio): {skipped}")

    return dataset_entries


def create_splits(
    entries: List[Dict],
    output_dir: Path,
    test_ratio: float = 0.05,
    val_ratio: float = 0.05
) -> None:
    """Create train/dev/test splits with tier-aware stratification."""
    random.seed(42)

    # Stratify by tier for balanced splits
    by_tier = defaultdict(list)
    for entry in entries:
        by_tier[entry.get("tier", "unknown")].append(entry)

    train_set = []
    val_set = []
    test_set = []

    for tier, tier_entries in by_tier.items():
        random.shuffle(tier_entries)

        n = len(tier_entries)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_set.extend(tier_entries[:n_test])
        val_set.extend(tier_entries[n_test:n_test + n_val])
        train_set.extend(tier_entries[n_test + n_val:])

    # Shuffle final sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    print(f"\nTrain: {len(train_set)}")
    print(f"Val:   {len(val_set)}")
    print(f"Test:  {len(test_set)}")

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_jsonl(train_set, output_dir / "train.jsonl")
    save_jsonl(val_set, output_dir / "valid.jsonl")
    save_jsonl(test_set, output_dir / "test.jsonl")

    # Metadata
    total = len(entries)
    metadata = {
        "total_entries": total,
        "train_size": len(train_set),
        "valid_size": len(val_set),
        "test_size": len(test_set),
        "total_duration_hours": sum(e["duration"] for e in entries) / 3600,
        "avg_confidence": sum(e["confidence"] for e in entries) / total if total else 0,
        "avg_llm_score": sum(e["llm_score"] for e in entries) / total if total else 0,
        "avg_align_score": sum(e["align_score"] for e in entries) / total if total else 0,
        "tier_distribution": {
            tier: len(items) for tier, items in by_tier.items()
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare TuvaLLM dataset V6")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR,
                        help="Output directory")
    parser.add_argument("--tier1-llm", type=float, default=TIER1_LLM,
                        help="Tier 1 LLM score threshold")
    parser.add_argument("--tier2-llm", type=float, default=TIER2_LLM,
                        help="Tier 2 LLM score threshold")
    parser.add_argument("--no-rough-patches", action="store_true",
                        help="Exclude rough patches (tier 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Preparation V6 (Tiered Inclusion)")
    print("=" * 60)
    print(f"Tier 1 (high confidence): LLM >= {TIER1_LLM}, Align >= {TIER1_ALIGN}")
    print(f"Tier 2 (good quality): LLM >= {TIER2_LLM}, Align >= {TIER2_ALIGN}, WPS {MIN_WPS}-{MAX_WPS}")
    print(f"Tier 3 (rough patch): LLM >= {TIER3_LLM}, Align >= {TIER3_ALIGN}, neighbor >= {NEIGHBOR_THRESHOLD}")

    # Load segments
    segments = load_segments()

    if not segments:
        print("\nNo segments found!")
        print("Run the alignment pipeline first:")
        print("  1. python scripts/word_align.py")
        print("  2. python scripts/llm_score_segments.py")
        return

    print(f"\nLoaded {len(segments)} segments")

    # Assign tiers
    segments = assign_tiers(segments)

    # Count by tier
    tier_counts = defaultdict(int)
    for seg in segments:
        tier_counts[seg["tier"]] += 1

    print(f"\nTier distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count}")

    included = sum(1 for s in segments if s.get("include", False))
    print(f"\nTotal included: {included}")

    if args.no_rough_patches:
        # Exclude rough patches
        for seg in segments:
            if seg["tier"] in ("rough_patch", "bridge_segment"):
                seg["include"] = False
        included = sum(1 for s in segments if s.get("include", False))
        print(f"After excluding rough patches: {included}")

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Process segments
    entries = process_segments(segments, clips_dir)

    if not entries:
        print("No valid entries created!")
        return

    # Create splits
    create_splits(entries, args.output_dir)

    print(f"\nDataset ready in {args.output_dir}/")


if __name__ == "__main__":
    main()
