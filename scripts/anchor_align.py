#!/usr/bin/env python3
"""
Anchor-Based Alignment v3: Uses merged anchor sources and sequence-based matching.

This version integrates with:
- merge_anchor_sources.py: Combined MMS + Whisper English anchors
- sequence_align.py: LCS-based sequence matching

Generates confident segments only from verified sequence matches.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import docx
import soundfile as sf
from tqdm import tqdm


def get_day_take(path):
    """Extract (day, take) tuple from audio filename for proper chronological sorting."""
    name = path.name
    day_match = re.search(r'DAY[_\s]*(\d+)', name, re.IGNORECASE)
    take_match = re.search(r'Take[_\s]*(\d+)', name, re.IGNORECASE)
    day = int(day_match.group(1)) if day_match else 0
    take = int(take_match.group(1)) if take_match else 0
    return (day, take)


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MMS_DIR = PROCESSED_DIR / "mms_transcripts"
MERGED_ANCHORS_DIR = PROCESSED_DIR / "merged_anchors"
MATCHED_SEQUENCES_DIR = PROCESSED_DIR / "matched_sequences"
AUDIO_DIR = PROCESSED_DIR / "audio_16k"
SEGMENTS_DIR = PROCESSED_DIR / "segments_confident"

SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MIN_CONFIDENCE = 0.45  # Minimum sequence confidence for segment generation
MIN_SEQUENCE_LENGTH = 5  # Minimum anchors in a sequence
MAX_SEGMENT_DURATION = 15.0  # Maximum segment duration in seconds
MAX_SEGMENT_WORDS = 30  # Maximum words per segment
MIN_WPS = 0.5  # Minimum words per second
MAX_WPS = 6.0  # Maximum words per second


def load_matched_sequences(json_path: Path) -> List[Dict]:
    """Load matched sequences from sequence_align.py output."""
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_docx_words(docx_path: Path) -> List[str]:
    """Load transcript as list of words."""
    doc = docx.Document(docx_path)
    words = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            words.extend(text.split())
    return words


def create_segments_from_sequence(
    sequence: Dict,
    words: List[str],
    audio_path: Path
) -> List[Dict]:
    """
    Create training segments from a matched sequence.

    Uses anchor points as alignment boundaries and creates segments between them.
    """
    anchors = sequence.get("anchors", [])
    if len(anchors) < 2:
        return []

    segments = []
    sequence_confidence = sequence.get("confidence", 0.5)

    for i in range(len(anchors) - 1):
        a1 = anchors[i]
        a2 = anchors[i + 1]

        audio_start = a1["audio_time"]
        audio_end = a2["audio_time"]
        seg_duration = audio_end - audio_start

        word_start = a1["word_idx"]
        word_end = a2["word_idx"]

        if word_end <= word_start or seg_duration <= 0:
            continue

        # Include anchor_end word (+1 because Python slicing is exclusive)
        seg_words = words[word_start:word_end + 1]
        if not seg_words:
            continue

        wps = len(seg_words) / seg_duration

        # Skip unreasonable WPS
        if wps < MIN_WPS or wps > MAX_WPS:
            continue

        # Calculate segment confidence (inherit from sequence, adjusted by WPS)
        wps_penalty = 0
        if wps < 1.0 or wps > 4.5:
            wps_penalty = 0.1
        segment_confidence = sequence_confidence - wps_penalty

        # Split long segments
        if seg_duration <= MAX_SEGMENT_DURATION and len(seg_words) <= MAX_SEGMENT_WORDS:
            segments.append({
                "text": " ".join(w.lower() for w in seg_words),
                "start": audio_start,
                "end": audio_end,
                "audio_path": str(audio_path),
                "wps": wps,
                "confidence": segment_confidence,
                "anchor_start": a1["word"],
                "anchor_end": a2["word"],
                "sequence_length": len(anchors),
            })
        else:
            # Split into smaller segments
            n_splits = max(
                2,
                int(seg_duration / MAX_SEGMENT_DURATION) + 1,
                int(len(seg_words) / MAX_SEGMENT_WORDS) + 1
            )
            words_per = len(seg_words) // n_splits
            time_per = seg_duration / n_splits

            for s in range(n_splits):
                start_word_idx = s * words_per
                end_word_idx = (s + 1) * words_per if s < n_splits - 1 else len(seg_words)
                sub_words = seg_words[start_word_idx:end_word_idx]

                if not sub_words:
                    continue

                sub_start = audio_start + s * time_per
                sub_end = audio_start + (s + 1) * time_per
                sub_wps = len(sub_words) / time_per if time_per > 0 else 0

                segments.append({
                    "text": " ".join(w.lower() for w in sub_words),
                    "start": sub_start,
                    "end": sub_end,
                    "audio_path": str(audio_path),
                    "wps": sub_wps,
                    "confidence": segment_confidence - 0.05,  # Slight penalty for split
                    "anchor_start": a1["word"] if s == 0 else None,
                    "anchor_end": a2["word"] if s == n_splits - 1 else None,
                    "sequence_length": len(anchors),
                })

    return segments


def process_from_sequences(
    sequences_path: Path,
    transcript_words_cache: Dict[str, List[str]]
) -> List[Dict]:
    """
    Process matched sequences to generate confident segments.
    """
    sequences = load_matched_sequences(sequences_path)

    if not sequences:
        print("No matched sequences found. Run sequence_align.py first.")
        return []

    print(f"Loaded {len(sequences)} matched sequences")

    # Filter by confidence
    confident_sequences = [s for s in sequences if s.get("confidence", 0) >= MIN_CONFIDENCE]
    print(f"High-confidence sequences (>= {MIN_CONFIDENCE}): {len(confident_sequences)}")

    all_segments = []

    # Group by session for word lookup
    by_session = defaultdict(list)
    for seq in confident_sequences:
        session = seq.get("session", "unknown")
        by_session[session].append(seq)

    for session, session_sequences in by_session.items():
        words = transcript_words_cache.get(session, [])
        if not words:
            print(f"  Warning: No transcript for session {session}")
            continue

        print(f"\nProcessing {session}: {len(session_sequences)} sequences")

        for seq in tqdm(session_sequences, desc=f"  {session}"):
            audio_path = Path(seq.get("audio_file", ""))
            segments = create_segments_from_sequence(seq, words, audio_path)
            all_segments.extend(segments)

    return all_segments


def load_transcripts() -> Dict[str, List[str]]:
    """Load all session transcripts into cache."""
    cache = {}

    # June session
    june_docx = next(RAW_DIR.rglob("*June*.docx"), None)
    if june_docx:
        cache["june"] = load_docx_words(june_docx)
        print(f"Loaded June transcript: {len(cache['june']):,} words")

    # December session
    dec_docx = next(RAW_DIR.rglob("*Tesema*.docx"), None)
    if dec_docx:
        cache["december"] = load_docx_words(dec_docx)
        print(f"Loaded December transcript: {len(cache['december']):,} words")

    return cache


def main():
    print("=" * 60)
    print("Anchor-Based Alignment v3 (Sequence-Based)")
    print("=" * 60)
    print(f"Min confidence: {MIN_CONFIDENCE}")
    print(f"Min sequence length: {MIN_SEQUENCE_LENGTH}")

    # Load transcripts
    print("\nLoading transcripts...")
    transcript_cache = load_transcripts()

    # Check for matched sequences
    sequences_path = MATCHED_SEQUENCES_DIR / "matched_sequences.json"

    if not sequences_path.exists():
        print(f"\nNo matched sequences found at {sequences_path}")
        print("Run the following pipeline first:")
        print("  1. python scripts/transcribe_whisper_english.py")
        print("  2. python scripts/merge_anchor_sources.py")
        print("  3. python scripts/sequence_align.py")
        return

    # Process sequences
    all_segments = process_from_sequences(sequences_path, transcript_cache)

    # Summary and save
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total segments: {len(all_segments)}")

    if all_segments:
        total_duration = sum(s["end"] - s["start"] for s in all_segments)
        avg_wps = sum(s["wps"] for s in all_segments) / len(all_segments)
        avg_conf = sum(s["confidence"] for s in all_segments) / len(all_segments)

        print(f"Total duration: {total_duration/3600:.2f} hours")
        print(f"Avg WPS: {avg_wps:.2f}")
        print(f"Avg confidence: {avg_conf:.3f}")

        # Confidence distribution
        high_conf = sum(1 for s in all_segments if s["confidence"] >= 0.7)
        med_conf = sum(1 for s in all_segments if 0.6 <= s["confidence"] < 0.7)
        print(f"\nConfidence distribution:")
        print(f"  High (>= 0.7): {high_conf}")
        print(f"  Medium (0.6-0.7): {med_conf}")

        # Save
        save_path = SEGMENTS_DIR / "confident_segments.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {save_path}")

        # Also save metadata
        metadata = {
            "total_segments": len(all_segments),
            "total_duration_hours": total_duration / 3600,
            "avg_wps": avg_wps,
            "avg_confidence": avg_conf,
            "min_confidence_threshold": MIN_CONFIDENCE,
            "high_confidence_count": high_conf,
            "medium_confidence_count": med_conf,
        }
        meta_path = SEGMENTS_DIR / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")
    else:
        print("No segments generated!")


if __name__ == "__main__":
    main()
