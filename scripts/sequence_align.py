#!/usr/bin/env python3
"""
Sequence-Based Alignment for TuvaLLM.

Core Algorithm:
If anchors [A, B, C, D, E] appear in the SAME ORDER in both:
- Audio (by timestamp)
- Transcript (by word position)

This is strong evidence of correct alignment.

Uses Longest Increasing Subsequence (LIS) algorithm to find the longest
chain of anchors that maintain monotonic order in both domains.

Confidence scoring based on:
- Sequence length (longer = better)
- Anchor density (anchors per minute)
- WPS consistency (stable speech rate)
- Anchor diversity (different words, not just repeated)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import docx
import soundfile as sf
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio_16k"
MERGED_ANCHORS_DIR = PROCESSED_DIR / "merged_anchors"
OUTPUT_DIR = PROCESSED_DIR / "matched_sequences"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum sequence length for confident match
MIN_SEQUENCE_LENGTH = 5


@dataclass
class AnchorMatch:
    """An anchor matched between audio and transcript."""
    word: str
    audio_time: float
    word_idx: int
    confidence: float


@dataclass
class MatchedSequence:
    """A sequence of matched anchors with metadata."""
    anchors: List[AnchorMatch]
    audio_start: float
    audio_end: float
    word_start: int
    word_end: int
    confidence: float
    metrics: Dict

    def to_dict(self):
        return {
            "anchors": [asdict(a) for a in self.anchors],
            "audio_start": self.audio_start,
            "audio_end": self.audio_end,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "confidence": self.confidence,
            "metrics": self.metrics,
        }


def load_merged_anchors(json_path: Path) -> List[Dict]:
    """Load merged anchors from JSON."""
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


def find_anchor_in_transcript(
    words: List[str],
    anchor: str,
    start_idx: int = 0
) -> List[int]:
    """Find all positions of anchor word in transcript from start_idx."""
    anchor_lower = anchor.lower()
    positions = []

    for i in range(start_idx, len(words)):
        # Clean the word (remove punctuation for matching)
        word_clean = re.sub(r'[^\w]', '', words[i].lower())
        if word_clean == anchor_lower:
            positions.append(i)

    return positions


def find_all_anchor_positions(
    words: List[str],
    unique_anchors: set
) -> Dict[str, List[int]]:
    """Pre-compute all positions for each anchor in transcript."""
    positions = defaultdict(list)
    words_lower = [re.sub(r'[^\w]', '', w.lower()) for w in words]

    for i, word in enumerate(words_lower):
        if word in unique_anchors:
            positions[word].append(i)

    return positions


def longest_increasing_subsequence_with_indices(
    audio_anchors: List[Dict],
    transcript_positions: Dict[str, List[int]],
    word_start_hint: int = 0,
    word_end_hint: int = float('inf')
) -> List[Tuple[Dict, int]]:
    """
    Find longest subsequence where both audio times AND transcript positions
    are monotonically increasing.

    Returns: List of (audio_anchor, word_idx) tuples
    """
    # Build candidate matches: for each audio anchor, find possible transcript positions
    candidates = []

    for anchor in audio_anchors:
        word = anchor["word"]
        positions = transcript_positions.get(word, [])

        # Filter positions to estimated range
        filtered_positions = [
            p for p in positions
            if word_start_hint <= p <= word_end_hint
        ]

        # If no positions in range, include nearest ones
        if not filtered_positions and positions:
            # Find positions closest to the range
            filtered_positions = sorted(
                positions,
                key=lambda p: min(abs(p - word_start_hint), abs(p - word_end_hint))
            )[:3]

        for pos in filtered_positions:
            candidates.append({
                "anchor": anchor,
                "word_idx": pos,
                "audio_time": anchor["time"],
            })

    if not candidates:
        return []

    # Sort by audio time
    candidates.sort(key=lambda c: c["audio_time"])

    # Dynamic programming for LIS on (audio_time, word_idx) pairs
    # We need word_idx to be increasing as audio_time increases
    n = len(candidates)
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            # Check if word_idx is increasing (audio_time is already sorted)
            if candidates[j]["word_idx"] < candidates[i]["word_idx"]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

    # Find the longest sequence
    max_len = max(dp) if dp else 0
    max_idx = dp.index(max_len) if dp else -1

    if max_len < MIN_SEQUENCE_LENGTH:
        return []

    # Backtrack to get the sequence
    sequence = []
    idx = max_idx
    while idx != -1:
        c = candidates[idx]
        sequence.append((c["anchor"], c["word_idx"]))
        idx = parent[idx]

    sequence.reverse()
    return sequence


def calculate_sequence_confidence(
    sequence: List[Tuple[Dict, int]],
    words: List[str],
    audio_duration: float
) -> Tuple[float, Dict]:
    """
    Calculate confidence score for a matched sequence.

    Factors:
    - Sequence length (longer = better)
    - Anchor density (anchors per minute)
    - WPS consistency (stable speech rate)
    - Anchor diversity (different words)
    """
    if len(sequence) < 2:
        return 0.0, {}

    # Extract metrics
    audio_start = sequence[0][0]["time"]
    audio_end = sequence[-1][0]["time"]
    word_start = sequence[0][1]
    word_end = sequence[-1][1]

    audio_span = audio_end - audio_start
    word_span = word_end - word_start

    if audio_span <= 0 or word_span <= 0:
        return 0.0, {}

    # Words per second
    wps = word_span / audio_span

    # Anchor density (anchors per minute of audio)
    anchor_density = len(sequence) / (audio_span / 60) if audio_span > 0 else 0

    # Anchor diversity (unique words / total)
    unique_words = len(set(a[0]["word"] for a in sequence))
    diversity = unique_words / len(sequence)

    # Calculate per-segment WPS consistency
    wps_values = []
    for i in range(len(sequence) - 1):
        seg_audio = sequence[i + 1][0]["time"] - sequence[i][0]["time"]
        seg_words = sequence[i + 1][1] - sequence[i][1]
        if seg_audio > 0:
            wps_values.append(seg_words / seg_audio)

    wps_std = 0
    if len(wps_values) > 1:
        mean_wps = sum(wps_values) / len(wps_values)
        wps_std = (sum((w - mean_wps) ** 2 for w in wps_values) / len(wps_values)) ** 0.5

    # Confidence scoring
    scores = []

    # Length score (5 anchors = 0.5, 10 = 0.8, 20+ = 1.0)
    length_score = min(1.0, (len(sequence) - MIN_SEQUENCE_LENGTH) / 15 + 0.5)
    scores.append(length_score * 0.3)

    # WPS score (ideal range 1.5-4.0)
    if 1.0 <= wps <= 5.0:
        wps_score = 1.0 - abs(wps - 2.5) / 2.5
    else:
        wps_score = 0.3
    scores.append(wps_score * 0.25)

    # WPS consistency score (lower std = better)
    consistency_score = max(0, 1.0 - wps_std / 2.0)
    scores.append(consistency_score * 0.2)

    # Diversity score
    scores.append(diversity * 0.15)

    # Anchor confidence score (average of individual anchor confidences)
    avg_anchor_conf = sum(a[0].get("confidence", 0.5) for a in sequence) / len(sequence)
    scores.append(avg_anchor_conf * 0.1)

    confidence = sum(scores)

    metrics = {
        "sequence_length": len(sequence),
        "audio_span": audio_span,
        "word_span": word_span,
        "wps": wps,
        "wps_std": wps_std,
        "anchor_density": anchor_density,
        "diversity": diversity,
        "unique_anchors": unique_words,
    }

    return confidence, metrics


def find_matched_sequences(
    audio_anchors: List[Dict],
    transcript_positions: Dict[str, List[int]],
    words: List[str],
    audio_duration: float,
    word_start_hint: int = 0,
    word_end_hint: int = None
) -> List[MatchedSequence]:
    """
    Find all confident matched sequences between audio and transcript.
    """
    if word_end_hint is None:
        word_end_hint = len(words)

    # Find the longest increasing subsequence
    sequence = longest_increasing_subsequence_with_indices(
        audio_anchors,
        transcript_positions,
        word_start_hint,
        word_end_hint
    )

    if len(sequence) < MIN_SEQUENCE_LENGTH:
        return []

    confidence, metrics = calculate_sequence_confidence(sequence, words, audio_duration)

    if confidence < 0.4:  # Minimum threshold
        return []

    # Convert to MatchedSequence
    anchors = [
        AnchorMatch(
            word=a[0]["word"],
            audio_time=a[0]["time"],
            word_idx=a[1],
            confidence=a[0].get("confidence", 0.5)
        )
        for a in sequence
    ]

    matched = MatchedSequence(
        anchors=anchors,
        audio_start=sequence[0][0]["time"],
        audio_end=sequence[-1][0]["time"],
        word_start=sequence[0][1],
        word_end=sequence[-1][1],
        confidence=confidence,
        metrics=metrics,
    )

    return [matched]


def get_day_take(path: Path) -> Tuple[int, int]:
    """Extract (day, take) tuple from audio filename."""
    name = path.name
    day_match = re.search(r'DAY[_\s]*(\d+)', name, re.IGNORECASE)
    take_match = re.search(r'Take[_\s]*(\d+)', name, re.IGNORECASE)
    day = int(day_match.group(1)) if day_match else 0
    take = int(take_match.group(1)) if take_match else 0
    return (day, take)


def process_session(
    audio_files: List[Path],
    transcript_path: Path,
    session_name: str
) -> List[Dict]:
    """Process a parliament session, finding matched sequences for each audio file."""
    print(f"\n{'=' * 60}")
    print(f"Processing {session_name}")
    print(f"{'=' * 60}")

    # Load transcript
    words = load_docx_words(transcript_path)
    total_words = len(words)
    print(f"Transcript: {total_words:,} words")

    # Get unique anchors from all merged anchor files
    unique_anchors = set()
    for af in audio_files:
        anchor_path = MERGED_ANCHORS_DIR / (af.stem + ".json")
        if anchor_path.exists():
            anchors = load_merged_anchors(anchor_path)
            for a in anchors:
                unique_anchors.add(a["word"])

    print(f"Unique anchor types: {len(unique_anchors)}")

    # Pre-compute all anchor positions in transcript
    transcript_positions = find_all_anchor_positions(words, unique_anchors)
    total_positions = sum(len(v) for v in transcript_positions.values())
    print(f"Transcript anchor positions: {total_positions}")

    # Calculate total audio duration
    total_audio_duration = 0
    file_durations = {}
    for af in audio_files:
        info = sf.info(af)
        file_durations[af] = info.duration
        total_audio_duration += info.duration

    overall_wps = total_words / total_audio_duration if total_audio_duration > 0 else 2.0
    print(f"Audio files: {len(audio_files)} ({total_audio_duration/3600:.2f} hours)")
    print(f"Estimated WPS: {overall_wps:.2f}")

    all_sequences = []
    cumulative_duration = 0

    for audio_file in tqdm(audio_files, desc=f"Processing {session_name}"):
        # Estimate transcript range
        file_duration = file_durations[audio_file]
        word_start_hint = int(cumulative_duration * overall_wps)
        word_end_hint = int((cumulative_duration + file_duration) * overall_wps)
        word_end_hint = min(word_end_hint, total_words)

        # Load merged anchors
        anchor_path = MERGED_ANCHORS_DIR / (audio_file.stem + ".json")
        if not anchor_path.exists():
            cumulative_duration += file_duration
            continue

        audio_anchors = load_merged_anchors(anchor_path)

        if len(audio_anchors) < MIN_SEQUENCE_LENGTH:
            cumulative_duration += file_duration
            continue

        # Find matched sequences
        sequences = find_matched_sequences(
            audio_anchors,
            transcript_positions,
            words,
            file_duration,
            word_start_hint,
            word_end_hint
        )

        for seq in sequences:
            seq_dict = seq.to_dict()
            seq_dict["audio_file"] = str(audio_file)
            seq_dict["session"] = session_name
            all_sequences.append(seq_dict)

        cumulative_duration += file_duration

    return all_sequences


def main():
    print("=" * 60)
    print("Sequence-Based Alignment")
    print("=" * 60)
    print(f"Merged anchors: {MERGED_ANCHORS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Min sequence length: {MIN_SEQUENCE_LENGTH}")

    all_sequences = []

    # June session
    june_docx = next(RAW_DIR.rglob("*June*.docx"), None)
    if june_docx:
        june_audios = sorted(
            [f for f in AUDIO_DIR.glob("*.wav")
             if "june" in f.name.lower() or "24-06-24" in f.name],
            key=get_day_take
        )
        if june_audios:
            sequences = process_session(june_audios, june_docx, "june")
            all_sequences.extend(sequences)

    # December session
    dec_docx = next(RAW_DIR.rglob("*Tesema*.docx"), None)
    if dec_docx:
        dec_audios = sorted(
            [f for f in AUDIO_DIR.glob("*.wav")
             if "december" in f.name.lower() or "12-24" in f.name],
            key=get_day_take
        )
        if dec_audios:
            sequences = process_session(dec_audios, dec_docx, "december")
            all_sequences.extend(sequences)

    # Save all sequences
    output_path = OUTPUT_DIR / "matched_sequences.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_sequences, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total sequences: {len(all_sequences)}")

    if all_sequences:
        high_conf = [s for s in all_sequences if s["confidence"] >= 0.6]
        print(f"High-confidence (>= 0.6): {len(high_conf)}")

        avg_conf = sum(s["confidence"] for s in all_sequences) / len(all_sequences)
        print(f"Average confidence: {avg_conf:.3f}")

        avg_len = sum(s["metrics"]["sequence_length"] for s in all_sequences) / len(all_sequences)
        print(f"Average sequence length: {avg_len:.1f} anchors")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
