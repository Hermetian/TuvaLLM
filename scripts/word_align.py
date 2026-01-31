#!/usr/bin/env python3
"""
Word-Level Alignment for TuvaLLM.

Uses MMS Samoan transcripts with timestamp estimation from chunk positions,
anchors as synchronization points, and DTW alignment within regions.

Pipeline:
1. Load MMS transcripts + transcript + anchors for each audio file
2. Estimate word-level timestamps from chunk boundaries
3. Use anchors to synchronize MMS words with transcript positions
4. Run DTW alignment within anchor regions
5. Find natural segment boundaries (anchor pairs, sentence endings)
6. Output candidate segments with alignment scores
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
import docx
from tqdm import tqdm
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MMS_TRANSCRIPTS_DIR = PROCESSED_DIR / "mms_transcripts"
MERGED_ANCHORS_DIR = PROCESSED_DIR / "merged_anchors"
MATCHED_SEQUENCES_DIR = PROCESSED_DIR / "matched_sequences"
AUDIO_DIR = PROCESSED_DIR / "audio_16k"
OUTPUT_DIR = PROCESSED_DIR / "segments_word_aligned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MIN_SEGMENT_DURATION = 2.0
MAX_SEGMENT_DURATION = 15.0
IDEAL_SEGMENT_DURATION = 8.0
MIN_SEGMENT_WORDS = 3
MAX_SEGMENT_WORDS = 40
MIN_WPS = 0.8
MAX_WPS = 5.0
DTW_BAND_RATIO = 0.3  # Sakoe-Chiba band as ratio of sequence length


@dataclass
class MMSWord:
    """A word from MMS transcript with estimated timing."""
    word: str
    start: float
    end: float
    chunk_idx: int


@dataclass
class CandidateSegment:
    """A candidate training segment."""
    id: str
    text: str
    audio_path: str
    start: float
    end: float
    duration: float
    wps: float
    word_count: int
    align_score: float
    mms_text: str
    anchor_start: Optional[str] = None
    anchor_end: Optional[str] = None
    session: str = ""

    def to_dict(self):
        return asdict(self)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity between MMS (Samoan) word and transcript (Tuvaluan) word.

    Uses normalized Levenshtein + Polynesian cognate patterns.
    """
    w1 = word1.lower().strip()
    w2 = word2.lower().strip()

    if not w1 or not w2:
        return 0.0

    # Exact match
    if w1 == w2:
        return 1.0

    # Levenshtein-based similarity
    max_len = max(len(w1), len(w2))
    lev_dist = levenshtein_distance(w1, w2)
    lev_sim = 1.0 - (lev_dist / max_len)

    # Polynesian cognate adjustments
    # Common Samoan ↔ Tuvaluan sound correspondences
    cognate_bonus = 0.0

    # k/t interchange (common in Polynesian)
    w1_normalized = w1.replace('k', 't')
    w2_normalized = w2.replace('k', 't')
    if w1_normalized == w2_normalized:
        cognate_bonus = 0.3

    # ng/n variations
    w1_normalized = w1.replace('ng', 'n')
    w2_normalized = w2.replace('ng', 'n')
    if w1_normalized == w2_normalized and cognate_bonus == 0:
        cognate_bonus = 0.2

    # l/r variations
    w1_normalized = w1.replace('l', 'r').replace('r', 'l')
    if w1_normalized == w2 and cognate_bonus == 0:
        cognate_bonus = 0.15

    return min(1.0, lev_sim + cognate_bonus)


def dtw_align(
    mms_words: List[Dict],
    transcript_words: List[str],
    band_ratio: float = DTW_BAND_RATIO
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Dynamic Time Warping alignment between MMS words and transcript words.

    Uses Sakoe-Chiba band constraint for efficiency.
    Returns: (alignment_path, average_similarity)
    """
    n = len(mms_words)
    m = len(transcript_words)

    if n == 0 or m == 0:
        return [], 0.0

    # Compute similarity matrix
    sim_matrix = np.zeros((n, m))
    for i, mms_w in enumerate(mms_words):
        mms_text = mms_w["word"].lower()
        for j, trans_w in enumerate(transcript_words):
            sim_matrix[i, j] = word_similarity(mms_text, trans_w.lower())

    # DTW with Sakoe-Chiba band
    band = max(3, int(max(n, m) * band_ratio))

    # Cost matrix (we want to maximize similarity, so use negative)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, n + 1):
        # Band constraint
        j_start = max(1, int((i - 1) * m / n) - band)
        j_end = min(m + 1, int((i - 1) * m / n) + band + 1)

        for j in range(j_start, j_end):
            # Convert similarity to cost (1 - sim)
            local_cost = 1.0 - sim_matrix[i - 1, j - 1]
            cost[i, j] = local_cost + min(
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1]   # match/substitution
            )

    # Backtrack to get path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        # Find which direction we came from
        candidates = []
        if i > 0 and j > 0:
            candidates.append((cost[i - 1, j - 1], (i - 1, j - 1)))
        if i > 0:
            candidates.append((cost[i - 1, j], (i - 1, j)))
        if j > 0:
            candidates.append((cost[i, j - 1], (i, j - 1)))

        if not candidates:
            break

        _, (new_i, new_j) = min(candidates, key=lambda x: x[0])
        i, j = new_i, new_j

    path.reverse()

    # Calculate average similarity along path
    if path:
        total_sim = sum(sim_matrix[i, j] for i, j in path)
        avg_sim = total_sim / len(path)
    else:
        avg_sim = 0.0

    return path, avg_sim


def load_mms_transcripts(json_path: Path) -> List[Dict]:
    """Load MMS transcripts with chunk boundaries."""
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def mms_chunks_to_words(chunks: List[Dict]) -> List[MMSWord]:
    """
    Convert MMS transcript chunks to word list with estimated timestamps.

    Each chunk has text, start, end times. We distribute word timing
    evenly within each chunk.
    """
    all_words = []

    for chunk_idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue

        start = chunk["start"]
        end = chunk["end"]
        duration = end - start

        # Split into words
        words = text.split()
        if not words:
            continue

        # Estimate time per word
        time_per_word = duration / len(words)

        for i, word in enumerate(words):
            word_start = start + i * time_per_word
            word_end = start + (i + 1) * time_per_word

            all_words.append(MMSWord(
                word=word.lower().strip(),
                start=round(word_start, 3),
                end=round(word_end, 3),
                chunk_idx=chunk_idx
            ))

    return all_words


def is_sentence_boundary(word: str) -> bool:
    """Check if word ends with sentence-ending punctuation."""
    return word.rstrip().endswith(('.', '!', '?', ':'))


def load_docx_words(docx_path: Path) -> List[str]:
    """Load transcript as list of words."""
    doc = docx.Document(docx_path)
    words = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            words.extend(text.split())
    return words


def load_matched_sequences(json_path: Path) -> List[Dict]:
    """Load matched sequences from sequence_align.py output."""
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_mms_words_in_range(
    mms_words: List[MMSWord],
    start_time: float,
    end_time: float
) -> List[MMSWord]:
    """Find MMS words within a time range."""
    return [
        w for w in mms_words
        if w.start >= start_time - 0.5 and w.end <= end_time + 0.5
    ]


def create_segments_from_anchor_pair(
    anchor1: Dict,
    anchor2: Dict,
    mms_words: List[MMSWord],
    transcript_words: List[str],
    audio_path: Path,
    session: str,
    segment_id_base: str,
    global_counter: List[int]
) -> List[CandidateSegment]:
    """
    Create segments between two anchors using DTW alignment.

    1. Find MMS words in the anchor time range
    2. Get transcript words in the anchor word range
    3. DTW align them
    4. Split into reasonable segment sizes
    """
    segments = []

    audio_start = anchor1["audio_time"]
    audio_end = anchor2["audio_time"]
    word_start = anchor1["word_idx"]
    word_end = anchor2["word_idx"]

    # Basic validation
    if audio_end <= audio_start or word_end <= word_start:
        return []

    total_duration = audio_end - audio_start
    total_words = word_end - word_start + 1

    # Skip if too short
    if total_duration < MIN_SEGMENT_DURATION:
        return []

    # Find MMS words in this time range
    region_mms_words = find_mms_words_in_range(mms_words, audio_start, audio_end)

    if len(region_mms_words) < MIN_SEGMENT_WORDS:
        return []

    # Get transcript words (inclusive of end)
    region_transcript = transcript_words[word_start:word_end + 1]

    if len(region_transcript) < MIN_SEGMENT_WORDS:
        return []

    # Convert MMSWord to dict format for DTW
    mms_word_dicts = [{"word": w.word, "start": w.start, "end": w.end} for w in region_mms_words]

    # DTW alignment
    alignment_path, align_score = dtw_align(mms_word_dicts, region_transcript)

    if not alignment_path or align_score < 0.25:
        return []

    # Find sentence boundaries from transcript
    sentence_boundaries = []
    for i, word in enumerate(region_transcript):
        if is_sentence_boundary(word):
            # Find corresponding MMS index via alignment
            for mms_i, trans_i in alignment_path:
                if trans_i == i:
                    sentence_boundaries.append(mms_i)
                    break

    # If segment is within limits, create single segment
    if total_duration <= MAX_SEGMENT_DURATION and total_words <= MAX_SEGMENT_WORDS:
        wps = total_words / total_duration
        if MIN_WPS <= wps <= MAX_WPS:
            seg_id = f"{session}_R{global_counter[0]:05d}_t{int(audio_start)}"
            global_counter[0] += 1
            segment = CandidateSegment(
                id=seg_id,
                text=" ".join(w.lower() for w in region_transcript),
                audio_path=str(audio_path),
                start=audio_start,
                end=audio_end,
                duration=total_duration,
                wps=wps,
                word_count=total_words,
                align_score=align_score,
                mms_text=" ".join(w.word for w in region_mms_words),
                anchor_start=anchor1["word"],
                anchor_end=anchor2["word"],
                session=session,
            )
            return [segment]

    # Split long segments at sentence boundaries or evenly
    if sentence_boundaries:
        all_boundaries = sorted(set(sentence_boundaries))
    else:
        # Split evenly
        n_splits = max(2, int(total_duration / IDEAL_SEGMENT_DURATION) + 1)
        step = len(region_mms_words) // n_splits
        all_boundaries = [i * step for i in range(1, n_splits)]

    # Create segments
    boundary_indices = [0] + all_boundaries + [len(region_mms_words) - 1]

    for i in range(len(boundary_indices) - 1):
        start_idx = boundary_indices[i]
        end_idx = boundary_indices[i + 1]

        if end_idx <= start_idx:
            continue

        seg_mms = region_mms_words[start_idx:end_idx + 1]
        if len(seg_mms) < MIN_SEGMENT_WORDS:
            continue

        seg_start = seg_mms[0].start
        seg_end = seg_mms[-1].end
        seg_duration = seg_end - seg_start

        if seg_duration < MIN_SEGMENT_DURATION or seg_duration > MAX_SEGMENT_DURATION:
            continue

        # Find corresponding transcript words via alignment
        trans_indices = set()
        for mms_i, trans_i in alignment_path:
            if start_idx <= mms_i <= end_idx:
                trans_indices.add(trans_i)

        if not trans_indices:
            continue

        trans_start = min(trans_indices)
        trans_end = max(trans_indices)
        seg_transcript = region_transcript[trans_start:trans_end + 1]

        if len(seg_transcript) < MIN_SEGMENT_WORDS:
            continue

        wps = len(seg_transcript) / seg_duration
        if wps < MIN_WPS or wps > MAX_WPS:
            continue

        # Calculate segment-specific alignment score
        seg_path = [(m, t) for m, t in alignment_path if start_idx <= m <= end_idx]
        if seg_path:
            seg_sim = sum(
                word_similarity(
                    mms_word_dicts[m]["word"],
                    region_transcript[t]
                )
                for m, t in seg_path
            ) / len(seg_path)
        else:
            seg_sim = align_score

        seg_id = f"{session}_R{global_counter[0]:05d}_t{int(seg_start)}"
        global_counter[0] += 1

        segment = CandidateSegment(
            id=seg_id,
            text=" ".join(w.lower() for w in seg_transcript),
            audio_path=str(audio_path),
            start=seg_start,
            end=seg_end,
            duration=seg_duration,
            wps=wps,
            word_count=len(seg_transcript),
            align_score=seg_sim,
            mms_text=" ".join(w.word for w in seg_mms),
            anchor_start=anchor1["word"] if i == 0 else None,
            anchor_end=anchor2["word"] if i == len(boundary_indices) - 2 else None,
            session=session,
        )

        segments.append(segment)

    return segments


def process_audio_file(
    audio_stem: str,
    sequence: Dict,
    transcript_words: List[str],
    session: str,
    global_counter: List[int]
) -> Tuple[List[CandidateSegment], List[int]]:
    """Process a single audio file using its matched sequence and MMS transcripts."""

    # Find corresponding MMS transcript file
    transcript_files = list(MMS_TRANSCRIPTS_DIR.glob(f"{audio_stem}.json"))
    if not transcript_files:
        # Try partial match
        for tf in MMS_TRANSCRIPTS_DIR.glob("*.json"):
            # Match by significant parts of the name
            if audio_stem in tf.stem or tf.stem in audio_stem:
                transcript_files.append(tf)
                break

    if not transcript_files:
        return [], global_counter

    # Load MMS transcript chunks
    mms_chunks = load_mms_transcripts(transcript_files[0])
    if not mms_chunks:
        return [], global_counter

    # Convert chunks to word list with estimated timestamps
    mms_words = mms_chunks_to_words(mms_chunks)
    if not mms_words:
        return [], global_counter

    # Get audio path
    audio_path = AUDIO_DIR / f"{audio_stem}.wav"
    if not audio_path.exists():
        # Try finding it with the MMS transcript file stem
        audio_path = AUDIO_DIR / f"{transcript_files[0].stem}.wav"
        if not audio_path.exists():
            return [], global_counter

    # Get anchors from sequence
    anchors = sequence.get("anchors", [])
    if len(anchors) < 2:
        return [], global_counter

    segments = []

    # Process each pair of consecutive anchors
    for i in range(len(anchors) - 1):
        anchor1 = anchors[i]
        anchor2 = anchors[i + 1]

        pair_segments = create_segments_from_anchor_pair(
            anchor1, anchor2, mms_words, transcript_words,
            audio_path, session, f"{session}_{audio_stem[:20]}_{i:03d}",
            global_counter
        )

        segments.extend(pair_segments)

    return segments, global_counter


def load_transcripts() -> Dict[str, List[str]]:
    """Load all session transcripts."""
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
    print("Word-Level Alignment Pipeline")
    print("=" * 60)

    # Check for MMS transcript files
    transcript_files = list(MMS_TRANSCRIPTS_DIR.glob("*.json"))
    print(f"\nMMS transcript files available: {len(transcript_files)}")

    if not transcript_files:
        print("\nNo MMS transcript files found!")
        print("Run MMS transcription first.")
        return

    # Load transcripts
    print("\nLoading transcripts...")
    transcripts = load_transcripts()

    if not transcripts:
        print("No transcripts found!")
        return

    # Load matched sequences
    sequences_path = MATCHED_SEQUENCES_DIR / "matched_sequences.json"
    sequences = load_matched_sequences(sequences_path)

    if not sequences:
        print(f"\nNo matched sequences found at {sequences_path}")
        print("Run: python scripts/sequence_align.py")
        return

    print(f"Loaded {len(sequences)} matched sequences")

    # Process each sequence
    all_segments = []
    # Global counter as mutable list to pass by reference
    global_counter = [0]

    for seq in tqdm(sequences, desc="Processing sequences"):
        audio_file = seq.get("audio_file", "")
        session = seq.get("session", "unknown")

        if not audio_file:
            continue

        audio_stem = Path(audio_file).stem
        transcript_words = transcripts.get(session, [])

        if not transcript_words:
            continue

        segments, global_counter = process_audio_file(
            audio_stem, seq, transcript_words, session, global_counter
        )
        all_segments.extend(segments)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total candidate segments: {len(all_segments)}")

    if all_segments:
        total_duration = sum(s.duration for s in all_segments)
        avg_align = sum(s.align_score for s in all_segments) / len(all_segments)
        avg_wps = sum(s.wps for s in all_segments) / len(all_segments)

        print(f"Total duration: {total_duration / 3600:.2f} hours")
        print(f"Average alignment score: {avg_align:.3f}")
        print(f"Average WPS: {avg_wps:.2f}")

        # Alignment score distribution
        high_align = sum(1 for s in all_segments if s.align_score >= 0.6)
        med_align = sum(1 for s in all_segments if 0.4 <= s.align_score < 0.6)
        low_align = sum(1 for s in all_segments if s.align_score < 0.4)

        print(f"\nAlignment score distribution:")
        print(f"  High (>= 0.6): {high_align}")
        print(f"  Medium (0.4-0.6): {med_align}")
        print(f"  Low (< 0.4): {low_align}")

        # Save
        output_data = [s.to_dict() for s in all_segments]
        output_path = OUTPUT_DIR / "candidate_segments.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved to {output_path}")

        # Also save metadata
        metadata = {
            "total_segments": len(all_segments),
            "total_duration_hours": total_duration / 3600,
            "avg_alignment_score": avg_align,
            "avg_wps": avg_wps,
            "high_align_count": high_align,
            "medium_align_count": med_align,
            "low_align_count": low_align,
            "mms_transcript_files": len(transcript_files),
        }

        meta_path = OUTPUT_DIR / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {meta_path}")
    else:
        print("No segments generated!")
        print("\nPossible issues:")
        print("- MMS transcripts may not match audio files")
        print("- Anchors may not have word_idx mappings")
        print("- Time ranges may be too short/long")


if __name__ == "__main__":
    main()
