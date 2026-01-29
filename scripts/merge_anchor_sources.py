#!/usr/bin/env python3
"""
Merge Anchor Sources for TuvaLLM.

Combines anchors detected from:
- MMS-Samoan transcripts (Tuvaluan/Samoan words)
- Whisper-English transcripts (English loanwords, names, titles)

If same anchor detected by both models within 2s tolerance → HIGH confidence.
Single-source anchors included with lower confidence.

Output: Unified anchor stream sorted by time.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MMS_DIR = PROCESSED_DIR / "mms_transcripts"
WHISPER_DIR = PROCESSED_DIR / "whisper_english_transcripts"
OUTPUT_DIR = PROCESSED_DIR / "merged_anchors"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Anchor categories
# MMS-Samoan anchors (Tuvaluan words that survive MMS transcription)
MMS_ANCHORS = [
    # Island names
    "funafuti", "nukufetau", "nanumea", "vaitupu", "nukulaelae", "niutao", "nui",
    # High-frequency Tuvaluan words
    "tatou", "tenei", "latou", "mafai", "fenua", "tulaga", "tuvalu",
    "penei", "tausaga", "matou", "tulafono", "vaega", "fesili",
    "fakatasi", "kiloga", "talia", "manako", "fakatokaga", "fakafetai",
    "galuega", "manakoga", "kiluga", "atufenua", "mataupu",
    # Tuvaluanized loanwords
    "palamene", "komiti", "minista", "minisita", "miliona", "pili",
    "ofisa", "pasene", "fono",
]

# English anchors (words Whisper-English will detect)
ENGLISH_ANCHORS = [
    # Government/Parliament terms
    "parliament", "minister", "speaker", "budget", "government",
    "committee", "bill", "policy", "public", "member", "motion",
    "report", "general", "national", "private", "prime",
    # Modern terms
    "climate", "project", "development", "economic", "financial",
    # Proper names (common in Tuvalu parliament)
    "teo", "kofe", "telavi", "sopoaga", "tausi", "enele",
    # Titles
    "honourable", "honorable",
]

# Cross-model mappings (same concept, different transcriptions)
ANCHOR_EQUIVALENTS = {
    "palamene": "parliament",
    "parliament": "palamene",
    "minista": "minister",
    "minisita": "minister",
    "minister": "minista",
    "komiti": "committee",
    "committee": "komiti",
}

# Tolerance for matching anchors from different sources
TIME_TOLERANCE = 2.0  # seconds


@dataclass
class Anchor:
    """Represents a detected anchor with metadata."""
    word: str
    time: float
    source: str  # 'mms', 'whisper_en', 'both'
    confidence: float  # 0.0-1.0
    chunk_text: str = ""  # Surrounding context

    def to_dict(self):
        return asdict(self)


def load_transcript(json_path: Path) -> List[Dict]:
    """Load transcript JSON (MMS or Whisper format)."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_anchors_in_transcript(
    transcript: List[Dict],
    anchor_list: List[str],
    source: str
) -> List[Anchor]:
    """Find anchors in transcript, return Anchor objects."""
    found = []

    for chunk in transcript:
        text = chunk.get("text", "").lower()
        start = chunk.get("start", 0.0)
        end = chunk.get("end", start + 10.0)

        for anchor in anchor_list:
            anchor_lower = anchor.lower()
            # Use word boundary matching for accuracy
            pattern = r'\b' + re.escape(anchor_lower) + r'\b'
            matches = list(re.finditer(pattern, text))

            for match in matches:
                # Estimate time within chunk based on character position
                char_ratio = match.start() / len(text) if len(text) > 0 else 0
                anchor_time = start + (end - start) * char_ratio

                found.append(Anchor(
                    word=anchor_lower,
                    time=anchor_time,
                    source=source,
                    confidence=0.5,  # Single-source default
                    chunk_text=text[:100],  # Context snippet
                ))

    return found


def merge_anchor_sources(
    mms_anchors: List[Anchor],
    whisper_anchors: List[Anchor],
    tolerance: float = TIME_TOLERANCE
) -> List[Anchor]:
    """
    Merge anchors from MMS and Whisper sources.

    - Anchors detected by both within tolerance → HIGH confidence (0.9)
    - Single-source anchors → LOWER confidence (0.5 for MMS, 0.4 for Whisper)
    """
    merged = []
    used_whisper = set()

    # First pass: Find matches between MMS and Whisper
    for mms_anchor in mms_anchors:
        matched = False

        for i, whisper_anchor in enumerate(whisper_anchors):
            if i in used_whisper:
                continue

            # Check if same word or equivalent
            words_match = (
                mms_anchor.word == whisper_anchor.word or
                ANCHOR_EQUIVALENTS.get(mms_anchor.word) == whisper_anchor.word or
                ANCHOR_EQUIVALENTS.get(whisper_anchor.word) == mms_anchor.word
            )

            time_close = abs(mms_anchor.time - whisper_anchor.time) <= tolerance

            if words_match and time_close:
                # High-confidence match!
                avg_time = (mms_anchor.time + whisper_anchor.time) / 2
                merged.append(Anchor(
                    word=mms_anchor.word,
                    time=avg_time,
                    source="both",
                    confidence=0.9,
                    chunk_text=mms_anchor.chunk_text or whisper_anchor.chunk_text,
                ))
                used_whisper.add(i)
                matched = True
                break

        if not matched:
            # MMS-only anchor
            mms_anchor.confidence = 0.5
            merged.append(mms_anchor)

    # Add unmatched Whisper anchors
    for i, whisper_anchor in enumerate(whisper_anchors):
        if i not in used_whisper:
            whisper_anchor.confidence = 0.4
            merged.append(whisper_anchor)

    # Sort by time
    merged.sort(key=lambda a: a.time)

    return merged


def process_audio_file(
    audio_stem: str,
    mms_dir: Path,
    whisper_dir: Path
) -> Optional[List[Anchor]]:
    """Process a single audio file, merging anchors from both sources."""
    mms_path = mms_dir / (audio_stem + ".json")
    whisper_path = whisper_dir / (audio_stem + ".json")

    mms_anchors = []
    whisper_anchors = []

    # Load MMS transcript
    if mms_path.exists():
        mms_transcript = load_transcript(mms_path)
        mms_anchors = find_anchors_in_transcript(
            mms_transcript, MMS_ANCHORS, "mms"
        )

    # Load Whisper transcript
    if whisper_path.exists():
        whisper_transcript = load_transcript(whisper_path)
        whisper_anchors = find_anchors_in_transcript(
            whisper_transcript, ENGLISH_ANCHORS, "whisper_en"
        )

    if not mms_anchors and not whisper_anchors:
        return None

    # Merge sources
    merged = merge_anchor_sources(mms_anchors, whisper_anchors)

    return merged


def main():
    print("=" * 60)
    print("Merge Anchor Sources")
    print("=" * 60)
    print(f"MMS dir: {MMS_DIR}")
    print(f"Whisper dir: {WHISPER_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Find all audio stems from MMS (primary source)
    mms_files = list(MMS_DIR.glob("*.json"))
    whisper_files = list(WHISPER_DIR.glob("*.json"))

    print(f"\nMMS transcripts: {len(mms_files)}")
    print(f"Whisper transcripts: {len(whisper_files)}")

    # Get all unique audio stems
    all_stems = set()
    for f in mms_files:
        all_stems.add(f.stem)
    for f in whisper_files:
        all_stems.add(f.stem)

    print(f"Unique audio files: {len(all_stems)}")

    # Process each audio file
    total_anchors = 0
    total_high_conf = 0
    files_processed = 0

    for audio_stem in tqdm(sorted(all_stems), desc="Merging anchors"):
        merged = process_audio_file(audio_stem, MMS_DIR, WHISPER_DIR)

        if not merged:
            continue

        # Save merged anchors
        output_path = OUTPUT_DIR / (audio_stem + ".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in merged], f, indent=2)

        total_anchors += len(merged)
        total_high_conf += sum(1 for a in merged if a.confidence >= 0.9)
        files_processed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files processed: {files_processed}")
    print(f"Total anchors: {total_anchors}")
    print(f"High-confidence (both sources): {total_high_conf}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
