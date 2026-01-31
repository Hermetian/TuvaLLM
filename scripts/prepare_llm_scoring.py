#!/usr/bin/env python3
"""Prepare segment pairs for LLM-based confidence scoring."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEGMENTS_FILE = PROJECT_ROOT / "data/processed/segments_confident/confident_segments.json"
MMS_DIR = PROJECT_ROOT / "data/processed/mms_transcripts"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/segments_for_llm_scoring.json"


def get_mms_transcript_for_audio(audio_path: str) -> list:
    """Find MMS transcript file matching an audio file."""
    audio_name = Path(audio_path).stem

    # Try to find matching MMS transcript
    for mms_file in MMS_DIR.glob("*.json"):
        # Match by checking if the audio name is similar to MMS filename
        mms_name = mms_file.stem
        # Normalize both names for comparison
        if audio_name.replace("_", "").lower() in mms_name.replace("_", "").lower() or \
           mms_name.replace("_", "").lower() in audio_name.replace("_", "").lower():
            with open(mms_file) as f:
                return json.load(f)

    return []


def get_asr_text_for_timerange(mms_chunks: list, start: float, end: float) -> str:
    """Extract ASR text for a specific time range."""
    texts = []
    for chunk in mms_chunks:
        chunk_start = chunk.get("start", 0)
        chunk_end = chunk.get("end", 0)
        # Check for overlap
        if chunk_start < end and chunk_end > start:
            texts.append(chunk.get("text", ""))
    return " ".join(texts)


def main():
    print("Loading segments...")
    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)

    print(f"Loaded {len(segments)} segments")

    # Cache MMS transcripts by audio path
    mms_cache = {}

    pairs = []
    for i, seg in enumerate(segments):
        audio_path = seg.get("audio_path", "")

        # Get MMS transcript (cached)
        if audio_path not in mms_cache:
            mms_cache[audio_path] = get_mms_transcript_for_audio(audio_path)

        mms_chunks = mms_cache[audio_path]

        if not mms_chunks:
            continue

        # Get ASR text for this segment's time range
        asr_text = get_asr_text_for_timerange(
            mms_chunks,
            seg.get("start", 0),
            seg.get("end", 0)
        )

        if not asr_text.strip():
            continue

        pairs.append({
            "id": i,
            "tuvaluan_transcript": seg.get("text", ""),
            "samoan_asr": asr_text,
            "original_confidence": seg.get("confidence", 0),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "audio_path": audio_path
        })

    print(f"Created {len(pairs)} pairs for LLM scoring")

    # Save for LLM processing
    with open(OUTPUT_FILE, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

    # Show some examples
    print("\nSample pairs:")
    for p in pairs[:3]:
        print(f"\n--- Segment {p['id']} (conf: {p['original_confidence']:.3f}) ---")
        print(f"Tuvaluan: {p['tuvaluan_transcript'][:100]}...")
        print(f"Samoan:   {p['samoan_asr'][:100]}...")


if __name__ == "__main__":
    main()
