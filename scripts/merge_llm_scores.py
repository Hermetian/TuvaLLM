#!/usr/bin/env python3
"""
Merge LLM semantic scores into confident segments.

The LLM scores provide semantic confidence (how well Tuvaluan transcript
matches Samoan ASR output). This script merges those scores back into
the segment data for training filtering.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    # Load corrected segments
    segments_path = DATA_DIR / "segments_confident" / "confident_segments.json"
    with open(segments_path) as f:
        segments = json.load(f)
    print(f"Loaded {len(segments)} segments")

    # Load LLM scores
    llm_scores_path = DATA_DIR / "llm_scores_combined.json"
    if not llm_scores_path.exists():
        print(f"LLM scores not found: {llm_scores_path}")
        return

    with open(llm_scores_path) as f:
        llm_scores = json.load(f)
    print(f"Loaded {len(llm_scores)} LLM scores")

    # Create lookup by ID
    llm_by_id = {s["id"]: s["llm_confidence"] for s in llm_scores}

    # Merge scores into segments
    merged_count = 0
    for i, seg in enumerate(segments):
        if i in llm_by_id:
            seg["llm_confidence"] = llm_by_id[i]
            merged_count += 1
        else:
            # Keep sequence confidence as fallback
            seg["llm_confidence"] = seg.get("confidence", 0.5)

    print(f"Merged {merged_count} LLM scores")
    print(f"Segments without LLM score: {len(segments) - merged_count}")

    # Update confidence to use LLM confidence
    for seg in segments:
        # Store original sequence confidence
        seg["sequence_confidence"] = seg.get("confidence", 0.5)
        # Use LLM confidence as primary
        seg["confidence"] = seg["llm_confidence"]

    # Save merged segments
    output_path = DATA_DIR / "segments_confident" / "confident_segments.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    # Report distribution
    from collections import Counter
    conf_ranges = Counter()
    for s in segments:
        conf = s.get("confidence", 0)
        if conf >= 0.7:
            conf_ranges["0.7+"] += 1
        elif conf >= 0.6:
            conf_ranges["0.6-0.7"] += 1
        elif conf >= 0.5:
            conf_ranges["0.5-0.6"] += 1
        else:
            conf_ranges["<0.5"] += 1

    print("\n=== Final confidence distribution ===")
    for k in sorted(conf_ranges.keys(), reverse=True):
        print(f"  {k}: {conf_ranges[k]}")

    high_conf = conf_ranges.get("0.7+", 0) + conf_ranges.get("0.6-0.7", 0)
    print(f"\nSegments with conf >= 0.6: {high_conf}")


if __name__ == "__main__":
    main()
