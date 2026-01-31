#!/usr/bin/env python3
"""
LLM Semantic Scoring for TuvaLLM Segments.

Uses Claude (via API or direct prompting) to score alignment quality
between MMS ASR output and Tuvaluan transcript.

This script prepares batches for LLM scoring. The actual scoring
should be done via Task agents with model="haiku" for efficiency.

Output format matches what prepare_dataset_v6.py expects.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEGMENTS_DIR = PROCESSED_DIR / "segments_word_aligned"
OUTPUT_DIR = PROCESSED_DIR / "llm_scored"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 50  # Segments per batch for LLM scoring


def load_candidate_segments(segments_path: Path) -> List[Dict]:
    """Load candidate segments from word_align.py output."""
    if not segments_path.exists():
        return []
    with open(segments_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_scoring_batches(
    segments: List[Dict],
    batch_size: int = BATCH_SIZE
) -> List[List[Dict]]:
    """Split segments into batches for parallel LLM scoring."""
    batches = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        batches.append(batch)
    return batches


def format_batch_for_llm(batch: List[Dict], batch_idx: int) -> str:
    """
    Format a batch of segments for LLM scoring.

    Returns a prompt string that can be used with Task agents.
    """
    prompt = f"""You are a Polynesian language expert scoring ASR alignment quality (batch {batch_idx}).

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. You must NOT write any code or algorithms
2. You must NOT calculate Jaccard similarity, Levenshtein distance, or ANY numerical metric
3. You must NOT use Python, JavaScript, or any programming
4. You MUST use your semantic understanding of Polynesian languages only
5. Just READ each pair and give your expert linguistic judgment

For each pair below, compare:
- MMS: Samoan ASR output (approximation of Tuvaluan speech)
- Transcript: Actual Tuvaluan text from parliament records

Score 0.0-1.0 based on YOUR semantic judgment:
- 0.9-1.0: Nearly identical meaning, minor spelling differences
- 0.7-0.8: Same general meaning, some word variations
- 0.5-0.6: Partial overlap, about half the content matches
- 0.3-0.4: Some matching words but different meaning
- 0.0-0.2: Little to no meaningful connection

Polynesian cognate patterns to consider:
- Samoan 'k' often = Tuvaluan 't' (e.g., tatou/katou)
- Word boundaries may differ (fakatasi vs faka tasi)
- Loan words from English appear in both

OUTPUT FORMAT - Return ONLY a JSON array, nothing else:
[{{"id": "june_001", "score": 0.75}}, {{"id": "june_002", "score": 0.6}}]

DO NOT explain your reasoning. DO NOT write code. Just output the JSON array.

Pairs to score:
"""

    for seg in batch:
        prompt += f"""
---
ID: {seg["id"]}
MMS: {seg["mms_text"]}
Transcript: {seg["text"]}
"""

    return prompt


def save_batches(batches: List[List[Dict]], output_dir: Path) -> List[Path]:
    """Save batches to JSON files for processing."""
    batch_paths = []

    for i, batch in enumerate(batches):
        batch_data = {
            "batch_idx": i,
            "segments": batch,
            "prompt": format_batch_for_llm(batch, i),
        }

        batch_path = output_dir / f"batch_{i:03d}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        batch_paths.append(batch_path)

    return batch_paths


def merge_scores(
    segments: List[Dict],
    scores_dir: Path
) -> List[Dict]:
    """
    Merge LLM scores back into segments.

    Looks for score files in scores_dir matching pattern: scores_batch_NNN.json
    """
    # Build ID -> segment mapping
    id_to_segment = {s["id"]: s for s in segments}

    # Load all score files
    score_files = sorted(scores_dir.glob("scores_batch_*.json"))

    if not score_files:
        print("No score files found. Run LLM scoring first.")
        return segments

    scores_merged = 0
    for score_file in score_files:
        with open(score_file, "r", encoding="utf-8") as f:
            try:
                scores = json.load(f)
                for item in scores:
                    seg_id = item.get("id")
                    score = item.get("score", 0.0)

                    if seg_id in id_to_segment:
                        id_to_segment[seg_id]["llm_score"] = score
                        scores_merged += 1
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {score_file}")

    print(f"Merged {scores_merged} LLM scores")

    # For segments without scores, use alignment score as fallback
    for seg in segments:
        if "llm_score" not in seg:
            seg["llm_score"] = seg.get("align_score", 0.5)

    return segments


def calculate_combined_confidence(segments: List[Dict]) -> List[Dict]:
    """
    Calculate combined confidence from alignment and LLM scores.

    Combined confidence = 0.4 * align_score + 0.6 * llm_score
    """
    for seg in segments:
        align_score = seg.get("align_score", 0.5)
        llm_score = seg.get("llm_score", align_score)

        # Weighted combination (LLM is more reliable for semantic match)
        combined = 0.4 * align_score + 0.6 * llm_score

        seg["confidence"] = round(combined, 3)

    return segments


def main():
    parser = argparse.ArgumentParser(description="LLM Scoring for TuvaLLM segments")
    parser.add_argument("--input", type=Path, default=SEGMENTS_DIR / "candidate_segments.json",
                        help="Input segments JSON file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for batches and scored segments")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Segments per batch")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare batches, don't merge scores")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing scores")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Semantic Scoring Pipeline")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load segments
    segments = load_candidate_segments(args.input)

    if not segments:
        print(f"No segments found in {args.input}")
        print("Run word_align.py first.")
        return

    print(f"Loaded {len(segments)} segments")

    if args.merge_only:
        # Just merge scores and save
        segments = merge_scores(segments, args.output_dir)
        segments = calculate_combined_confidence(segments)

        output_path = args.output_dir / "scored_segments.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"Saved scored segments to {output_path}")
        return

    # Create batches
    batches = create_scoring_batches(segments, args.batch_size)
    print(f"Created {len(batches)} batches of ~{args.batch_size} segments")

    # Save batches
    batch_paths = save_batches(batches, args.output_dir)
    print(f"Saved batch files to {args.output_dir}")

    if args.prepare_only:
        print("\nBatches prepared. To score, use Claude Code Task agents:")
        print("""
For each batch file, launch a Task agent with:
- subagent_type: "general-purpose"
- model: "haiku"
- prompt: [contents of batch prompt field]

Save results as scores_batch_NNN.json in the same directory.

Then run: python scripts/llm_score_segments.py --merge-only
""")
        return

    # If not prepare-only, we would score here
    # For now, just merge any existing scores
    segments = merge_scores(segments, args.output_dir)
    segments = calculate_combined_confidence(segments)

    # Save final output
    output_path = args.output_dir / "scored_segments.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    scored = sum(1 for s in segments if "llm_score" in s and s["llm_score"] != s.get("align_score"))
    print(f"Segments with LLM scores: {scored}")

    if segments:
        avg_conf = sum(s["confidence"] for s in segments) / len(segments)
        high_conf = sum(1 for s in segments if s["confidence"] >= 0.7)
        med_conf = sum(1 for s in segments if 0.5 <= s["confidence"] < 0.7)

        print(f"Average confidence: {avg_conf:.3f}")
        print(f"High confidence (>= 0.7): {high_conf}")
        print(f"Medium confidence (0.5-0.7): {med_conf}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
