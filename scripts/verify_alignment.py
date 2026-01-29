#!/usr/bin/env python3
"""
Verification Script for TuvaLLM Alignment.

- Export 20 random audio clips with transcripts for manual spot-check
- Generate quality report with confidence distribution
- Flag suspicious segments (very slow/fast WPS, low confidence)
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import soundfile as sf
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEGMENTS_DIR = PROCESSED_DIR / "segments_confident"
VERIFICATION_DIR = PROCESSED_DIR / "verification_samples"

VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def load_segments(json_path: Path) -> list:
    """Load segments from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_sample_clips(
    segments: list,
    n_samples: int = 20,
    output_dir: Path = VERIFICATION_DIR
) -> list:
    """
    Export random sample of audio clips with transcripts for manual verification.

    Returns list of exported samples with metadata.
    """
    # Filter for segments with valid audio paths
    valid_segments = [s for s in segments if Path(s["audio_path"]).exists()]

    if len(valid_segments) < n_samples:
        print(f"Warning: Only {len(valid_segments)} valid segments available")
        n_samples = len(valid_segments)

    if not valid_segments:
        print("No valid segments to export!")
        return []

    # Sample with stratification by confidence
    random.seed(42)

    # Stratify: half high-confidence, half medium-confidence
    high_conf = [s for s in valid_segments if s.get("confidence", 0) >= 0.7]
    med_conf = [s for s in valid_segments if 0.6 <= s.get("confidence", 0) < 0.7]

    samples = []

    # Sample from high confidence
    n_high = min(n_samples // 2, len(high_conf))
    if high_conf:
        samples.extend(random.sample(high_conf, n_high))

    # Sample from medium confidence
    n_med = min(n_samples - len(samples), len(med_conf))
    if med_conf:
        samples.extend(random.sample(med_conf, n_med))

    # Fill remaining with any valid segments
    remaining = n_samples - len(samples)
    if remaining > 0:
        available = [s for s in valid_segments if s not in samples]
        samples.extend(random.sample(available, min(remaining, len(available))))

    print(f"Exporting {len(samples)} sample clips...")

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    exported = []

    for i, seg in enumerate(tqdm(samples, desc="Exporting clips")):
        audio_path = Path(seg["audio_path"])

        try:
            # Load audio
            audio, sr = sf.read(audio_path, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Extract segment
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)

            if end_sample > len(audio):
                end_sample = len(audio)

            clip = audio[start_sample:end_sample]

            # Save clip
            clip_name = f"sample_{i+1:02d}.wav"
            clip_path = clips_dir / clip_name
            sf.write(clip_path, clip, sr)

            # Record metadata
            export_info = {
                "clip_file": clip_name,
                "text": seg["text"],
                "duration": seg["end"] - seg["start"],
                "wps": seg.get("wps", 0),
                "confidence": seg.get("confidence", 0),
                "anchor_start": seg.get("anchor_start"),
                "anchor_end": seg.get("anchor_end"),
                "source_audio": audio_path.name,
                "source_start": seg["start"],
                "source_end": seg["end"],
            }
            exported.append(export_info)

        except Exception as e:
            print(f"Error exporting clip {i+1}: {e}")

    return exported


def generate_quality_report(segments: list, output_dir: Path = VERIFICATION_DIR) -> dict:
    """Generate quality report with confidence distribution and flagged segments."""

    report = {
        "summary": {},
        "confidence_distribution": {},
        "wps_distribution": {},
        "flagged_segments": [],
    }

    if not segments:
        return report

    # Summary stats
    total = len(segments)
    total_duration = sum(s["end"] - s["start"] for s in segments)
    avg_wps = sum(s.get("wps", 0) for s in segments) / total
    avg_conf = sum(s.get("confidence", 0) for s in segments) / total

    report["summary"] = {
        "total_segments": total,
        "total_duration_hours": total_duration / 3600,
        "avg_wps": avg_wps,
        "avg_confidence": avg_conf,
    }

    # Confidence distribution
    conf_buckets = {
        "very_high (>= 0.8)": 0,
        "high (0.7-0.8)": 0,
        "medium (0.6-0.7)": 0,
        "low (< 0.6)": 0,
    }

    for s in segments:
        conf = s.get("confidence", 0)
        if conf >= 0.8:
            conf_buckets["very_high (>= 0.8)"] += 1
        elif conf >= 0.7:
            conf_buckets["high (0.7-0.8)"] += 1
        elif conf >= 0.6:
            conf_buckets["medium (0.6-0.7)"] += 1
        else:
            conf_buckets["low (< 0.6)"] += 1

    report["confidence_distribution"] = conf_buckets

    # WPS distribution
    wps_buckets = {
        "very_slow (< 1.0)": 0,
        "slow (1.0-1.5)": 0,
        "normal (1.5-3.5)": 0,
        "fast (3.5-4.5)": 0,
        "very_fast (> 4.5)": 0,
    }

    for s in segments:
        wps = s.get("wps", 0)
        if wps < 1.0:
            wps_buckets["very_slow (< 1.0)"] += 1
        elif wps < 1.5:
            wps_buckets["slow (1.0-1.5)"] += 1
        elif wps < 3.5:
            wps_buckets["normal (1.5-3.5)"] += 1
        elif wps < 4.5:
            wps_buckets["fast (3.5-4.5)"] += 1
        else:
            wps_buckets["very_fast (> 4.5)"] += 1

    report["wps_distribution"] = wps_buckets

    # Flag suspicious segments
    flagged = []

    for i, s in enumerate(segments):
        flags = []

        wps = s.get("wps", 0)
        conf = s.get("confidence", 0)
        duration = s["end"] - s["start"]

        # Flag conditions
        if wps < 1.0:
            flags.append("very_slow_wps")
        if wps > 4.5:
            flags.append("very_fast_wps")
        if conf < 0.55:
            flags.append("low_confidence")
        if duration < 1.0:
            flags.append("very_short")
        if duration > 12.0:
            flags.append("very_long")

        # Check for suspiciously repetitive text
        words = s["text"].split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                flags.append("repetitive_text")

        if flags:
            flagged.append({
                "index": i,
                "text": s["text"][:50] + "..." if len(s["text"]) > 50 else s["text"],
                "wps": wps,
                "confidence": conf,
                "duration": duration,
                "flags": flags,
            })

    report["flagged_segments"] = flagged[:50]  # Limit to top 50
    report["flagged_count"] = len(flagged)

    return report


def create_verification_checklist(exported_samples: list, output_dir: Path) -> None:
    """Create a markdown checklist for manual verification."""

    checklist_path = output_dir / "verification_checklist.md"

    with open(checklist_path, "w", encoding="utf-8") as f:
        f.write("# TuvaLLM Alignment Verification Checklist\n\n")
        f.write("Listen to each clip and verify the transcript matches the audio.\n\n")
        f.write("## Scoring Guide\n")
        f.write("- **Perfect (5)**: Transcript matches audio exactly\n")
        f.write("- **Good (4)**: Minor errors (1-2 words)\n")
        f.write("- **Acceptable (3)**: Some errors but understandable\n")
        f.write("- **Poor (2)**: Many errors, partially correct\n")
        f.write("- **Wrong (1)**: Transcript doesn't match audio\n\n")
        f.write("---\n\n")

        for i, sample in enumerate(exported_samples):
            f.write(f"## Sample {i+1}: `{sample['clip_file']}`\n\n")
            f.write(f"**Transcript:** {sample['text']}\n\n")
            f.write(f"**Duration:** {sample['duration']:.2f}s | ")
            f.write(f"**WPS:** {sample['wps']:.2f} | ")
            f.write(f"**Confidence:** {sample['confidence']:.3f}\n\n")
            f.write(f"**Source:** {sample['source_audio']} @ {sample['source_start']:.1f}s\n\n")
            f.write("- [ ] Score: _____ (1-5)\n")
            f.write("- [ ] Notes: \n\n")
            f.write("---\n\n")

    print(f"Created verification checklist: {checklist_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify TuvaLLM alignment quality")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples to export")
    parser.add_argument("--segments-file", type=Path,
                        default=SEGMENTS_DIR / "confident_segments.json",
                        help="Path to segments JSON")
    parser.add_argument("--output-dir", type=Path, default=VERIFICATION_DIR,
                        help="Output directory for verification files")
    args = parser.parse_args()

    print("=" * 60)
    print("TuvaLLM Alignment Verification")
    print("=" * 60)

    if not args.segments_file.exists():
        print(f"Segments file not found: {args.segments_file}")
        print("Run anchor_align.py first to generate segments.")
        return

    # Load segments
    print(f"\nLoading segments from {args.segments_file}...")
    segments = load_segments(args.segments_file)
    print(f"Loaded {len(segments)} segments")

    # Generate quality report
    print("\nGenerating quality report...")
    report = generate_quality_report(segments, args.output_dir)

    # Save report
    report_path = args.output_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved quality report: {report_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("QUALITY REPORT SUMMARY")
    print(f"{'=' * 60}")

    summary = report["summary"]
    print(f"Total segments: {summary['total_segments']}")
    print(f"Total duration: {summary['total_duration_hours']:.2f} hours")
    print(f"Average WPS: {summary['avg_wps']:.2f}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")

    print(f"\nConfidence distribution:")
    for bucket, count in report["confidence_distribution"].items():
        pct = count / summary['total_segments'] * 100 if summary['total_segments'] else 0
        print(f"  {bucket}: {count} ({pct:.1f}%)")

    print(f"\nWPS distribution:")
    for bucket, count in report["wps_distribution"].items():
        pct = count / summary['total_segments'] * 100 if summary['total_segments'] else 0
        print(f"  {bucket}: {count} ({pct:.1f}%)")

    print(f"\nFlagged segments: {report['flagged_count']}")

    # Export sample clips
    print(f"\n{'=' * 60}")
    print("EXPORTING SAMPLE CLIPS")
    print(f"{'=' * 60}")

    exported = export_sample_clips(segments, args.n_samples, args.output_dir)

    if exported:
        # Save exported info
        exported_path = args.output_dir / "exported_samples.json"
        with open(exported_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)
        print(f"Saved sample metadata: {exported_path}")

        # Create verification checklist
        create_verification_checklist(exported, args.output_dir)

    print(f"\n{'=' * 60}")
    print("VERIFICATION FILES")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - quality_report.json")
    print(f"  - exported_samples.json")
    print(f"  - verification_checklist.md")
    print(f"  - clips/sample_*.wav")


if __name__ == "__main__":
    main()
