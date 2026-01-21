#!/usr/bin/env python3
"""
TuvaLLM Iterative Improvement Pipeline
Bootstrapping loop for expanding training data with model predictions.

Pipeline:
1. Train initial model on corrected segments
2. Run inference on remaining uncorrected segments
3. Review predictions vs original MMS output
4. Correct additional segments where model improves
5. Retrain with expanded dataset
6. Repeat until diminishing returns
"""

import os
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch
import torchaudio
import numpy as np

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"

# Iteration tracking
ITERATIONS_DIR = PROCESSED_DIR / "iterations"


def load_mms_transcript() -> Dict[str, Any]:
    """Load original MMS transcript."""
    mms_file = TRANSCRIPTS_DIR / "mms_samoan_transcript.json"
    if mms_file.exists():
        with open(mms_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_corrections() -> Dict[str, Any]:
    """Load current corrections."""
    corrections_file = PROCESSED_DIR / "corrections" / "corrections.json"
    if corrections_file.exists():
        with open(corrections_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"segments": {}, "vocabulary": {}, "metadata": {}}


def save_corrections(corrections: Dict[str, Any]):
    """Save updated corrections."""
    corrections_file = PROCESSED_DIR / "corrections" / "corrections.json"
    corrections_file.parent.mkdir(parents=True, exist_ok=True)
    corrections["metadata"]["last_modified"] = datetime.now().isoformat()
    with open(corrections_file, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


def get_uncorrected_segments(
    mms_transcript: Dict[str, Any],
    corrections: Dict[str, Any]
) -> List[Dict]:
    """Get segments that haven't been manually corrected."""
    corrected_keys = set(corrections.get("segments", {}).keys())
    uncorrected = []

    for seg in mms_transcript.get("segments", []):
        seg_key = f"{seg['start']:.1f}-{seg['end']:.1f}"
        if seg_key not in corrected_keys:
            uncorrected.append({
                "key": seg_key,
                "start": seg["start"],
                "end": seg["end"],
                "mms_text": seg["text"]
            })

    return uncorrected


def transcribe_with_model(
    model_path: str,
    audio_path: str,
    segments: List[Dict]
) -> List[Dict]:
    """
    Transcribe segments using fine-tuned model.

    Returns segments with model predictions added.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"Loading model from {model_path}")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load full audio
    print(f"Loading audio from {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.squeeze().numpy()

    # Transcribe each segment
    results = []
    for i, seg in enumerate(segments):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        audio_segment = waveform[start_sample:end_sample]

        # Process
        inputs = processor(
            audio_segment,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        results.append({
            **seg,
            "model_prediction": transcription
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(segments)}")

    return results


def compute_similarity(text1: str, text2: str) -> float:
    """Compute simple word-level similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def identify_improvements(
    segments_with_predictions: List[Dict],
    similarity_threshold: float = 0.3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Identify segments where model prediction might be better than original MMS.

    Returns (improved, unchanged) segments.
    """
    improved = []
    unchanged = []

    for seg in segments_with_predictions:
        mms_text = seg["mms_text"]
        model_pred = seg["model_prediction"]

        # Compute similarity
        similarity = compute_similarity(mms_text, model_pred)

        # Heuristics for "improved":
        # - Model output is non-empty
        # - Model output differs from original (but not completely)
        # - Model output has reasonable length

        is_improved = (
            model_pred.strip() and
            0.1 < similarity < 0.9 and
            len(model_pred) > 5
        )

        seg_with_analysis = {
            **seg,
            "similarity": similarity,
            "potentially_improved": is_improved
        }

        if is_improved:
            improved.append(seg_with_analysis)
        else:
            unchanged.append(seg_with_analysis)

    return improved, unchanged


def generate_review_file(
    improved_segments: List[Dict],
    output_file: str
) -> None:
    """Generate a review file for manual inspection of model improvements."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Model Improvement Review\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total segments for review: {len(improved_segments)}\n\n")

        for i, seg in enumerate(improved_segments):
            f.write(f"## Segment {i+1}: {seg['key']}\n")
            f.write(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s\n\n")
            f.write(f"**Original MMS:**\n{seg['mms_text']}\n\n")
            f.write(f"**Model Prediction:**\n{seg['model_prediction']}\n\n")
            f.write(f"Similarity: {seg['similarity']:.2%}\n")
            f.write(f"\n---\n\n")

    print(f"Review file generated: {output_file}")


def run_iteration(
    iteration_num: int,
    model_path: str,
    audio_file: str,
    auto_accept_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Run one iteration of the improvement pipeline.

    Args:
        iteration_num: Current iteration number
        model_path: Path to trained model
        audio_file: Audio file to process
        auto_accept_threshold: Similarity threshold for auto-accepting improvements

    Returns:
        Statistics about the iteration
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*70}\n")

    # Create iteration directory
    iter_dir = ITERATIONS_DIR / f"iter_{iteration_num:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    mms_transcript = load_mms_transcript()
    corrections = load_corrections()

    # Get uncorrected segments
    uncorrected = get_uncorrected_segments(mms_transcript, corrections)
    print(f"Uncorrected segments: {len(uncorrected)}")

    if not uncorrected:
        print("No uncorrected segments remaining!")
        return {"status": "complete", "remaining": 0}

    # Transcribe with model
    print("\nTranscribing with model...")
    audio_path = RAW_DIR / audio_file
    segments_with_predictions = transcribe_with_model(
        model_path,
        str(audio_path),
        uncorrected
    )

    # Identify improvements
    print("\nAnalyzing improvements...")
    improved, unchanged = identify_improvements(segments_with_predictions)

    print(f"Potentially improved: {len(improved)}")
    print(f"Unchanged: {len(unchanged)}")

    # Save predictions
    predictions_file = iter_dir / "predictions.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(segments_with_predictions, f, ensure_ascii=False, indent=2)

    # Generate review file
    review_file = iter_dir / "review.md"
    generate_review_file(improved, str(review_file))

    # Auto-accept high-confidence improvements
    auto_accepted = 0
    for seg in improved:
        if seg["similarity"] >= auto_accept_threshold:
            # Add to corrections with moderate quality score
            seg_key = seg["key"]
            corrections["segments"][seg_key] = {
                "original_mms": seg["mms_text"],
                "corrected_text": seg["model_prediction"],
                "start": seg["start"],
                "end": seg["end"],
                "quality": 3,  # Moderate quality - model-generated
                "source": f"iteration_{iteration_num}",
                "auto_accepted": True
            }
            auto_accepted += 1

    if auto_accepted > 0:
        print(f"\nAuto-accepted {auto_accepted} segments with similarity >= {auto_accept_threshold:.0%}")
        save_corrections(corrections)

    # Save iteration stats
    stats = {
        "iteration": iteration_num,
        "timestamp": datetime.now().isoformat(),
        "uncorrected_segments": len(uncorrected),
        "potentially_improved": len(improved),
        "auto_accepted": auto_accepted,
        "model_path": model_path
    }

    stats_file = iter_dir / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nIteration {iteration_num} complete.")
    print(f"Review file: {review_file}")
    print(f"Manual review needed for {len(improved) - auto_accepted} segments")

    return stats


def get_current_iteration() -> int:
    """Get the next iteration number."""
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(ITERATIONS_DIR.glob("iter_*"))
    if not existing:
        return 1
    numbers = [int(p.name.split("_")[1]) for p in existing]
    return max(numbers) + 1


def show_progress():
    """Show overall progress across iterations."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Load data
    mms_transcript = load_mms_transcript()
    corrections = load_corrections()

    total_segments = len(mms_transcript.get("segments", []))
    corrected_segments = len(corrections.get("segments", {}))

    # Quality breakdown
    quality_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    sources = {}
    for seg in corrections.get("segments", {}).values():
        q = seg.get("quality", 0)
        if q in quality_counts:
            quality_counts[q] += 1
        src = seg.get("source", "manual")
        sources[src] = sources.get(src, 0) + 1

    # Display
    console.print("\n[bold]TuvaLLM Progress Summary[/bold]\n")

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total segments", str(total_segments))
    table.add_row("Corrected", str(corrected_segments))
    table.add_row("Remaining", str(total_segments - corrected_segments))
    table.add_row("Progress", f"{corrected_segments/total_segments:.1%}" if total_segments else "N/A")

    console.print(table)

    # Quality breakdown
    console.print("\n[bold]Quality Distribution:[/bold]")
    for q in range(5, 0, -1):
        count = quality_counts[q]
        bar = "█" * (count // 2) if count > 0 else ""
        console.print(f"  {q}: {bar} {count}")

    # Source breakdown
    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for src, count in sorted(sources.items()):
            console.print(f"  {src}: {count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TuvaLLM Iterative Improvement Pipeline")
    parser.add_argument("--action", choices=["iterate", "progress", "prepare-retrain"],
                       default="progress", help="Action to perform")
    parser.add_argument("--model-path", type=str, default=str(MODELS_DIR / "mms-tuvaluan"),
                       help="Path to trained model")
    parser.add_argument("--audio", type=str, default="grn_tuvalu_01.mp3",
                       help="Audio file to process")
    parser.add_argument("--auto-threshold", type=float, default=0.7,
                       help="Similarity threshold for auto-accepting (default: 0.7)")
    args = parser.parse_args()

    if args.action == "progress":
        show_progress()

    elif args.action == "iterate":
        # Check model exists
        if not Path(args.model_path).exists():
            print(f"Error: Model not found at {args.model_path}")
            print("Train a model first with finetune_mms.py")
            return

        iteration_num = get_current_iteration()
        run_iteration(
            iteration_num,
            args.model_path,
            args.audio,
            args.auto_threshold
        )

    elif args.action == "prepare-retrain":
        # Prepare data for retraining after manual review
        print("Preparing data for retraining...")

        # Run prepare_training_data with updated corrections
        import subprocess
        subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_training_data.py"),
            "--min-quality", "3"
        ])

        print("\nData prepared. Run finetune_mms.py to retrain.")


if __name__ == "__main__":
    main()
