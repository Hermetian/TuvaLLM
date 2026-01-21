#!/usr/bin/env python3
"""
TuvaLLM UDHR Benchmark Script
Quick validation that the pipeline works end-to-end using the UDHR sample.

This script:
1. Loads the UDHR audio sample (Omniglot)
2. Runs MMS baseline (Samoan adapter) transcription
3. Runs fine-tuned model transcription (if available)
4. Compares both against ground truth
5. Reports WER for both orthographic systems (System 4 and 6)

Usage:
    python scripts/benchmark_udhr.py
    python scripts/benchmark_udhr.py --download  # Download audio first
    python scripts/benchmark_udhr.py --model path/to/model  # Test specific model
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torchaudio
import numpy as np

# Import orthography module
from orthography import to_system6, detect_system, normalize_text

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
UDHR_DIR = DATA_DIR / "test" / "udhr_sample"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    transcription: str
    wer_system4: float
    wer_system6: float
    cer_system4: float
    cer_system6: float


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    try:
        from jiwer import wer
        if not reference.strip():
            return 1.0 if hypothesis.strip() else 0.0
        if not hypothesis.strip():
            return 1.0
        return wer(reference, hypothesis)
    except ImportError:
        print("Warning: jiwer not installed. Install with: pip install jiwer")
        # Fallback to simple word-level comparison
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        # Simple edit distance approximation
        errors = abs(len(ref_words) - len(hyp_words))
        for i, word in enumerate(ref_words):
            if i >= len(hyp_words) or word != hyp_words[i]:
                errors += 1
        return min(1.0, errors / len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    try:
        from jiwer import cer
        if not reference.strip():
            return 1.0 if hypothesis.strip() else 0.0
        if not hypothesis.strip():
            return 1.0
        return cer(reference, hypothesis)
    except ImportError:
        # Fallback
        ref_chars = list(reference.lower().replace(" ", ""))
        hyp_chars = list(hypothesis.lower().replace(" ", ""))
        if not ref_chars:
            return 1.0 if hyp_chars else 0.0
        errors = abs(len(ref_chars) - len(hyp_chars))
        for i, char in enumerate(ref_chars):
            if i >= len(hyp_chars) or char != hyp_chars[i]:
                errors += 1
        return min(1.0, errors / len(ref_chars))


def load_udhr_sample() -> Tuple[np.ndarray, int, Dict[str, str]]:
    """
    Load the UDHR audio sample and reference texts.

    Returns:
        Tuple of (waveform, sample_rate, reference_texts)
        reference_texts has keys 'system4' and 'system6'
    """
    audio_path = UDHR_DIR / "audio.mp3"
    metadata_path = UDHR_DIR / "metadata.json"
    reference_path = UDHR_DIR / "reference.txt"

    # Check if files exist
    if not audio_path.exists():
        raise FileNotFoundError(
            f"UDHR audio not found at {audio_path}\n"
            "Download it with: python scripts/benchmark_udhr.py --download\n"
            "Or manually from: https://www.omniglot.com/soundfiles/udhr/udhr_tuvaluan.mp3"
        )

    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.squeeze().numpy()

    # Load reference texts
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        reference_texts = metadata.get("reference_text", {})
    else:
        # Fall back to reference.txt
        with open(reference_path, "r", encoding="utf-8") as f:
            system4_text = f.read().strip()
        reference_texts = {
            "system4": system4_text,
            "system6": to_system6(system4_text)
        }

    return waveform, sample_rate, reference_texts


def download_udhr_audio() -> bool:
    """Download the UDHR audio sample from Omniglot."""
    import urllib.request

    url = "https://www.omniglot.com/soundfiles/udhr/udhr_tuvaluan.mp3"
    output_path = UDHR_DIR / "audio.mp3"

    print(f"Downloading UDHR audio from {url}...")
    UDHR_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(output_path))
        print(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please download manually from the URL above.")
        return False


def transcribe_mms_baseline(waveform: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe using MMS baseline (Samoan adapter)."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print("Loading MMS baseline (Samoan adapter)...")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    processor.tokenizer.set_target_lang("smo")
    model.load_adapter("smo")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Process
    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


def transcribe_finetuned(
    waveform: np.ndarray,
    model_path: str,
    sample_rate: int = 16000
) -> str:
    """Transcribe using fine-tuned MMS model."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"Loading fine-tuned model from {model_path}...")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


def evaluate_transcription(
    transcription: str,
    reference_texts: Dict[str, str],
    model_name: str
) -> BenchmarkResult:
    """Evaluate transcription against both orthographic systems."""
    # Normalize transcription
    trans_normalized = normalize_text(transcription, target_system="system4")
    trans_system6 = to_system6(trans_normalized)

    # Get reference texts
    ref_system4 = normalize_text(reference_texts["system4"], target_system="system4")
    ref_system6 = normalize_text(reference_texts["system6"], target_system="system6")

    # Compute metrics
    wer_s4 = compute_wer(ref_system4, trans_normalized)
    wer_s6 = compute_wer(ref_system6, trans_system6)
    cer_s4 = compute_cer(ref_system4, trans_normalized)
    cer_s6 = compute_cer(ref_system6, trans_system6)

    return BenchmarkResult(
        model_name=model_name,
        transcription=transcription,
        wer_system4=wer_s4,
        wer_system6=wer_s6,
        cer_system4=cer_s4,
        cer_system6=cer_s6
    )


def print_results(results: list, reference_texts: Dict[str, str]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("UDHR BENCHMARK RESULTS")
    print("=" * 70)

    print("\nReference Text (System 4):")
    print(f"  {reference_texts['system4'][:80]}...")

    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'WER (S4)':>10} {'WER (S6)':>10} {'CER (S4)':>10} {'CER (S6)':>10}")
    print("-" * 70)

    for result in results:
        print(f"{result.model_name:<25} {result.wer_system4:>10.1%} {result.wer_system6:>10.1%} "
              f"{result.cer_system4:>10.1%} {result.cer_system6:>10.1%}")

    print("-" * 70)

    # Show transcriptions
    print("\nTranscriptions:")
    for result in results:
        print(f"\n  [{result.model_name}]")
        print(f"  {result.transcription[:100]}...")

    print("\n" + "=" * 70)


def save_results(results: list, output_path: Path):
    """Save results to JSON file."""
    output = {
        "benchmark": "UDHR Article 1",
        "results": []
    }

    for result in results:
        output["results"].append({
            "model": result.model_name,
            "transcription": result.transcription,
            "wer_system4": result.wer_system4,
            "wer_system6": result.wer_system6,
            "cer_system4": result.cer_system4,
            "cer_system6": result.cer_system6
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")


def benchmark():
    """Run the full benchmark."""
    print("=" * 70)
    print("TuvaLLM UDHR Benchmark")
    print("=" * 70)
    print()

    # Load sample
    print("Loading UDHR sample...")
    waveform, sample_rate, reference_texts = load_udhr_sample()
    print(f"Audio: {len(waveform)/sample_rate:.1f}s at {sample_rate}Hz")

    results = []

    # MMS Baseline
    print("\n--- MMS Baseline (Samoan) ---")
    try:
        trans_baseline = transcribe_mms_baseline(waveform, sample_rate)
        result_baseline = evaluate_transcription(
            trans_baseline, reference_texts, "MMS Baseline (Samoan)"
        )
        results.append(result_baseline)
    except Exception as e:
        print(f"Error running baseline: {e}")

    # Fine-tuned model (if available)
    finetuned_path = MODELS_DIR / "mms-tuvaluan"
    if finetuned_path.exists():
        print("\n--- MMS Fine-tuned ---")
        try:
            trans_finetuned = transcribe_finetuned(waveform, str(finetuned_path), sample_rate)
            result_finetuned = evaluate_transcription(
                trans_finetuned, reference_texts, "MMS Fine-tuned"
            )
            results.append(result_finetuned)
        except Exception as e:
            print(f"Error running fine-tuned model: {e}")
    else:
        print(f"\nFine-tuned model not found at {finetuned_path}")
        print("Train one with: python scripts/finetune_mms.py")

    # Print and save results
    if results:
        print_results(results, reference_texts)
        output_path = UDHR_DIR / "benchmark_results.json"
        save_results(results, output_path)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TuvaLLM UDHR Benchmark")
    parser.add_argument("--download", action="store_true",
                       help="Download UDHR audio sample")
    parser.add_argument("--model", type=str,
                       help="Path to specific model to test")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline model evaluation")
    args = parser.parse_args()

    if args.download:
        download_udhr_audio()
        return

    try:
        if args.model:
            # Test specific model
            print("Loading UDHR sample...")
            waveform, sample_rate, reference_texts = load_udhr_sample()

            print(f"\nTesting model: {args.model}")
            trans = transcribe_finetuned(waveform, args.model, sample_rate)
            result = evaluate_transcription(trans, reference_texts, Path(args.model).name)
            print_results([result], reference_texts)
        else:
            # Run full benchmark
            benchmark()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
