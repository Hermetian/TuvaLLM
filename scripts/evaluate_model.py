#!/usr/bin/env python3
"""
TuvaLLM Model Evaluation Script
Evaluates trained ASR models using WER/CER metrics.

Features:
- Evaluate MMS fine-tuned model
- Evaluate Whisper LoRA model
- Compare baseline vs fine-tuned
- Linguistic validation (check for valid Tuvaluan words)
- Per-segment analysis
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import torch
import torchaudio
import numpy as np

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    wer: float
    cer: float
    num_samples: int
    predictions: List[str]
    references: List[str]
    per_sample_wer: List[float]
    per_sample_cer: List[float]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    from jiwer import wer
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    if not hypothesis.strip():
        return 1.0
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    from jiwer import cer
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    if not hypothesis.strip():
        return 1.0
    return cer(reference, hypothesis)


def load_test_data(test_file: str) -> List[Dict]:
    """Load test dataset."""
    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocabulary() -> set:
    """Load confirmed Tuvaluan vocabulary."""
    vocab_file = PROCESSED_DIR / "corrections" / "tuvaluan_vocabulary.json"
    if vocab_file.exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            return set(vocab_data.keys())
    return set()


class MMSEvaluator:
    """Evaluator for MMS models."""

    def __init__(self, model_path: Optional[str] = None, use_baseline: bool = False):
        """
        Initialize MMS evaluator.

        Args:
            model_path: Path to fine-tuned model, or None for baseline
            use_baseline: If True, use Samoan adapter baseline
        """
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.use_baseline = use_baseline

        if use_baseline or model_path is None:
            print("Loading MMS baseline (Samoan adapter)...")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
            self.processor.tokenizer.set_target_lang("smo")
            self.model.load_adapter("smo")
        else:
            print(f"Loading fine-tuned MMS from {model_path}...")
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        # Check if file exists
        if not Path(audio_path).exists():
            print(f"Warning: Audio file not found: {audio_path}")
            return ""

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.squeeze().numpy()

        # Process
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        return transcription

    def evaluate(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate model on test set."""
        predictions = []
        references = []
        per_sample_wer = []
        per_sample_cer = []

        print(f"Evaluating {len(test_data)} samples...")

        for i, item in enumerate(test_data):
            # Transcribe
            pred = self.transcribe(item["audio"])
            ref = item["sentence"]

            predictions.append(pred)
            references.append(ref)

            # Compute metrics
            sample_wer = compute_wer(ref, pred)
            sample_cer = compute_cer(ref, pred)
            per_sample_wer.append(sample_wer)
            per_sample_cer.append(sample_cer)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")

        # Aggregate metrics
        avg_wer = np.mean(per_sample_wer)
        avg_cer = np.mean(per_sample_cer)

        return EvaluationResult(
            wer=avg_wer,
            cer=avg_cer,
            num_samples=len(test_data),
            predictions=predictions,
            references=references,
            per_sample_wer=per_sample_wer,
            per_sample_cer=per_sample_cer
        )


class WhisperEvaluator:
    """Evaluator for Whisper models."""

    def __init__(self, model_path: Optional[str] = None, use_baseline: bool = False):
        """
        Initialize Whisper evaluator.

        Args:
            model_path: Path to fine-tuned LoRA model
            use_baseline: If True, use base Whisper
        """
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.use_baseline = use_baseline

        try:
            if use_baseline or model_path is None:
                print("Loading Whisper baseline...")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            else:
                print(f"Loading fine-tuned Whisper from {model_path}...")
                from peft import PeftModel

                base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.processor = WhisperProcessor.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Set decoding config
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="sm",
            task="transcribe"
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        if not Path(audio_path).exists():
            print(f"Warning: Audio file not found: {audio_path}")
            return ""

        waveform, sr = torchaudio.load(audio_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.squeeze().numpy()

        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                max_length=225
            )

        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def evaluate(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate model on test set."""
        predictions = []
        references = []
        per_sample_wer = []
        per_sample_cer = []

        print(f"Evaluating {len(test_data)} samples...")

        for i, item in enumerate(test_data):
            pred = self.transcribe(item["audio"])
            ref = item["sentence"]

            predictions.append(pred)
            references.append(ref)

            sample_wer = compute_wer(ref, pred)
            sample_cer = compute_cer(ref, pred)
            per_sample_wer.append(sample_wer)
            per_sample_cer.append(sample_cer)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")

        avg_wer = np.mean(per_sample_wer)
        avg_cer = np.mean(per_sample_cer)

        return EvaluationResult(
            wer=avg_wer,
            cer=avg_cer,
            num_samples=len(test_data),
            predictions=predictions,
            references=references,
            per_sample_wer=per_sample_wer,
            per_sample_cer=per_sample_cer
        )


def linguistic_analysis(predictions: List[str], vocabulary: set) -> Dict[str, Any]:
    """Analyze predictions for valid Tuvaluan words."""
    all_words = []
    for pred in predictions:
        words = pred.lower().split()
        all_words.extend(words)

    word_counts = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counts)

    # Check vocabulary coverage
    if vocabulary:
        known_words = sum(1 for w in all_words if w in vocabulary)
        vocab_coverage = known_words / total_words if total_words > 0 else 0
    else:
        known_words = 0
        vocab_coverage = 0

    # Most common words
    most_common = word_counts.most_common(20)

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "vocabulary_coverage": vocab_coverage,
        "known_words": known_words,
        "most_common": most_common
    }


def compare_models(results: Dict[str, EvaluationResult]) -> None:
    """Print comparison of multiple model results."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Table header
    print(f"\n{'Model':<30} {'WER':>10} {'CER':>10} {'Samples':>10}")
    print("-" * 60)

    for name, result in results.items():
        print(f"{name:<30} {result.wer:>10.2%} {result.cer:>10.2%} {result.num_samples:>10}")

    print("-" * 60)


def save_results(results: Dict[str, EvaluationResult], output_file: str) -> None:
    """Save evaluation results to JSON."""
    output = {}
    for name, result in results.items():
        output[name] = {
            "wer": result.wer,
            "cer": result.cer,
            "num_samples": result.num_samples,
            "predictions": result.predictions,
            "references": result.references,
            "per_sample_wer": result.per_sample_wer,
            "per_sample_cer": result.per_sample_cer
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TuvaLLM ASR models")
    parser.add_argument("--model", choices=["mms", "whisper", "both"], default="mms",
                       help="Model to evaluate")
    parser.add_argument("--mms-path", type=str, default=str(MODELS_DIR / "mms-tuvaluan"),
                       help="Path to fine-tuned MMS model")
    parser.add_argument("--whisper-path", type=str, default=str(MODELS_DIR / "whisper-tuvaluan-lora"),
                       help="Path to fine-tuned Whisper model")
    parser.add_argument("--test-file", type=str, default=str(PROCESSED_DIR / "test.json"),
                       help="Test data file")
    parser.add_argument("--baseline", action="store_true",
                       help="Also evaluate baseline models")
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR / "evaluation_results.json"),
                       help="Output file for results")
    parser.add_argument("--linguistic", action="store_true",
                       help="Perform linguistic analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("TuvaLLM Model Evaluation")
    print("=" * 70)
    print()

    # Load test data
    if not Path(args.test_file).exists():
        print(f"Error: Test file not found at {args.test_file}")
        print("Run prepare_training_data.py first.")
        return

    test_data = load_test_data(args.test_file)
    print(f"Loaded {len(test_data)} test samples\n")

    results = {}

    # Evaluate MMS
    if args.model in ["mms", "both"]:
        if args.baseline:
            print("\n--- MMS Baseline (Samoan) ---")
            evaluator = MMSEvaluator(use_baseline=True)
            results["MMS Baseline (Samoan)"] = evaluator.evaluate(test_data)

        if Path(args.mms_path).exists():
            print("\n--- MMS Fine-tuned ---")
            evaluator = MMSEvaluator(model_path=args.mms_path)
            results["MMS Fine-tuned"] = evaluator.evaluate(test_data)
        else:
            print(f"Fine-tuned MMS not found at {args.mms_path}")

    # Evaluate Whisper
    if args.model in ["whisper", "both"]:
        if args.baseline:
            print("\n--- Whisper Baseline ---")
            evaluator = WhisperEvaluator(use_baseline=True)
            results["Whisper Baseline"] = evaluator.evaluate(test_data)

        if Path(args.whisper_path).exists():
            print("\n--- Whisper LoRA Fine-tuned ---")
            evaluator = WhisperEvaluator(model_path=args.whisper_path)
            results["Whisper LoRA"] = evaluator.evaluate(test_data)
        else:
            print(f"Fine-tuned Whisper not found at {args.whisper_path}")

    # Compare results
    if results:
        compare_models(results)

        # Linguistic analysis
        if args.linguistic:
            vocabulary = load_vocabulary()
            print("\n" + "=" * 70)
            print("LINGUISTIC ANALYSIS")
            print("=" * 70)

            for name, result in results.items():
                analysis = linguistic_analysis(result.predictions, vocabulary)
                print(f"\n{name}:")
                print(f"  Total words: {analysis['total_words']}")
                print(f"  Unique words: {analysis['unique_words']}")
                if vocabulary:
                    print(f"  Vocabulary coverage: {analysis['vocabulary_coverage']:.1%}")
                print(f"  Most common words: {', '.join(w for w, _ in analysis['most_common'][:10])}")

        # Save results
        save_results(results, args.output)

    else:
        print("\nNo models found to evaluate.")
        print("Train models with finetune_mms.py or finetune_whisper_lora.py first.")


if __name__ == "__main__":
    main()
