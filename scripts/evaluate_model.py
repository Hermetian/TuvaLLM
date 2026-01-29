#!/usr/bin/env python3
"""
Evaluate fine-tuned MMS model on Tuvaluan test set.
"""
import torch
import json
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
from tqdm import tqdm
import evaluate
import re
import glob
import os
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_FILE = PROCESSED_DIR / "dataset" / "test.jsonl"

def normalize_text(text, allowed_chars):
    """Normalize text using same whitelist as training."""
    text = text.lower()
    clean_text = []
    for char in text:
        if char in allowed_chars:
            clean_text.append(char)
        elif char == " ":
            clean_text.append(" ")
    return re.sub(r'\s+', ' ', "".join(clean_text)).strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned MMS model")
    parser.add_argument("--model-dir", type=str, default="models/mms-tuvaluan-v2",
                        help="Path to model directory (relative to project root or absolute)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()

    # Resolve model path
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    effective_model_path = model_dir
    if not (model_dir / "config.json").exists():
        checkpoints = sorted(glob.glob(str(model_dir / "checkpoint-*")), key=os.path.getmtime)
        if checkpoints:
            effective_model_path = Path(checkpoints[-1])
            print(f"Root model not found. Using latest checkpoint: {effective_model_path}")
        else:
            print(f"Error: No model found at {model_dir}")
            return

    print(f"Loading model from: {effective_model_path}")

    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained(effective_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(effective_model_path)

    # CRITICAL: Ensure adapters are applied
    model.config.add_adapter = True
    print(f"Vocab size: {model.config.vocab_size} (Expected ~30 for v2)")
    if model.config.vocab_size > 50:
        print("WARNING: Model vocab size is too large. Random re-initialization might have occurred!")

    # Check device (MPS fallback for CTC not needed for inference, but model runs on device)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval() # Ensure eval mode

    # Load vocab for normalization - use vocab from model directory
    vocab_file = model_dir / "vocab" / "vocab.json"
    if not vocab_file.exists():
        vocab_file = effective_model_path / "vocab.json"
    if not vocab_file.exists():
        vocab_file = PROCESSED_DIR / "vocab.json"
    print(f"Using vocab from: {vocab_file}")

    with open(vocab_file, "r") as f:
        vocab = json.load(f)
        allowed_chars = set(vocab.keys())
        if "|" in allowed_chars: allowed_chars.add(" ")
        allowed_chars = {c for c in allowed_chars if c not in ["<s>", "</s>", "<pad>", "<unk>"]}

    # Load test data
    test_samples = []
    with open(TEST_FILE, "r") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))

    # Limit samples if requested
    if args.num_samples:
        test_samples = test_samples[:args.num_samples]

    print(f"Evaluating on {len(test_samples)} samples...")

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    refs = []
    preds = []
    
    print("\nSample Predictions:")
    print("-" * 80)
    
    for i, sample in enumerate(tqdm(test_samples)):
        # Load audio
        audio_path = sample["audio"]
        speech, sr = sf.read(audio_path)
        
        # Resample
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(torch.tensor(speech, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
            
        # Process
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Decode
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]

        # Post-process: remove special tokens that CTC decoder doesn't filter
        for special in ['<s>', '</s>', '<unk>', '<pad>']:
            pred_str = pred_str.replace(special, '')
        
        # Reference
        ref_str = sample.get("text", sample.get("sentence", ""))
        ref_norm = normalize_text(ref_str, allowed_chars)
        
        # Skip empty references
        if not ref_norm.strip():
            continue
            
        refs.append(ref_norm)
        preds.append(pred_str)
        
        if i < 10:
            print(f"\nRef:  {ref_norm}")
            print(f"Pred: {pred_str}")

    print("-" * 80)
    
    import jiwer
    
    # Calculate metrics
    wer = jiwer.wer(refs, preds)
    cer = jiwer.cer(refs, preds)
    
    print(f"\nResults:")
    print(f"WER: {wer:.2%}")
    print(f"CER: {cer:.2%}")
    
    # Save results
    results = {
        "wer": wer,
        "cer": cer,
        "samples": [{"ref": r, "pred": p} for r, p in zip(refs[:20], preds[:20])]
    }
    with open(PROJECT_ROOT / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
