#!/usr/bin/env python3
"""Compare trained TuvaLLM model vs raw Samoan MMS model."""

import json
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def load_audio(path, target_sr=16000):
    """Load and resample audio."""
    waveform, sr = librosa.load(path, sr=target_sr)
    return torch.tensor(waveform, dtype=torch.float32)

def transcribe(model, processor, audio_path):
    """Transcribe audio file."""
    waveform = load_audio(audio_path)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def main():
    # Load trained TuvaLLM model
    print("Loading TuvaLLM model...")
    tuva_model = Wav2Vec2ForCTC.from_pretrained(
        "/Users/discordwell/Projects/TuvaLLM/models/mms-tuvaluan-v4"
    )
    tuva_processor = Wav2Vec2Processor.from_pretrained(
        "/Users/discordwell/Projects/TuvaLLM/models/mms-tuvaluan-v4"
    )
    tuva_model.eval()

    # Load raw Samoan MMS model
    print("Loading Samoan MMS model...")
    samoan_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    samoan_processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    samoan_processor.tokenizer.set_target_lang("smo")
    samoan_model.load_adapter("smo")
    samoan_model.eval()

    # Load test samples
    test_samples = []
    with open("/Users/discordwell/Projects/TuvaLLM/data/processed/dataset/test.jsonl") as f:
        for line in f:
            test_samples.append(json.loads(line))

    print("\n" + "="*80)
    print("COMPARISON: TuvaLLM (trained) vs Samoan MMS (baseline)")
    print("="*80)

    # Test first 5 samples
    for i, sample in enumerate(test_samples[:5]):
        audio_path = sample["file"]
        reference = sample["text"].strip()

        print(f"\n{'='*80}")
        print(f"Sample {i+1}: {audio_path.split('/')[-1]}")
        print(f"{'='*80}")

        print(f"\nREFERENCE:")
        print(f"  {reference[:100]}...")

        # TuvaLLM prediction
        tuva_pred = transcribe(tuva_model, tuva_processor, audio_path)
        print(f"\nTUVALLM (trained):")
        print(f"  {tuva_pred[:100]}...")

        # Samoan prediction
        samoan_pred = transcribe(samoan_model, samoan_processor, audio_path)
        print(f"\nSAMOAN MMS (baseline):")
        print(f"  {samoan_pred[:100]}...")

if __name__ == "__main__":
    main()
