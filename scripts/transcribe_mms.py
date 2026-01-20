#!/usr/bin/env python3
"""
TuvaLLM MMS Transcription Script
Uses Meta's MMS (Massively Multilingual Speech) with Samoan as proxy.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, '/Users/discordwell/Library/Python/3.9/lib/python/site-packages')

import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def transcribe_with_mms(audio_path: str, language: str = "smo"):
    """
    Transcribe audio using Meta's MMS ASR model.

    Args:
        audio_path: Path to audio file
        language: ISO 639-3 language code (smo = Samoan)

    Returns:
        dict with transcription results
    """
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    print(f"Loading MMS model for language: {language}")

    # Load MMS model
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # Set target language
    processor.tokenizer.set_target_lang(language)
    model.load_adapter(language)

    print(f"Loading audio: {audio_path}")

    # Load and resample audio to 16kHz
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Process in chunks (30 second segments)
    chunk_length = 30 * 16000  # 30 seconds at 16kHz
    waveform = waveform.squeeze()

    transcripts = []
    total_samples = waveform.shape[0]

    print(f"Processing {total_samples / 16000:.1f} seconds of audio in 30s chunks...")

    for i in range(0, total_samples, chunk_length):
        chunk = waveform[i:i + chunk_length]

        # Skip very short chunks
        if len(chunk) < 1600:  # < 0.1 seconds
            continue

        # Process chunk
        inputs = processor(chunk.numpy(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs).logits

        # Decode
        ids = torch.argmax(outputs, dim=-1)[0]
        text = processor.decode(ids)

        if text.strip():
            start_time = i / 16000
            end_time = min((i + chunk_length) / 16000, total_samples / 16000)
            transcripts.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })

        # Progress indicator
        progress = min(100, (i + chunk_length) / total_samples * 100)
        print(f"  Progress: {progress:.0f}%", end='\r')

    print()  # New line after progress

    # Combine all text
    full_text = " ".join([t["text"] for t in transcripts])

    return {
        "text": full_text,
        "language": language,
        "segments": transcripts
    }

def main():
    raw_dir = Path("/Users/discordwell/TuvaLLM/data/raw")
    transcript_dir = Path("/Users/discordwell/TuvaLLM/data/transcripts")
    transcript_dir.mkdir(exist_ok=True)

    # Just process first file for now (faster testing)
    audio_file = raw_dir / "grn_tuvalu_01.mp3"

    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return

    print("=" * 70)
    print("TuvaLLM MMS Transcription (Samoan Proxy)")
    print("=" * 70)
    print()

    try:
        result = transcribe_with_mms(str(audio_file), language="smo")

        print(f"\nSegments transcribed: {len(result['segments'])}")
        print(f"Total characters: {len(result['text'])}")

        # Save result
        output_file = transcript_dir / "mms_samoan_transcript.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nSaved to: {output_file}")

        # Display transcript
        print("\n" + "=" * 70)
        print("MMS TRANSCRIPT (Samoan proxy)")
        print("=" * 70)
        print()

        # Show segments
        print("Segments:")
        for seg in result['segments'][:10]:
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

        if len(result['segments']) > 10:
            print(f"  ... ({len(result['segments']) - 10} more segments)")

        print("\n" + "-" * 70)
        print("Full transcript:")
        print("-" * 70)
        print(result['text'][:2000])
        if len(result['text']) > 2000:
            print(f"\n... [truncated, {len(result['text'])} chars total]")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
