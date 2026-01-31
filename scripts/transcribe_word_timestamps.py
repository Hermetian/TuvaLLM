#!/usr/bin/env python3
"""
Generate word-level timestamps from MMS Samoan ASR.

Outputs JSON with precise word boundaries for alignment:
[
  {"word": "palamene", "start": 2.10, "end": 2.85},
  {"word": "tuvalu", "start": 3.12, "end": 3.58},
  ...
]
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mms_word_timestamps"

# MMS uses 320x downsampling: 16kHz audio -> 50 frames/sec
FRAME_DURATION = 320 / 16000  # 0.02 seconds per frame


def load_model(device="cpu"):
    """Load MMS Samoan model."""
    print("Loading MMS Samoan model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    model.load_adapter("smo")
    model.to(device)
    model.eval()
    return model, processor


def extract_word_timestamps(logits, processor):
    """
    Extract word-level timestamps from CTC logits.

    Groups consecutive non-blank tokens into words,
    using | as word delimiter.
    """
    pred_ids = torch.argmax(logits, dim=-1)[0]

    vocab = processor.tokenizer.get_vocab()
    id_to_char = {v: k for k, v in vocab.items()}

    # Get blank and word delimiter IDs
    blank_id = vocab.get("<pad>", 0)
    delimiter_id = vocab.get("|", -1)

    words = []
    current_chars = []
    word_start_frame = None
    word_end_frame = None
    prev_id = -1

    for i, token_id in enumerate(pred_ids.tolist()):
        token_id_int = int(token_id)
        char = id_to_char.get(token_id_int, "")

        # Word delimiter (|) or blank after characters = end of word
        is_blank = (token_id_int == blank_id)
        is_delimiter = (token_id_int == delimiter_id)

        if is_delimiter:
            # End current word if we have one
            if current_chars and word_start_frame is not None:
                word = "".join(current_chars)
                words.append({
                    "word": word,
                    "start": round(word_start_frame * FRAME_DURATION, 3),
                    "end": round(word_end_frame * FRAME_DURATION, 3)
                })
                current_chars = []
                word_start_frame = None
                word_end_frame = None
            prev_id = token_id_int
            continue

        if is_blank:
            # Blanks don't end words - just skip
            prev_id = token_id_int
            continue

        # Regular character
        if char and char not in ["<s>", "</s>", "<unk>", "<pad>"]:
            # CTC deduplication: only add if different from previous
            if token_id_int != prev_id:
                if word_start_frame is None:
                    word_start_frame = i
                current_chars.append(char)
                word_end_frame = i

        prev_id = token_id_int

    # Handle last word
    if current_chars and word_start_frame is not None:
        word = "".join(current_chars)
        words.append({
            "word": word,
            "start": round(word_start_frame * FRAME_DURATION, 3),
            "end": round(word_end_frame * FRAME_DURATION, 3)
        })

    return words


def process_audio_file(audio_path, model, processor, device, chunk_duration=30.0):
    """
    Process an audio file and extract word-level timestamps.

    Processes in chunks to handle long files.
    """
    # Load audio
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        import torchaudio
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio_tensor).squeeze().numpy()
        sr = 16000

    total_duration = len(audio) / sr
    all_words = []

    # Process in chunks
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(2.0 * sr)  # 2 second overlap

    pos = 0
    chunk_idx = 0

    with torch.no_grad():
        while pos < len(audio):
            # Extract chunk
            end_pos = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end_pos]

            # Skip very short chunks
            if len(chunk) < sr * 0.5:
                break

            # Process
            inputs = processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits

            # Extract words
            words = extract_word_timestamps(logits.cpu(), processor)

            # Adjust timestamps for chunk position
            chunk_start_time = pos / sr
            for w in words:
                w["start"] = round(w["start"] + chunk_start_time, 3)
                w["end"] = round(w["end"] + chunk_start_time, 3)

            # Add words (handling overlap)
            if all_words and words:
                # Remove words from overlap region
                overlap_start = chunk_start_time
                # Keep words that end before overlap, or start in this chunk
                all_words = [w for w in all_words if w["end"] < overlap_start]

            all_words.extend(words)

            # Move to next chunk
            pos += chunk_samples - overlap_samples
            chunk_idx += 1

    return all_words, total_duration


def main():
    parser = argparse.ArgumentParser(description="Generate word-level timestamps from MMS")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR, help="Audio directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--chunk-duration", type=float, default=30.0, help="Chunk duration in seconds")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = list(args.audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return

    # Load model
    model, processor = load_model(args.device)

    # Process each file
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        output_path = args.output_dir / f"{audio_path.stem}.json"

        # Skip if already processed
        if output_path.exists():
            print(f"Skipping {audio_path.name} (already processed)")
            continue

        try:
            words, duration = process_audio_file(
                audio_path, model, processor, args.device, args.chunk_duration
            )

            # Save results
            result = {
                "audio_file": str(audio_path.name),
                "duration": round(duration, 2),
                "word_count": len(words),
                "words": words
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  {audio_path.name}: {len(words)} words, {duration:.1f}s")

        except Exception as e:
            print(f"  Error processing {audio_path.name}: {e}")
            continue

    print(f"\nDone! Word timestamps saved to {args.output_dir}")


if __name__ == "__main__":
    main()
