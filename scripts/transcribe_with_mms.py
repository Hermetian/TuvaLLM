#!/usr/bin/env python3
"""
Transcribe audio using MMS-1b-all (Samoan adapter) to get rough timestamps.
This is Step 1 of the alignment process.
"""

import os
import json
import torch
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2ForCTC, AutoProcessor
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mms_transcripts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_audio_files():
    files = sorted(list(AUDIO_DIR.rglob("*.wav")))
    # Sort by size (smallest first) to unblock pipeline verification
    files.sort(key=lambda f: f.stat().st_size)
    return files

def transcribe_file(model, processor, audio_path, device):
    print(f"Loading {audio_path.name}...")
    
    # Load audio using soundfile (more robust)
    import soundfile as sf
    waveform_np, sr = sf.read(audio_path, dtype="float32")
    
    # Convert to mono if stereo
    if len(waveform_np.shape) > 1:
        waveform_np = waveform_np.mean(axis=1)
        
    waveform = torch.tensor(waveform_np).unsqueeze(0) # (1, time)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Process in chunks (e.g. 30 seconds) to avoid OOM
    CHUNK_SEC = 10.0
    OVERLAP_SEC = 0.0 # Strict slicing for simplicity (might lose boundary words, but acceptable for rough alignment)
    chunk_samples = int(CHUNK_SEC * 16000)
    
    total_samples = waveform.size(1)
    
    all_words = []
    
    model.eval()
    
    # Use torch.no_grad
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, chunk_samples), desc="Transcribing chunks"):
            chunk = waveform[:, i:i+chunk_samples]
            if chunk.size(1) < 1600: # Skip tiny chunks (<0.1s)
                continue
                
            # Normalize
            input_values = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
            input_values = input_values.to(device)
            
            # Forward
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode with timestamps?
            # Wav2Vec2 processor batch_decode doesn't give timestamps easily.
            # We can use CTC decode manually or just get the text and interpolate time.
            # For simplicity, let's just get the text for now.
            # Actually, we NEED timestamps for alignment.
            
            # Use torchaudio CTC decoder or similar?
            # A simpler approach: Just get the text string.
            # Since we know the chunk start time (i / 16000), we can offset the words.
            # BUT: Getting word timestamps from CTC logits is non-trivial without a decoder.
            
            # Let's use `processor.batch_decode` to get text, and just assign the
            # timestamp to the CHUNK.
            # Then in text alignment, we know this text belongs to [T, T+10s].
            # This coarse alignment might be enough to slice the audio?
            # No, we want <30s segments. 10s chunks are perfect.
            
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Save: {text: "...", start: T, end: T+10}
            start_time = i / 16000.0
            end_time = (i + chunk.size(1)) / 16000.0
            
            if transcription.strip():
                all_words.append({
                    "text": transcription,
                    "start": start_time,
                    "end": end_time
                })

    return all_words

def main():
    print("="*60)
    print("MMS-Samoan Transcription (Chunked)")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    model_id = "facebook/mms-1b-all"
    print(f"Loading model {model_id} (smo adapter)...")
    processor = AutoProcessor.from_pretrained(model_id, target_lang="smo")
    model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang="smo")
    
    model.to(device)
    
    files = get_audio_files()
    print(f"Found {len(files)} files.")
    
    for f in files:
        output_json = OUTPUT_DIR / (f.stem + ".json")
        if output_json.exists():
            print(f"Skipping {f.name} (already done)")
            continue
            
        print(f"\nProcessing {f.name}...")
        results = transcribe_file(model, processor, f, device)
        
        with open(output_json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2, ensure_ascii=False)
            
    print("\nDone!")

if __name__ == "__main__":
    main()
