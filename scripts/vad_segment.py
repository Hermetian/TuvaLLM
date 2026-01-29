#!/usr/bin/env python3
"""
VAD-based segmentation: Split audio at silences and assign transcript text sequentially.
This approach doesn't require perfect text-audio alignment upfront.
"""

import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import re
import unicodedata
import docx
import soundfile as sf

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio_16k"
SEGMENTS_DIR = PROCESSED_DIR / "segments_vad"

SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

def load_docx_paragraphs(docx_path):
    """Load transcript as list of non-empty paragraphs."""
    doc = docx.Document(docx_path)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        # Skip headers/short lines
        if text and len(text.split()) >= 3:
            paragraphs.append(text)
    return paragraphs

def detect_speech_segments(audio_path, min_speech_duration=1.0, min_silence_duration=0.3):
    """
    Detect speech segments using energy-based VAD.
    Returns list of (start, end) tuples in seconds.
    """
    # Load audio
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Parameters
    frame_size = int(0.025 * sr)  # 25ms frames
    hop_size = int(0.010 * sr)    # 10ms hop
    
    # Compute energy per frame
    n_frames = (len(audio) - frame_size) // hop_size + 1
    energies = []
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        energy = (frame ** 2).mean()
        energies.append(energy)
    
    energies = torch.tensor(energies)
    
    # Adaptive threshold (dynamic based on percentiles)
    threshold = energies.median() * 2  # Speech is typically louder than silence
    
    # Binary speech/silence
    is_speech = energies > threshold
    
    # Smooth with median filter
    kernel_size = int(0.1 / 0.01)  # 100ms window
    if kernel_size % 2 == 0:
        kernel_size += 1
    is_speech_smooth = torch.nn.functional.max_pool1d(
        is_speech.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    ).squeeze() > 0.5
    
    # Find contiguous speech regions
    segments = []
    in_speech = False
    seg_start = 0
    
    for i, speech in enumerate(is_speech_smooth):
        time = i * 0.01  # 10ms per frame
        if speech and not in_speech:
            seg_start = time
            in_speech = True
        elif not speech and in_speech:
            seg_end = time
            duration = seg_end - seg_start
            if duration >= min_speech_duration:
                segments.append((seg_start, seg_end))
            in_speech = False
    
    # Handle final segment
    if in_speech:
        seg_end = n_frames * 0.01
        if seg_end - seg_start >= min_speech_duration:
            segments.append((seg_start, seg_end))
    
    # Merge segments that are close together
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] < min_silence_duration:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    
    return merged

def estimate_words_per_second(total_text_words, total_audio_duration):
    """Estimate speaking rate for text-audio mapping."""
    if total_audio_duration <= 0:
        return 2.0  # Default: 2 words per second
    return total_text_words / total_audio_duration

def assign_text_to_segments(segments, paragraphs, wps):
    """
    Assign text to audio segments based on duration and word count.
    Uses a simple sequential mapping.
    """
    if not segments or not paragraphs:
        return []
    
    # Flatten paragraphs into words
    all_words = []
    for para in paragraphs:
        words = para.split()
        all_words.extend(words)
    
    results = []
    word_idx = 0
    
    for start, end in segments:
        duration = end - start
        # Estimate how many words fit in this segment
        n_words = max(3, int(duration * wps))
        
        # Get words for this segment
        segment_words = all_words[word_idx:word_idx + n_words]
        if not segment_words:
            continue
        
        text = " ".join(segment_words)
        
        # Clean text (lowercase, basic normalization)
        text = unicodedata.normalize("NFC", text).lower()
        text = re.sub(r"[^\w\s]", "", text)
        
        if len(text.split()) >= 3:  # Minimum 3 words
            results.append({
                "text": text,
                "start": start,
                "end": end,
                "duration": duration
            })
        
        word_idx += n_words
    
    return results

def process_session(audio_files, transcript_path, session_name):
    """Process an entire parliament session."""
    print(f"\nProcessing {session_name}...")
    
    # Load transcript
    paragraphs = load_docx_paragraphs(transcript_path)
    total_words = sum(len(p.split()) for p in paragraphs)
    print(f"  Transcript: {len(paragraphs)} paragraphs, {total_words} words")
    
    # Get total audio duration
    total_duration = 0
    for af in audio_files:
        info = sf.info(af)
        total_duration += info.duration
    print(f"  Audio: {len(audio_files)} files, {total_duration/3600:.2f} hours")
    
    # Estimate words per second
    wps = estimate_words_per_second(total_words, total_duration)
    print(f"  Estimated WPS: {wps:.2f}")
    
    all_segments = []
    para_idx = 0
    
    for audio_file in tqdm(audio_files, desc=session_name):
        # Detect speech segments in this file
        try:
            speech_segs = detect_speech_segments(audio_file)
        except Exception as e:
            print(f"    VAD failed for {audio_file.name}: {e}")
            continue
        
        file_duration = sf.info(audio_file).duration
        
        # Estimate how many paragraphs belong to this file
        file_words_estimate = file_duration * wps
        paras_for_file = []
        words_so_far = 0
        
        while para_idx < len(paragraphs) and words_so_far < file_words_estimate:
            paras_for_file.append(paragraphs[para_idx])
            words_so_far += len(paragraphs[para_idx].split())
            para_idx += 1
        
        # Assign text to segments
        file_segments = assign_text_to_segments(speech_segs, paras_for_file, wps)
        
        # Add audio path
        for seg in file_segments:
            seg["audio_path"] = str(audio_file)
            seg["session"] = session_name
        
        all_segments.extend(file_segments)
        print(f"    {audio_file.name}: {len(speech_segs)} speech regions -> {len(file_segments)} segments")
    
    return all_segments

def main():
    print("=" * 60)
    print("VAD-Based Segmentation")
    print("=" * 60)
    
    all_segments = []
    
    # Process June session
    june_docx = next(RAW_DIR.rglob("*June*.docx"), None)
    if june_docx:
        june_audios = sorted([
            f for f in AUDIO_DIR.glob("*.wav")
            if "june" in f.name.lower() or "24-06-24" in f.name
        ])
        if june_audios:
            segments = process_session(june_audios, june_docx, "june")
            all_segments.extend(segments)
    
    # Process December session
    dec_docx = next(RAW_DIR.rglob("*Tesema*.docx"), None)
    if dec_docx:
        dec_audios = sorted([
            f for f in AUDIO_DIR.glob("*.wav")
            if "december" in f.name.lower() or "12-24" in f.name
        ])
        if dec_audios:
            segments = process_session(dec_audios, dec_docx, "december")
            all_segments.extend(segments)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Total segments: {len(all_segments)}")
    total_duration = sum(s["duration"] for s in all_segments)
    print(f"Total duration: {total_duration/3600:.2f} hours")
    
    # Filter by reasonable duration (1-30 seconds)
    filtered = [s for s in all_segments if 1.0 <= s["duration"] <= 30.0]
    print(f"Filtered (1-30s): {len(filtered)} segments")
    filtered_duration = sum(s["duration"] for s in filtered)
    print(f"Filtered duration: {filtered_duration/3600:.2f} hours")
    
    # Save
    save_path = SEGMENTS_DIR / "vad_segments.json"
    with open(save_path, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()
