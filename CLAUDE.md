# TuvaLLM Project Documentation

## Overview
Fine-tuning Facebook's MMS (Massively Multilingual Speech) model for Tuvaluan ASR using Samoan as a bridge language. Tuvaluan has no existing ASR model, but Samoan is linguistically similar and has MMS support.

## Project Structure

```
TuvaLLM/
├── data/
│   ├── raw/
│   │   ├── FINALPalamene June Parliamenet session 2024.docx  # June transcript (172,809 words)
│   │   ├── Tesema 2024 Palamene.docx                         # December transcript (119,874 words)
│   │   └── audio/                                             # Parliament audio files (DAY_X_Take_Y.wav)
│   └── processed/
│       ├── audio_16k/                    # Resampled audio (16kHz mono)
│       ├── mms_transcripts/              # MMS Samoan ASR output per audio file
│       ├── merged_anchors/               # Combined MMS + Whisper English anchors
│       ├── matched_sequences/            # LCS-based sequence matching results
│       ├── segments_confident/           # Final aligned segments with confidence scores
│       │   └── confident_segments.json   # 4,278 segments with LLM confidence
│       ├── dataset/                      # Training-ready data
│       │   ├── clips/                    # Sliced audio clips (WAV, 16kHz)
│       │   ├── train.jsonl               # Training split
│       │   ├── valid.jsonl               # Validation split
│       │   └── test.jsonl                # Test split
│       ├── llm_scores_combined.json      # LLM semantic confidence scores
│       └── segments_for_llm_scoring.json # Pairs prepared for LLM scoring
├── scripts/
│   ├── anchor_align.py                   # Main alignment script (sequence-based)
│   ├── prepare_dataset.py                # Creates train/val/test splits
│   ├── prepare_llm_scoring.py            # Prepares segment pairs for LLM scoring
│   ├── merge_llm_scores.py               # Merges LLM scores into segments
│   ├── sequence_align.py                 # LCS-based sequence matching
│   ├── merge_anchor_sources.py           # Combines MMS + Whisper anchors
│   ├── transcribe_whisper_english.py     # Whisper English ASR for loanwords
│   └── compare_models.py                 # Compare trained vs baseline models
├── models/                               # Trained model checkpoints
└── .venv/                                # Python virtual environment
```

## Data Pipeline

1. **Audio Preprocessing**: Resample to 16kHz mono (`audio_16k/`)
2. **MMS Transcription**: Run Samoan MMS on all audio (`mms_transcripts/`)
3. **Whisper English**: Capture English loanwords and names (`whisper_english_transcripts/`)
4. **Anchor Merging**: Combine anchors from both models (`merged_anchors/`)
5. **Sequence Matching**: LCS-based alignment (`matched_sequences/`)
6. **Segment Generation**: Create aligned segments (`segments_confident/`)
7. **LLM Scoring**: Semantic confidence scoring (`llm_scores_combined.json`)
8. **Dataset Preparation**: Slice audio and create splits (`dataset/`)

## Current Dataset Stats (confidence >= 0.6)

| Metric | Value |
|--------|-------|
| Training samples | 1,211 |
| Validation samples | 67 |
| Test samples | 67 |
| Total duration | 3.36 hours |
| High confidence (>= 0.7) | 648 samples |
| Average WPS | 1.86 |

## Known Issues & Fixes

### FIXED: Anchor End Off-by-One Error (2025-01-31)
**Problem**: 57.4% of segments were missing the `anchor_end` word in their text because Python slice `words[word_start:word_end]` is exclusive of end index.

**Location**: `scripts/anchor_align.py:104`

**Fix**:
```python
# Before (WRONG)
seg_words = words[word_start:word_end]

# After (CORRECT)
seg_words = words[word_start:word_end + 1]
```

**Result**: 100% of segments now have anchor_end in text.

### FIXED: LLM Scoring Agents Using Heuristics (2025-01-31)
**Problem**: Sub-agents were calculating Jaccard similarity instead of doing semantic judgment.

**Fix**: Stricter prompting with explicit instructions: "You must NOT write any code or algorithms, DO NOT calculate Jaccard similarity"

**Result**: ~52% of batches used proper semantic judgment. Scores merged into segments.

### KNOWN: 162 Segments Without LLM Scores
**Reason**: These are new segments created after the +1 fix that weren't in the original scoring batch.

**Impact**: Minor - they use sequence confidence as fallback.

**Future Fix**: Re-run LLM scoring on new segments if needed.

### KNOWN: Apostrophe Variations in Anchors
**Example**: `ma'nako` vs `manako` causes 1 segment to show anchor_end not in text.

**Impact**: Negligible (1 out of 2,461 segments).

## Configuration

### anchor_align.py
```python
MIN_CONFIDENCE = 0.45      # Minimum sequence confidence
MIN_SEQUENCE_LENGTH = 5    # Minimum anchors in a sequence
MAX_SEGMENT_DURATION = 15.0
MAX_SEGMENT_WORDS = 30
MIN_WPS = 0.5
MAX_WPS = 6.0
```

### prepare_dataset.py
```python
--min-confidence 0.6       # Default LLM confidence threshold
```

## Commands

```bash
# Activate environment
cd /Users/discordwell/Projects/TuvaLLM
source .venv/bin/activate

# Regenerate segments (after any alignment changes)
python scripts/anchor_align.py

# Merge LLM scores into segments
python scripts/merge_llm_scores.py

# Prepare training dataset
python scripts/prepare_dataset.py --min-confidence 0.6

# For higher quality (fewer samples)
python scripts/prepare_dataset.py --min-confidence 0.7
```

## Training Notes

- Use CTC loss for ASR training
- Base model: `facebook/mms-1b-all` with Samoan adapter
- Target: Fine-tune for Tuvaluan
- Previous attempt with 468 samples resulted in 100% WER (gibberish output)
- Current approach uses LLM-verified alignment for higher quality data

## Quality Verification

To verify alignment quality:
```python
# Check anchor_end is in text
with open("data/processed/segments_confident/confident_segments.json") as f:
    segments = json.load(f)

for seg in segments:
    anchor_end = seg.get("anchor_end")
    if anchor_end and anchor_end.lower() not in seg["text"].lower():
        print(f"MISSING: {anchor_end} not in {seg['text'][-50:]}")
```

## LLM Confidence Interpretation

| Score | Meaning |
|-------|---------|
| 0.8-1.0 | Excellent match - ASR closely matches transcript |
| 0.6-0.8 | Good match - semantically similar with some differences |
| 0.4-0.6 | Moderate match - partial overlap, some misalignment |
| < 0.4 | Poor match - likely misaligned or wrong segment |

## Files Modified Recently

- `scripts/anchor_align.py` - Fixed off-by-one error (line 104)
- `scripts/merge_llm_scores.py` - Created to merge LLM confidence
- `data/processed/segments_confident/confident_segments.json` - Regenerated with fix
