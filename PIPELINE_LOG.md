# TuvaLLM Pipeline Log

## 2026-01-29: Improved Alignment Pipeline Implementation

### Problem 1: Whisper large-v3 NaN errors on MPS (Apple Silicon)

**Symptom:** All audio files fail with:
```
Error processing X.wav: Expected parameter logits (Tensor of shape (1, 51866)) of distribution Categorical(logits: torch.Size([1, 51866])) to satisfy the constraint IndependentConstraint(Real(), 1), but found invalid values:
tensor([[nan, nan, nan, ..., nan, nan, nan]], device='mps:0')
```

**Root Cause:** Whisper large-v3 has numerical stability issues with MPS (Metal Performance Shaders) on Apple Silicon. The fp16 computations produce NaN values.

**Solution:** Use fp32 computation by setting `fp16=False` in transcribe call, or fall back to CPU for large-v3, or use a smaller model like medium.

**Fix Applied (Attempt 1):** Updated `transcribe_whisper_english.py` to:
1. Add `--fp16/--no-fp16` flag
2. Default to `fp16=False` on MPS to avoid NaN issues

**Result:** New error - "Cannot convert a MPS Tensor to float64 dtype"

### Problem 2: MPS float64 not supported

**Symptom:** After fixing fp16 issue, new error appears:
```
Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
```

**Root Cause:** The `word_timestamps=True` feature in Whisper uses float64 operations which MPS doesn't support.

**Solution:** Run on CPU instead of MPS for Whisper large-v3. MPS has too many compatibility issues with large models.

**Fix Applied:** Use `--device cpu` for Whisper transcription

---

### Problem 3: O(n^2) LIS algorithm too slow

**Symptom:** sequence_align.py stuck at 9% after 10+ minutes, CPU at 98%

**Root Cause:** O(n^2) dynamic programming LIS algorithm with thousands of candidates per file

**Solution:** Replaced with O(n log n) binary search algorithm using bisect

**Result:** Completed in <1 second instead of hours

---

## Pipeline Status

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 1 | transcribe_whisper_english.py | RUNNING | CPU medium model, still on file 2/19 |
| 2 | merge_anchor_sources.py | DONE | 13,788 MMS-only anchors |
| 3 | sequence_align.py | DONE | 19 sequences, avg 627 anchors each |
| 4 | anchor_align.py | DONE | 1809 segments, 4.44 hours (lowered threshold to 0.55) |
| 5 | verify_alignment.py | DONE | 10 samples exported for QA |
| 6 | prepare_dataset.py | DONE | 1225 clips, 2.78 hours (min confidence 0.5) |

## First Run Results (MMS-only)

**Dataset created:** 2.78 hours of aligned audio
- Train: 1103 clips
- Val: 61 clips
- Test: 61 clips

**Quality Notes:**
- 88.2% of segments have confidence < 0.6 (MMS-only)
- 43.3% have normal WPS (1.5-3.5)
- 1284 segments flagged for potential issues

**Next Steps:**
1. Wait for Whisper English transcription to complete (~hours remaining)
2. Re-run merge_anchor_sources.py with both MMS + Whisper
3. Re-run pipeline - expect higher confidence scores with dual-model anchors
4. Manually verify sample clips before training
