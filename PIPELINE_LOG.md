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

## Pipeline Status

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 1 | transcribe_whisper_english.py | FIXING | MPS NaN issue |
| 2 | merge_anchor_sources.py | PENDING | |
| 3 | sequence_align.py | PENDING | |
| 4 | anchor_align.py | PENDING | |
| 5 | verify_alignment.py | PENDING | |
| 6 | prepare_dataset.py | PENDING | |
