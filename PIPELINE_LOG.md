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

### Problem 4: CTC loss not implemented on MPS

**Symptom:** Training fails with:
```
NotImplementedError: The operator 'aten::_ctc_loss' is not currently implemented for the MPS device.
```

**Root Cause:** PyTorch's CTC loss function is not implemented for Apple's MPS (Metal Performance Shaders).

**Solution Attempts:**
1. `PYTORCH_ENABLE_MPS_FALLBACK=1` - Works but very slow (91s/step)
2. Added `--device cpu` flag - Initially failed because HuggingFace Trainer ignores manual device setting

**Final Fix:** Added `use_cpu=True` to `TrainingArguments` when `--device cpu` is passed. This forces the Trainer to use CPU properly.

```python
use_cpu = (device == "cpu")
training_args = TrainingArguments(
    ...
    use_cpu=use_cpu,  # Force CPU when --device cpu is passed
)
```

**Result:** Training runs on CPU at ~60s/step (slow but stable).

---

## Pipeline Status

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 1 | transcribe_whisper_english.py | DONE | 5h 13min on CPU medium model |
| 2 | merge_anchor_sources.py | DONE | 13,788 MMS-only anchors |
| 3 | sequence_align.py | DONE | 19 sequences, avg 627 anchors each |
| 4 | anchor_align.py | DONE | 1809 segments, 4.44 hours (lowered threshold to 0.55) |
| 5 | verify_alignment.py | DONE | 10 samples exported for QA |
| 6 | prepare_dataset.py | DONE | 1,430 clips, 3.38 hours |
| 7 | finetune_mms.py | RUNNING | CPU mode, 10 epochs, ~7 hours total |

## Final Dataset (MMS + Whisper combined)

**Dataset created:** 3.38 hours of aligned audio
- Train: 1,288 clips
- Val: 71 clips
- Test: 71 clips

## Training Configuration

- **Model**: facebook/mms-1b-all with Samoan adapter as starting point
- **Device**: CPU (forced due to MPS CTC loss issue)
- **Epochs**: 10
- **Batch size**: 4 × 8 = 32 effective batch
- **Two-phase training**:
  - Phase 1 (epochs 1-2): Train lm_head only (adapters frozen)
  - Phase 2 (epochs 3-10): Unfreeze adapters and train all
- **Trainable params**: 2.2M / 965M (0.23%)

## Quality Notes

- 88.2% of segments have confidence < 0.6 (MMS-only)
- 43.3% have normal WPS (1.5-3.5)
- 1284 segments flagged for potential issues

## Next Steps

1. Wait for training to complete (~7 hours on CPU)
2. Evaluate model on test set
3. If quality is poor, may need to increase training data or adjust hyperparameters

## TODO: Recover First 50 Minutes of Day 1

The alignment only matched audio from 50-123 minutes. The first 50 minutes likely contains usable speech beyond just ceremonial music. Possible approaches:
- Lower similarity threshold for early portions
- Manual spot-check to identify where speech starts
- Use Whisper English to catch English loanwords/names as anchors

Revisit after initial training run succeeds.
