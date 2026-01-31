# AGENTS.md

## Data Preparation Pipeline

### 1. Preprocess Audio
Convert all MP3/MP4 files in `data/raw` to 16kHz Mono WAV in `data/processed/audio_16k`.
```bash
python scripts/preprocess_audio.py
```

### 2. Transcribe Audio (MMS-Samoan)
Run ASR (Samoan adapter) on all audio files to generate timestamped rough transcripts.
**Time:** ~2-4 hours (depending on GPU/MPS).
```bash
# This requires ~16GB RAM for long files
python scripts/transcribe_with_mms.py
```
*Output: `data/processed/mms_transcripts/*.json`*

### 3. Align Transcripts
Align the Official Transcripts (DOCX) to the MMS Audio Transcripts to generate accurate training segments.
```bash
python scripts/align_text_to_transcript.py
```
*Output: `data/processed/segments/*.json`*

### 4. Create Dataset
Slice audio into clips and create HuggingFace-compatible JSONL splits.
```bash
python scripts/prepare_dataset.py
```
*Output: `data/processed/dataset/{train,valid,test}.jsonl` & `clips/`*

---

## Notes: PyTorch MPS on macOS 26.x

### Problem
- `torch.backends.mps.is_built()` returns True but `mps_available()` returns False
- RuntimeError: "MPS backend is supported on MacOS 14.0+"
- Cause: PyTorch stable wheels mis-detect macOS 26.x

### Solution: Build PyTorch from Source

```bash
# 1. Clone PyTorch (shallow to save time)
git clone --depth 1 https://github.com/pytorch/pytorch.git ~/Projects/pytorch
cd ~/Projects/pytorch
git submodule update --init --recursive --depth 1

# 2. Install build deps in your venv
source /path/to/your/.venv/bin/activate
pip install cmake ninja

# 3. Build with MPS enabled, CUDA disabled
export TMPDIR=/tmp/pytorch_build_tmp  # Fix clang temp file permission errors
mkdir -p $TMPDIR
USE_MPS=1 USE_METAL=1 USE_CUDA=0 USE_CUDNN=0 BUILD_TEST=0 \
  MACOSX_DEPLOYMENT_TARGET=14.0 MAX_JOBS=8 \
  python setup.py develop
```

Build takes ~30-45 min on M4 Max. Verify with:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
torch.tensor([1.0]).to('mps')  # Should work
```

### References
- https://github.com/pytorch/pytorch/issues/167679
- https://developer.apple.com/metal/pytorch/

### CTC Loss on MPS
CTC loss (`aten::_ctc_loss`) is not implemented on MPS. Run training with:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_mms.py ...
```
This uses CPU fallback for CTC loss while keeping other ops on GPU.

## Training Tips

### Two-Phase Training
The script uses two-phase training (HuggingFace best practice):
1. **Phase 1**: Train lm_head only (adapters frozen) - learns output mapping
2. **Phase 2**: Unfreeze adapters - fine-tunes acoustic features

### Hyperparameters (tested on LatinLLM)
- `--epochs 4`
- `--batch-size 2` with gradient accumulation 8 (effective batch 16)
- `--lr 1e-3` for adapters
- `--lm-head-lr 1e-3` for output head
- `--warmup-epochs 1` for phase 1

### Example Training Command
```bash
source .venv/bin/activate
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_mms.py \
    --epochs 4 \
    --lr 1e-3 \
    --lm-head-lr 1e-3 \
    --output-dir models/mms-tuvaluan
```
