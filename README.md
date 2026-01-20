# TuvaLLM

Tuvaluan Speech-to-Text using fine-tuned MMS (Meta's Massively Multilingual Speech).

## Overview

TuvaLLM builds Tuvaluan ASR by fine-tuning MMS from its Samoan adapter. Samoan and Tuvaluan share ~80% mutual intelligibility, making transfer learning highly effective.

**Approach**: Fine-tune MMS adapters (not train from scratch)
- Only ~2.5M parameters to train (vs 1B base)
- Initialize from Samoan adapter for maximum transfer
- CTC-based architecture proven for low-resource languages

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Transcript Correction (Critical First Step)

```bash
# Interactive correction interface
python scripts/transcript_corrector.py --mode interactive

# Batch scoring mode
python scripts/transcript_corrector.py --mode batch

# View statistics
python scripts/transcript_corrector.py --mode stats
```

### 2. Prepare Training Data

```bash
python scripts/prepare_training_data.py --min-quality 3
```

Output:
```
data/processed/
├── segments/          # Individual WAV files (16kHz mono)
├── train.json         # HuggingFace datasets format
├── validation.json
├── test.json
└── vocab.json         # Tuvaluan character vocabulary
```

### 3. Fine-tune MMS (Primary Model)

```bash
python scripts/finetune_mms.py --epochs 10 --batch-size 8 --lr 1e-3
```

### 4. Fine-tune Whisper LoRA (Optional Secondary)

```bash
python scripts/finetune_whisper_lora.py --epochs 10 --lora-r 8
```

### 5. Evaluate Models

```bash
# Evaluate fine-tuned model
python scripts/evaluate_model.py --model mms

# Compare with baseline
python scripts/evaluate_model.py --model both --baseline --linguistic
```

### 6. Iterative Improvement

```bash
# View progress
python scripts/iterative_improve.py --action progress

# Run improvement iteration
python scripts/iterative_improve.py --action iterate

# Prepare for retraining
python scripts/iterative_improve.py --action prepare-retrain
```

### 7. Inference Server

```bash
python scripts/inference_server.py --port 8000
```

API endpoints:
- `POST /transcribe` - Upload audio, return transcript
- `GET /health` - Health check
- `GET /models` - List available models

## Project Structure

```
TuvaLLM/
├── data/
│   ├── raw/                    # Source audio files
│   ├── processed/              # Training data
│   │   ├── corrections/        # Manual corrections
│   │   ├── segments/           # Segmented audio
│   │   └── iterations/         # Improvement iterations
│   └── transcripts/            # Initial transcripts
├── models/                     # Trained models
├── scripts/
│   ├── transcript_corrector.py # Interactive correction
│   ├── prepare_training_data.py# Data preparation
│   ├── finetune_mms.py         # MMS fine-tuning
│   ├── finetune_whisper_lora.py# Whisper LoRA
│   ├── evaluate_model.py       # Evaluation
│   ├── iterative_improve.py    # Bootstrapping
│   └── inference_server.py     # Production API
└── requirements.txt
```

## Expected Results

| Metric | Before (Samoan zero-shot) | After Fine-tuning |
|--------|---------------------------|-------------------|
| WER    | 60-80%                    | 25-35%            |
| Usable segments | ~30%             | ~70-80%           |

## Data Sources

- **GRN Tuvaluan recordings** - Primary audio source
- **Gerd Koch recordings** - TIB Hanover (potential expansion)

## References

- [MMS: Scaling Speech Technology to 1000+ Languages](https://arxiv.org/abs/2305.13516)
- [Tuvaluan language (Wikipedia)](https://en.wikipedia.org/wiki/Tuvaluan_language)
- [Hiroshi Tachibana's Tuvaluan Dictionary](https://tuvalu.aa-ken.jp/en)

## License

MIT
