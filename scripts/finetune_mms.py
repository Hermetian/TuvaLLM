#!/usr/bin/env python3
"""
TuvaLLM MMS Fine-Tuning Script
Fine-tunes MMS adapter layers for Tuvaluan ASR.

Key approach:
1. Load MMS with Samoan adapter as starting point (~80% mutual intelligibility)
2. Create Tuvaluan vocabulary from corrected transcripts
3. Reinitialize lm_head AFTER loading adapter (critical fix)
4. Two-phase training: train lm_head first, then unfreeze adapters
5. Freeze base model, train only adapter (~2.5M params)

Ported from LatinLLM with all stability fixes for MPS.
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure model directory exists
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_id: str = "facebook/mms-1b-all"
    source_language: str = "smo"  # Samoan as starting point
    target_language: str = "tvl"  # Tuvaluan (custom)

    # Training (based on HuggingFace MMS adapter best practices)
    num_epochs: int = 15
    batch_size: int = 2  # Smaller batch for MPS memory
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    learning_rate: float = 3e-4  # Standard LR for fine-tuning
    lm_head_lr: float = 1e-3  # Higher LR for lm_head (init from scratch)
    warmup_ratio: float = 0.1
    weight_decay: float = 0.005
    lm_head_weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    freeze_lm_head: bool = False
    # Two-phase: Train lm_head FIRST (adapters frozen), then unfreeze adapters
    warmup_epochs_lm_head_only: int = 2

    # Device
    device: str = None  # None = auto-detect, or force "cpu"/"mps"/"cuda"

    # Memory optimization
    fp16: bool = True
    gradient_checkpointing: bool = False  # Disabled - can cause issues on MPS
    max_audio_length: float = 30.0  # seconds - increased to avoid audio/text mismatch
    # How to handle samples longer than max_audio_length:
    # "truncate_both" = truncate audio AND text proportionally (default)
    # "skip" = skip samples longer than max_audio_length
    long_audio_strategy: str = "truncate_both"

    # Output
    output_dir: str = str(MODELS_DIR / "mms-tuvaluan")
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    max_steps: int = -1  # Set >0 to cap total optimizer steps (debugging)

    # Data
    train_file: str = str(PROCESSED_DIR / "dataset" / "train.jsonl")
    val_file: str = str(PROCESSED_DIR / "dataset" / "valid.jsonl")
    vocab_file: str = str(PROCESSED_DIR / "vocab.json")


class TuvaluanASRDataset(Dataset):
    """Dataset for Tuvaluan ASR training."""

    def __init__(
        self,
        data_file: str,
        processor,
        max_audio_length: float = 30.0,
        sample_rate: int = 16000,
        long_audio_strategy: str = "truncate_both"
    ):
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.long_audio_strategy = long_audio_strategy

        # Load data (JSONL format)
        raw_data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        
        # Filter samples if using "skip" strategy
        if long_audio_strategy == "skip":
            self.data = []
            skipped = 0
            for item in raw_data:
                duration = item.get("duration", 0)
                if duration <= max_audio_length:
                    self.data.append(item)
                else:
                    skipped += 1
            if skipped > 0:
                print(f"Skipped {skipped} samples longer than {max_audio_length}s")
        else:
            self.data = raw_data
        
        # Load vocab for normalization
        with open(PROCESSED_DIR / "vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            # Create whitelist set (space represented as |)
            self.allowed_chars = set(self.vocab.keys())
            if "|" in self.allowed_chars:
                self.allowed_chars.add(" ") # mapping space to | later
            # Remove special tokens from allowed chars for text filtering
            self.allowed_chars = {c for c in self.allowed_chars if c not in ["<s>", "</s>", "<pad>", "<unk>"]}

        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def normalize_text(self, text):
        """Strict whitelist normalization using vocabulary."""
        text = text.lower()
        
        # Normalize curly quotes/apostrophes to straight apostrophe (glottal stop)
        text = text.replace("'", "'").replace("'", "'").replace("ʻ", "'").replace("`", "'")
        
        # Filter chars
        clean_text = []
        for char in text:
            if char in self.allowed_chars:
                clean_text.append(char)
            elif char == " ":
                clean_text.append(" ")
        
        text = "".join(clean_text)
        
        # Collapse spaces
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load audio using soundfile (avoids TorchCodec dependency)
        import soundfile as sf
        audio_path = item["audio"]
        waveform, sr = sf.read(audio_path, dtype="float32")

        # Resample if needed
        if sr != self.sample_rate:
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform_tensor).squeeze().numpy()

        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[1] == 2:
            waveform = waveform.mean(axis=1)

        # Get text and normalize
        text = item.get("text", item.get("sentence", ""))
        text = self.normalize_text(text)

        # Handle audio longer than max_audio_length
        max_samples = int(self.max_audio_length * self.sample_rate)
        original_length = len(waveform)
        
        if original_length > max_samples:
            if self.long_audio_strategy == "truncate_both":
                # CRITICAL FIX: Truncate BOTH audio AND text proportionally
                # This ensures CTC can still align the truncated audio to truncated text
                truncation_ratio = max_samples / original_length
                waveform = waveform[:max_samples]
                
                # Truncate text proportionally (by character count)
                # Add small buffer to avoid cutting mid-word when possible
                target_text_len = int(len(text) * truncation_ratio)
                if target_text_len < len(text):
                    # Try to cut at a word boundary
                    truncated = text[:target_text_len]
                    last_space = truncated.rfind(' ')
                    if last_space > target_text_len * 0.7:  # Only if we don't lose too much
                        text = truncated[:last_space].strip()
                    else:
                        text = truncated.strip()
            else:
                # Just truncate audio (old behavior - causes mismatch)
                waveform = waveform[:max_samples]

        return {
            "input_values": waveform,  # Key required by Trainer's LengthGroupedSampler
            "text": text
        }


_DEBUG_BATCH_COUNT = [0]

def collate_fn(batch, processor, tuvaluan_vocab):
    """Collate function for DataLoader.

    CRITICAL: Uses tuvaluan_vocab for label encoding, NOT processor.tokenizer.
    The processor.tokenizer is Samoan but our lm_head is Tuvaluan.
    Using Samoan tokenizer causes label IDs > vocab size, which breaks CTC loss.
    """
    # Process audio
    audio_arrays = [item["input_values"] for item in batch]
    texts = [item["text"] for item in batch]

    # Track actual audio lengths BEFORE padding (critical for CTC loss)
    audio_lengths = [len(arr) for arr in audio_arrays]

    # Check for NaN/Inf in input audio
    for i, arr in enumerate(audio_arrays):
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"Warning: NaN/Inf in audio sample {i}, replacing with zeros")
            audio_arrays[i] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Process inputs without attention mask (workaround for MPS boolean indexing issue)
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=False
    )

    # Store original audio lengths for CTC input_lengths calculation
    inputs["audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)

    # Ensure float32 dtype
    inputs["input_values"] = inputs["input_values"].float()

    # CRITICAL FIX: Encode labels using TUVALUAN vocab, not Samoan tokenizer
    pad_id = tuvaluan_vocab.get("<pad>", 0)
    unk_id = tuvaluan_vocab.get("<unk>", 3)
    tuvaluan_vocab_size = len(tuvaluan_vocab)

    # Convert each text to Tuvaluan token IDs
    # CRITICAL: Encode spaces as pipe '|' (token 4), not literal space
    # The Wav2Vec2 decoder expects pipe as word delimiter and converts it to space
    word_delimiter_id = tuvaluan_vocab.get('|', 4)

    label_ids_list = []
    for text in texts:
        # Convert text to lowercase and encode character by character
        text_lower = text.lower()
        ids = []
        for char in text_lower:
            if char == ' ':
                # Use pipe for word boundary (CTC convention)
                ids.append(word_delimiter_id)
            else:
                ids.append(tuvaluan_vocab.get(char, unk_id))
        label_ids_list.append(ids)

    # Pad to same length
    max_len = max(len(ids) for ids in label_ids_list)
    padded_labels = []
    for ids in label_ids_list:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_labels.append(padded)

    inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)

    # Debug: print label info for first few batches
    _DEBUG_BATCH_COUNT[0] += 1

    if _DEBUG_BATCH_COUNT[0] <= 3:
        print(f"\n[DEBUG batch {_DEBUG_BATCH_COUNT[0]}]")
        print(f"  Input shape: {inputs['input_values'].shape}")
        print(f"  Labels shape: {inputs['labels'].shape}")
        print(f"  Labels min/max: {inputs['labels'].min().item()}/{inputs['labels'].max().item()}")
        print(f"  Tuvaluan vocab size: {tuvaluan_vocab_size}")
        print(f"  Sample text: {texts[0][:50]}")
        print(f"  Sample labels: {inputs['labels'][0][:20].tolist()}")
        # Verify all labels are in valid range
        if inputs['labels'].max().item() >= tuvaluan_vocab_size:
            print(f"  ERROR: Label {inputs['labels'].max().item()} >= vocab size {tuvaluan_vocab_size}!")
        else:
            print(f"  ✓ All labels in valid range [0, {tuvaluan_vocab_size-1}]")

    # Replace padding with -100 for loss computation
    inputs["labels"] = inputs["labels"].masked_fill(
        inputs["labels"] == pad_id, -100
    )

    return inputs


def create_tuvaluan_vocabulary(vocab_file: str) -> Dict[str, int]:
    """Load or create Tuvaluan vocabulary."""
    if Path(vocab_file).exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both formats (direct vocab or nested)
            if "vocabulary" in data:
                return data["vocabulary"]
            return data

    # Default Tuvaluan character set (includes macrons for long vowels and apostrophe for glottal stop)
    chars = list("abcdefghijklmnopqrstuvwxyz'āēīōū")
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
    }
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def setup_model_and_processor(config: TrainingConfig):
    """Initialize model and processor for fine-tuning."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
    from transformers import Wav2Vec2FeatureExtractor

    print(f"Loading MMS model: {config.model_id}")
    print(f"Source adapter: {config.source_language}")

    # Load Tuvaluan vocabulary FIRST (needed for model config)
    vocab = create_tuvaluan_vocabulary(config.vocab_file)
    print(f"Tuvaluan vocab ({len(vocab)} tokens): {vocab}")

    # Create tokenizer with Tuvaluan vocabulary
    vocab_dir = Path(config.output_dir) / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = vocab_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )

    # Load model with CRITICAL CTC fixes (per HuggingFace best practices)
    use_fp16 = config.fp16 and torch.cuda.is_available()
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        # CRITICAL CTC fixes
        ctc_loss_reduction="mean",      # Use mean, not sum
        ctc_zero_infinity=True,         # Zero out infinite losses (prevents NaN)
        # Disable all dropout for small datasets
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        # New vocabulary size
        vocab_size=len(vocab),
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,   # Allow lm_head resize
    )
    print(f"Model loaded with ctc_zero_infinity=True, ctc_loss_reduction='mean', all dropouts=0.0")

    # Disable masking - causes MPS crashes with boolean indexing
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0

    # Load Samoan adapter as starting point for acoustic features
    model.load_adapter(config.source_language)
    model.config.add_adapter = True  # CRITICAL: ensure adapters are active in saved model
    print(f"Loaded Samoan adapter as base")

    # CRITICAL FIX: load_adapter() overwrites lm_head with Samoan vocab
    # We need to replace it with a fresh lm_head for Tuvaluan vocab
    import torch.nn as nn
    hidden_size = model.config.hidden_size  # 1280

    # Create new lm_head for Tuvaluan vocab
    model.lm_head = nn.Linear(hidden_size, len(vocab), bias=True)

    # Initialize with Xavier (as recommended for CTC output layers)
    nn.init.xavier_uniform_(model.lm_head.weight, gain=1.0)
    model.lm_head.bias.data.zero_()

    # Set moderate negative blank bias to prevent CTC collapse
    # Blank token (index 0) should be discouraged early in training
    # -2.0 is more moderate than -5.0 to avoid over-penalizing blank
    model.lm_head.bias.data[0] = -2.0

    # Update config to match new vocab size
    model.config.vocab_size = len(vocab)

    print(f"Reinitialized lm_head for Tuvaluan vocab:")
    print(f"  Shape: {model.lm_head.weight.shape}")
    print(f"  Blank (pad) bias: {model.lm_head.bias.data[0].item():.2f}")

    # Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # Freeze base model, only train adapter and new head
    model.freeze_base_model()

    # IMPORTANT: Unfreeze adapter layers (freeze_base_model freezes them too)
    adapter_count = 0
    for name, param in model.named_parameters():
        if 'adapter' in name.lower():
            param.requires_grad = True
            adapter_count += 1
    print(f"Unfroze {adapter_count} adapter layer parameters")

    # Optionally freeze lm_head to only train adapter (for stability)
    if config.freeze_lm_head:
        print("FREEZING lm_head - only training adapter layers")
        for param in model.lm_head.parameters():
            param.requires_grad = False

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, processor, vocab


def compute_metrics(pred, processor):
    """Compute WER and CER metrics."""
    from evaluate import load

    wer_metric = load("wer")
    cer_metric = load("cer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 with pad token
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Filter empty strings
    pred_str = [p if p else " " for p in pred_str]
    label_str = [l if l else " " for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


from transformers import Trainer

class CTCTrainer(Trainer):
    """Custom Trainer that computes CTC loss manually to avoid label validation issues on MPS."""

    _last_valid_loss = 5.0  # Track last valid loss for recovery
    _nan_reported = False
    _nan_loss_reported = False

    def _log_nan(self, kind: str):
        if kind == "logits" and self._nan_reported:
            return
        if kind == "loss" and self._nan_loss_reported:
            return
        if kind == "logits":
            self._nan_reported = True
        if kind == "loss":
            self._nan_loss_reported = True
        step = getattr(getattr(self, "state", None), "global_step", "n/a")
        epoch = getattr(getattr(self, "state", None), "epoch", "n/a")
        print(f"WARNING: NaN/Inf in {kind} at step {step}, epoch {epoch}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch.nn.functional as F

        labels = inputs.pop("labels", None)
        audio_lengths = inputs.pop("audio_lengths", None)
        outputs = model(**inputs)

        if labels is not None:
            logits = outputs.logits  # (batch, time, vocab)

            # Check for NaN in logits before proceeding
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                self._log_nan("logits")
                loss = torch.tensor(self._last_valid_loss, device=logits.device, requires_grad=True)
                return (loss, outputs) if return_outputs else loss

            # Clamp logits to prevent extreme values causing NaN in log_softmax
            logits = logits.clamp(min=-50, max=50)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (time, batch, vocab)

            # Check for NaN in log_probs
            if torch.isnan(log_probs).any():
                self._log_nan("logits")
                loss = torch.tensor(self._last_valid_loss, device=logits.device, requires_grad=True)
                return (loss, outputs) if return_outputs else loss

            # CRITICAL FIX: Compute actual input lengths from audio lengths
            # The feature extractor downsamples audio; use model's method to get output lengths
            if audio_lengths is not None:
                # Move to same device as logits
                audio_lengths = audio_lengths.to(logits.device)
                # Use the model's built-in method to compute feature extractor output lengths
                input_lengths = model._get_feat_extract_output_lengths(audio_lengths)
                # Clamp to max sequence length (in case of any mismatch)
                input_lengths = input_lengths.clamp(max=logits.size(1))
            else:
                # Fallback: assume all sequences are full length (old behavior)
                input_lengths = torch.full(
                    (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
                )

            # Get target lengths (count non -100 values)
            target_lengths = (labels != -100).sum(dim=1)

            # Replace -100 with 0 for CTC loss (it ignores based on target_lengths)
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0

            # Compute CTC loss
            loss = F.ctc_loss(
                log_probs,
                labels_for_ctc,
                input_lengths,
                target_lengths,
                blank=0,  # pad token is blank
                reduction="mean",
                zero_infinity=True  # Handle inf gracefully
            )

            # Replace NaN/Inf loss with last valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                self._log_nan("loss")
                print(f"WARNING: Loss is {loss.item()}, using last valid loss {self._last_valid_loss}")
                loss = torch.tensor(self._last_valid_loss, device=logits.device, requires_grad=True)
            elif loss.item() > 0.01:  # Only update if loss is meaningful
                CTCTrainer._last_valid_loss = loss.item()
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


from transformers import TrainerCallback

class UnfreezeAdaptersCallback(TrainerCallback):
    """Callback to unfreeze adapter layers after warmup epochs.

    Two-phase training (per HuggingFace best practices):
    - Phase 1: Train lm_head only (adapters frozen)
    - Phase 2: Unfreeze adapters and train everything
    """

    def __init__(self, warmup_epochs: int, adapter_params: list):
        self.warmup_epochs = warmup_epochs
        self.adapter_params = adapter_params
        self.unfrozen = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.unfrozen and state.epoch >= self.warmup_epochs:
            print(f"\n>>> UNFREEZING ADAPTERS at epoch {state.epoch:.1f} <<<")
            for param in self.adapter_params:
                param.requires_grad = True
            self.unfrozen = True


def train(config: TrainingConfig):
    """Main training function with two-phase training."""
    from transformers import TrainingArguments
    from functools import partial

    print("=" * 70)
    print("TuvaLLM MMS Fine-Tuning (Two-Phase)")
    print("=" * 70)
    print()

    # Check for GPU (or use forced device)
    if config.device:
        device = config.device
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup model
    model, processor, vocab = setup_model_and_processor(config)

    # Move to device (ensure float32 for MPS/CPU)
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.float()  # Ensure float32 before moving to MPS
        model = model.to("mps")
    elif device == "cpu":
        model = model.float()  # Ensure float32
        model = model.to("cpu")
        print("FORCED CPU MODE: CTC loss will run on CPU (slower but stable)")

    # Create datasets
    print("\nLoading datasets...")
    print(f"Audio length handling: max={config.max_audio_length}s, strategy={config.long_audio_strategy}")
    train_dataset = TuvaluanASRDataset(
        config.train_file,
        processor,
        max_audio_length=config.max_audio_length,
        long_audio_strategy=config.long_audio_strategy
    )

    val_dataset = TuvaluanASRDataset(
        config.val_file,
        processor,
        max_audio_length=config.max_audio_length,
        long_audio_strategy=config.long_audio_strategy
    )

    # Training arguments
    # CRITICAL: Force CPU if device is set to cpu (HuggingFace Trainer ignores our device var otherwise)
    use_cpu = (device == "cpu")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        group_by_length=False,  # Disabled - custom dataset format
        remove_unused_columns=False,  # Keep 'text' column for our collate_fn
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="no",  # Disable eval - causes NaN on MPS
        save_strategy="epoch",  # Only save at end of epoch
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,  # Base LR (overridden by custom optimizer)
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        fp16=False,  # Disable fp16 entirely - causes issues on MPS
        gradient_checkpointing=False,  # Disable - causes issues on MPS
        save_total_limit=2,
        load_best_model_at_end=False,  # Disabled - uses eval which causes NaN
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=0,  # Disable multiprocessing - MPS issues
        dataloader_pin_memory=False,  # MPS doesn't support pinned memory
        use_cpu=use_cpu,  # Force CPU when --device cpu is passed
    )

    # Create custom optimizer with parameter groups (different LRs for lm_head vs adapters)
    # CRITICAL FIX: New lm_head needs much higher LR than pre-trained adapter layers
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    lm_head_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lm_head" in name:
                lm_head_params.append(param)
            else:
                adapter_params.append(param)

    # TWO-PHASE TRAINING (HuggingFace best practice): Train lm_head FIRST
    # Phase 1: Freeze adapters, train only lm_head (learns Tuvaluan output mapping)
    # Phase 2: Unfreeze adapters to fine-tune acoustic features
    if config.warmup_epochs_lm_head_only > 0:
        print(f"\nTWO-PHASE TRAINING: Freezing ADAPTERS for first {config.warmup_epochs_lm_head_only} epoch(s)")
        print("  Phase 1: Train lm_head only (adapters frozen)")
        print("  Phase 2: Unfreeze adapters (train everything)")
        for param in adapter_params:
            param.requires_grad = False

    optimizer_grouped_parameters = [
        {
            "params": lm_head_params,
            "lr": config.lm_head_lr,
            "weight_decay": config.lm_head_weight_decay,
            "name": "lm_head"
        },
        {
            "params": adapter_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "adapter"
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    # Calculate total training steps for scheduler
    num_training_steps = (
        len(train_dataset) // config.batch_size // config.gradient_accumulation_steps
    ) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\nOptimizer setup:")
    print(f"  lm_head params: {len(lm_head_params)}, lr={config.lm_head_lr}, wd={config.lm_head_weight_decay}")
    print(f"  adapter params: {len(adapter_params)}, lr={config.learning_rate}, wd={config.weight_decay}")
    print(f"  Total steps: {num_training_steps}, warmup: {num_warmup_steps}")

    # Setup callbacks for two-phase training
    callbacks = []
    if config.warmup_epochs_lm_head_only > 0:
        callbacks.append(UnfreezeAdaptersCallback(
            warmup_epochs=config.warmup_epochs_lm_head_only,
            adapter_params=adapter_params
        ))

    # Create Trainer (using custom CTCTrainer to avoid label validation issues on MPS)
    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=partial(collate_fn, processor=processor, tuvaluan_vocab=vocab),
        compute_metrics=partial(compute_metrics, processor=processor),
        optimizers=(optimizer, scheduler),  # Pass custom optimizer
        callbacks=callbacks,
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rates: adapter={config.learning_rate}, lm_head={config.lm_head_lr}")
    if config.warmup_epochs_lm_head_only > 0:
        print(f"Two-phase training: lm_head-only for {config.warmup_epochs_lm_head_only} epoch(s), then full training")
    print()

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Save training config
    config_path = Path(config.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nModel saved to: {config.output_dir}")

    # Skip HuggingFace evaluation (causes NaN on MPS)
    # Run benchmark.py instead for proper evaluation
    print("\nTraining complete. Run evaluate_model.py to evaluate the model.")

    return trainer, {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune MMS for Tuvaluan")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for adapter layers")
    parser.add_argument("--lm-head-lr", type=float, default=1e-3,
                       help="Learning rate for lm_head")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "mms-tuvaluan"),
                       help="Output directory")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--source-lang", type=str, default="smo",
                       help="Source language adapter (default: smo/Samoan)")
    parser.add_argument("--freeze-lm-head", action="store_true",
                       help="Freeze lm_head and only train adapter layers")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                       help="Epochs to train lm_head only before unfreezing adapters")
    parser.add_argument("--max-steps", type=int, default=-1,
                       help="Cap total optimizer steps (debugging; default: no cap)")
    parser.add_argument("--max-audio-length", type=float, default=30.0,
                       help="Maximum audio length in seconds (default: 30)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/mps/cuda). Default: auto-detect")
    parser.add_argument("--long-audio-strategy", type=str, default="truncate_both",
                       choices=["truncate_both", "skip"],
                       help="How to handle samples longer than max-audio-length: "
                            "truncate_both (truncate audio AND text proportionally) or "
                            "skip (skip long samples)")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lm_head_lr=args.lm_head_lr,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        fp16=not args.no_fp16,
        source_language=args.source_lang,
        freeze_lm_head=args.freeze_lm_head,
        warmup_epochs_lm_head_only=args.warmup_epochs,
        max_steps=args.max_steps,
        max_audio_length=args.max_audio_length,
        long_audio_strategy=args.long_audio_strategy,
        device=args.device
    )

    # Check data exists
    if not Path(config.train_file).exists():
        print(f"Error: Training data not found at {config.train_file}")
        print("Run prepare_training_data.py first.")
        return

    train(config)


if __name__ == "__main__":
    main()
