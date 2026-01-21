#!/usr/bin/env python3
"""
TuvaLLM Whisper LoRA Fine-Tuning Script
Fine-tunes Whisper using LoRA for Tuvaluan ASR as secondary model.

Key features:
- LoRA: Only 1% trainable parameters
- Data augmentation: speed perturbation, noise injection
- Can be used for ensemble voting with MMS

Note: Whisper needs more data than MMS for good results,
so this is lower priority than MMS fine-tuning.
"""

import os
import sys
from pathlib import Path
import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Base paths - computed relative to this script's location
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class WhisperLoRAConfig:
    """Training configuration for Whisper LoRA."""
    # Model
    model_name: str = "openai/whisper-base"
    language: str = "sm"  # Samoan as closest available

    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    # Training
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01

    # Augmentation
    use_augmentation: bool = True
    speed_perturbation_range: tuple = (0.9, 1.1)
    noise_snr_range: tuple = (15, 30)  # dB

    # Memory
    fp16: bool = True
    max_audio_length: float = 30.0

    # Output
    output_dir: str = str(MODELS_DIR / "whisper-tuvaluan-lora")
    save_steps: int = 100
    eval_steps: int = 50

    # Data
    train_file: str = str(PROCESSED_DIR / "train.json")
    val_file: str = str(PROCESSED_DIR / "validation.json")

    def __post_init__(self):
        if self.target_modules is None:
            # LoRA targets for Whisper encoder and decoder
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "out_proj",
                "fc1", "fc2"
            ]


class AudioAugmentor:
    """Audio augmentation for training data."""

    def __init__(self, config: WhisperLoRAConfig):
        self.config = config

    def speed_perturbation(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply random speed perturbation."""
        if not self.config.use_augmentation:
            return waveform

        speed_factor = random.uniform(*self.config.speed_perturbation_range)

        # Use torchaudio for resampling (speed change)
        if speed_factor != 1.0:
            effects = [
                ["speed", str(speed_factor)],
                ["rate", str(sample_rate)]
            ]
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.unsqueeze(0) if waveform.dim() == 1 else waveform,
                sample_rate,
                effects
            )
            waveform = waveform.squeeze(0)

        return waveform

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        if not self.config.use_augmentation:
            return waveform

        # Random SNR
        snr_db = random.uniform(*self.config.noise_snr_range)

        # Calculate noise level
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise

    def augment(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply random augmentations."""
        if not self.config.use_augmentation:
            return waveform

        # Apply with some probability
        if random.random() < 0.5:
            waveform = self.speed_perturbation(waveform, sample_rate)
        if random.random() < 0.3:
            waveform = self.add_noise(waveform)

        return waveform


class WhisperDataset(Dataset):
    """Dataset for Whisper fine-tuning."""

    def __init__(
        self,
        data_file: str,
        processor,
        config: WhisperLoRAConfig,
        augment: bool = False
    ):
        self.processor = processor
        self.config = config
        self.augment = augment
        self.augmentor = AudioAugmentor(config) if augment else None

        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load audio
        waveform, sr = torchaudio.load(item["audio"])

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Augment if training
        if self.augment and self.augmentor:
            waveform = self.augmentor.augment(waveform, sr)

        # Truncate
        max_samples = int(self.config.max_audio_length * sr)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        # Process audio
        input_features = self.processor(
            waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.squeeze(0)

        # Process text
        labels = self.processor.tokenizer(
            item["sentence"],
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels
        }


def collate_fn(batch, processor):
    """Collate batch for Whisper training."""
    input_features = torch.stack([item["input_features"] for item in batch])

    # Pad labels
    labels = [item["labels"] for item in batch]
    max_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l

    return {
        "input_features": input_features,
        "labels": padded_labels
    }


def setup_lora_model(config: WhisperLoRAConfig):
    """Setup Whisper model with LoRA."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading Whisper model: {config.model_name}")

    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32
    )
    processor = WhisperProcessor.from_pretrained(config.model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Set language for decoding
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.language,
        task="transcribe"
    )

    return model, processor


def train_whisper_lora(config: WhisperLoRAConfig):
    """Main training function for Whisper LoRA."""
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    from functools import partial
    import evaluate

    print("=" * 70)
    print("TuvaLLM Whisper LoRA Fine-Tuning")
    print("=" * 70)
    print()

    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup model
    model, processor = setup_lora_model(config)

    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = WhisperDataset(
        config.train_file,
        processor,
        config,
        augment=True
    )
    val_dataset = WhisperDataset(
        config.val_file,
        processor,
        config,
        augment=False
    )

    # Metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=10,
        fp16=config.fp16 and device == "cuda",
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=partial(collate_fn, processor=processor),
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {config.num_epochs}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Augmentation: {config.use_augmentation}")
    print()

    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Also save the LoRA config
    model.save_pretrained(config.output_dir)

    print(f"Model saved to: {config.output_dir}")

    # Evaluate
    print("\nFinal evaluation...")
    results = trainer.evaluate()
    print(f"WER: {results.get('eval_wer', 'N/A'):.2%}")

    return trainer, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA for Tuvaluan")
    parser.add_argument("--model", type=str, default="openai/whisper-base",
                       help="Whisper model to fine-tune")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "whisper-tuvaluan-lora"),
                       help="Output directory")
    args = parser.parse_args()

    config = WhisperLoRAConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        use_augmentation=not args.no_augment,
        output_dir=args.output_dir
    )

    # Check data exists
    if not Path(config.train_file).exists():
        print(f"Error: Training data not found at {config.train_file}")
        print("Run prepare_training_data.py first.")
        return

    train_whisper_lora(config)


if __name__ == "__main__":
    main()
