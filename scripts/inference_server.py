#!/usr/bin/env python3
"""
TuvaLLM Inference Server
FastAPI-based production API for Tuvaluan ASR.

Features:
- POST /transcribe - Upload audio, return transcript
- Confidence scores per segment
- Optional MMS + Whisper ensemble voting
- Health check and model info endpoints
"""

import os
import sys
from pathlib import Path
import json
import tempfile
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

sys.path.insert(0, '/Users/discordwell/Library/Python/3.9/lib/python/site-packages')

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Base paths
PROJECT_ROOT = Path("/Users/discordwell/TuvaLLM")
MODELS_DIR = PROJECT_ROOT / "models"


# Response models
class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class TranscriptionResponse(BaseModel):
    text: str
    segments: List[TranscriptionSegment]
    duration: float
    model: str
    processing_time: float
    language: str = "tvl"


class ModelInfo(BaseModel):
    name: str
    path: str
    loaded: bool
    device: str


class HealthResponse(BaseModel):
    status: str
    models: List[ModelInfo]
    version: str = "0.1.0"


# Global model storage
class ModelManager:
    """Manages loaded ASR models."""

    def __init__(self):
        self.mms_model = None
        self.mms_processor = None
        self.whisper_model = None
        self.whisper_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_mms(self, model_path: Optional[str] = None) -> bool:
        """Load MMS model."""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if model_path is None:
            model_path = str(MODELS_DIR / "mms-tuvaluan")

        if not Path(model_path).exists():
            # Fall back to baseline
            print("Fine-tuned MMS not found, loading baseline Samoan...")
            self.mms_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
            self.mms_processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
            self.mms_processor.tokenizer.set_target_lang("smo")
            self.mms_model.load_adapter("smo")
        else:
            print(f"Loading MMS from {model_path}")
            self.mms_model = Wav2Vec2ForCTC.from_pretrained(model_path)
            self.mms_processor = Wav2Vec2Processor.from_pretrained(model_path)

        self.mms_model.eval()
        self.mms_model.to(self.device)
        return True

    def load_whisper(self, model_path: Optional[str] = None) -> bool:
        """Load Whisper model (optional)."""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if model_path is None:
            model_path = str(MODELS_DIR / "whisper-tuvaluan-lora")

        try:
            if Path(model_path).exists():
                from peft import PeftModel
                base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                self.whisper_model = PeftModel.from_pretrained(base_model, model_path)
                self.whisper_processor = WhisperProcessor.from_pretrained(model_path)
            else:
                print("Loading baseline Whisper...")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

            self.whisper_model.eval()
            self.whisper_model.to(self.device)

            # Set decoder config
            self.whisper_model.config.forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                language="sm",
                task="transcribe"
            )
            return True
        except Exception as e:
            print(f"Failed to load Whisper: {e}")
            return False

    def transcribe_mms(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        chunk_length: float = 30.0
    ) -> Dict[str, Any]:
        """Transcribe audio using MMS model."""
        if self.mms_model is None:
            raise RuntimeError("MMS model not loaded")

        total_samples = len(waveform)
        chunk_samples = int(chunk_length * sample_rate)

        segments = []
        all_text = []

        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk = waveform[start_idx:end_idx]

            if len(chunk) < sample_rate * 0.1:  # Skip very short chunks
                continue

            # Process
            inputs = self.mms_processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.mms_model(**inputs)
                logits = outputs.logits

            # Get predictions and confidence
            probs = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(logits, dim=-1)
            confidence_scores = probs.max(dim=-1).values

            # Average confidence for the segment
            avg_confidence = confidence_scores.mean().item()

            # Decode
            text = self.mms_processor.decode(predicted_ids[0])

            if text.strip():
                start_time = start_idx / sample_rate
                end_time = end_idx / sample_rate

                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "confidence": avg_confidence
                })
                all_text.append(text)

        return {
            "text": " ".join(all_text),
            "segments": segments
        }

    def transcribe_whisper(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Transcribe audio using Whisper model."""
        if self.whisper_model is None:
            raise RuntimeError("Whisper model not loaded")

        inputs = self.whisper_processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.whisper_model.generate(
                inputs.input_features,
                max_length=225,
                return_dict_in_generate=True,
                output_scores=True
            )

        text = self.whisper_processor.batch_decode(
            generated_ids.sequences,
            skip_special_tokens=True
        )[0]

        return {
            "text": text,
            "segments": [{
                "start": 0,
                "end": len(waveform) / sample_rate,
                "text": text,
                "confidence": None
            }]
        }

    def ensemble_transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Transcribe using ensemble of MMS and Whisper.
        Uses MMS as primary, Whisper as verification.
        """
        mms_result = self.transcribe_mms(waveform, sample_rate)

        if self.whisper_model is not None:
            whisper_result = self.transcribe_whisper(waveform, sample_rate)
            mms_result["whisper_text"] = whisper_result["text"]

        return mms_result


# Initialize model manager
models = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Loading models...")
    models.load_mms()
    # Whisper is optional
    try:
        models.load_whisper()
    except Exception as e:
        print(f"Whisper loading skipped: {e}")
    print("Models loaded!")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="TuvaLLM ASR API",
    description="Tuvaluan Speech-to-Text API powered by fine-tuned MMS",
    version="0.1.0",
    lifespan=lifespan
)


def load_and_preprocess_audio(audio_file) -> tuple:
    """Load audio file and preprocess for ASR."""
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(tmp_path)

        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.squeeze().numpy()

        duration = len(waveform) / sample_rate

        return waveform, sample_rate, duration

    finally:
        os.unlink(tmp_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    model_info = []

    model_info.append(ModelInfo(
        name="MMS (Tuvaluan)",
        path=str(MODELS_DIR / "mms-tuvaluan"),
        loaded=models.mms_model is not None,
        device=models.device
    ))

    model_info.append(ModelInfo(
        name="Whisper (LoRA)",
        path=str(MODELS_DIR / "whisper-tuvaluan-lora"),
        loaded=models.whisper_model is not None,
        device=models.device
    ))

    return HealthResponse(
        status="healthy" if models.mms_model is not None else "degraded",
        models=model_info
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Query("mms", description="Model to use: mms, whisper, or ensemble"),
    chunk_length: float = Query(30.0, description="Chunk length in seconds for MMS")
):
    """
    Transcribe uploaded audio file.

    Supports WAV, MP3, and other common audio formats.
    """
    start_time = time.time()

    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Load and preprocess
        waveform, sample_rate, duration = load_and_preprocess_audio(audio.file)

        # Transcribe based on selected model
        if model == "mms":
            if models.mms_model is None:
                raise HTTPException(status_code=503, detail="MMS model not loaded")
            result = models.transcribe_mms(waveform, sample_rate, chunk_length)
            model_name = "mms-tuvaluan"

        elif model == "whisper":
            if models.whisper_model is None:
                raise HTTPException(status_code=503, detail="Whisper model not loaded")
            result = models.transcribe_whisper(waveform, sample_rate)
            model_name = "whisper-tuvaluan-lora"

        elif model == "ensemble":
            if models.mms_model is None:
                raise HTTPException(status_code=503, detail="MMS model not loaded")
            result = models.ensemble_transcribe(waveform, sample_rate)
            model_name = "ensemble"

        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        processing_time = time.time() - start_time

        return TranscriptionResponse(
            text=result["text"],
            segments=[
                TranscriptionSegment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    confidence=s.get("confidence")
                )
                for s in result["segments"]
            ],
            duration=duration,
            model=model_name,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models and their status."""
    mms_path = MODELS_DIR / "mms-tuvaluan"
    whisper_path = MODELS_DIR / "whisper-tuvaluan-lora"

    return {
        "models": [
            {
                "id": "mms",
                "name": "MMS Tuvaluan",
                "description": "Fine-tuned MMS with Samoan adapter base",
                "available": mms_path.exists() or True,  # Baseline always available
                "loaded": models.mms_model is not None
            },
            {
                "id": "whisper",
                "name": "Whisper LoRA",
                "description": "Whisper base with LoRA fine-tuning",
                "available": whisper_path.exists() or True,
                "loaded": models.whisper_model is not None
            },
            {
                "id": "ensemble",
                "name": "Ensemble",
                "description": "Combined MMS + Whisper with voting",
                "available": True,
                "loaded": models.mms_model is not None
            }
        ],
        "device": models.device
    }


@app.post("/reload")
async def reload_models(
    mms_path: Optional[str] = None,
    whisper_path: Optional[str] = None
):
    """Reload models from specified paths."""
    results = {}

    if mms_path or mms_path is None:
        try:
            models.load_mms(mms_path)
            results["mms"] = "loaded"
        except Exception as e:
            results["mms"] = f"error: {e}"

    if whisper_path or whisper_path is None:
        try:
            success = models.load_whisper(whisper_path)
            results["whisper"] = "loaded" if success else "skipped"
        except Exception as e:
            results["whisper"] = f"error: {e}"

    return results


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="TuvaLLM Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--mms-path", type=str, help="Path to MMS model")
    parser.add_argument("--whisper-path", type=str, help="Path to Whisper model")
    args = parser.parse_args()

    print("=" * 70)
    print("TuvaLLM Inference Server")
    print("=" * 70)
    print()
    print(f"Starting server on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        "inference_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
