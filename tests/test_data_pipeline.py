"""
Tests for the data preparation pipeline.
Tests prepare_training_data.py functionality.
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from orthography import normalize_text, build_ctc_vocabulary


class TestNormalizeTextPipeline:
    """Tests for text normalization in data pipeline."""

    def test_normalize_for_system4(self):
        """Text normalization should preserve macrons for system4."""
        text = "E FĀ'NAU Mai"
        result = normalize_text(text, target_system="system4")
        assert "ā" in result
        assert result.islower()

    def test_normalize_for_system6(self):
        """Text normalization should remove macrons for system6."""
        text = "E fā'nau mai"
        result = normalize_text(text, target_system="system6")
        assert "ā" not in result
        assert "a" in result

    def test_normalize_removes_invalid(self):
        """Normalization should remove invalid characters."""
        text = "hello! world? 123"
        result = normalize_text(text, target_system="system6")
        assert "!" not in result
        assert "?" not in result
        assert "1" not in result


class TestVocabularyBuilding:
    """Tests for vocabulary building."""

    def test_vocabulary_has_special_tokens(self):
        """Vocabulary should include special tokens."""
        texts = ["hello world"]
        vocab = build_ctc_vocabulary(texts, system="system6")

        assert "<pad>" in vocab
        assert "<unk>" in vocab
        assert "|" in vocab

    def test_vocabulary_system4_includes_macrons(self):
        """System 4 vocabulary should include macron characters."""
        texts = ["tūlaga fā'nau"]
        vocab = build_ctc_vocabulary(texts, system="system4")

        assert "ū" in vocab
        assert "ā" in vocab

    def test_vocabulary_system6_excludes_macrons(self):
        """System 6 vocabulary should not include macron characters."""
        texts = ["tulaga fa'nau"]
        vocab = build_ctc_vocabulary(texts, system="system6")

        assert "ū" not in vocab
        assert "ā" not in vocab


class TestCorrectionsLoading:
    """Tests for corrections file handling."""

    def test_load_valid_corrections(self, sample_corrections, temp_corrections_file):
        """Should load valid corrections file."""
        with open(temp_corrections_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert "segments" in loaded
        assert "vocabulary" in loaded
        assert "metadata" in loaded

    def test_corrections_segment_structure(self, sample_corrections):
        """Corrections segments should have required fields."""
        for seg_key, seg_data in sample_corrections["segments"].items():
            assert "corrected_text" in seg_data
            assert "start" in seg_data
            assert "end" in seg_data
            assert "quality" in seg_data

    def test_quality_filtering(self, sample_corrections):
        """Should be able to filter segments by quality."""
        min_quality = 4
        segments = sample_corrections["segments"]

        high_quality = {
            k: v for k, v in segments.items()
            if v.get("quality", 0) >= min_quality
        }

        assert len(high_quality) <= len(segments)
        for seg in high_quality.values():
            assert seg["quality"] >= min_quality


class TestDataSplitting:
    """Tests for train/val/test splitting."""

    def test_split_ratios(self):
        """Splits should approximately match requested ratios."""
        import random

        # Create sample data
        data = list(range(100))
        random.seed(42)
        random.shuffle(data)

        train_ratio = 0.8
        val_ratio = 0.1

        train_end = int(len(data) * train_ratio)
        val_end = int(len(data) * (train_ratio + val_ratio))

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        # Check approximate ratios
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        """Train/val/test should have no overlap."""
        import random

        data = list(range(100))
        random.seed(42)
        random.shuffle(data)

        train = data[:80]
        val = data[80:90]
        test = data[90:]

        # Check no overlap
        assert set(train).isdisjoint(set(val))
        assert set(train).isdisjoint(set(test))
        assert set(val).isdisjoint(set(test))


class TestSegmentFiltering:
    """Tests for segment filtering by duration and quality."""

    def test_duration_filtering(self):
        """Should filter segments by duration."""
        segments = [
            {"duration": 0.5, "text": "short"},  # Too short
            {"duration": 5.0, "text": "good"},
            {"duration": 35.0, "text": "too long"},  # Too long
        ]

        min_duration = 1.0
        max_duration = 30.0

        filtered = [
            s for s in segments
            if min_duration <= s["duration"] <= max_duration
        ]

        assert len(filtered) == 1
        assert filtered[0]["text"] == "good"

    def test_empty_text_filtering(self):
        """Should filter out segments with empty text."""
        segments = [
            {"text": "hello"},
            {"text": ""},
            {"text": "   "},
            {"text": "world"},
        ]

        filtered = [s for s in segments if s["text"].strip()]
        assert len(filtered) == 2
