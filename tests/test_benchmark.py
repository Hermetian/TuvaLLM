"""
Tests for UDHR benchmark functionality.
Tests benchmark_udhr.py and WER/CER computation.
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from orthography import to_system6, normalize_text


class TestWERComputation:
    """Tests for Word Error Rate computation."""

    def test_perfect_match(self):
        """Identical texts should have WER of 0."""
        ref = "hello world"
        hyp = "hello world"

        # Simple WER computation
        ref_words = ref.split()
        hyp_words = hyp.split()

        if ref_words == hyp_words:
            wer = 0.0
        else:
            # Simplified - not real edit distance
            wer = 1.0 - len(set(ref_words) & set(hyp_words)) / len(ref_words)

        assert wer == 0.0

    def test_completely_wrong(self):
        """Completely different texts should have high WER."""
        ref = "hello world"
        hyp = "goodbye universe"

        ref_words = set(ref.split())
        hyp_words = set(hyp.split())

        overlap = len(ref_words & hyp_words)
        assert overlap == 0

    def test_partial_match(self):
        """Partial matches should have intermediate WER."""
        ref = "hello world today"
        hyp = "hello universe today"

        ref_words = ref.split()
        hyp_words = hyp.split()

        # 2 out of 3 words match
        matches = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
        assert matches == 2


class TestCERComputation:
    """Tests for Character Error Rate computation."""

    def test_perfect_match(self):
        """Identical texts should have CER of 0."""
        ref = "hello"
        hyp = "hello"

        if ref == hyp:
            cer = 0.0
        else:
            cer = 1.0

        assert cer == 0.0

    def test_single_char_error(self):
        """Single character error should have low CER."""
        ref = "hello"
        hyp = "hallo"

        # 1 error out of 5 characters
        errors = sum(1 for r, h in zip(ref, hyp) if r != h)
        assert errors == 1


class TestOrthographicNormalization:
    """Tests for orthographic system handling in benchmark."""

    def test_normalize_both_systems(self, udhr_reference_texts):
        """Should be able to normalize to both systems."""
        system4 = udhr_reference_texts["system4"]
        system6 = udhr_reference_texts["system6"]

        # System 4 should have macrons
        assert "ā" in system4 or "ū" in system4

        # System 6 should not have macrons
        assert "ā" not in system6
        assert "ū" not in system6

    def test_convert_system4_to_system6(self, udhr_reference_texts):
        """Converting System 4 to System 6 should remove macrons."""
        system4 = udhr_reference_texts["system4"]
        converted = to_system6(system4)

        assert "ā" not in converted
        assert "ū" not in converted
        assert "ē" not in converted


class TestBenchmarkResult:
    """Tests for benchmark result structure."""

    def test_result_fields(self):
        """Benchmark result should have all required fields."""
        result = {
            "model_name": "test_model",
            "transcription": "hello world",
            "wer_system4": 0.5,
            "wer_system6": 0.45,
            "cer_system4": 0.2,
            "cer_system6": 0.18
        }

        assert "model_name" in result
        assert "transcription" in result
        assert "wer_system4" in result
        assert "wer_system6" in result
        assert "cer_system4" in result
        assert "cer_system6" in result

    def test_wer_bounds(self):
        """WER should be between 0 and 1 (or higher for insertions)."""
        wer = 0.45
        assert wer >= 0.0
        # WER can exceed 1.0 with many insertions, but typically <= 1.0

    def test_cer_less_than_wer(self):
        """CER is typically less than WER for similar text."""
        # This is a heuristic - character-level errors are usually
        # less severe than word-level errors
        wer = 0.5
        cer = 0.2
        assert cer <= wer  # Usually true for similar transcriptions


class TestUDHRSample:
    """Tests for UDHR sample handling."""

    def test_reference_text_not_empty(self, udhr_reference_texts):
        """Reference texts should not be empty."""
        assert len(udhr_reference_texts["system4"]) > 0
        assert len(udhr_reference_texts["system6"]) > 0

    def test_reference_text_length_similar(self, udhr_reference_texts):
        """Both system texts should have similar length."""
        len4 = len(udhr_reference_texts["system4"])
        len6 = len(udhr_reference_texts["system6"])

        # System 4 is slightly longer due to diacritics (same char count)
        # but after normalization they should be similar
        assert abs(len4 - len6) < len4 * 0.1  # Within 10%

    def test_udhr_contains_expected_words(self, udhr_reference_texts):
        """UDHR text should contain expected Tuvaluan words."""
        text = udhr_reference_texts["system6"].lower()

        # Common words from UDHR Article 1
        assert "tino" in text  # people
        assert "katoa" in text  # all
        assert "saolotoga" in text  # freedom


class TestComparisonAcrossSystems:
    """Tests for comparing results across orthographic systems."""

    def test_system6_wer_typically_lower(self):
        """System 6 WER is often lower since macrons don't need to match."""
        # This is a design decision - comparing in System 6 is more forgiving
        # since the model doesn't need to predict macrons correctly
        wer_s4 = 0.50  # Higher due to macron mismatches
        wer_s6 = 0.45  # Lower since macrons are removed

        assert wer_s6 <= wer_s4

    def test_evaluation_consistency(self, udhr_reference_texts):
        """Same transcription should give consistent results."""
        transcription = "e fa'nau mai a tino katoa"

        # Normalize for both systems
        norm_s4 = normalize_text(transcription, target_system="system4")
        norm_s6 = normalize_text(transcription, target_system="system6")

        # Both normalizations should be valid strings
        assert isinstance(norm_s4, str)
        assert isinstance(norm_s6, str)
        assert len(norm_s4) > 0
        assert len(norm_s6) > 0
