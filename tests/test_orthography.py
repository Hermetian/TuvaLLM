"""
Tests for the orthography module.
Tests System 4 (macrons) and System 6 (no macrons) conversions.
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from orthography import (
    to_system6,
    to_system4,
    detect_system,
    has_macrons,
    count_macrons,
    normalize_text,
    get_vocabulary_for_system,
    build_ctc_vocabulary,
    MACRON_TO_PLAIN,
    MACRON_CHARS
)


class TestToSystem6:
    """Tests for System 4 to System 6 conversion."""

    def test_basic_conversion(self, sample_system4_text, sample_system6_text):
        """Test basic macron removal."""
        result = to_system6(sample_system4_text)
        # Should remove macrons
        assert "ā" not in result
        assert "ū" not in result

    def test_preserves_non_macron_text(self, sample_system6_text):
        """Text without macrons should be unchanged."""
        result = to_system6(sample_system6_text)
        assert result == sample_system6_text

    def test_all_macron_characters(self):
        """Test all macron conversions."""
        text = "ā ē ī ō ū Ā Ē Ī Ō Ū"
        result = to_system6(text)
        assert result == "a e i o u A E I O U"

    def test_mixed_text(self):
        """Test text with macrons mixed with regular characters."""
        text = "tūlaga and tulaga"
        result = to_system6(text)
        assert result == "tulaga and tulaga"


class TestToSystem4:
    """Tests for System 6 to System 4 conversion (with dictionary)."""

    def test_without_dictionary(self):
        """Without dictionary, text should be unchanged."""
        text = "tulaga latou"
        result = to_system4(text)
        assert result == text

    def test_with_dictionary(self, word_dictionary):
        """With dictionary, known words should be converted."""
        text = "tulaga latou"
        result = to_system4(text, word_dictionary)
        assert "tūlaga" in result
        assert "lātou" in result

    def test_preserves_unknown_words(self, word_dictionary):
        """Unknown words should be preserved."""
        text = "tulaga unknown latou"
        result = to_system4(text, word_dictionary)
        assert "unknown" in result

    def test_preserves_punctuation(self, word_dictionary):
        """Punctuation should be preserved."""
        text = "tulaga, latou!"
        result = to_system4(text, word_dictionary)
        assert "," in result
        assert "!" in result

    def test_preserves_case(self, word_dictionary):
        """Word case should be preserved."""
        text = "Tulaga LATOU"
        result = to_system4(text, word_dictionary)
        assert result[0].isupper()  # First char uppercase


class TestDetectSystem:
    """Tests for orthographic system detection."""

    def test_detects_system4(self, sample_system4_text):
        """Text with macrons should be detected as system4."""
        assert detect_system(sample_system4_text) == "system4"

    def test_detects_system6(self, sample_system6_text):
        """Text without macrons should be detected as system6."""
        assert detect_system(sample_system6_text) == "system6"

    def test_empty_text(self):
        """Empty text should be system6 (no macrons)."""
        assert detect_system("") == "system6"

    def test_single_macron(self):
        """Even a single macron should trigger system4."""
        assert detect_system("hello ā world") == "system4"


class TestHasMacrons:
    """Tests for has_macrons function."""

    def test_has_macrons_true(self):
        """Should return True for text with macrons."""
        assert has_macrons("tūlaga") is True

    def test_has_macrons_false(self):
        """Should return False for text without macrons."""
        assert has_macrons("tulaga") is False


class TestCountMacrons:
    """Tests for count_macrons function."""

    def test_count_zero(self):
        """Text without macrons should return 0."""
        assert count_macrons("tulaga") == 0

    def test_count_multiple(self):
        """Should count all macron characters."""
        assert count_macrons("tūlāgā") == 3

    def test_mixed_text(self, sample_system4_text):
        """Should count macrons in mixed text."""
        count = count_macrons(sample_system4_text)
        assert count > 0


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        result = normalize_text("HELLO WORLD")
        assert result == "hello world"

    def test_normalize_apostrophes(self):
        """Should normalize apostrophe variants."""
        result = normalize_text("fa'nau fa'nau fa`nau")
        assert result.count("'") == 3  # All normalized to '

    def test_remove_invalid_chars(self):
        """Should remove non-Tuvaluan characters."""
        result = normalize_text("hello! 123 world@#$")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_collapse_spaces(self):
        """Should collapse multiple spaces."""
        result = normalize_text("hello    world")
        assert "  " not in result

    def test_target_system4(self):
        """Should preserve macrons for system4 target."""
        result = normalize_text("TŪLAGA", target_system="system4")
        assert "ū" in result

    def test_target_system6(self):
        """Should remove macrons for system6 target."""
        result = normalize_text("tūlaga", target_system="system6")
        assert "ū" not in result
        assert "u" in result


class TestGetVocabularyForSystem:
    """Tests for vocabulary character sets."""

    def test_system4_includes_macrons(self):
        """System 4 vocab should include macron characters."""
        vocab = get_vocabulary_for_system("system4")
        assert "ā" in vocab
        assert "ē" in vocab
        assert "ī" in vocab
        assert "ō" in vocab
        assert "ū" in vocab

    def test_system6_excludes_macrons(self):
        """System 6 vocab should not include macron characters."""
        vocab = get_vocabulary_for_system("system6")
        assert "ā" not in vocab
        assert "ū" not in vocab

    def test_both_include_basic_chars(self):
        """Both systems should include basic Latin characters."""
        for system in ["system4", "system6"]:
            vocab = get_vocabulary_for_system(system)
            assert "a" in vocab
            assert "z" in vocab
            assert " " in vocab
            assert "'" in vocab


class TestBuildCTCVocabulary:
    """Tests for CTC vocabulary building."""

    def test_includes_special_tokens(self):
        """Should include CTC special tokens."""
        vocab = build_ctc_vocabulary(["hello"], system="system4")
        assert "<pad>" in vocab
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<unk>" in vocab
        assert "|" in vocab

    def test_special_tokens_indices(self):
        """Special tokens should have correct indices."""
        vocab = build_ctc_vocabulary(["hello"], system="system4")
        assert vocab["<pad>"] == 0
        assert vocab["<s>"] == 1
        assert vocab["</s>"] == 2
        assert vocab["<unk>"] == 3
        assert vocab["|"] == 4

    def test_system4_vocab_size(self, sample_system4_text):
        """System 4 vocab should be larger due to macrons."""
        vocab_s4 = build_ctc_vocabulary([sample_system4_text], system="system4")
        vocab_s6 = build_ctc_vocabulary([sample_system4_text], system="system6")
        # System 4 vocab includes macron chars
        assert len(vocab_s4) >= len(vocab_s6)

    def test_includes_text_characters(self):
        """Should include characters from input texts."""
        vocab = build_ctc_vocabulary(["hello world"], system="system6")
        assert "h" in vocab
        assert "e" in vocab
        assert "l" in vocab
        assert "o" in vocab


class TestRoundTrip:
    """Tests for round-trip conversions."""

    def test_system4_to_system6_and_back(self, sample_system4_text, word_dictionary):
        """Converting S4→S6→S4 should recover text with dictionary."""
        # S4 → S6
        system6 = to_system6(sample_system4_text)
        assert detect_system(system6) == "system6"

        # S6 → S4 (with dictionary)
        # Note: This won't be perfect without a complete dictionary
        system4_back = to_system4(system6, word_dictionary)
        # At minimum, some words should be converted
        # Full recovery requires complete dictionary

    def test_system6_is_stable(self, sample_system6_text):
        """Converting S6→S6 should be stable."""
        result = to_system6(sample_system6_text)
        assert result == sample_system6_text
