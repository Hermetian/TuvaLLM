#!/usr/bin/env python3
"""
Tests for the Tuvaluan Grammar module.
Based on Kennedy, D.G. (1945) "Handbook on the Language of the Tuvalu Islands"
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tuvaluan_grammar import (
    validate_text,
    ValidationLevel,
    ValidationResult,
    get_possessive_class,
    PossessiveClass,
    conjugate_verb,
    analyze_verb_phrase,
    analyze_pronouns,
    analyze_text,
    apply_alliteration,
    number_for_people,
    KENNEDY_VOWELS,
    KENNEDY_CONSONANTS_CHARS,
    TENSE_PARTICLES,
    NEGATIVE_PARTICLES,
)


class TestCharacterValidation:
    """Test Kennedy's 16-letter alphabet validation."""

    def test_valid_tuvaluan_text(self):
        """Text with only valid Tuvaluan characters should pass."""
        text = "E fa'nau mai a tino katoa"
        result = validate_text(text, ValidationLevel.STRICT)
        # Should have no character errors (word-ending warnings are separate)
        char_errors = [i for i in result.issues if i.category == "character"]
        assert len(char_errors) == 0

    def test_invalid_character_strict(self):
        """Non-Tuvaluan characters should be flagged in strict mode."""
        text = "computer"  # 'c' and 'r' are not in Tuvaluan alphabet
        result = validate_text(text, ValidationLevel.STRICT)
        char_issues = [i for i in result.issues if i.category == "character"]
        assert len(char_issues) >= 2  # 'c' and 'r'

    def test_invalid_character_moderate(self):
        """Non-Tuvaluan characters should warn in moderate mode."""
        text = "radio"  # 'r' and 'd' are not traditional
        result = validate_text(text, ValidationLevel.MODERATE)
        char_issues = [i for i in result.issues if i.category == "character"]
        assert all(i.severity == "warning" for i in char_issues)

    def test_macrons_valid(self):
        """Macron vowels should be valid."""
        text = "tūlaga fā'nau lātou"
        result = validate_text(text, ValidationLevel.STRICT)
        char_errors = [i for i in result.issues
                      if i.category == "character" and i.severity == "error"]
        assert len(char_errors) == 0

    def test_apostrophe_variants(self):
        """Various apostrophe styles should be accepted."""
        texts = ["fa'nau", "fa'nau", "fa`nau"]
        for text in texts:
            result = validate_text(text, ValidationLevel.STRICT)
            # Apostrophe should not cause character errors
            char_errors = [i for i in result.issues if i.category == "character"]
            assert len(char_errors) == 0


class TestPhonotactics:
    """Test Tuvaluan phonotactic constraints."""

    def test_word_ends_in_vowel(self):
        """Kennedy §2: Every word ends in a vowel."""
        # Valid - ends in vowels
        result = validate_text("tangata fafine tama", ValidationLevel.STRICT)
        phonotactic_issues = [i for i in result.issues if i.category == "phonotactic"]
        word_ending_issues = [i for i in phonotactic_issues if "end in a vowel" in i.message]
        assert len(word_ending_issues) == 0

    def test_word_not_ending_in_vowel(self):
        """Words not ending in vowels should be flagged."""
        result = validate_text("tangat", ValidationLevel.STRICT)  # Missing final 'a'
        phonotactic_issues = [i for i in result.issues if "end in a vowel" in i.message]
        assert len(phonotactic_issues) == 1


class TestPossessiveClass:
    """Test A/O possessive distinction (Kennedy §22-23)."""

    def test_a_class_food(self):
        """Food words should be A-class."""
        pclass, _ = get_possessive_class("meakai")
        assert pclass == PossessiveClass.A_CLASS

        pclass, _ = get_possessive_class("ika")
        assert pclass == PossessiveClass.A_CLASS

    def test_a_class_children(self):
        """Child words should be A-class."""
        pclass, _ = get_possessive_class("tama")
        assert pclass == PossessiveClass.A_CLASS

        pclass, _ = get_possessive_class("tamaliki")
        assert pclass == PossessiveClass.A_CLASS

    def test_o_class_body_parts(self):
        """Body part words should be O-class."""
        pclass, _ = get_possessive_class("vae")  # leg
        assert pclass == PossessiveClass.O_CLASS

        pclass, _ = get_possessive_class("lima")  # hand/arm
        assert pclass == PossessiveClass.O_CLASS

    def test_o_class_house(self):
        """House/land should be O-class."""
        pclass, _ = get_possessive_class("fale")  # house
        assert pclass == PossessiveClass.O_CLASS

        pclass, _ = get_possessive_class("fenua")  # homeland
        assert pclass == PossessiveClass.O_CLASS

    def test_o_class_relations(self):
        """Parents/relations (except children) should be O-class."""
        pclass, _ = get_possessive_class("tamana")  # father
        assert pclass == PossessiveClass.O_CLASS

        pclass, _ = get_possessive_class("tupuna")  # ancestor
        assert pclass == PossessiveClass.O_CLASS

    def test_unknown_word(self):
        """Unknown words should return UNKNOWN class."""
        pclass, _ = get_possessive_class("xyzabc")
        assert pclass == PossessiveClass.UNKNOWN


class TestVerbConjugation:
    """Test verb tense particles (Kennedy §28-30)."""

    def test_present_tense(self):
        """Present tense uses 'e' particle."""
        result = conjugate_verb("tuku", "present")
        assert result == "e tuku"

    def test_future_tense(self):
        """Future tense uses 'ka' particle."""
        result = conjugate_verb("tuku", "future")
        assert result == "ka tuku"

    def test_past_tense(self):
        """Past tense uses 'ne' particle."""
        result = conjugate_verb("tuku", "past")
        assert result == "ne tuku"

    def test_negative_present(self):
        """Negative present uses 'se' particle."""
        result = conjugate_verb("tuku", "present", negative=True)
        assert result == "se tuku"

    def test_negative_past(self):
        """Negative past uses 'seki' particle."""
        result = conjugate_verb("tuku", "past", negative=True)
        assert result == "seki tuku"

    def test_negative_imperative(self):
        """Negative imperative uses 'sa' particle."""
        result = conjugate_verb("tuku", "imperative", negative=True)
        assert result == "sa tuku"

    def test_strips_infinitive_marker(self):
        """Should strip 'o ' infinitive marker."""
        result = conjugate_verb("o tuku", "present")
        assert result == "e tuku"

    def test_conditional(self):
        """Conditional uses 'ma' particle."""
        result = conjugate_verb("tuku", "future_conditional")
        assert result == "ma tuku"


class TestVerbPhraseAnalysis:
    """Test identification of verb phrases in text."""

    def test_finds_present_particle(self):
        """Should find 'e' present particle."""
        results = analyze_verb_phrase("E tuku ne au")
        particles = [r["particle"] for r in results]
        assert "e" in particles

    def test_finds_past_particle(self):
        """Should find 'ne' past particle."""
        results = analyze_verb_phrase("Ne tuku ne au")
        particles = [r["particle"] for r in results]
        assert "ne" in particles

    def test_finds_multiple_particles(self):
        """Should find multiple particles in complex sentence."""
        results = analyze_verb_phrase("Ka tuku ne ia ke fano")
        particles = [r["particle"] for r in results]
        assert "ka" in particles
        assert "ke" in particles


class TestPronounAnalysis:
    """Test pronoun identification."""

    def test_finds_first_person_singular(self):
        """Should find 'au' (I/me)."""
        results = analyze_pronouns("E tuku ne au")
        pronouns = [r["pronoun"] for r in results]
        assert "au" in pronouns

    def test_finds_third_person(self):
        """Should find 'ia' (he/she)."""
        results = analyze_pronouns("Ko ia tenei")
        pronouns = [r["pronoun"] for r in results]
        assert "ia" in pronouns

    def test_finds_plural_inclusive(self):
        """Should find 'tatou' (we all inclusive)."""
        results = analyze_pronouns("Ta olo tatou")
        pronouns = [r["pronoun"] for r in results]
        assert "tatou" in pronouns

    def test_finds_dual(self):
        """Should find dual pronouns."""
        results = analyze_pronouns("E a taua te vaka")
        pronouns = [r["pronoun"] for r in results]
        assert "taua" in pronouns


class TestAlliteration:
    """Test Kennedy's alliteration rules (§37)."""

    def test_alliteration_after_s(self):
        """After 's' words, toku becomes soku."""
        result = apply_alliteration("seai", "toku")
        assert result == "soku"

    def test_alliteration_tau_to_sau(self):
        """After 's' words, tau becomes sau."""
        result = apply_alliteration("salasala", "tau")
        assert result == "sau"

    def test_no_alliteration_without_s(self):
        """Without 's', pronouns stay unchanged."""
        result = apply_alliteration("fale", "toku")
        assert result == "toku"


class TestNumbers:
    """Test number system (Kennedy §17)."""

    def test_number_for_people(self):
        """Counting people uses 'toko' prefix."""
        assert number_for_people(1) == "tokotasi"
        assert number_for_people(2) == "tokolua"
        assert number_for_people(3) == "tokotolu"


class TestComprehensiveAnalysis:
    """Test full text analysis."""

    def test_analyze_simple_sentence(self):
        """Should analyze a simple Tuvaluan sentence."""
        text = "E tuku ne au te meakai"
        analysis = analyze_text(text)

        assert analysis.text == text
        assert len(analysis.verb_phrases) > 0
        assert len(analysis.pronouns) > 0

    def test_analyze_generates_summary(self):
        """Should generate readable summary."""
        text = "Ne fano ne ia ki te fale"
        analysis = analyze_text(text)
        summary = analysis.summary()

        assert "Analysis of" in summary
        assert text in summary


class TestKennedyAlphabet:
    """Verify Kennedy's alphabet constants are correct."""

    def test_vowel_count(self):
        """Should have exactly 5 vowels."""
        assert len(KENNEDY_VOWELS) == 5
        assert KENNEDY_VOWELS == {"a", "e", "i", "o", "u"}

    def test_consonant_chars_count(self):
        """Should have 10 single consonant chars (ng is digraph)."""
        assert len(KENNEDY_CONSONANTS_CHARS) == 10
        assert "n" in KENNEDY_CONSONANTS_CHARS
        assert "g" not in KENNEDY_CONSONANTS_CHARS  # g only in 'ng' digraph

    def test_tense_particles_complete(self):
        """Should have all Kennedy's tense particles."""
        expected = {"present", "future", "past", "future_conditional",
                   "consequential", "perfect_conditional", "precautionary"}
        assert set(TENSE_PARTICLES.keys()) == expected

    def test_negative_particles_complete(self):
        """Should have all Kennedy's negative particles."""
        expected = {"present", "past", "imperative"}
        assert set(NEGATIVE_PARTICLES.keys()) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
