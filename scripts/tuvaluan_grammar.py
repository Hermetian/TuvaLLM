#!/usr/bin/env python3
"""
TuvaLLM Tuvaluan Grammar Module
Based on Kennedy, D.G. (1945) "Handbook on the Language of the Tuvalu (Ellice) Islands"

This module encodes grammatical knowledge from the 1945 Kennedy grammar, treating rules
as guidance rather than strict enforcement. Transcriptions may vary by dialect, era,
and individual style - Kennedy documents the Vaitupu dialect circa 1945.

Key features:
- Character set validation (Kennedy's 16-letter alphabet)
- A/O possessive classification
- Verb tense particle system
- Pronoun system (singular/dual/plural, inclusive/exclusive)
- Phonotactic constraints (words end in vowels, etc.)
"""

from typing import Dict, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import re


# =============================================================================
# PHONOLOGY - Kennedy's Alphabet
# =============================================================================

# Kennedy (1945) §1: "The alphabet consists of sixteen letters"
# 5 vowels + 11 consonants (including the digraph ng)
KENNEDY_VOWELS = set("aeiou")
KENNEDY_VOWELS_WITH_MACRONS = set("aeiouāēīōū")
KENNEDY_CONSONANTS = {"f", "h", "k", "l", "m", "n", "ng", "p", "s", "t", "v"}

# Single-character consonants (for character-level validation)
KENNEDY_CONSONANTS_CHARS = set("fhklmnpstv")

# Characters NOT in traditional Tuvaluan (may appear in loanwords/modern usage)
NON_TUVALUAN_CHARS = set("bcddgjqrwxyz")

# Valid character set for strict Kennedy validation
KENNEDY_VALID_CHARS = KENNEDY_VOWELS_WITH_MACRONS | KENNEDY_CONSONANTS_CHARS | {" ", "'", "g"}  # g for ng

# Extended set allowing some modern variation
MODERN_VALID_CHARS = KENNEDY_VALID_CHARS | set("bcdrw")  # common in loanwords


class ValidationLevel(Enum):
    """How strictly to apply Kennedy's rules."""
    STRICT = "strict"      # Only Kennedy's 16 letters
    MODERATE = "moderate"  # Allow common loanword characters
    PERMISSIVE = "permissive"  # Flag issues but accept anything


@dataclass
class ValidationIssue:
    """A potential issue found during validation."""
    severity: str  # "error", "warning", "info"
    category: str  # "character", "phonotactic", "grammar"
    message: str
    position: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating Tuvaluan text."""
    text: str
    is_valid: bool
    issues: List[ValidationIssue]
    kennedy_conformant: bool  # Strictly follows Kennedy 1945

    def __str__(self) -> str:
        if not self.issues:
            return f"Valid Tuvaluan text (Kennedy conformant: {self.kennedy_conformant})"
        issues_str = "\n".join(f"  [{i.severity}] {i.message}" for i in self.issues)
        return f"Validation issues:\n{issues_str}"


def validate_text(text: str, level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """
    Validate Tuvaluan text against Kennedy's grammar.

    Args:
        text: Text to validate
        level: How strictly to apply rules

    Returns:
        ValidationResult with issues found
    """
    issues = []
    text_lower = text.lower()

    # Check for invalid characters
    for i, char in enumerate(text_lower):
        if char in {" ", "'", "'", "'", "`"}:
            continue
        if char in "āēīōū":  # macrons are valid
            continue
        if char not in KENNEDY_VALID_CHARS:
            if char in NON_TUVALUAN_CHARS:
                if level == ValidationLevel.STRICT:
                    issues.append(ValidationIssue(
                        severity="error",
                        category="character",
                        message=f"Character '{char}' at position {i} is not in Kennedy's Tuvaluan alphabet",
                        position=i,
                        suggestion=f"'{char}' may be a loanword or transcription variant"
                    ))
                elif level == ValidationLevel.MODERATE:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="character",
                        message=f"Character '{char}' at position {i} is not traditional Tuvaluan",
                        position=i,
                        suggestion="May be a loanword or modern borrowing"
                    ))

    # Check words end in vowels (Kennedy §2: "Every word ends in a vowel")
    words = re.findall(r"[a-zA-ZāēīōūĀĒĪŌŪ']+", text_lower)
    for word in words:
        # Remove trailing apostrophes for this check
        clean_word = word.rstrip("'")
        if clean_word and clean_word[-1] not in KENNEDY_VOWELS_WITH_MACRONS:
            issues.append(ValidationIssue(
                severity="warning" if level != ValidationLevel.STRICT else "error",
                category="phonotactic",
                message=f"Word '{word}' does not end in a vowel (Kennedy §2)",
                suggestion="Tuvaluan words traditionally end in vowels"
            ))

    # Check for consonant clusters (Tuvaluan typically has CV syllable structure)
    consonant_cluster = re.search(r'[fhklmnpstv]{2,}', text_lower.replace("ng", "N"))
    if consonant_cluster and "ng" not in text_lower[consonant_cluster.start():consonant_cluster.end()]:
        issues.append(ValidationIssue(
            severity="info",
            category="phonotactic",
            message=f"Possible consonant cluster found: '{consonant_cluster.group()}'",
            suggestion="Tuvaluan typically has CV syllable structure; may indicate elision"
        ))

    kennedy_conformant = not any(i.severity == "error" for i in issues)
    is_valid = level == ValidationLevel.PERMISSIVE or kennedy_conformant

    return ValidationResult(
        text=text,
        is_valid=is_valid,
        issues=issues,
        kennedy_conformant=kennedy_conformant
    )


# =============================================================================
# A/O POSSESSIVE DISTINCTION - Kennedy §22-23
# =============================================================================

class PossessiveClass(Enum):
    """The A/O distinction in Tuvaluan possessive constructions."""
    A_CLASS = "a"  # Transitive/acquired possession
    O_CLASS = "o"  # Inherent/inalienable possession
    UNKNOWN = "?"


# Kennedy §22-23: Classification of nouns by possessive form
# "A" form: food, children, states of being, things acquired through action
# "O" form: body parts, feelings, houses, land, canoes, clothing, parents/relations (except children)

A_CLASS_CATEGORIES = {
    "food": ["meakai", "ika", "popo", "pi", "vai"],
    "children": ["tama", "tamaliki", "tamafafine", "tamatangata"],
    "actions_states": [
        "ngaluenga", "faifainga", "tongafiti", "olonga", "omamainga",
        "teleatunga", "fangongo", "lakau", "fatu", "toki"
    ],
}

O_CLASS_CATEGORIES = {
    "body_parts": [
        "vae", "lima", "mata", "ulu", "nifo", "tua", "manava",
        "isu", "taliga", "loto"
    ],
    "feelings": ["manatu", "alofa", "ita", "fiafia", "mataku"],
    "possessions": [
        "fale", "vaka", "paopao", "manafa", "fenua", "umanga",
        "pulou", "taka", "moenga"
    ],
    "relations_except_children": [
        "tamana", "tina", "matua", "tupuna", "avanga", "taina", "tuangane"
    ],
}

# Flat lookups
A_CLASS_WORDS: Set[str] = set()
for words in A_CLASS_CATEGORIES.values():
    A_CLASS_WORDS.update(words)

O_CLASS_WORDS: Set[str] = set()
for words in O_CLASS_CATEGORIES.values():
    O_CLASS_WORDS.update(words)


def get_possessive_class(word: str) -> Tuple[PossessiveClass, str]:
    """
    Determine the possessive class (A or O) for a noun.

    Kennedy §22: "The 'a' form is used when the state or degree of possession
    is the result of some transitive action on the part of the possessor;
    and the 'o' form when the state or degree of possession is the normal
    state of being, through inheritance or otherwise."

    Args:
        word: Tuvaluan noun

    Returns:
        Tuple of (PossessiveClass, explanation)
    """
    word_lower = word.lower().strip()

    if word_lower in A_CLASS_WORDS:
        for category, words in A_CLASS_CATEGORIES.items():
            if word_lower in words:
                return PossessiveClass.A_CLASS, f"A-class ({category})"
        return PossessiveClass.A_CLASS, "A-class"

    if word_lower in O_CLASS_WORDS:
        for category, words in O_CLASS_CATEGORIES.items():
            if word_lower in words:
                return PossessiveClass.O_CLASS, f"O-class ({category})"
        return PossessiveClass.O_CLASS, "O-class"

    return PossessiveClass.UNKNOWN, "Unknown - not in Kennedy's examples"


def get_possessive_pronoun(person: str, number: str, possessed_class: PossessiveClass,
                           possessed_count: str = "singular") -> str:
    """
    Get the correct possessive pronoun form.

    Kennedy §20-21: Pronouns vary by A/O class of possessed noun.

    Args:
        person: "1", "2", "3" (first, second, third person)
        number: "singular", "dual_inclusive", "dual_exclusive", "plural_inclusive", "plural_exclusive"
        possessed_class: A_CLASS or O_CLASS
        possessed_count: "singular" or "plural" (of the thing possessed)

    Returns:
        The possessive pronoun
    """
    # Simplified - full tables in Kennedy §20-21
    vowel = "a" if possessed_class == PossessiveClass.A_CLASS else "o"

    PRONOUNS = {
        ("1", "singular", "singular"): f"t{vowel}ku",
        ("1", "singular", "plural"): f"{vowel}ku",
        ("2", "singular", "singular"): f"t{vowel}u",
        ("2", "singular", "plural"): f"{vowel}u",
        ("3", "singular", "singular"): f"t{vowel}na",
        ("3", "singular", "plural"): f"{vowel}na",
        # Dual and plural forms follow similar patterns
    }

    key = (person, number, possessed_count)
    return PRONOUNS.get(key, f"t{vowel}?? (see Kennedy §20-21)")


# =============================================================================
# VERB SYSTEM - Kennedy §28-30
# =============================================================================

class TenseParticle(NamedTuple):
    """A Tuvaluan tense/mood particle."""
    particle: str
    name: str
    description: str
    example: str


# Kennedy §28: Tense particles
TENSE_PARTICLES = {
    "present": TenseParticle("e", "Present Indicative",
        "Current action or state", "E tuku ne au - I give"),
    "future": TenseParticle("ka", "Future Indicative",
        "Future action", "Ka tuku ne au - I shall give"),
    "past": TenseParticle("ne", "Past Indicative",
        "Completed action", "Ne tuku ne au - I gave"),
    "future_conditional": TenseParticle("ma", "Future Contingent",
        "If someone should do", "Ma tuku ne au - If I should give"),
    "consequential": TenseParticle("ke", "Future Consequential",
        "That someone may do / let someone do", "Ke tuku ne au - That I may give"),
    "perfect_conditional": TenseParticle("moi", "Perfect Conditional",
        "If someone had done", "Moi tuku ne au - If I had given"),
    "precautionary": TenseParticle("mana", "Future Precautionary",
        "Lest someone should do", "Mana tuku ne au - Lest I should give"),
}

# Kennedy §30: Negative particles
NEGATIVE_PARTICLES = {
    "present": TenseParticle("se", "Negative Present",
        "Does not / is not", "Se tuku ne au - I do not give"),
    "past": TenseParticle("seki", "Negative Past",
        "Did not", "Seki tuku ne au - I did not give"),
    "imperative": TenseParticle("sa", "Negative Imperative",
        "Don't!", "Sa tuku - Don't give"),
}


def conjugate_verb(verb_stem: str, tense: str = "present", negative: bool = False) -> str:
    """
    Show verb conjugation with tense particle.

    Kennedy §28-30: Tuvaluan verbs have no inflections; tense is indicated by particles.

    Args:
        verb_stem: The verb (usually starts with 'o' for infinitive, e.g., "o tuku")
        tense: One of the tense keys
        negative: Whether to use negative form

    Returns:
        Conjugated form with particle
    """
    # Strip infinitive marker if present
    if verb_stem.startswith("o "):
        verb_stem = verb_stem[2:]

    particles = NEGATIVE_PARTICLES if negative else TENSE_PARTICLES

    if tense not in particles:
        available = ", ".join(particles.keys())
        return f"Unknown tense '{tense}'. Available: {available}"

    particle = particles[tense]
    return f"{particle.particle} {verb_stem}"


def analyze_verb_phrase(text: str) -> List[Dict]:
    """
    Identify verb phrases and their tense/mood in text.

    Returns list of found verb particles with their grammatical information.
    """
    results = []
    text_lower = text.lower()

    # Check for particles at word boundaries
    all_particles = {**TENSE_PARTICLES, **{f"neg_{k}": v for k, v in NEGATIVE_PARTICLES.items()}}

    for name, particle_info in all_particles.items():
        pattern = rf'\b{particle_info.particle}\b'
        for match in re.finditer(pattern, text_lower):
            results.append({
                "particle": particle_info.particle,
                "position": match.start(),
                "tense": name,
                "description": particle_info.description,
            })

    return sorted(results, key=lambda x: x["position"])


# =============================================================================
# PRONOUN SYSTEM - Kennedy §20
# =============================================================================

PRONOUNS = {
    # Singular
    "au": {"person": 1, "number": "singular", "gloss": "I, me"},
    "koe": {"person": 2, "number": "singular", "gloss": "you (singular)"},
    "ia": {"person": 3, "number": "singular", "gloss": "he, she, him, her"},

    # Dual inclusive (you and I)
    "taua": {"person": 1, "number": "dual_inclusive", "gloss": "we two (you and I)"},
    # Dual exclusive (he/she and I)
    "maua": {"person": 1, "number": "dual_exclusive", "gloss": "we two (he/she and I)"},
    # Dual 2nd person
    "koulua": {"person": 2, "number": "dual", "gloss": "you two"},
    # Dual 3rd person
    "laua": {"person": 3, "number": "dual", "gloss": "they two"},

    # Plural inclusive (you all and I)
    "tatou": {"person": 1, "number": "plural_inclusive", "gloss": "we all (you and I)"},
    # Plural exclusive (they and I, not you)
    "matou": {"person": 1, "number": "plural_exclusive", "gloss": "we all (they and I)"},
    # Plural 2nd person
    "koutou": {"person": 2, "number": "plural", "gloss": "you all"},
    # Plural 3rd person
    "latou": {"person": 3, "number": "plural", "gloss": "they all"},
}


def analyze_pronouns(text: str) -> List[Dict]:
    """Find and analyze pronouns in text."""
    results = []
    text_lower = text.lower()

    for pronoun, info in PRONOUNS.items():
        pattern = rf'\b{pronoun}\b'
        for match in re.finditer(pattern, text_lower):
            results.append({
                "pronoun": pronoun,
                "position": match.start(),
                **info
            })

    return sorted(results, key=lambda x: x["position"])


# =============================================================================
# ARTICLES - Kennedy §7
# =============================================================================

ARTICLES = {
    "te": {"type": "definite", "number": "singular", "gloss": "the"},
    "se": {"type": "indefinite", "number": "singular", "gloss": "a, an"},
    "a": {"type": "personal", "number": "singular", "gloss": "before names/pronouns"},
    "ni": {"type": "partitive", "number": "plural", "gloss": "some"},
    "ne": {"type": "partitive", "number": "plural", "gloss": "some (variant)"},
}


# =============================================================================
# NUMBERS - Kennedy §17
# =============================================================================

CARDINAL_NUMBERS = {
    1: "tasi", 2: "lua", 3: "tolu", 4: "fa", 5: "lima",
    6: "ono", 7: "fitu", 8: "valu", 9: "iva", 10: "angafulu",
    11: "angafulu ma tasi", 20: "lua ngafulu", 100: "se lau", 1000: "afe"
}

# Kennedy §17: "toko must precede the numeral when applied to persons"
def number_for_people(n: int) -> str:
    """Get the number form for counting people."""
    if n in CARDINAL_NUMBERS:
        return f"toko{CARDINAL_NUMBERS[n]}"
    return f"toko{n}"


# =============================================================================
# SIBLING TERMINOLOGY - Kennedy §41
# =============================================================================

# Kennedy §41: "Brother to brother, or sister to sister is taina.
# Brother to sister, or sister to brother is tuangane."

SIBLING_TERMS = {
    "taina": "same-sex sibling (brother's brother or sister's sister)",
    "tuangane": "opposite-sex sibling (brother's sister or sister's brother)",
}


# =============================================================================
# ALLITERATION RULES - Kennedy §37
# =============================================================================

# Kennedy §37: After words containing 's', possessive pronouns change initial 't' to 's'
ALLITERATION_CHANGES = {
    "toku": "soku",
    "taku": "saku",
    "tou": "sou",
    "tau": "sau",
    "tona": "sona",
    "tena": "sena",
}


def apply_alliteration(preceding_word: str, pronoun: str) -> str:
    """
    Apply Kennedy's alliteration rule for pronouns after 's' words.

    Kennedy §37: For euphony, possessive pronouns change t→s after words with 's'.
    Example: "Seai soku fale" (I have no house) - not "Seai toku fale"
    """
    if "s" in preceding_word.lower() and pronoun.lower() in ALLITERATION_CHANGES:
        return ALLITERATION_CHANGES[pronoun.lower()]
    return pronoun


# =============================================================================
# COMMON EXPRESSIONS - Kennedy §39
# =============================================================================

EXCLAMATIONS = {
    "e": "hey, there",
    "tapa": "astonishment",
    "tape": "surprise and satisfaction",
    "aue": "satisfaction, awe",
    "ai aue": "disappointment, sadness",
    "mea": "surprise and resentment",
    "ko foki": "surprise and delight",
    "tafanga loa": "excellent!",
    "kiloke": "behold! I say!",
    "te": "nonsense",
}


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

@dataclass
class GrammarAnalysis:
    """Complete grammatical analysis of Tuvaluan text."""
    text: str
    validation: ValidationResult
    verb_phrases: List[Dict]
    pronouns: List[Dict]
    possessive_analysis: List[Dict]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Analysis of: \"{self.text}\"", ""]

        # Validation
        if self.validation.issues:
            lines.append("Validation Notes:")
            for issue in self.validation.issues:
                lines.append(f"  [{issue.severity}] {issue.message}")
            lines.append("")

        # Verb phrases
        if self.verb_phrases:
            lines.append("Verb Phrases Found:")
            for vp in self.verb_phrases:
                lines.append(f"  '{vp['particle']}' - {vp['description']}")
            lines.append("")

        # Pronouns
        if self.pronouns:
            lines.append("Pronouns Found:")
            for p in self.pronouns:
                lines.append(f"  '{p['pronoun']}' - {p['gloss']}")
            lines.append("")

        return "\n".join(lines)


def analyze_text(text: str, validation_level: ValidationLevel = ValidationLevel.MODERATE) -> GrammarAnalysis:
    """
    Perform comprehensive grammatical analysis of Tuvaluan text.

    Args:
        text: Tuvaluan text to analyze
        validation_level: How strictly to validate

    Returns:
        GrammarAnalysis with all findings
    """
    return GrammarAnalysis(
        text=text,
        validation=validate_text(text, validation_level),
        verb_phrases=analyze_verb_phrase(text),
        pronouns=analyze_pronouns(text),
        possessive_analysis=[],  # Would need more context to analyze
    )


# =============================================================================
# REFERENCE OUTPUT
# =============================================================================

def print_grammar_reference():
    """Print a reference guide based on Kennedy 1945."""
    print("""
================================================================================
TUVALUAN GRAMMAR REFERENCE
Based on Kennedy, D.G. (1945) "Handbook on the Language of the Tuvalu Islands"
================================================================================

ALPHABET (§1)
  Vowels: a, e, i, o, u (each has long and short forms; macrons mark long vowels)
  Consonants: f, h, k, l, m, n, ng, p, s, t, v
  Note: Every word ends in a vowel (§2)

ARTICLES (§7)
  te - the (singular)          ni, ne - some (plural)
  se - a, an                   a - before names/pronouns

A/O POSSESSIVE DISTINCTION (§22-23)
  A-class (acquired): food, children, actions, things you make/acquire
    Example: taku meakai (my food), taku tama (my child)
  O-class (inherent): body parts, feelings, house, land, parents, relations
    Example: toku vae (my leg), toku fale (my house), toku tamana (my father)

PRONOUNS (§20)
  Singular:    au (I), koe (you), ia (he/she)
  Dual incl:   taua (we two, you and I)
  Dual excl:   maua (we two, he and I)
  Plural incl: tatou (we all, you and I)
  Plural excl: matou (we all, they and I)
  2nd plural:  koulua (you two), koutou (you all)
  3rd plural:  laua (they two), latou (they all)

VERB TENSE PARTICLES (§28-30)
  e  - present     (E tuku ne au - I give)
  ka - future      (Ka tuku ne au - I shall give)
  ne - past        (Ne tuku ne au - I gave)
  ma - conditional (Ma tuku ne au - If I should give)
  ke - consequential (Ke tuku ne au - That I may give)

  Negatives:
  se   - present negative  (Se tuku ne au - I don't give)
  seki - past negative     (Seki tuku ne au - I didn't give)
  sa   - imperative neg    (Sa tuku - Don't give!)

SIBLING TERMS (§41)
  taina - same-sex sibling (brother to brother, sister to sister)
  tuangane - opposite-sex sibling (brother to sister, sister to brother)

NUMBERS (§17)
  1: tasi    6: ono       For people, prefix 'toko':
  2: lua     7: fitu        tokotasi (one person)
  3: tolu    8: valu        tokolua (two people)
  4: fa      9: iva
  5: lima   10: angafulu

================================================================================
""")


# =============================================================================
# MAIN / CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print_grammar_reference()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "reference":
        print_grammar_reference()

    elif command == "validate" and len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        result = validate_text(text)
        print(result)

    elif command == "analyze" and len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        analysis = analyze_text(text)
        print(analysis.summary())

    elif command == "possessive" and len(sys.argv) > 2:
        word = sys.argv[2]
        pclass, explanation = get_possessive_class(word)
        print(f"{word}: {pclass.value}-class ({explanation})")
        if pclass != PossessiveClass.UNKNOWN:
            print(f"  'my {word}' = t{pclass.value}ku {word}")

    elif command == "conjugate" and len(sys.argv) > 2:
        verb = sys.argv[2]
        print(f"Conjugation of '{verb}':")
        for tense in TENSE_PARTICLES:
            print(f"  {tense}: {conjugate_verb(verb, tense)}")
        print("  Negatives:")
        for tense in NEGATIVE_PARTICLES:
            print(f"  {tense}: {conjugate_verb(verb, tense, negative=True)}")

    else:
        print("Usage:")
        print("  python tuvaluan_grammar.py reference")
        print("  python tuvaluan_grammar.py validate <text>")
        print("  python tuvaluan_grammar.py analyze <text>")
        print("  python tuvaluan_grammar.py possessive <noun>")
        print("  python tuvaluan_grammar.py conjugate <verb>")
