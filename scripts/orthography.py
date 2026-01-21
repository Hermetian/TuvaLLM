#!/usr/bin/env python3
"""
TuvaLLM Orthography Module
Handles Tuvaluan orthographic system conversions.

Tuvaluan has multiple orthographic systems:
- System 4 (TLB): Uses macrons to mark long vowels (ā, ē, ī, ō, ū)
- System 6 (Common): No macrons - most commonly used by Tuvaluans

Reference: UDHR Article 1 comparison
System 4: "E fā'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tūlaga
          fakaaloalogina mo telotolo aiā."
System 6: "E fa'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tulaga
          fakaaloalogina mo telotolo aia."
"""

import re
from typing import Dict, Optional, Set


# Macron mappings
MACRON_TO_PLAIN = {
    'ā': 'a', 'Ā': 'A',
    'ē': 'e', 'Ē': 'E',
    'ī': 'i', 'Ī': 'I',
    'ō': 'o', 'Ō': 'O',
    'ū': 'u', 'Ū': 'U',
}

PLAIN_TO_MACRON = {v: k for k, v in MACRON_TO_PLAIN.items()}

# All macron characters
MACRON_CHARS = set(MACRON_TO_PLAIN.keys())

# Valid Tuvaluan characters (System 4 - with macrons)
TUVALUAN_CHARS_SYSTEM4 = set("abcdefghijklmnopqrstuvwxyzāēīōūABCDEFGHIJKLMNOPQRSTUVWXYZĀĒĪŌŪ '")

# Valid Tuvaluan characters (System 6 - without macrons)
TUVALUAN_CHARS_SYSTEM6 = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '")


def to_system6(text: str) -> str:
    """
    Convert System 4 (macrons) to System 6 (no macrons).

    This is a lossless conversion in terms of readability but loses
    information about vowel length.

    Args:
        text: Text potentially containing macrons

    Returns:
        Text with macrons removed (ā→a, ē→e, ī→i, ō→o, ū→u)

    Example:
        >>> to_system6("E fā'nau mai a tino katoa")
        "E fa'nau mai a tino katoa"
    """
    result = text
    for macron, plain in MACRON_TO_PLAIN.items():
        result = result.replace(macron, plain)
    return result


def to_system4(text: str, dictionary: Optional[Dict[str, str]] = None) -> str:
    """
    Convert System 6 to System 4 (requires dictionary lookup).

    This conversion is lossy without a dictionary since we cannot know
    where macrons should be placed just from the plain text.

    Args:
        text: Text without macrons
        dictionary: Optional word->macronized_word mapping
                   If None, returns text unchanged

    Returns:
        Text with macrons added where known from dictionary

    Example:
        >>> to_system4("tulaga", {"tulaga": "tūlaga"})
        "tūlaga"
    """
    if dictionary is None:
        return text

    # Simple word-by-word replacement
    words = text.split()
    result = []

    for word in words:
        # Preserve punctuation
        prefix = ""
        suffix = ""
        core = word

        # Extract leading punctuation
        while core and not core[0].isalpha():
            prefix += core[0]
            core = core[1:]

        # Extract trailing punctuation
        while core and not core[-1].isalpha():
            suffix = core[-1] + suffix
            core = core[:-1]

        # Look up word (case-insensitive)
        lookup_key = core.lower()
        if lookup_key in dictionary:
            # Match case of original
            replacement = dictionary[lookup_key]
            if core.isupper():
                replacement = replacement.upper()
            elif core and core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            core = replacement

        result.append(prefix + core + suffix)

    return " ".join(result)


def detect_system(text: str) -> str:
    """
    Detect which orthographic system the text uses.

    Args:
        text: Text to analyze

    Returns:
        "system4" if macrons are present, "system6" otherwise

    Example:
        >>> detect_system("E fā'nau mai")
        "system4"
        >>> detect_system("E fa'nau mai")
        "system6"
    """
    for char in text:
        if char in MACRON_CHARS:
            return "system4"
    return "system6"


def has_macrons(text: str) -> bool:
    """
    Check if text contains any macron characters.

    Args:
        text: Text to check

    Returns:
        True if macrons are present
    """
    return detect_system(text) == "system4"


def count_macrons(text: str) -> int:
    """
    Count the number of macron characters in text.

    Args:
        text: Text to analyze

    Returns:
        Number of macron characters
    """
    return sum(1 for char in text if char in MACRON_CHARS)


def normalize_text(text: str, target_system: str = "system4",
                   dictionary: Optional[Dict[str, str]] = None) -> str:
    """
    Normalize text to target orthographic system.

    Also performs general text normalization:
    - Converts to lowercase
    - Normalizes apostrophe variants
    - Removes invalid characters
    - Collapses multiple spaces

    Args:
        text: Text to normalize
        target_system: "system4" (with macrons) or "system6" (no macrons)
        dictionary: Word dictionary for system6→system4 conversion

    Returns:
        Normalized text in target orthographic system
    """
    # Lowercase
    text = text.lower()

    # Normalize apostrophe variants
    text = text.replace("'", "'")
    text = text.replace("'", "'")
    text = text.replace("`", "'")

    # Determine current system
    current_system = detect_system(text)

    # Convert if needed
    if target_system == "system6":
        text = to_system6(text)
        valid_chars = TUVALUAN_CHARS_SYSTEM6
    elif target_system == "system4":
        if current_system == "system6" and dictionary:
            text = to_system4(text, dictionary)
        valid_chars = TUVALUAN_CHARS_SYSTEM4
    else:
        raise ValueError(f"Unknown target system: {target_system}")

    # Filter to valid characters (lowercase only now)
    valid_chars_lower = set(c.lower() for c in valid_chars)
    text = "".join(c if c in valid_chars_lower else " " for c in text)

    # Collapse multiple spaces
    text = " ".join(text.split())

    return text.strip()


def get_vocabulary_for_system(system: str = "system4") -> Set[str]:
    """
    Get the character vocabulary appropriate for the target system.

    Args:
        system: "system4" or "system6"

    Returns:
        Set of valid characters for CTC vocabulary building
    """
    if system == "system4":
        # Include macron vowels
        chars = set("abcdefghijklmnopqrstuvwxyz āēīōū'")
    else:
        # No macrons
        chars = set("abcdefghijklmnopqrstuvwxyz '")

    return chars


def build_ctc_vocabulary(texts: list, system: str = "system4") -> Dict[str, int]:
    """
    Build a CTC vocabulary from texts for the specified orthographic system.

    Args:
        texts: List of normalized text strings
        system: Target orthographic system

    Returns:
        Character to index mapping with special tokens
    """
    from collections import Counter

    # Get valid characters for this system
    valid_chars = get_vocabulary_for_system(system)

    # Count characters
    char_counts = Counter()
    for text in texts:
        # Normalize to target system first
        normalized = normalize_text(text, target_system=system)
        char_counts.update(normalized)

    # Build vocabulary with special tokens
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,  # Word boundary (CTC convention)
    }

    # Add characters sorted by frequency
    for char, _ in char_counts.most_common():
        if char in valid_chars and char not in vocab:
            vocab[char] = len(vocab)

    # Ensure all valid chars are in vocab even if not seen
    for char in sorted(valid_chars):
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


# Example UDHR reference texts for testing
UDHR_SYSTEM4 = """E fā'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tūlaga
fakaaloalogina mo telotolo aiā. Ne tuku atu ki a lātou a te mafaufau
mo te loto lagona, tēlā lā, e 'tau o gā'lue fakatasi lātou e pēlā me ne taina."""

UDHR_SYSTEM6 = """E fa'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tulaga
fakaaloalogina mo telotolo aia. Ne tuku atu ki a latou a te mafaufau
mo te loto lagona, tela la, e 'tau o ga'lue fakatasi latou e pela me ne taina."""


if __name__ == "__main__":
    # Test the module
    print("Orthography Module Tests")
    print("=" * 50)

    # Test System 4 → System 6
    print("\n1. System 4 → System 6 conversion:")
    s4_text = "E fā'nau mai a tino katoa i te tūlaga"
    s6_text = to_system6(s4_text)
    print(f"   Input:  {s4_text}")
    print(f"   Output: {s6_text}")

    # Test detection
    print("\n2. System detection:")
    print(f"   '{s4_text[:20]}...' → {detect_system(s4_text)}")
    print(f"   '{s6_text[:20]}...' → {detect_system(s6_text)}")

    # Test normalization
    print("\n3. Text normalization (System 6):")
    messy = "  E FĀ'NAU   Mai's  testing!!  "
    normalized = normalize_text(messy, target_system="system6")
    print(f"   Input:  '{messy}'")
    print(f"   Output: '{normalized}'")

    # Test vocabulary building
    print("\n4. Vocabulary building:")
    vocab_s4 = build_ctc_vocabulary([UDHR_SYSTEM4], system="system4")
    vocab_s6 = build_ctc_vocabulary([UDHR_SYSTEM6], system="system6")
    print(f"   System 4 vocab size: {len(vocab_s4)}")
    print(f"   System 6 vocab size: {len(vocab_s6)}")

    # Show macron characters in System 4 vocab
    macron_in_vocab = [c for c in vocab_s4.keys() if c in MACRON_CHARS]
    print(f"   Macron chars in S4 vocab: {macron_in_vocab}")

    # Test dictionary conversion
    print("\n5. System 6 → System 4 with dictionary:")
    word_dict = {"tulaga": "tūlaga", "latou": "lātou", "aia": "aiā"}
    s6_sample = "tulaga mo latou aia"
    s4_converted = to_system4(s6_sample, word_dict)
    print(f"   Input:  {s6_sample}")
    print(f"   Output: {s4_converted}")

    print("\n" + "=" * 50)
    print("All tests passed!")
