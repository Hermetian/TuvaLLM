# Tuvaluan Grammar Reference Skill

This skill provides Tuvaluan language grammar reference and validation based on Kennedy, D.G. (1945) "Handbook on the Language of the Tuvalu (Ellice) Islands".

## Usage

```
/tuvaluan-grammar [command] [arguments]
```

## Commands

### `reference` (default)
Display the complete grammar reference guide.

### `validate <text>`
Check if text follows Tuvaluan phonotactic rules. Reports:
- Invalid characters (not in Kennedy's 16-letter alphabet)
- Words not ending in vowels
- Potential consonant clusters

Note: Validation is advisory - transcriptions may vary by dialect and era.

### `analyze <text>`
Perform grammatical analysis of Tuvaluan text:
- Identify verb tense particles (e, ka, ne, ma, ke, etc.)
- Find and gloss pronouns
- Note validation issues

### `possessive <noun>`
Determine the A/O possessive class for a noun:
- A-class: food, children, acquired things
- O-class: body parts, feelings, house, land, relations

### `conjugate <verb>`
Show verb conjugation with all tense particles.

## Implementation

Run the grammar module directly:
```bash
cd scripts && python tuvaluan_grammar.py [command] [args]
```

Or import in Python:
```python
from scripts.tuvaluan_grammar import (
    validate_text,
    analyze_text,
    get_possessive_class,
    conjugate_verb,
    print_grammar_reference
)
```

## Key Grammar Points (Kennedy 1945)

### Alphabet
- 5 vowels: a, e, i, o, u (with macron variants for long vowels)
- 11 consonants: f, h, k, l, m, n, ng, p, s, t, v
- Every word ends in a vowel

### A/O Distinction
The most complex feature - possessive pronouns vary based on the noun:
- **A-class** (taku, tau, tana): food, children, actions
- **O-class** (toku, tou, tona): body parts, house, land, parents

### Verb Particles
No verb inflections - tense indicated by particles:
- `e` = present
- `ka` = future
- `ne` = past
- `se/seki/sa` = negatives

### Pronouns
Three numbers (singular, dual, plural) with inclusive/exclusive distinction in 1st person.

## Notes on Variation

Kennedy documents the Vaitupu dialect circa 1945. Modern usage may vary:
- Loanwords introduce non-traditional characters
- Spelling conventions have evolved
- Island-to-island dialect differences exist

The validation is informative, not prescriptive.
