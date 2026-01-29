#!/usr/bin/env python3
"""
Analyze Tuvaluan transcripts to build vocabulary and check for loan characters.
"""

import os
import json
from pathlib import Path
from collections import Counter
import docx
import unicodedata

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure processed dir exists
PROCESSED_DIR.mkdir(exist_ok=True)

def get_docx_files():
    """Find all DOCX transcripts."""
    return list(RAW_DIR.rglob("*.docx"))

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(docx_path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)

def analyze_characters(text):
    """Analyze character distribution."""
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # Count characters
    counter = Counter(text)
    
    return counter

def is_standard_tuvaluan(char):
    """Check if character is standard Tuvaluan."""
    standard = "abcdefghijklmnopqrstuvwxyzāēīōū' \n\r\t.,!?:;\"-()0123456789"
    return char.lower() in standard

def main():
    print("="*60)
    print("Tuvaluan Transcript Analysis")
    print("="*60)
    
    docx_files = get_docx_files()
    print(f"Found {len(docx_files)} transcript files:")
    for f in docx_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")
    
    all_text = ""
    for f in docx_files:
        print(f"\nProcessing {f.name}...")
        try:
            text = extract_text_from_docx(f)
            print(f"  Extracted {len(text)} characters")
            all_text += text + "\n"
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")
            
    # Analyze
    print("\nAnalyzing character distribution...")
    counter = analyze_characters(all_text)
    
    # Sort by frequency
    sorted_chars = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCharacter Frequency:")
    print(f"{'Char':<6} {'Count':<8} {'Standard?':<10} {'Name'}")
    print("-" * 40)
    
    vocab_chars = set()
    non_standard_chars = []
    
    for char, count in sorted_chars:
        # Skip invisible controls for display
        display_char = repr(char)[1:-1]
        if char == " ": display_char = "<space>"
        if char == "\n": display_char = "<newline>"
        
        is_std = is_standard_tuvaluan(char)
        name = unicodedata.name(char, "UNKNOWN")
        
        print(f"{display_char:<6} {count:<8} {'Yes' if is_std else 'NO':<10} {name}")
        
        if not is_std and char.strip() and not char in "0123456789.,!?:;\"-()":
            non_standard_chars.append((char, count, name))
            
        # Add to vocab if it's a letter (including English loan letters)
        if char.isalpha() or char in "'āēīōū":
            vocab_chars.add(char.lower())
            
    print("\n" + "="*60)
    print("POTENTIAL LOAN CHARACTERS (Non-Standard Tuvaluan):")
    for char, count, name in non_standard_chars:
        print(f"  {repr(char)} ({count}): {name}")
        
    # Build Vocab
    print("\nBuilding Vocabulary...")
    # Base special tokens
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,  # Word delimiter
    }
    
    # Add found characters (alphabetical order)
    sorted_vocab_chars = sorted(list(vocab_chars))
    for c in sorted_vocab_chars:
        if c not in vocab:
            vocab[c] = len(vocab)
            
    print(f"Final Vocab Size: {len(vocab)}")
    print(f"Vocab: {sorted_vocab_chars}")
    
    # Save vocab
    vocab_path = PROCESSED_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved vocabulary to {vocab_path}")

if __name__ == "__main__":
    main()
