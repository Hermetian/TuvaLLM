#!/usr/bin/env python3
"""
Inspect transcripts to understand structure (headers, dates) for alignment mapping.
"""

import docx
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "raw"

def get_docx_files():
    """Find all DOCX transcripts."""
    return sorted(list(DATA_DIR.rglob("*.docx")))

def inspect_docx(docx_path):
    """Print first 50 paragraphs of DOCX."""
    print(f"\n{'='*20} {docx_path.name} {'='*20}")
    doc = docx.Document(docx_path)
    
    count = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            print(f"{count}: {text[:100]}")
            count += 1
            if count > 50:
                print("... (truncated)")
                break

def main():
    files = get_docx_files()
    for f in files:
        inspect_docx(f)

if __name__ == "__main__":
    main()
