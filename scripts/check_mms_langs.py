#!/usr/bin/env python3
"""
Check if Tuvaluan (tvl) is supported in MMS vocab/adapters.
"""
from transformers import Wav2Vec2ForCTC, AutoTokenizer

def main():
    model_id = "facebook/mms-1b-all"
    print(f"Checking {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    print(f"Base vocab size: {len(vocab)}")
    
    try:
        # Check if we can load the 'tvl' adapter
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        model.load_adapter("tvl")
        print("SUCCESS: Tuvaluan ('tvl') adapter loaded!")
    except Exception as e:
        print(f"FAILED to load 'tvl' adapter: {e}")
        
    try:
        # Check 'smo' (Samoan) as fallback
        model.load_adapter("smo")
        print("SUCCESS: Samoan ('smo') adapter loaded!")
    except Exception as e:
        print(f"FAILED to load 'smo' adapter: {e}")

if __name__ == "__main__":
    main()
