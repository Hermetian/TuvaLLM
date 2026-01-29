#!/usr/bin/env python3
"""
Check Samoan adapter vocabulary to see if it covers Tuvaluan characters/loans.
"""
from transformers import AutoTokenizer

def main():
    # Load Samoan tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-1b-all", target_lang="smo")
    vocab = tokenizer.get_vocab()
    
    print(f"Samoan Vocab Size: {len(vocab)}")
    
    # Check Tuvaluan chars (including loans found earlier)
    # Tuvaluan found: a-z, ō
    tvl_chars = "abcdefghijklmnopqrstuvwxyzō"
    
    missing = []
    present = []
    
    for c in tvl_chars:
        # Tokenizer might use 'a' or 'a|' etc.
        # MMS usually character based vocabulary
        # Let's check direct char presence or tokenization
        ids = tokenizer.encode(c, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        print(f"'{c}' -> {ids} {tokens}")
        
        # If it encodes to <unk>, it's missing
        if tokenizer.unk_token_id in ids:
            missing.append(c)
        else:
            present.append(c)
            
    print("\n" + "="*40)
    print(f"Supported chars ({len(present)}): {''.join(present)}")
    print(f"Unknown/Missing ({len(missing)}): {''.join(missing)}")
    print("="*40)

if __name__ == "__main__":
    main()
