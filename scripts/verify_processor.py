from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import json
import torch

MODEL_PATH = "models/mms-tuvaluan"

print(f"Loading processor from {MODEL_PATH}...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Failed to load processor: {e}")
    # Try loading from base components if possible, or exit
    exit(1)

text = "talo fa"
print(f"\nOriginal text: '{text}'")

# Encode
encoded = processor(text=text, return_tensors="pt")
ids = encoded.input_ids[0].tolist()
print(f"Encoded IDs: {ids}")

# Verify IDs against vocab
with open("data/processed/vocab.json") as f:
    vocab = json.load(f)
    id2char = {v: k for k, v in vocab.items()}

print("\nID Mapping check:")
decoded_manual = []
for i in ids:
    char = id2char.get(i, "?")
    print(f"  {i} -> '{char}'")
    decoded_manual.append(char)

print(f"Manual Decode: '{''.join(decoded_manual)}'")

# Processor Decode
decoded_processor = processor.batch_decode([ids])[0]
print(f"Processor Decode: '{decoded_processor}'")

if decoded_processor == text:
    print("\n✅ Processor working correctly.")
else:
    print("\n❌ ID mismatch! Processor might be stripping space or using wrong delimiter.")
