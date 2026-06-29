"""
Model ve processor'ı local'e indir.
Sonra model/ klasörünü sunucuya kopyalayın.

Kullanım:
    python download_model.py
"""

import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "tr"
TASK = "transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print(f"Model indiriliyor: {MODEL_NAME}")
print("=" * 60)

# Processor indir
print("\n1. Processor indiriliyor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor.save_pretrained(MODEL_DIR)
print(f"   ✓ Processor kaydedildi: {MODEL_DIR}")

# Model indir
print("\n2. Model indiriliyor (bu biraz sürebilir, ~3GB)...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.save_pretrained(MODEL_DIR)
print(f"   ✓ Model kaydedildi: {MODEL_DIR}")

print("\n" + "=" * 60)
print("İNDİRME TAMAMLANDI!")
print(f"Klasör: {MODEL_DIR}")
print(f"Boyut: ~3-4 GB")
print("\nŞimdi bu klasörü sunucuya kopyalayın:")
print(f"  scp -r {MODEL_DIR} sunucu:/home/mkys-yz/eren_whisper_finetune/")
print("=" * 60)
