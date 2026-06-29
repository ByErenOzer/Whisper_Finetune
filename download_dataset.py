"""
Dataset'i local'e indir ve işle.
Sonra data/ klasörünü sunucuya kopyalayın.

Kullanım:
    python download_dataset.py
"""

import os
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer

MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "tr"
TASK = "transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Dataset indiriliyor: ysdede/commonvoice_17_tr_fixed")
print("=" * 60)

# Feature extractor ve tokenizer yükle
print("\n1. Processor yükleniyor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
print("   ✓ Processor yüklendi")

# Dataset indir
print("\n2. Dataset indiriliyor (bu biraz sürebilir)...")
common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "ysdede/commonvoice_17_tr_fixed",
    split="train+validation",
    trust_remote_code=True,
)
common_voice["test"] = load_dataset(
    "ysdede/commonvoice_17_tr_fixed",
    split="test",
    trust_remote_code=True,
)

print(f"   ✓ Train: {len(common_voice['train'])} örnek")
print(f"   ✓ Test:  {len(common_voice['test'])} örnek")

# Gereksiz sütunları kaldır
print("\n3. Sütunlar temizleniyor...")
keep_columns = {"audio", "sentence"}
existing_columns = set(common_voice["train"].column_names)
columns_to_remove = list(existing_columns - keep_columns)

if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)
    print(f"   ✓ Kaldırılan: {columns_to_remove}")

# Audio 16kHz'e resample
print("\n4. Audio 16kHz'e resample ediliyor...")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
print("   ✓ Resample tamamlandı")

# Feature extraction & tokenization
print("\n5. Dataset işleniyor (num_proc=1, bu UZUN sürebilir - 30-60 dk)...")
print("   İlerleme göstergesi görünmeyebilir, lütfen bekleyin...")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1,
    load_from_cache_file=False,
    desc="Processing dataset",
)

# Dataset'i kaydet
print(f"\n6. Dataset kaydediliyor: {DATA_DIR}")
common_voice.save_to_disk(DATA_DIR)

print("\n" + "=" * 60)
print("DATASET HAZIRLAMA TAMAMLANDI!")
print(f"  Train: {len(common_voice['train'])} örnek")
print(f"  Test:  {len(common_voice['test'])} örnek")
print(f"  Klasör: {DATA_DIR}")
print(f"  Boyut: ~5-10 GB")
print("\nŞimdi bu klasörü sunucuya kopyalayın:")
print(f"  scp -r {DATA_DIR} sunucu:/home/mkys-yz/eren_whisper_finetune/")
print("=" * 60)
