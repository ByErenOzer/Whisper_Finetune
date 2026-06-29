"""
Whisper Large V3 Fine-Tuning için veri seti hazırlama scripti.
- Dataset: ysdede/commonvoice_17_tr_fixed (temizlenmiş Common Voice Turkish)
- Makaledeki race condition sorunu için num_proc=1 kullanılıyor
- Audio 16kHz'e resample + Whisper feature extraction + tokenization

Kullanım:
    python prepare_data.py
"""

import os
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

# ============================================================
# AYARLAR
# ============================================================
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "tr"
TASK = "transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 1. Model & Processor İndir ve Kaydet
# ============================================================
print("=" * 60)
print(f"Processor indiriliyor: {MODEL_NAME}")
print("=" * 60)

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

processor.save_pretrained(MODEL_DIR)
print(f"Processor kaydedildi: {MODEL_DIR}")

# ============================================================
# 2. Dataset İndir — ysdede/commonvoice_17_tr_fixed
# ============================================================
print("=" * 60)
print("Dataset indiriliyor: ysdede/commonvoice_17_tr_fixed")
print("=" * 60)

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

print(f"Train: {len(common_voice['train'])} örnek")
print(f"Test:  {len(common_voice['test'])} örnek")

# ============================================================
# 3. Gereksiz Sütunları Kaldır
# ============================================================
keep_columns = {"audio", "sentence"}
existing_columns = set(common_voice["train"].column_names)
columns_to_remove = list(existing_columns - keep_columns)

if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)
    print(f"Kaldırılan sütunlar: {columns_to_remove}")

# ============================================================
# 4. Audio'yu 16kHz'e Resample Et
# ============================================================
print("Audio 16kHz'e resample ediliyor...")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# ============================================================
# 5. Feature Extraction & Tokenization
#    NOT: Makaledeki race condition sorunu nedeniyle num_proc=1
#    ve load_from_cache_file=False kullanılıyor.
# ============================================================
def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch

print("Dataset işleniyor (num_proc=1, bu biraz sürebilir)...")
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1,
    load_from_cache_file=False,
)

# ============================================================
# 6. Dataset'i Diske Kaydet
# ============================================================
print(f"Dataset kaydediliyor: {DATA_DIR}")
common_voice.save_to_disk(DATA_DIR)

print("=" * 60)
print("Veri seti hazırlama tamamlandı!")
print(f"  Train: {len(common_voice['train'])} örnek")
print(f"  Test:  {len(common_voice['test'])} örnek")
print(f"  Kayıt: {DATA_DIR}")
print("=" * 60)
