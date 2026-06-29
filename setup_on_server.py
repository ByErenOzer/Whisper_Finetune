"""
Sunucuda çalıştırılacak setup scripti.
Model ve dataset'i HuggingFace'den indirir ve işler.

Kullanım (sunucuda):
    pip install soundfile librosa tqdm
    python3 setup_on_server.py

NOT: Model zaten model/ klasöründe varsa tekrar indirmez.
"""

import os
import io
import time
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer
)

MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "tr"
TASK = "transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("WHISPER LARGE V3 FINE-TUNING SETUP")
print("=" * 70)

# ============================================================
# 1. MODEL VE PROCESSOR
# ============================================================
print("\n[1/3] MODEL VE PROCESSOR...")
print("-" * 70)

processor_config = os.path.join(MODEL_DIR, "preprocessor_config.json")
model_file = os.path.join(MODEL_DIR, "model.safetensors")

if os.path.exists(processor_config) and os.path.exists(model_file):
    print("    ✓ Model ve processor zaten mevcut, atlanıyor.")
else:
    print("  → Processor indiriliyor...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor.save_pretrained(MODEL_DIR)
    print(f"    ✓ Processor kaydedildi: {MODEL_DIR}")

    print("\n  → Model indiriliyor (~3GB, biraz sürebilir)...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_DIR)
    print(f"    ✓ Model kaydedildi: {MODEL_DIR}")
    del model  # bellek temizle

# ============================================================
# 2. DATASET İNDİR
# ============================================================
print("\n[2/3] DATASET İNDİRİLİYOR...")
print("-" * 70)

print("  → ysdede/commonvoice_17_tr_fixed indiriliyor...")
common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "ysdede/commonvoice_17_tr_fixed",
    split="train+validation",
)
common_voice["test"] = load_dataset(
    "ysdede/commonvoice_17_tr_fixed",
    split="test",
)

print(f"    ✓ Train: {len(common_voice['train'])} örnek")
print(f"    ✓ Test:  {len(common_voice['test'])} örnek")

# Sütun adlarını göster (debug)
col_names = common_voice["train"].column_names
print(f"    Sütunlar: {col_names}")

# Metin sütununu bul (transcription, sentence, text)
if "transcription" in col_names:
    TEXT_COL = "transcription"
elif "sentence" in col_names:
    TEXT_COL = "sentence"
elif "text" in col_names:
    TEXT_COL = "text"
else:
    raise ValueError(f"Metin sütunu bulunamadı! Sütunlar: {col_names}")

print(f"    Metin sütunu: '{TEXT_COL}'")

# Audio auto-decode'u kapat — torchcodec bypass (ÖNCE bunu yap!)
print("\n  → Audio decode=False ayarlanıyor (torchcodec bypass)...")
for split in common_voice:
    common_voice[split] = common_voice[split].cast_column("audio", Audio(decode=False))
print("    ✓ Tamamlandı")

# Gereksiz sütunları kaldır (audio ve TEXT_COL hariç) - cast sonrası yap
keep_columns = {"audio", TEXT_COL}
columns_to_remove = [c for c in col_names if c not in keep_columns]
if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)
    print(f"    ✓ Kaldırılan: {len(columns_to_remove)} sütun ({columns_to_remove})")

# İlk satırı test et (decode=False olduğu için torchcodec çağrılmaz)
test_row = common_voice["train"][0]
print(f"    Test row keys: {list(test_row.keys())}")
print(f"    Audio keys: {list(test_row['audio'].keys())}")
print(f"    Text: '{test_row[TEXT_COL][:50]}...'")

# ============================================================
# 3. DATASET İŞLE (FEATURE EXTRACTION)
# ============================================================
print("\n[3/3] DATASET İŞLENİYOR (FEATURE EXTRACTION)...")
print("-" * 70)
print("  → soundfile + librosa ile audio decode & 16kHz resample")

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)


import pyarrow as pa
from datasets.arrow_writer import ArrowWriter


def process_split(dataset, split_name, output_dir):
    """Bir split'i for loop + tqdm ile işle, parça parça diske yaz (overflow önleme)."""
    os.makedirs(output_dir, exist_ok=True)
    
    skipped = 0
    written = 0
    BATCH_SIZE = 500  # Her 500 örnekte bir diske yaz
    
    batch_features = []
    batch_labels = []
    shard_idx = 0

    for i in tqdm(range(len(dataset)), desc=f"  {split_name}", unit="örnek"):
        try:
            row = dataset[i]
            audio_data = row["audio"]
            
            # Audio bytes al
            audio_bytes = audio_data.get("bytes", None)
            if audio_bytes is None:
                audio_path = audio_data.get("path", None)
                if audio_path and os.path.exists(audio_path):
                    audio_array, sr = sf.read(audio_path)
                else:
                    raise ValueError(f"Audio bytes veya path bulunamadı: {list(audio_data.keys())}")
            else:
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))

            # Mono'ya çevir
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # float32'ye çevir
            audio_array = audio_array.astype(np.float32)

            # 16kHz'e resample
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            # Whisper feature extraction
            feats = feature_extractor(
                audio_array, sampling_rate=16000
            ).input_features[0]

            # Tokenize
            text = row[TEXT_COL]
            labels = tokenizer(text).input_ids

            batch_features.append(feats)
            batch_labels.append(labels)
            written += 1
            
            # Batch doldu, diske yaz
            if len(batch_features) >= BATCH_SIZE:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.parquet")
                shard_ds = Dataset.from_dict({
                    "input_features": batch_features,
                    "labels": batch_labels,
                })
                shard_ds.to_parquet(shard_path)
                batch_features = []
                batch_labels = []
                shard_idx += 1
                
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\n    ⚠️  Satır {i} atlandı: {e}")
            if skipped == 1:
                print(f"    DEBUG row keys: {list(row.keys())}")

    # Kalan batch'i yaz
    if batch_features:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.parquet")
        shard_ds = Dataset.from_dict({
            "input_features": batch_features,
            "labels": batch_labels,
        })
        shard_ds.to_parquet(shard_path)
        shard_idx += 1

    if skipped > 0:
        print(f"\n    ⚠️  Toplam {skipped}/{len(dataset)} satır atlandı")
    
    print(f"    ✓ {written} örnek, {shard_idx} shard yazıldı")
    
    # Tüm shard'ları birleştir
    from datasets import concatenate_datasets
    all_shards = []
    for s in range(shard_idx):
        shard_path = os.path.join(output_dir, f"shard_{s:04d}.parquet")
        shard = Dataset.from_parquet(shard_path)
        all_shards.append(shard)
    
    combined = concatenate_datasets(all_shards)
    
    # Shard dosyalarını temizle
    for s in range(shard_idx):
        os.remove(os.path.join(output_dir, f"shard_{s:04d}.parquet"))
    
    return combined


start_time = time.time()

train_tmp = os.path.join(BASE_DIR, "tmp_train_shards")
test_tmp = os.path.join(BASE_DIR, "tmp_test_shards")

processed = DatasetDict()
processed["train"] = process_split(common_voice["train"], "Train", train_tmp)
processed["test"] = process_split(common_voice["test"], "Test", test_tmp)

# Temp klasörleri temizle
import shutil
if os.path.exists(train_tmp):
    shutil.rmtree(train_tmp)
if os.path.exists(test_tmp):
    shutil.rmtree(test_tmp)

elapsed = time.time() - start_time
print(f"\n    ✓ İşleme tamamlandı: {elapsed/60:.1f} dakika")
print(f"    Train: {len(processed['train'])} örnek")
print(f"    Test:  {len(processed['test'])} örnek")

if len(processed["train"]) == 0:
    print("\n    ❌ HATA: Dataset boş! Lütfen çıktıyı kontrol edin.")
    exit(1)

# Dataset'i kaydet
print(f"\n  → Dataset kaydediliyor: {DATA_DIR}")
processed.save_to_disk(DATA_DIR)
print(f"    ✓ Dataset kaydedildi")

# ============================================================
# ÖZET
# ============================================================
print("\n" + "=" * 70)
print("SETUP TAMAMLANDI!")
print("=" * 70)
print(f"Model:    {MODEL_DIR}")
print(f"Dataset:  {DATA_DIR}")
print(f"  Train:  {len(processed['train'])} örnek")
print(f"  Test:   {len(processed['test'])} örnek")
print("\nŞimdi eğitimi başlatabilirsiniz:")
print("  python3 finetune.py")
print("=" * 70)
