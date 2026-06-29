"""
Whisper Model Karşılaştırma — Orijinal vs LoRA Fine-Tuned
==========================================================
whisperv3.mp4 dosyasını iki modelle transkript eder ve karşılaştırır:
  1. Orijinal Whisper Large V3 (model/ klasörü)
  2. LoRA Fine-Tuned (best_model_lora_lr1e-4_linear/ klasörü)

Kullanım:
    python3 compare_models.py

Not: whisperv3.mp4 dosyası script ile aynı klasörde olmalı.
"""

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = (
    "MIG-55f875b9-19ba-5515-bff2-b4ed23162a0c,"
    "MIG-89fc3b54-087b-507a-b507-2715c4e673a2,"
    "MIG-5713d4a9-971f-55ef-b4d4-00678e40a66a,"
    "MIG-0c838429-3606-5d8b-95fb-885963a6ab47"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import numpy as np
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# ============================================================
# YOLLAR
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILE = os.path.join(BASE_DIR, "whisperv3.wav")
ORIGINAL_MODEL_DIR = os.path.join(BASE_DIR, "model")
LORA_MODEL_DIR = os.path.join(BASE_DIR, "best_model_lora_lr1e-4_linear")

LANGUAGE = "tr"
TASK = "transcribe"

# ============================================================
# SES DOSYASINI YÜKLE
# ============================================================
def load_audio(file_path):
    """WAV ses dosyasını 16kHz mono olarak yükle."""
    print(f"Ses dosyası yükleniyor: {file_path}")

    if not os.path.exists(file_path):
        print(f"HATA: Dosya bulunamadı: {file_path}")
        sys.exit(1)

    audio, sr = sf.read(file_path)

    # Stereo ise mono'ya çevir
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 16kHz'e resample et (gerekirse)
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
        sr = 16000

    duration = len(audio) / sr
    print(f"  Süre: {duration:.1f} saniye ({duration/60:.1f} dakika)")

    return audio.astype(np.float32)


# ============================================================
# TRANSKRIPT FONKSIYONU
# ============================================================
def transcribe_audio(model, processor, audio, model_name="Model"):
    """Ses verisini parçalara ayırıp transkript et (30 saniyelik chunk'lar)."""

    print(f"\n{'='*60}")
    print(f"Transkript yapılıyor: {model_name}")
    print(f"{'='*60}")

    sample_rate = 16000
    chunk_duration = 30  # Whisper 30 saniyelik pencere kullanır
    chunk_size = chunk_duration * sample_rate

    full_text = []
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size

    start_time = time.time()

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        chunk_idx = i // chunk_size + 1

        # Feature extract
        input_features = processor.feature_extractor(
            chunk, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(torch.bfloat16)

        # Model'in bulunduğu device'a gönder
        if hasattr(model, "device"):
            input_features = input_features.to(model.device)
        elif hasattr(model, "hf_device_map"):
            # device_map="auto" durumunda ilk device'ı kullan
            first_device = next(iter(set(model.hf_device_map.values())))
            input_features = input_features.to(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)

        # Generate
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
                no_repeat_ngram_size=5,
            )

        # Decode
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        full_text.append(transcription)
        print(f"  Chunk {chunk_idx}/{total_chunks}: {transcription[:80]}{'...' if len(transcription) > 80 else ''}")

    elapsed = time.time() - start_time
    result = " ".join(full_text)

    print(f"\n  Transkript süresi: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika)")
    return result, elapsed


# ============================================================
# ANA PROGRAM
# ============================================================
def main():
    print("=" * 60)
    print("WHISPER MODEL KARŞILAŞTIRMA")
    print("  Orijinal vs LoRA Fine-Tuned (lr=1e-4, linear)")
    print(f"  Ses dosyası: {AUDIO_FILE}")
    print("=" * 60)

    # Ses yükle
    audio = load_audio(AUDIO_FILE)

    # ----------------------------------------------------------
    # 1. ORİJİNAL MODEL
    # ----------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 1. ORİJİNAL WHISPER LARGE V3")
    print("#" * 60)

    processor = WhisperProcessor.from_pretrained(ORIGINAL_MODEL_DIR)
    print(f"Processor yüklendi: {ORIGINAL_MODEL_DIR}")

    original_model = WhisperForConditionalGeneration.from_pretrained(
        ORIGINAL_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Orijinal model yüklendi (device_map=auto)")

    original_text, original_time = transcribe_audio(original_model, processor, audio, "Orijinal Whisper Large V3")

    # Bellek temizle
    del original_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 2. LoRA FINE-TUNED MODEL
    # ----------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 2. LoRA FINE-TUNED (lr=1e-4, linear)")
    print("#" * 60)

    # Base model'i tekrar yükle
    base_model = WhisperForConditionalGeneration.from_pretrained(
        ORIGINAL_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # LoRA adaptörünü yükle
    lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    print(f"LoRA adaptör yüklendi: {LORA_MODEL_DIR}")

    lora_text, lora_time = transcribe_audio(lora_model, processor, audio, "LoRA Fine-Tuned (lr=1e-4)")

    # Bellek temizle
    del lora_model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # KARŞILAŞTIRMA
    # ----------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("                         KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 80)

    print("\n📌 ORİJİNAL MODEL (Whisper Large V3):")
    print("-" * 60)
    print(original_text)
    print(f"\n⏱️  Transkript süresi: {original_time:.1f} saniye ({original_time/60:.1f} dakika)")

    print(f"\n📌 LoRA FINE-TUNED (lr=1e-4, linear scheduler):")
    print("-" * 60)
    print(lora_text)
    print(f"\n⏱️  Transkript süresi: {lora_time:.1f} saniye ({lora_time/60:.1f} dakika)")

    print("\n" + "=" * 80)

    # Farkları göster
    orig_words = original_text.split()
    lora_words = lora_text.split()
    print(f"\nOrijinal kelime sayısı: {len(orig_words)}")
    print(f"LoRA kelime sayısı:     {len(lora_words)}")

    speed_diff = ((lora_time - original_time) / original_time) * 100
    if speed_diff > 0:
        print(f"\n⚡ Hız farkı: LoRA modeli %{speed_diff:.1f} DAHA YAVAŞ")
    elif speed_diff < 0:
        print(f"\n⚡ Hız farkı: LoRA modeli %{abs(speed_diff):.1f} DAHA HIZLI")
    else:
        print(f"\n⚡ Hız farkı: İki model AYNI HIZDA")

    if original_text == lora_text:
        print("\n⚠️  İki transkript BİREBİR AYNI!")
    else:
        print("\n✅ Transkriptler FARKLI — fine-tune etkisi var.")

    # Sonuçları dosyaya kaydet
    results_file = os.path.join(BASE_DIR, "transcription_comparison.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("WHISPER MODEL KARŞILAŞTIRMA\n")
        f.write(f"Ses dosyası: {AUDIO_FILE}\n")
        f.write("=" * 60 + "\n\n")
        f.write("ORİJİNAL MODEL (Whisper Large V3):\n")
        f.write("-" * 40 + "\n")
        f.write(original_text + "\n")
        f.write(f"Transkript süresi: {original_time:.1f} saniye\n\n")
        f.write("LoRA FINE-TUNED (lr=1e-4, linear):\n")
        f.write("-" * 40 + "\n")
        f.write(lora_text + "\n")
        f.write(f"Transkript süresi: {lora_time:.1f} saniye\n\n")
        f.write(f"Hız farkı: {speed_diff:+.1f}%\n")

    print(f"\nSonuçlar kaydedildi: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
