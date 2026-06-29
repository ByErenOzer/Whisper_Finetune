"""
Whisper Large V3 Full Fine-Tuning Script
=========================================
- openai/whisper-large-v3 modeli (quantize YOK, full precision)
- Tek MIG GPU (20GB) üzerinde çalışır
- Dataset: ysdede/commonvoice_17_tr_fixed
- Best model eval WER'e göre best_model/ klasörüne kaydedilir
- Makaledeki anti-halüsinasyon ayarları uygulanmıştır

Kullanım:
    python3 finetune.py
"""

import os
import sys

# ÖNEMLİ: torch import'undan ÖNCE tek MIG'i ayarla
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-55f875b9-19ba-5515-bff2-b4ed23162a0c"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

# ============================================================
# AYARLAR
# ============================================================
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "tr"
TASK = "transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BEST_MODEL_DIR = os.path.join(BASE_DIR, "best_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Training hyperparameters
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # 20GB MIG için güvenli
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 2*16 = 32
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
EVAL_STEPS = 250
SAVE_STEPS = 250
LOGGING_STEPS = 25
FP16 = False
BF16 = True  # H100 native bf16 destekliyor
GRADIENT_CHECKPOINTING = True

NUM_GPU = 1
is_main = True

print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"CUDA devices visible: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ============================================================
# 1. Veri Seti Kontrol
# ============================================================
dataset_dict_path = os.path.join(DATA_DIR, "dataset_dict.json")
if not os.path.exists(dataset_dict_path):
    print("HATA: Veri seti bulunamadı! Önce setup_on_server.py çalıştırın.")
    sys.exit(1)

# ============================================================
# 2. Processor & Model Yükle
# ============================================================
print("=" * 60)
print(f"Model yükleniyor: {MODEL_NAME}")
print("=" * 60)

# Processor
processor_config_path = os.path.join(MODEL_DIR, "preprocessor_config.json")
if os.path.exists(processor_config_path):
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    print(f"Processor local'den yüklendi: {MODEL_DIR}")
else:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor.save_pretrained(MODEL_DIR)
    print(f"Processor indirildi ve kaydedildi: {MODEL_DIR}")

# Model
model_file = os.path.join(MODEL_DIR, "model.safetensors")
if os.path.exists(model_file):
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    print(f"Model local'den yüklendi: {MODEL_DIR}")
else:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_DIR)
    print(f"Model indirildi ve kaydedildi: {MODEL_DIR}")

# Whisper generation ayarları + anti-halüsinasyon (makaleden)
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.no_repeat_ngram_size = 5
model.generation_config.condition_on_prev_tokens = False

if GRADIENT_CHECKPOINTING:
    model.config.use_cache = False

if is_main:
    print(f"Model parametreleri: {model.num_parameters() / 1e6:.1f}M")
    print(f"Model boyutu (yaklaşık): {model.num_parameters() * 2 / 1e9:.1f}GB (FP16)")

# ============================================================
# 3. Dataset Yükle
# ============================================================
dataset = load_from_disk(DATA_DIR)
if is_main:
    print(f"Train: {len(dataset['train'])} örnek")
    print(f"Test:  {len(dataset['test'])} örnek")

# ============================================================
# 4. Data Collator
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ============================================================
# 5. Evaluation Metric (WER)
# ============================================================
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ============================================================
# 6. Training Arguments
# ============================================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # Epochs & Batch
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Optimizer + Cosine LR Schedule
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    optim="adamw_torch",

    # Precision — BF16 (H100 native, daha stabil)
    fp16=FP16,
    bf16=BF16,
    bf16_full_eval=True,

    # Evaluation & Saving — her 250 step'te eval + best model kaydet
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    # Logging
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],

    # Generation (eval sırasında decode yapar — WER hesabı için gerekli)
    predict_with_generate=True,
    generation_max_length=225,

    # Gradient Checkpointing — VRAM tasarrufu
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Tek GPU (MIG 1g.20gb — DDP desteklenmiyor)
    # ddp_find_unused_parameters=False,

    # Dataloader
    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    push_to_hub=False,
)

# ============================================================
# 7. Trainer
# ============================================================
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# ============================================================
# 8. Eğitimi Başlat
# ============================================================
if is_main:
    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPU * GRADIENT_ACCUMULATION_STEPS
    print("=" * 60)
    print("EĞİTİM BAŞLIYOR")
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Epochs:         {NUM_TRAIN_EPOCHS}")
    print(f"  Batch/GPU:      {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  GPU sayısı:     {NUM_GPU}")
    print(f"  Grad accum:     {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  LR:             {LEARNING_RATE}")
    print(f"  BF16:           {BF16}")
    print(f"  Grad CP:        {GRADIENT_CHECKPOINTING}")
    print(f"  Eval/Save step: {EVAL_STEPS}")
    print("=" * 60)

train_result = trainer.train()

# ============================================================
# 9. Best Model'i Kaydet
# ============================================================
if is_main:
    print("=" * 60)
    print(f"Best model kaydediliyor: {BEST_MODEL_DIR}")
    print("=" * 60)

    trainer.save_model(BEST_MODEL_DIR)
    processor.save_pretrained(BEST_MODEL_DIR)

    # Metrics kaydet
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final eval
    print("=" * 60)
    print("Final evaluation...")
    print("=" * 60)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("=" * 60)
    print("EĞİTİM TAMAMLANDI!")
    print(f"  Best model: {BEST_MODEL_DIR}")
    print(f"  Final Eval WER: {eval_metrics.get('eval_wer', 'N/A'):.2f}%")
    print("=" * 60)
