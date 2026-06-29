"""
Whisper Large V3 Full Fine-Tuning — Multi-MIG (4x 20GB)
=========================================================
MIG 1g.20gb slice'larda NCCL DDP desteklenmiyor.
Bu script model'i 4 MIG'e device_map ile dağıtarak pipeline parallelism yapar.
Böylece daha büyük batch_size kullanılabilir ve eğitim hızlanır.

NOT: Bu yaklaşımda model katmanları farklı MIG'lere bölünür (model parallelism).
     Data parallelism DEĞİLDİR — her step tek batch işlenir ama model 4 MIG'e sığar.

Kullanım:
    python3 finetune_multi_mig.py
"""

import os
import sys

# ÖNEMLİ: torch import'undan ÖNCE tüm MIG'leri görünür yap
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
OUTPUT_DIR = os.path.join(BASE_DIR, "output_multi")
BEST_MODEL_DIR = os.path.join(BASE_DIR, "best_model_multi")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Training hyperparameters
# 4 MIG x 20GB = 80GB toplam — model dağıtılınca batch büyütülebilir
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 32  # Maksimum hız için batch=32, grad_accum=1
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch = 32*1 = 32
LEARNING_RATE = 1e-5  # Full fine-tune için 1e-5 (LoRA'da 1e-4 kullanılır, full'de catastrophic forgetting yapar!)
WARMUP_STEPS = 500
EVAL_STEPS = 250  # Tek MIG ile aynı
SAVE_STEPS = 250  # Tek MIG ile aynı
LOGGING_STEPS = 25
BF16 = True
GRADIENT_CHECKPOINTING = True

print("=" * 60)
print("MULTI-MIG MODE (Model Parallelism)")
print(f"Model: {MODEL_NAME}")
print(f"CUDA devices visible: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("=" * 60)

# ============================================================
# 1. Veri Seti Kontrol
# ============================================================
dataset_dict_path = os.path.join(DATA_DIR, "dataset_dict.json")
if not os.path.exists(dataset_dict_path):
    print("HATA: Veri seti bulunamadı! Önce setup_on_server.py çalıştırın.")
    sys.exit(1)

# ============================================================
# 2. Processor & Model Yükle (device_map="auto" ile 4 MIG'e dağıt)
# ============================================================
print("=" * 60)
print(f"Model yükleniyor: {MODEL_NAME} (4 MIG'e dağıtılıyor)")
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

# Model — device_map="auto" ile katmanlar otomatik dağıtılır
model_file = os.path.join(MODEL_DIR, "model.safetensors")
if os.path.exists(model_file):
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Model local'den yüklendi ve 4 MIG'e dağıtıldı")
else:
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.save_pretrained(MODEL_DIR)
    print(f"Model indirildi ve kaydedildi: {MODEL_DIR}")

# Model device map göster
if hasattr(model, "hf_device_map"):
    devices_used = set(str(v) for v in model.hf_device_map.values())
    print(f"Model dağıtıldığı cihazlar: {devices_used}")

# Model parametreleri
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parametreleri: {total_params/1e6:.1f}M")

# Whisper generation ayarları + anti-halüsinasyon
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.no_repeat_ngram_size = 5
model.generation_config.condition_on_prev_tokens = False

if GRADIENT_CHECKPOINTING:
    model.config.use_cache = False

# ============================================================
# 3. Dataset Yükle
# ============================================================
dataset = load_from_disk(DATA_DIR)
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

        # input_features'ı BF16'ya cast et (device_map="auto" ile model BF16'da)
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

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
# 5. Metrics (WER)
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

    # Precision — BF16 (H100 native)
    fp16=False,
    bf16=BF16,
    bf16_full_eval=True,

    # Evaluation & Saving
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    # Logging
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],

    # Generation
    predict_with_generate=True,
    generation_max_length=225,

    # Gradient Checkpointing — VRAM tasarrufu
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Dataloader
    dataloader_num_workers=4,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Tek MIG ile aynı
)

# ============================================================
# 8. Eğitimi Başlat
# ============================================================
effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
print("=" * 60)
print("EĞİTİM BAŞLIYOR (MULTI-MIG)")
print(f"  Model:          {MODEL_NAME}")
print(f"  Epochs:         {NUM_TRAIN_EPOCHS}")
print(f"  Batch size:     {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"  Grad accum:     {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch: {effective_batch}")
print(f"  LR:             {LEARNING_RATE} (cosine)")
print(f"  BF16:           {BF16}")
print(f"  Grad CP:        {GRADIENT_CHECKPOINTING}")
print(f"  Eval/Save step: {EVAL_STEPS}")
print(f"  Early stopping: patience=5")
print("=" * 60)

# Checkpoint'tan devam etme desteği
import glob
checkpoint_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume_from = checkpoint_dirs[-1] if checkpoint_dirs else None
if resume_from:
    print(f"Checkpoint'tan devam ediliyor: {resume_from}")

train_result = trainer.train(resume_from_checkpoint=resume_from)

# ============================================================
# 9. Best Model'i Kaydet
# ============================================================
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

# Final eval — device_map modelde sorun çıkarabilir, try/except ile koru
print("=" * 60)
print("Final evaluation...")
print("=" * 60)
try:
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    print(f"  Final Eval WER: {eval_metrics.get('eval_wer', 'N/A'):.2f}%")
except Exception as e:
    print(f"  Final eval hatası (model kaydedildi, sorun yok): {e}")
    eval_metrics = {}

print("=" * 60)
print("EĞİTİM TAMAMLANDI!")
print(f"  Best model: {BEST_MODEL_DIR}")
if eval_metrics:
    print(f"  Final Eval WER: {eval_metrics.get('eval_wer', 'N/A'):.2f}%")
print("=" * 60)
