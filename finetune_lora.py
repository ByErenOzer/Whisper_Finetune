"""
Whisper Large V3 LoRA Fine-Tuning — Multi-MIG (4x 20GB)
=========================================================
3 deneyi sırayla çalıştırır ve sonunda karşılaştırma tablosu basar:
  Deney 1: LR=1e-4, Linear scheduler
  Deney 2: LR=1e-5, Linear scheduler
  Deney 3: LR=1e-4, Cosine scheduler

Kullanım:
    python3 finetune_lora.py
"""

import os
import sys
import gc
import json
import time

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
import glob
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
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# SABİT AYARLAR
# ============================================================
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "tr"
TASK = "transcribe"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Training hyperparameters (tüm deneyler için ortak)
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 500
EVAL_STEPS = 250
SAVE_STEPS = 250
LOGGING_STEPS = 25
BF16 = True
GRADIENT_CHECKPOINTING = True

# LoRA hyperparameters
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

# ============================================================
# 3 DENEY TANIMLA
# ============================================================
EXPERIMENTS = [
    {"name": "lr1e-4_linear",  "lr": 1e-4, "scheduler": "linear"},
    {"name": "lr1e-5_linear",  "lr": 1e-5, "scheduler": "linear"},
    {"name": "lr1e-4_cosine",  "lr": 1e-4, "scheduler": "cosine"},
]

print("=" * 60)
print("LoRA FINE-TUNE — 3 DENEY SIRALI ÇALIŞACAK")
print(f"Model: {MODEL_NAME}")
print(f"CUDA devices visible: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("-" * 60)
for i, exp in enumerate(EXPERIMENTS, 1):
    print(f"  Deney {i}: LR={exp['lr']}, Scheduler={exp['scheduler']} → output_lora_{exp['name']}/")
print("=" * 60)

# ============================================================
# Veri Seti Kontrol
# ============================================================
dataset_dict_path = os.path.join(DATA_DIR, "dataset_dict.json")
if not os.path.exists(dataset_dict_path):
    print("HATA: Veri seti bulunamadı! Önce setup_on_server.py çalıştırın.")
    sys.exit(1)

# Processor yükle (tüm deneyler için ortak)
processor_config_path = os.path.join(MODEL_DIR, "preprocessor_config.json")
if os.path.exists(processor_config_path):
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    print(f"Processor local'den yüklendi: {MODEL_DIR}")
else:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor.save_pretrained(MODEL_DIR)
    print(f"Processor indirildi ve kaydedildi: {MODEL_DIR}")

# Dataset yükle (tüm deneyler için ortak)
dataset = load_from_disk(DATA_DIR)
print(f"Train: {len(dataset['train'])} örnek")
print(f"Test:  {len(dataset['test'])} örnek")

# Metric (tüm deneyler için ortak)
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
# Data Collator
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
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


# ============================================================
# DENEY FONKSIYONU
# ============================================================
def run_experiment(exp_config, exp_index):
    """Tek bir LoRA deney çalıştırır, sonuçları döner."""

    exp_name = exp_config["name"]
    lr = exp_config["lr"]
    scheduler = exp_config["scheduler"]

    output_dir = os.path.join(BASE_DIR, f"output_lora_{exp_name}")
    best_model_dir = os.path.join(BASE_DIR, f"best_model_lora_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print("\n" + "#" * 60)
    print(f"# DENEY {exp_index}/3: {exp_name}")
    print(f"#   LR={lr}, Scheduler={scheduler}")
    print("#" * 60)

    # Model'i her deney için taze yükle (önceki LoRA ağırlıkları kalmasın)
    print(f"Model yükleniyor: {MODEL_NAME}")
    model_file = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(model_file):
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.save_pretrained(MODEL_DIR)

    if hasattr(model, "hf_device_map"):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Model cihazlar: {devices_used}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Base parametreler: {total_params/1e6:.1f}M")

    # Generation ayarları
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.generation_config.no_repeat_ngram_size = 5
    model.generation_config.condition_on_prev_tokens = False

    # Gradient checkpointing
    if GRADIENT_CHECKPOINTING:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # LoRA uygula
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Eğitilebilir: {trainable_params/1e6:.1f}M ({100*trainable_params/(trainable_params+frozen_params):.2f}%)")

    # decoder_start_token_id al
    if hasattr(model.config, "decoder_start_token_id") and model.config.decoder_start_token_id is not None:
        dec_start_id = model.config.decoder_start_token_id
    else:
        dec_start_id = model.base_model.model.config.decoder_start_token_id

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=dec_start_id,
    )

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=scheduler,
        weight_decay=0.01,
        optim="adamw_torch",
        fp16=False,
        bf16=BF16,
        bf16_full_eval=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        predict_with_generate=True,
        generation_max_length=225,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        push_to_hub=False,
    )

    # Trainer
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

    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    print(f"  Effective batch: {effective_batch}")
    print(f"  LR: {lr} ({scheduler})")
    print(f"  Early stopping: patience=5")
    print(f"  Output: {output_dir}")

    # Checkpoint'tan devam
    checkpoint_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    resume_from = checkpoint_dirs[-1] if checkpoint_dirs else None
    if resume_from:
        print(f"  Checkpoint'tan devam: {resume_from}")

    # EĞIT
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    train_time = time.time() - start_time

    # Kaydet
    model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # Final eval
    eval_results = {}
    try:
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)
    except Exception as e:
        print(f"  Final eval hatası: {e}")

    # trainer_state.json'dan best WER bul
    state_file = os.path.join(output_dir, "trainer_state.json")
    best_wer = eval_results.get("eval_wer", None)
    best_step = None
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        best_wer = state.get("best_metric", best_wer)
        best_step = state.get("best_global_step", None)

    result = {
        "name": exp_name,
        "lr": lr,
        "scheduler": scheduler,
        "train_loss": train_metrics.get("train_loss", None),
        "best_wer": best_wer,
        "best_step": best_step,
        "final_eval_wer": eval_results.get("eval_wer", None),
        "final_eval_loss": eval_results.get("eval_loss", None),
        "train_time_min": train_time / 60,
        "best_model_dir": best_model_dir,
    }

    print(f"\n  ✓ Deney {exp_index} tamamlandı!")
    print(f"    Best WER: {best_wer:.2f}% (step {best_step})")
    print(f"    Süre: {train_time/60:.1f} dakika")

    # Bellek temizle
    del model, trainer, training_args
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# 3 DENEYI SIRALI ÇALIŞTIR
# ============================================================
all_results = []

for i, exp in enumerate(EXPERIMENTS, 1):
    result = run_experiment(exp, i)
    all_results.append(result)

    # Sonuçları her deney sonrası kaydet (crash durumunda kaybolmasın)
    results_file = os.path.join(BASE_DIR, "experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

# ============================================================
# KARŞILAŞTIRMA TABLOSU
# ============================================================
print("\n\n" + "=" * 80)
print("                    3 DENEY KARŞILAŞTIRMA TABLOSU")
print("=" * 80)
print(f"{'Deney':<22} {'LR':<10} {'Scheduler':<10} {'Best WER%':<12} {'Best Step':<12} {'Train Loss':<12} {'Süre (dk)':<10}")
print("-" * 80)

best_exp = None
best_wer_overall = float("inf")

for r in all_results:
    wer_str = f"{r['best_wer']:.2f}" if r['best_wer'] is not None else "N/A"
    step_str = str(r['best_step']) if r['best_step'] is not None else "N/A"
    loss_str = f"{r['train_loss']:.4f}" if r['train_loss'] is not None else "N/A"
    time_str = f"{r['train_time_min']:.1f}"

    marker = ""
    if r['best_wer'] is not None and r['best_wer'] < best_wer_overall:
        best_wer_overall = r['best_wer']
        best_exp = r

    print(f"{r['name']:<22} {r['lr']:<10.0e} {r['scheduler']:<10} {wer_str:<12} {step_str:<12} {loss_str:<12} {time_str:<10}")

print("-" * 80)
if best_exp:
    print(f"\n🏆 EN İYİ DENEY: {best_exp['name']}")
    print(f"   WER: {best_exp['best_wer']:.2f}% (step {best_exp['best_step']})")
    print(f"   Model: {best_exp['best_model_dir']}")

print("\n" + "=" * 80)
print(f"Sonuçlar kaydedildi: {os.path.join(BASE_DIR, 'experiment_results.json')}")
print("=" * 80)
