#!/bin/bash
# ============================================================
# Whisper Large V3 Fine-Tuning - 4 MIG GPU ile Eğitim Başlatma
# ============================================================

# Dizinleri oluştur
mkdir -p model data output best_model

# MIG GPU'ları listele
echo "=========================================="
echo "Kullanılabilir GPU'lar:"
echo "=========================================="
nvidia-smi -L

# CUDA visible devices - 4 MIG cihazı
# MIG UUID'lerini otomatik algıla
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL ayarları (MIG uyumluluğu için)
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Tokenizers parallelism uyarısını kapat
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "Adım 1: Veri seti hazırlanıyor..."
echo "=========================================="

# Eğer data klasörü boşsa veri setini hazırla
if [ ! -f "./data/dataset_dict.json" ]; then
    python prepare_data.py
    if [ $? -ne 0 ]; then
        echo "HATA: Veri seti hazırlama başarısız!"
        exit 1
    fi
else
    echo "Veri seti zaten mevcut, atlanıyor."
fi

echo "=========================================="
echo "Adım 2: Fine-tuning başlatılıyor..."
echo "=========================================="

# accelerate ile 4 MIG GPU'da DDP eğitimi
accelerate launch \
    --config_file accelerate_config.yaml \
    finetune.py

echo "=========================================="
echo "Eğitim tamamlandı!"
echo "Best model: ./best_model/"
echo "=========================================="
