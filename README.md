# Whisper Large V3 Türkçe LoRA İnce Ayar (Fine-Tuning) Projesi

Bu proje, Türkçe ses tanıma (ASR) performansını artırmak amacıyla **OpenAI Whisper Large V3** modelinin Parametre Verimli İnce Ayar (**PEFT / LoRA**) yöntemiyle eğitilmesini sağlar. 

Eğitim süreçleri, çoklu GPU/MIG (Multi-Instance GPU) donanımları üzerinde optimize edilmiş olup, veri seti hazırlama aşamasından model karşılaştırmalarına kadar tüm uçtan uca akışı içerir.

---

## 📊 Kullanılan Veri Seti

Bu projede Türkçe konuşma tanıma modeli eğitimi için Hugging Face üzerindeki **[ysdede/commonvoice_17_tr_fixed](https://huggingface.co/datasets/ysdede/commonvoice_17_tr_fixed)** veri seti kullanılmıştır.

### Veri Seti Hakkında Önemli Bilgiler:
* **Kaynak:** Mozilla Common Voice 17.0 Türkçe verilerinin temizlenmiş ve düzeltilmiş versiyonudur.
* **İçerik:** Yaklaşık **45.000+** temizlenmiş Türkçe ses kaydı ve bunlara karşılık gelen metin transkriptlerini barındırır.
* **Bölümleme (Splits):**
  * **Train + Validation:** Eğitim ve doğrulama süreçlerinde bir arada kullanılır.
  * **Test:** Modelin performansını (WER - Word Error Rate) objektif olarak ölçmek için ayrılmıştır.

---

## ⚙️ Gereksinimler ve Kurulum

Projeyi çalıştırmadan önce gerekli kütüphaneleri yüklemeniz gerekir. PyTorch sürümünün CUDA desteği barındırdığından emin olun.

Bağımlılıkları yüklemek için:
```bash
pip install -r requirements.txt
```

*Detaylı bağımlılık listesi için [requirements.txt](file:///c:/Users/doganeren.ozer/Desktop/whisper%20fine%20tune/requirements.txt) dosyasını inceleyebilirsiniz.*

---

## 📂 Veri Dönüştürme ve Hazırlama Kodları

Whisper modelinin eğitimi için ses verilerinin 16kHz frekansına resample edilmesi, modelin anlayacağı özniteliklere dönüştürülmesi (Feature Extraction) ve metinlerin tokenize edilmesi gerekir. Bu işlemler için iki farklı script geliştirilmiştir:

### 1. Standart Veri Hazırlama: [prepare_data.py](file:///c:/Users/doganeren.ozer/Desktop/whisper%20fine%20tune/prepare_data.py)
Küçük veya orta ölçekli makinelerde hızlıca veri setini indirip işlemek için tasarlanmıştır.
* **İşleyiş:** Hugging Face `datasets` kütüphanesinin `.map()` fonksiyonunu kullanır.
* **Optimizasyonlar:** Hugging Face veri eşleme işlemlerinde karşılaşılan race-condition (yarış durumu) hatalarını önlemek için veri setini tek işlemci (`num_proc=1`) ile işler ve cache kullanımını devre dışı bırakır (`load_from_cache_file=False`).
* **Kullanım:**
  ```bash
  python prepare_data.py
  ```

### 2. Sunucu Seviyesinde Bellek Dostu Veri Hazırlama: [setup_on_server.py](file:///c:/Users/doganeren.ozer/Desktop/whisper%20fine%20tune/setup_on_server.py)
Büyük veri setleri işlenirken RAM (bellek) aşımını ve çökmeleri engellemek için tasarlanmış gelişmiş bir boru hattıdır (data pipeline).
* **İşleyiş:** Veri setinin ses dosyalarını manuel olarak `soundfile` ve `librosa` yardımıyla decode eder, 16kHz seviyesine resample eder ve 500'er örnekten oluşan **Parquet Shard'ları** halinde parça parça diske yazar.
* **Avantajı:** Tüm veri setini RAM'e yüklemek yerine disk tabanlı çalıştığı için RAM şişmesini tamamen engeller. İşlem bittikten sonra tüm shard'ları birleştirerek tek bir dataset oluşturur ve geçici dosyaları temizler.
* **Kullanım:**
  ```bash
  python setup_on_server.py
  ```

---

## 🚀 LoRA İnce Ayar (LoRA Fine-Tuning) Süreci: [finetune_lora.py](file:///c:/Users/doganeren.ozer/Desktop/whisper%20fine%20tune/finetune_lora.py)

Eğitimin belkemiğini oluşturan bu script, tüm model ağırlıklarını eğitmek yerine yalnızca belirli projeksiyon katmanlarına LoRA adaptörleri ekleyerek bellek kullanımını ve eğitim süresini dramatik ölçüde düşürür.

### 🔧 LoRA Konfigürasyonu
* **Model Ağırlık Tipi:** `bfloat16` (BF16) hassasiyeti ile eğitilir (H100/A100 GPU'lar için yerel destek).
* **LoRA Rank ($r$):** `64`
* **LoRA Alpha ($\alpha$):** `128`
* **LoRA Dropout:** `0.05`
* **Hedef Modüller (Target Modules):** `["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]` (Transformer mimarisindeki self-attention ve feed-forward katmanları).
* **Eğitilebilir Parametre Oranı:** Toplam parametrelerin yaklaşık **%1 - %2**'si eğitilir, bu da aşırı öğrenmeyi (overfitting) engeller.

### 🧪 Deney Tasarımı (Grid Search)
Eğitim scripti, en iyi performans gösteren hiperparametre kombinasyonunu bulmak için **3 farklı deneyi sıralı olarak** çalıştırır:

| Deney Adı | Öğrenme Oranı (LR) | Zamanlayıcı (Scheduler) | Çıktı Dizini |
| :--- | :--- | :--- | :--- |
| **lr1e-4_linear** | $1 \times 10^{-4}$ | Linear | `output_lora_lr1e-4_linear/` |
| **lr1e-5_linear** | $1 \times 10^{-5}$ | Linear | `output_lora_lr1e-5_linear/` |
| **lr1e-4_cosine** | $1 \times 10^{-4}$ | Cosine | `output_lora_lr1e-4_cosine/` |

Her deney sonunda elde edilen en iyi model ağırlıkları `best_model_lora_<deney_adi>/` dizinine kaydedilir.

### 💡 Gelişmiş Eğitim Optimizasyonları
* **Multi-MIG Desteği:** `CUDA_VISIBLE_DEVICES` ortam değişkeniyle Nvidia MIG (Multi-Instance GPU - örn. 4x 20GB) cihazlarını otomatik olarak tanımlar ve yönetir.
* **Early Stopping:** WER (Word Error Rate) değeri 5 değerlendirme adımı boyunca (5 x 250 = 1250 adım) iyileşmezse eğitimi otomatik olarak sonlandırır.
* **Gradient Checkpointing:** Bellek tasarrufu sağlamak için etkinleştirilmiştir (`use_reentrant=False`).
* **Anti-Halüsinasyon Ayarları:** Whisper modellerinde sık karşılaşılan tekrarlama ve halüsinasyonları engellemek için `no_repeat_ngram_size=5` ve `condition_on_prev_tokens=False` ayarları uygulanmıştır.

### 📈 Çalıştırma ve Karşılaştırma
Eğitimleri başlatmak için:
```bash
python finetune_lora.py
```
Tüm deneyler tamamlandığında konsola aşağıdaki gibi detaylı bir **karşılaştırma tablosu** basılır ve en iyi performansı (en düşük WER%) veren deney şampiyon seçilir:

```text
================================================================================
                    3 DENEY KARŞILAŞTIRMA TABLOSU
================================================================================
Deney                  LR         Scheduler  Best WER%    Best Step    Train Loss   Süre (dk) 
--------------------------------------------------------------------------------
lr1e-4_linear          1e-04      linear     x.xx         250          x.xxxx       xx.x      
lr1e-5_linear          1e-05      linear     y.yy         500          y.yyyy       yy.y      
lr1e-4_cosine          1e-04      cosine     z.zz         250          z.zzzz       zz.z      
--------------------------------------------------------------------------------

🏆 EN İYİ DENEY: lr1e-4_linear
   WER: x.xx% (step 250)
   Model: best_model_lora_lr1e-4_linear
================================================================================
```

Ayrıca eğitimlerin anlık kaybı (loss) ve WER gelişimleri **Tensorboard** ile `report_to=["tensorboard"]` parametresi sayesinde görselleştirilebilir.

---

## 🛠️ Temel İş Akışı Özeti

1. **Gereksinimleri Kurun:** `pip install -r requirements.txt`
2. **Veriyi Hazırlayın:** Sunucuda iseniz `python setup_on_server.py`, lokalde iseniz `python prepare_data.py` çalıştırarak verileri `data/` klasörüne kaydedin.
3. **Eğitimi Başlatın:** `python finetune_lora.py` komutu ile 3 adet LoRA deneyini sırayla koşturun ve en iyi modeli elde edin.
