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

---

## 📈 Eğitim ve Değerlendirme Sonuçları (WER)

Model eğitimi sırasında yapılan değerlendirme adımlarında elde edilen **WER (Word Error Rate)** ve **Kayıp (Loss)** değerleri aşağıdaki tabloda verilmiştir.

Eğitim boyunca 5 değerlendirme adımı boyunca (5 x 250 = 1250 adım) iyileşme gözlenmediği için Erken Durdurma (Early Stopping) tetiklenmiş ve eğitim 1500. adımda sonlandırılmıştır. En iyi model ağırlıkları **250. adımda** elde edilmiştir.

### 🎯 Değerlendirme (Evaluation) Sonuçları:

| Değerlendirme Adımı (Step) | Eğitim Epoch | Değerlendirme Kaybı (Eval Loss) | Kelime Hata Oranı (Eval WER %) |
| :--- | :--- | :--- | :--- |
| **250 (En İyi Checkpoint)** | 0.23 | **0.2190** | **17.71%** |
| 500 | 0.46 | 0.4162 | 29.59% |
| 750 | 0.68 | 0.4230 | 31.26% |
| 1000 | 0.91 | 0.4102 | 30.36% |
| 1250 | 1.14 | 0.4378 | 30.63% |
| 1500 | 1.36 | 0.4016 | 28.40% |

<details>
<summary><b>🔍 Detaylı Eğitim Adımları (Her 25 Adımda Bir Kaydedilen Training Loss) - Tıklayıp Açınız</b></summary>

| Step | Epoch | Training Loss | Learning Rate |
| :--- | :--- | :--- | :--- |
| 25 | 0.0227 | 0.8328 | 4.80e-06 |
| 50 | 0.0455 | 0.4866 | 9.80e-06 |
| 75 | 0.0682 | 0.1645 | 1.48e-05 |
| 100 | 0.0910 | 0.1210 | 1.98e-05 |
| 125 | 0.1137 | 0.1176 | 2.48e-05 |
| 150 | 0.1365 | 0.1345 | 2.98e-05 |
| 175 | 0.1592 | 0.1333 | 3.48e-05 |
| 200 | 0.1820 | 0.1447 | 3.98e-05 |
| 225 | 0.2047 | 0.1494 | 4.48e-05 |
| 250 | 0.2275 | 0.1673 | 4.98e-05 |
| 275 | 0.2502 | 0.1712 | 5.48e-05 |
| 300 | 0.2730 | 0.1947 | 5.98e-05 |
| 325 | 0.2957 | 0.1838 | 6.48e-05 |
| 350 | 0.3185 | 0.2139 | 6.98e-05 |
| 375 | 0.3412 | 0.2133 | 7.48e-05 |
| 400 | 0.3640 | 0.2279 | 7.98e-05 |
| 425 | 0.3867 | 0.2838 | 8.48e-05 |
| 450 | 0.4095 | 0.2730 | 8.98e-05 |
| 475 | 0.4322 | 0.2982 | 9.48e-05 |
| 500 | 0.4550 | 0.2967 | 9.98e-05 |
| 525 | 0.4777 | 0.3135 | 1.00e-04 |
| 550 | 0.5005 | 0.3063 | 1.00e-04 |
| 575 | 0.5232 | 0.3273 | 1.00e-04 |
| 600 | 0.5460 | 0.3039 | 1.00e-04 |
| 625 | 0.5687 | 0.3120 | 1.00e-04 |
| 650 | 0.5914 | 0.3008 | 1.00e-04 |
| 675 | 0.6142 | 0.3083 | 9.99e-05 |
| 700 | 0.6369 | 0.3116 | 9.99e-05 |
| 725 | 0.6597 | 0.3010 | 9.99e-05 |
| 750 | 0.6824 | 0.3060 | 9.99e-05 |
| 775 | 0.7052 | 0.2920 | 9.98e-05 |
| 800 | 0.7279 | 0.2969 | 9.98e-05 |
| 825 | 0.7507 | 0.2764 | 9.98e-05 |
| 850 | 0.7734 | 0.2923 | 9.97e-05 |
| 875 | 0.7962 | 0.2925 | 9.97e-05 |
| 900 | 0.8189 | 0.2803 | 9.96e-05 |
| 925 | 0.8417 | 0.2982 | 9.96e-05 |
| 950 | 0.8644 | 0.2670 | 9.95e-05 |
| 975 | 0.8872 | 0.2976 | 9.95e-05 |
| 1000 | 0.9099 | 0.2880 | 9.94e-05 |
| 1025 | 0.9327 | 0.2627 | 9.94e-05 |
| 1050 | 0.9554 | 0.2647 | 9.93e-05 |
| 1075 | 0.9782 | 0.2786 | 9.93e-05 |
| 1100 | 1.0009 | 0.2735 | 9.92e-05 |
| 1125 | 1.0237 | 0.1814 | 9.91e-05 |
| 1150 | 1.0464 | 0.1791 | 9.91e-05 |
| 1175 | 1.0692 | 0.1646 | 9.90e-05 |
| 1200 | 1.0919 | 0.1643 | 9.89e-05 |
| 1225 | 1.1146 | 0.1806 | 9.88e-05 |
| 1250 | 1.1374 | 0.1805 | 9.87e-05 |
| 1275 | 1.1601 | 0.1888 | 9.87e-05 |
| 1300 | 1.1829 | 0.1828 | 9.86e-05 |
| 1325 | 1.2056 | 0.1688 | 9.85e-05 |
| 1350 | 1.2284 | 0.1773 | 9.84e-05 |
| 1375 | 1.2511 | 0.1777 | 9.83e-05 |
| 1400 | 1.2739 | 0.1830 | 9.82e-05 |
| 1425 | 1.2966 | 0.1682 | 9.81e-05 |
| 1450 | 1.3194 | 0.1804 | 9.80e-05 |
| 1475 | 1.3421 | 0.1694 | 9.79e-05 |
| 1500 | 1.3649 | 0.1800 | 9.78e-05 |

</details>

### Eğitim İstatistikleri Özeti:
* **Toplam Eğitim Kaybı (Train Loss):** 0.2448
* **Eğitim Süresi:** ~7.95 saat (28,624 saniye)
* **Saniyedeki Eğitim Örneği Sayısı:** 12.276

---

## 🛠️ Temel İş Akışı Özeti

1. **Gereksinimleri Kurun:** `pip install -r requirements.txt`
2. **Veriyi Hazırlayın:** Sunucuda iseniz `python setup_on_server.py`, lokalde iseniz `python prepare_data.py` çalıştırarak verileri `data/` klasörüne kaydedin.
3. **Eğitimi Başlatın:** `python finetune_lora.py` komutu ile 3 adet LoRA deneyini sırayla koşturun ve en iyi modeli elde edin.
