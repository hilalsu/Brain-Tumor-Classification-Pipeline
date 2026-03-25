# Brain Tumor Classification Pipeline

MRI görüntülerinden beyin tümörü sınıflandırması yapan uçtan uca bir makine öğrenmesi pipeline'ı.
Derin özellik çıkarımı, Grey Wolf Optimization (GWO) tabanlı özellik seçimi ve çoklu sınıflandırıcı
değerlendirmesini bir arada sunar.

---

## İçindekiler

- [Proje Açıklaması](#proje-açıklaması)
- [Pipeline Mantığı](#pipeline-mantığı)
- [Klasör Yapısı](#klasör-yapısı)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Nasıl Çalıştırılır](#nasıl-çalıştırılır)
- [Çıktılar](#çıktılar)
- [Bilinen Sorun ve Yapılan Düzeltmeler](#bilinen-sorun-ve-yapılan-düzeltmeler)
- [Performans Notları](#performans-notları)

---

## Proje Açıklaması

Bu pipeline, 4 sınıflı beyin tümörü MRI veri seti üzerinde çalışır:

| Sınıf | Açıklama |
|---|---|
| `glioma` | Glioma tümörü |
| `meningioma` | Meningioma tümörü |
| `pituitary` | Hipofiz tümörü |
| `notumor` | Tümör yok |

Pipeline, klasik "end-to-end deep learning" yaklaşımı yerine hibrit bir yaklaşım kullanır:

- Pretrained CNN modellerinden derin özellik çıkarımı
- GWO (Grey Wolf Optimizer) ile özellik seçimi
- Klasik ML modelleri ile sınıflandırma

Bu yaklaşım:

- Daha hızlı deney yapmayı sağlar
- Feature-level kontrol sunar
- Küçük datasetlerde daha stabil sonuç verebilir

---

## Pipeline Mantığı

Pipeline 7 ana adımdan oluşur:

### 1. Veri Yükleme

`Training/` ve `Testing/` klasörlerinden görüntü yolları ve etiketler okunur.
Sınıf isimleri klasör adlarından otomatik çıkarılır. Veriler `%80 train / %20 validation`
olarak stratified split ile ayrılır.

### 2. Ön İşleme

- CLAHE ile kontrast artırma
- Resize (varsayılan: 224x224)
- Augmentation (opsiyonel): yatay çevirme, döndürme, zoom, parlaklık

### 3. Sınıf Dengeleme

Azınlık sınıflar oversampling ile çoğunluk sınıfın sayısına tamamlanır.

### 4. Feature Extraction

`timm` kütüphanesi üzerinden pretrained modellerle özellik çıkarımı:

- `efficientnet_b0` → 1280 boyutlu vektör
- `resnet50` → 2048 boyutlu vektör

### 5. Feature Fusion

Tüm backbone çıktıları birleştirilir (3328 boyut) ve `StandardScaler` ile normalize edilir.

### 6. Feature Selection (GWO)

Grey Wolf Optimizer, en bilgilendirici özellik alt kümesini seçer. Her kurt adayı
SVM cross-validation ile değerlendirilir. Fitness fonksiyonu:

```
fitness = accuracy - λ * (k / d)
```

Büyük özellik uzaylarında GWO öncesi `SelectKBest` ile hızlı ön eleme yapılır.

### 7. Model Training

5 model eğitilir ve test seti üzerinde değerlendirilir:

- KNN
- Random Forest
- SVM (RBF)
- MLP
- XGBoost (opsiyonel, kurulu değilse atlanır)

---

## Klasör Yapısı

```
beyin/
├── brain_tumor_pipeline/
│   ├── run_pipeline.py              # Ana giriş noktası, CLI argümanları
│   ├── requirements.txt             # Bağımlılıklar
│   ├── README.md                    # Bu dosya
│   ├── __init__.py
│   ├── outputs/
│   │   └── run_YYYYMMDD_HHMMSS_utc/
│   │       ├── run.log
│   │       ├── gwo_result.json
│   │       ├── gwo_convergence.png
│   │       ├── results_summary.json
│   │       ├── all_results.csv
│   │       ├── confusion_matrix_*.png
│   │       └── roc_*.png
│   └── src/
│       ├── config.py                # Tüm pipeline parametreleri (dataclass)
│       ├── data.py                  # Dataset, dataloader, oversampling
│       ├── preprocess.py            # CLAHE, augmentasyon, transform pipeline
│       ├── features.py              # timm backbone ile özellik çıkarımı
│       ├── fusion.py                # Özellik birleştirme ve StandardScaler
│       ├── gwo_feature_selector.py  # Grey Wolf Optimizer implementasyonu
│       ├── models.py                # Sınıflandırıcılar ve değerlendirme
│       ├── metrics.py               # Accuracy, F1, ROC-AUC hesaplama
│       ├── visualize.py             # Confusion matrix, ROC, GWO grafikleri
│       └── utils.py                 # Logger, seed, JSON/dosya yardımcıları
└── dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

---

## Veri Seti

Kaggle üzerinden indirilebilir:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download

İndirilen klasör yapısı doğrudan `dataset/` olarak yerleştirilebilir.

---

## Kurulum

```bash
pip install -r brain_tumor_pipeline/requirements.txt
```

> GPU kullanmak istiyorsanız PyTorch'u CUDA destekli kurmanız gerekir:
> https://pytorch.org/get-started/locally/

---

## Nasıl Çalıştırılır

Tüm komutlar workspace root'undan (`beyin/` klasöründen) çalıştırılmalıdır.

### Hızlı Test (Önerilen)

```bash
python brain_tumor_pipeline/run_pipeline.py \
  --max-images-per-class 50 \
  --max-iterations 5 \
  --population-size 3 \
  --max-features 20 \
  --min-features 5 \
  --no-augmentation
```

### Tam Dataset

```bash
python brain_tumor_pipeline/run_pipeline.py
```

### GPU ile

```bash
python brain_tumor_pipeline/run_pipeline.py --device cuda
```

### Tüm CLI Argümanları

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--dataset-root` | `dataset` | Veri seti kök klasörü |
| `--output-base-dir` | `brain_tumor_pipeline/outputs` | Çıktı klasörü |
| `--seed` | `42` | Rastgelelik tohumu |
| `--device` | otomatik | `cpu`, `cuda`, `cuda:0` |
| `--image-size` | `224` | Görüntü boyutu (px) |
| `--val-fraction` | `0.2` | Validasyon oranı |
| `--no-augmentation` | kapalı | Augmentasyonu devre dışı bırak |
| `--no-class-balance` | kapalı | Oversampling'i devre dışı bırak |
| `--imagenet-normalize` | kapalı | ImageNet mean/std normalizasyonu |
| `--contrast-method` | `clahe` | `clahe` veya `hist_equal` |
| `--max-images-per-class` | `None` | Sınıf başına max görüntü sayısı |
| `--min-features` | `20` | GWO min özellik sayısı |
| `--max-features` | `250` | GWO max özellik sayısı |
| `--population-size` | `30` | GWO popülasyon büyüklüğü |
| `--max-iterations` | `80` | GWO iterasyon sayısı |

---

## Çıktılar

Her çalıştırma `outputs/run_YYYYMMDD_HHMMSS_utc/` altında ayrı bir klasör oluşturur.

### Gerçek Başarılı Run

`run_20260325_132534_utc/` klasörü:

```
run_20260325_132534_utc/
├── run.log                          # Tüm adımların detaylı log kaydı
├── gwo_result.json                  # Seçilen özellikler, CV accuracy, iterasyon geçmişi
├── gwo_convergence.png              # Fitness ve özellik sayısı değişim grafiği
├── results_summary.json             # Tüm modellerin özet metrikleri
├── all_results.csv                  # Model karşılaştırma tablosu
├── confusion_matrix_knn.png
├── confusion_matrix_random_forest.png
├── confusion_matrix_svm_rbf.png
├── confusion_matrix_mlp.png
├── confusion_matrix_xgboost.png
├── roc_knn.png
├── roc_random_forest.png
├── roc_svm_rbf.png
├── roc_mlp.png
└── roc_xgboost.png
```

Bu klasörde:

- 5 model için confusion matrix
- 5 model için ROC curve
- GWO convergence plot
- JSON + CSV sonuçlar
- run.log

---

## Bilinen Sorun ve Yapılan Düzeltmeler

### Sorun: Outputs Klasörü Boş Görünüyordu

Pipeline aslında çalışıyordu fakat kullanıcı tarafında çıktı oluşmuyor gibi algılanıyordu.

### Gerçek Sebepler

#### 1. Ağır Model İndirme (timm / HuggingFace)

Pipeline başlangıçta şu modelleri kullanıyordu:

```python
backbones = [
    "efficientnet_b0",
    "resnet50",
    "vit_base_patch16_224",        # ~330 MB
    "swin_base_patch4_window7_224" # ~350 MB
]
```

Bu modeller ilk çalıştırmada HuggingFace Hub'dan indirilir. Yavaş bağlantıda:

- İndirme takılıyor
- Timeout olabiliyor
- Pipeline daha başlamadan kilitleniyor, output yok

#### 2. Tüm Dataset'in CPU'da İşlenmesi

`--max-images-per-class` argümanı yoktu. Pipeline her seferinde binlerce MRI görüntüsünü
4 backbone ile CPU'da işlemeye çalışıyordu. Bu durum saatler süren bir işleme yol açtı
ve kullanıcı tarafından "çalışmıyor" olarak algılandı.

#### 3. Yanlış Beklenti

Pipeline; feature extraction bitmeden, GWO çalışmadan, model training tamamlanmadan
hiçbir çıktı üretmez. Uzun süren işlemler sırasında outputs klasörü boş görünür.

---

### Yapılan Fixler

#### 1. Backbone Sadeleştirildi

```python
# ÖNCE
backbones = ["efficientnet_b0", "resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224"]

# SONRA
backbones = ["efficientnet_b0", "resnet50"]
```

#### 2. Yeni CLI Parametresi Eklendi

```python
p.add_argument("--max-images-per-class", type=int, default=None)
```

#### 3. Hızlı Test Modu

```bash
python brain_tumor_pipeline/run_pipeline.py --max-images-per-class 50
```

### Sonuç

Pipeline artık stabil şekilde çalışıyor ve tüm çıktıları üretiyor:
confusion matrix, ROC curves, gwo_convergence, JSON & CSV, run.log

---

### Önemli Uyarı

Eğer outputs klasörü boş görünüyorsa:

- Pipeline büyük ihtimalle hâlâ çalışıyordur
- Özellikle ilk run'da model indiriyor olabilir
- CPU kullanıyorsanız süreç çok uzun sürebilir

Çözüm:

```bash
python brain_tumor_pipeline/run_pipeline.py --max-images-per-class 50
```

---

## Performans Notları

| Senaryo | Tahmini Süre |
|---|---|
| Küçük test (50 görüntü/sınıf, CPU) | ~3-5 dk |
| Tam dataset, CPU | ~2-4 saat |
| Tam dataset, GPU | ~15-30 dk |

### Ipuçları

- İlk çalıştırma için `--max-images-per-class 50` ile başla
- GPU varsa mutlaka `--device cuda` kullan
- GWO'yu hızlandırmak için `--max-iterations 10 --population-size 5` dene
