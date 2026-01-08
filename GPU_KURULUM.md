# GPU Kurulum Talimatları (GTX 750 Ti)

## Sorun
GTX 750 Ti GPU'nuz CUDA capability 5.0'a sahip, ancak modern PyTorch sürümleri (2.x) CUDA 7.0+ gerektiriyor. Bu yüzden GPU uyarıları alıyorsunuz.

## Çözüm Seçenekleri

### Seçenek 1: Eski PyTorch Sürümü Kullan (Önerilen - GPU için)

GTX 750 Ti için uyumlu PyTorch 1.13.1 sürümünü kullanın:

```bash
# Önce mevcut PyTorch'u kaldırın
pip uninstall torch torchvision torchaudio

# CUDA 11.7 için PyTorch 1.13.1 yükleyin
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Diğer paketler
pip install ultralytics>=8.0.0 Pillow>=10.0.0 numpy>=1.24.0 opencv-python>=4.8.0
```

**Not:** Ultralytics YOLOv8 PyTorch 2.0+ gerektirebilir. Bu durumda Seçenek 2'yi kullanın.

### Seçenek 2: CPU Modunda Çalıştır (En Kolay)

Mevcut kurulumunuzla CPU'da çalıştırabilirsiniz. Program otomatik olarak CPU'yu kullanacak:

```bash
# Mevcut paketlerle çalışır
pip install -r requirements.txt
```

Program GPU'yu tespit edemezse otomatik olarak CPU'ya geçer.

### Seçenek 3: CUDA Toolkit Güncelle

Sisteminizde CUDA Toolkit 11.6 veya 11.7 yüklü olmalı:

1. NVIDIA'dan CUDA Toolkit 11.7 indirin: https://developer.nvidia.com/cuda-11-7-0-download-archive
2. CUDA Toolkit'i kurun
3. PyTorch 1.13.1'i yukarıdaki komutla yükleyin

## Program Kullanımı

Program otomatik olarak GPU'yu tespit eder:
- GPU uyumluysa → GPU kullanır
- GPU uyumlu değilse → CPU kullanır
- Cihaz bilgisi GUI'de gösterilir

## Test

GPU'nun çalışıp çalışmadığını test etmek için:

```python
import torch
print(f"CUDA mevcut: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA versiyonu: {torch.version.cuda}")
```

## Notlar

- GTX 750 Ti eski bir GPU'dur ve modern PyTorch sürümleriyle uyumlu değildir
- En iyi performans için PyTorch 1.13.1 + CUDA 11.7 kombinasyonunu kullanın
- Eğer ultralytics YOLOv8 yeni PyTorch gerektiriyorsa, CPU modunda çalıştırmanız gerekebilir
