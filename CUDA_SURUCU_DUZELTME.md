# NVIDIA Sürücü Versiyon Uyumsuzluğu Çözümü

## Sorun
```
nvidia-smi: Driver/library version mismatch
NVML library version: 590.48
Kernel modül: 535.274.02
```

## Durum
- Kernel modülü: **535.274.02** (çalışıyor)
- Kütüphaneler: **590.48.01** (yarım kurulu, `iU` durumunda)
- Eski kütüphaneler: **535.274.02** (kaldırılmış)

## Çözüm Seçenekleri

### Seçenek 1: Sistem Yeniden Başlatma (En Basit)
590 sürücüsü yarım kurulu. Sistem yeniden başlatıldığında düzgün yüklenebilir:

```bash
sudo reboot
```

Yeniden başlattıktan sonra:
```bash
nvidia-smi
```

### Seçenek 2: 590 Sürücüsünü Tam Kur
Yarım kalan kurulumu tamamla:

```bash
# Yarım kalan paketleri düzelt
sudo apt --fix-broken install

# 590 sürücüsünü tam kur
sudo apt install --reinstall nvidia-driver-590

# DKMS modülünü yeniden derle
sudo dkms install nvidia/590.48.01 -k $(uname -r)

# Sistem yeniden başlat
sudo reboot
```

### Seçenek 3: 535 Sürücüsüne Geri Dön (Önerilen - Daha Stabil)
590 sürücüsü GTX 750 Ti için çok yeni olabilir. 535'e geri dön:

```bash
# 590 paketlerini kaldır
sudo apt remove --purge nvidia-driver-590 nvidia-dkms-open-590

# 535 sürücüsünü kur
sudo apt install nvidia-driver-535

# Sistem yeniden başlat
sudo reboot
```

### Seçenek 4: Temiz Kurulum
Tüm NVIDIA paketlerini temizle ve yeniden kur:

```bash
# Tüm NVIDIA paketlerini kaldır
sudo apt remove --purge '^nvidia-.*'
sudo apt autoremove

# Sistem yeniden başlat
sudo reboot

# Otomatik uygun sürücüyü kur
sudo ubuntu-drivers autoinstall

# Veya manuel 535 kur
sudo apt install nvidia-driver-535

# Tekrar yeniden başlat
sudo reboot
```

## Kontrol Komutları

Kurulum sonrası kontrol:

```bash
# Sürücü durumu
nvidia-smi

# Kernel modül versiyonu
cat /proc/driver/nvidia/version

# CUDA toolkit (varsa)
nvcc --version

# PyTorch GPU testi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## GTX 750 Ti için Önerilen Sürücü
- **535.x** veya **470.x** serisi (daha eski ama stabil)
- 590.x çok yeni ve GTX 750 Ti için gerekli olmayabilir

## Not
Sürücü değişikliklerinden sonra **mutlaka sistem yeniden başlatılmalıdır**.
