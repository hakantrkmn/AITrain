"""
Dosya işlemleri utility fonksiyonları
"""

import os


def ensure_output_structure(output_base: str = "output"):
    """
    YOLOv8 çıktı klasör yapısını oluşturur.
    
    Args:
        output_base: Çıktı klasörü base yolu
    """
    # Normal mod klasörleri
    normal_images = os.path.join(output_base, "normal", "images", "train")
    normal_labels = os.path.join(output_base, "normal", "labels", "train")
    
    # BBMod klasörleri
    bbmod_images = os.path.join(output_base, "bbmod", "images", "train")
    bbmod_labels = os.path.join(output_base, "bbmod", "labels", "train")
    
    # UNet klasörleri
    unet_images = os.path.join(output_base, "unet", "images", "train")
    unet_masks = os.path.join(output_base, "unet", "masks", "train")
    unet_valid = os.path.join(output_base, "unet", "valid", "train")
    
    os.makedirs(normal_images, exist_ok=True)
    os.makedirs(normal_labels, exist_ok=True)
    os.makedirs(bbmod_images, exist_ok=True)
    os.makedirs(bbmod_labels, exist_ok=True)
    os.makedirs(unet_images, exist_ok=True)
    os.makedirs(unet_masks, exist_ok=True)
    os.makedirs(unet_valid, exist_ok=True)