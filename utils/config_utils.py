"""
Config dosyası yönetimi utility fonksiyonları
"""

import os
import json


def load_config(config_path: str = "config.json") -> dict:
    """
    Config dosyasından değerleri yükler.
    
    Args:
        config_path: Config dosyası yolu
        
    Returns:
        Config dictionary
    """
    config = {'folder': '', 'model': '', 'last_image_index': {}, 'epsilon': 0.002, 'shrink': 0.0, 'auto_save_on_next': False}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Sadece geçerli dosya yollarını yükle
                if 'folder' in loaded_config and loaded_config['folder'] and os.path.exists(loaded_config['folder']):
                    config['folder'] = loaded_config['folder']
                if 'model' in loaded_config and loaded_config['model'] and os.path.exists(loaded_config['model']):
                    config['model'] = loaded_config['model']
                # Son görsel indekslerini yükle (her mod için)
                if 'last_image_index' in loaded_config:
                    config['last_image_index'] = loaded_config['last_image_index']
                # Epsilon değerini yükle
                if 'epsilon' in loaded_config:
                    config['epsilon'] = loaded_config['epsilon']
                # Shrink değerini yükle
                if 'shrink' in loaded_config:
                    config['shrink'] = loaded_config['shrink']
                # Otomatik kaydet ayarını yükle
                if 'auto_save_on_next' in loaded_config:
                    config['auto_save_on_next'] = loaded_config['auto_save_on_next']
        except Exception as e:
            print(f"[UYARI] Config dosyası okunamadı: {str(e)}")
    
    return config


def save_config(folder: str = None, model: str = None, last_image_index: dict = None, 
                epsilon: float = None, shrink: float = None, auto_save_on_next: bool = None,
                config_path: str = "config.json"):
    """
    Mevcut seçimleri config dosyasına kaydeder.
    
    Args:
        folder: Klasör yolu
        model: Model dosyası yolu
        last_image_index: Mod'a göre son görsel indeksleri dict'i (örn: {'normal': 5, 'unet': 10})
        epsilon: Epsilon faktörü değeri
        shrink: Shrink yüzdesi değeri
        auto_save_on_next: Sonraki görsele geçerken otomatik kaydet
        config_path: Config dosyası yolu
    """
    try:
        # Mevcut config'i yükle (varsa)
        existing_config = load_config(config_path)
        
        config = {
            'folder': folder if folder is not None else existing_config.get('folder', ''),
            'model': model if model is not None else existing_config.get('model', ''),
            'last_image_index': last_image_index if last_image_index is not None else existing_config.get('last_image_index', {}),
            'epsilon': epsilon if epsilon is not None else existing_config.get('epsilon', 0.002),
            'shrink': shrink if shrink is not None else existing_config.get('shrink', 0.0),
            'auto_save_on_next': auto_save_on_next if auto_save_on_next is not None else existing_config.get('auto_save_on_next', False)
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[UYARI] Config dosyası kaydedilemedi: {str(e)}")
