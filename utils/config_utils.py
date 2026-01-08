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
    config = {'folder': '', 'model': ''}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Sadece geçerli dosya yollarını yükle
                if 'folder' in loaded_config and loaded_config['folder'] and os.path.exists(loaded_config['folder']):
                    config['folder'] = loaded_config['folder']
                if 'model' in loaded_config and loaded_config['model'] and os.path.exists(loaded_config['model']):
                    config['model'] = loaded_config['model']
        except Exception as e:
            print(f"[UYARI] Config dosyası okunamadı: {str(e)}")
    
    return config


def save_config(folder: str = None, model: str = None, config_path: str = "config.json"):
    """
    Mevcut seçimleri config dosyasına kaydeder.
    
    Args:
        folder: Klasör yolu
        model: Model dosyası yolu
        config_path: Config dosyası yolu
    """
    try:
        config = {
            'folder': folder if folder else '',
            'model': model if model else ''
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[UYARI] Config dosyası kaydedilemedi: {str(e)}")
