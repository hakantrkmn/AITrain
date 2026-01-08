"""
Base mod sınıfı
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
from PIL import Image


class BaseMode(ABC):
    """Tüm modlar için base sınıf"""
    
    def __init__(self, output_base: str = "output"):
        self.output_base = output_base
        self.mode_name = self.__class__.__name__.lower().replace('mode', '')
    
    @abstractmethod
    def load_image(
        self,
        image_path: str,
        masks_data: List[Dict],
        bounding_boxes: List[List[float]],
        temp_dir: str
    ) -> Tuple[str, List[Dict], Dict]:
        """
        Görseli yükler ve mod'a göre işler.
        
        Args:
            image_path: Görsel dosyası yolu
            masks_data: YOLO'dan gelen maske verileri
            bounding_boxes: YOLO'dan gelen bounding box'lar
            temp_dir: Geçici dosyalar için klasör
            
        Returns:
            (display_path, adjusted_masks, state) tuple
            display_path: Canvas'da gösterilecek görsel yolu
            adjusted_masks: Mod'a göre ayarlanmış maske verileri
            state: Mod'a özel state bilgisi (crop_box, letterbox_image, vb.)
        """
        pass
    
    @abstractmethod
    def save_image(
        self,
        image_path: str,
        edited_data: List[Dict],
        state: Dict
    ) -> bool:
        """
        Görseli ve etiketleri kaydeder.
        
        Args:
            image_path: Orijinal görsel yolu
            edited_data: Düzenlenmiş maske verileri
            state: Mod'a özel state bilgisi
            
        Returns:
            Başarılı ise True
        """
        pass
