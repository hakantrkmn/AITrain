"""
Normal mod - Orijinal görsel üzerinde düzenleme
"""

import os
import shutil
from PIL import Image
from typing import Tuple, Dict, List

from .base import BaseMode
from utils.image_utils import adjust_polygons_from_crop


class NormalMode(BaseMode):
    """Normal mod: Orijinal görsel üzerinde düzenleme"""
    
    def load_image(
        self,
        image_path: str,
        masks_data: List[Dict],
        bounding_boxes: List[List[float]],
        temp_dir: str
    ) -> Tuple[str, List[Dict], Dict]:
        """Normal mod: Orijinal görseli göster"""
        return image_path, masks_data, {}
    
    def save_image(
        self,
        image_path: str,
        edited_data: List[Dict],
        state: Dict
    ) -> bool:
        """Normal mod: Orijinal görseli ve YOLO formatında etiketleri kaydet"""
        try:
            output_images_dir = os.path.join(self.output_base, "normal", "images", "train")
            output_labels_dir = os.path.join(self.output_base, "normal", "labels", "train")
            
            # Klasörleri oluştur
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)
            
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Orijinal görseli kopyala
            output_image_path = os.path.join(output_images_dir, image_filename)
            shutil.copy2(image_path, output_image_path)
            
            # Görsel boyutlarını al
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Etiket dosyasını oluştur
            output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
            
            with open(output_label_path, 'w') as f:
                for mask_data in edited_data:
                    polygon = mask_data['polygon']
                    class_id = mask_data.get('class_id', 0)
                    
                    if len(polygon) < 3:
                        continue
                    
                    # Normalize koordinatları hesapla
                    normalized_coords = []
                    for x, y in polygon:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        normalized_coords.extend([norm_x, norm_y])
                    
                    line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
                    f.write(line + "\n")
            
            print(f"[BAŞARILI] Normal moduna kaydedildi: {output_image_path}")
            return True
            
        except Exception as e:
            print(f"[HATA] Normal moduna kaydetme hatası: {str(e)}")
            return False
