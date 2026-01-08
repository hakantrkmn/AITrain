"""
BBMod - Bounding box tabanlı kırpma modu
"""

import os
import shutil
from PIL import Image
from typing import Tuple, Dict, List
from tkinter import messagebox

from .base import BaseMode
from utils.image_utils import crop_image_by_bbox, adjust_polygons_to_crop


class BBModMode(BaseMode):
    """BBMod: Bounding box'a göre kırpılmış görsel üzerinde düzenleme"""
    
    def load_image(
        self,
        image_path: str,
        masks_data: List[Dict],
        bounding_boxes: List[List[float]],
        temp_dir: str
    ) -> Tuple[str, List[Dict], Dict]:
        """BBMod: Bounding box'a göre kırp ve göster"""
        if len(bounding_boxes) == 0:
            messagebox.showwarning("Uyarı", "BBMod için bounding box bulunamadı! Normal moda geçiliyor.")
            return image_path, masks_data, {'fallback': True}
        
        # En büyük bounding box'ı kullan
        largest_box = max(bounding_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cropped_image, crop_box = crop_image_by_bbox(image_path, largest_box)
        
        # Kırpılmış görseli geçici olarak kaydet
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"cropped_{os.path.basename(image_path)}")
        cropped_image.save(temp_path)
        
        # Polygon koordinatlarını kırpılmış görsele göre ayarla
        adjusted_masks = adjust_polygons_to_crop(masks_data, crop_box)
        
        state = {'crop_box': crop_box}
        return temp_path, adjusted_masks, state
    
    def save_image(
        self,
        image_path: str,
        edited_data: List[Dict],
        state: Dict
    ) -> bool:
        """BBMod: Kırpılmış görseli ve YOLO formatında etiketleri kaydet"""
        try:
            crop_box = state.get('crop_box')
            if not crop_box:
                print("[HATA] BBMod için crop_box bulunamadı")
                return False
            
            output_images_dir = os.path.join(self.output_base, "bbmod", "images", "train")
            output_labels_dir = os.path.join(self.output_base, "bbmod", "labels", "train")
            
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Kırpılmış görseli kaydet
            img = Image.open(image_path)
            cropped_img = img.crop(crop_box)
            output_image_path = os.path.join(output_images_dir, image_filename)
            cropped_img.save(output_image_path)
            
            # Kırpılmış görsel boyutlarını al
            img_width, img_height = cropped_img.size
            
            # Etiket dosyasını oluştur
            output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
            
            with open(output_label_path, 'w') as f:
                for mask_data in edited_data:
                    polygon = mask_data['polygon']
                    class_id = mask_data.get('class_id', 0)
                    
                    if len(polygon) < 3:
                        continue
                    
                    # Normalize koordinatları hesapla (kırpılmış görsele göre)
                    normalized_coords = []
                    for x, y in polygon:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        normalized_coords.extend([norm_x, norm_y])
                    
                    line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
                    f.write(line + "\n")
            
            print(f"[BAŞARILI] BBMod'a kaydedildi: {output_image_path}")
            return True
            
        except Exception as e:
            print(f"[HATA] BBMod'a kaydetme hatası: {str(e)}")
            return False
