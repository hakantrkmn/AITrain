"""
UNet mod - Bounding box kırpma + letterbox resize
"""

import os
from PIL import Image
from typing import Tuple, Dict, List
from tkinter import messagebox

from .base import BaseMode
from utils.image_utils import (
    crop_image_by_bbox,
    adjust_polygons_to_crop,
    letterbox_resize,
    create_binary_mask,
    create_valid_mask
)


class UNetMode(BaseMode):
    """UNet: Bounding box kırpma + 576x320 letterbox resize"""
    
    # Hedef boyut
    TARGET_SIZE = (576, 320)
    
    def load_image(
        self,
        image_path: str,
        masks_data: List[Dict],
        bounding_boxes: List[List[float]],
        temp_dir: str
    ) -> Tuple[str, List[Dict], Dict]:
        """UNet: Bounding box'a göre kırp + letterbox resize"""
        if len(bounding_boxes) == 0:
            messagebox.showwarning("Uyarı", "UNet için bounding box bulunamadı! Normal moda geçiliyor.")
            return image_path, masks_data, {'fallback': True}
        
        # En büyük bounding box'ı kullan
        largest_box = max(bounding_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cropped_image, crop_box = crop_image_by_bbox(image_path, largest_box)
        
        # 576x320'e letterbox resize (aspect ratio korunarak)
        letterbox_img, offset_x, offset_y, scale, new_width, new_height = letterbox_resize(cropped_image, self.TARGET_SIZE)
        
        # Polygon koordinatlarını önce kırpılmış görsele göre ayarla
        adjusted_masks_crop = adjust_polygons_to_crop(masks_data, crop_box)
        
        # Sonra letterbox resize'e göre ayarla (scale + offset)
        adjusted_masks = []
        for mask_data in adjusted_masks_crop:
            polygon = mask_data['polygon']
            # Önce scale uygula, sonra offset ekle
            adjusted_polygon = [(x * scale + offset_x, y * scale + offset_y) for x, y in polygon]
            adjusted_masks.append({
                'polygon': adjusted_polygon,
                'class_id': mask_data.get('class_id', 0),
                'confidence': mask_data.get('confidence', 0.0)
            })
        
        # Letterbox görseli geçici olarak kaydet
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"letterbox_{os.path.basename(image_path)}")
        letterbox_img.save(temp_path)
        
        state = {
            'crop_box': crop_box,
            'letterbox_image': letterbox_img,
            'letterbox_offset': (offset_x, offset_y),
            'letterbox_scale': scale,
            'letterbox_roi_size': (new_width, new_height),
            'target_size': self.TARGET_SIZE
        }
        return temp_path, adjusted_masks, state
    
    def save_image(
        self,
        image_path: str,
        edited_data: List[Dict],
        state: Dict
    ) -> bool:
        """UNet: 576x320 letterbox görseli, binary mask ve valid mask kaydet"""
        try:
            letterbox_image = state.get('letterbox_image')
            if not letterbox_image:
                print("[HATA] UNet için letterbox görsel bulunamadı")
                return False
            
            target_size = state.get('target_size', self.TARGET_SIZE)
            
            output_images_dir = os.path.join(self.output_base, "unet", "images", "train")
            output_masks_dir = os.path.join(self.output_base, "unet", "masks", "train")
            output_valid_dir = os.path.join(self.output_base, "unet", "valid", "train")
            
            # Klasörleri oluştur
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_masks_dir, exist_ok=True)
            os.makedirs(output_valid_dir, exist_ok=True)
            
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # 576x320 letterbox görseli kaydet
            output_image_path = os.path.join(output_images_dir, image_filename)
            letterbox_image.save(output_image_path)
            
            # Binary mask (object mask) oluştur ve kaydet (dosya adı aynı, sadece uzantı .png)
            binary_mask = create_binary_mask(edited_data, target_size[0], target_size[1])
            mask_filename = f"{image_name}.png"
            output_mask_path = os.path.join(output_masks_dir, mask_filename)
            binary_mask.save(output_mask_path)
            
            # Valid mask oluştur ve kaydet - tamamen beyaz (576x320)
            valid_mask = Image.new('L', target_size, 255)  # Tamamen beyaz
            valid_filename = f"{image_name}.png"
            output_valid_path = os.path.join(output_valid_dir, valid_filename)
            valid_mask.save(output_valid_path)
            
            print(f"[BAŞARILI] UNet'e kaydedildi: {output_image_path}, {output_mask_path}, {output_valid_path}")
            return True
            
        except Exception as e:
            print(f"[HATA] UNet'e kaydetme hatası: {str(e)}")
            return False
