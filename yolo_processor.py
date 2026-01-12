"""
YOLOv8 Model İşleme Modülü
YOLOv8 segmentasyon modelini yükler ve görselleri işler.
"""

from ultralytics import YOLO
import numpy as np
import cv2
import warnings
import torch
import sys
import traceback
from typing import List, Tuple, Optional

# CUDA uyarılarını bastır
warnings.filterwarnings('ignore', category=UserWarning)


def simplify_polygon(polygon: list, epsilon_factor: float = 0.01) -> list:
    """
    Polygon noktalarını azaltır (Douglas-Peucker algoritması).
    
    Args:
        polygon: [(x, y), ...] nokta listesi
        epsilon_factor: Basitleştirme faktörü (0.01 = çevrenin %1'i, daha büyük = daha az nokta)
                       Önerilen: 0.01-0.02 (az nokta), 0.005-0.01 (orta), <0.005 (çok nokta)
        
    Returns:
        Basitleştirilmiş polygon noktaları
    """
    if len(polygon) < 4:
        return polygon
    
    try:
        # NumPy array'e çevir
        points = np.array(polygon, dtype=np.float32)
        
        # Polygon çevresini hesapla
        perimeter = cv2.arcLength(points, closed=True)
        
        # Epsilon değeri (çevrenin belirli bir yüzdesi)
        epsilon = epsilon_factor * perimeter
        
        # Douglas-Peucker algoritması ile basitleştir
        simplified = cv2.approxPolyDP(points, epsilon, closed=True)
        
        # Listeye çevir
        simplified_polygon = [(float(pt[0]), float(pt[1])) for pt in simplified.reshape(-1, 2)]
        
        # En az 3 nokta olmalı
        if len(simplified_polygon) < 3:
            return polygon
        
        return simplified_polygon
    except Exception as e:
        print(f"[UYARI] Polygon basitleştirme hatası: {str(e)}")
        return polygon


class YOLOProcessor:
    """YOLOv8 segmentasyon modeli için işlemci sınıfı"""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        YOLOv8 modelini yükler.
        
        Args:
            model_path: .pt model dosyasının yolu
            device: 'auto', 'cpu', 'cuda', 'cuda:0' gibi cihaz seçimi
        """
        try:
            # Cihaz seçimi: sadece GPU
            if not torch.cuda.is_available():
                device = 'cpu'
                raise Exception("CUDA GPU bulunamadı. Lütfen uyumlu bir NVIDIA GPU ve uygun CUDA/PyTorch kurulumu sağlayın.")
            if device == 'auto':
                device = 'cuda:0'
            
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.device = device
            print(f"[BİLGİ] Model yüklendi: {model_path}")
            print(f"[BİLGİ] Cihaz: {device}")
        except Exception as e:
            error_msg = f"Model yüklenemedi: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            raise Exception(error_msg)
    
    def process_image(self, image_path: str, epsilon_factor: float = 0.002) -> dict:
        """
        Görseli YOLOv8 modeli ile işler ve segmentasyon sonuçlarını döndürür.
        
        Args:
            image_path: İşlenecek görselin yolu
            epsilon_factor: Polygon basitleştirme faktörü (0.001-0.1 arası, varsayılan: 0.002)
            
        Returns:
            {
                'masks': [
                    {
                        'polygon': [(x1, y1), (x2, y2), ...],  # Piksel koordinatları
                        'class_id': int,
                        'confidence': float
                    },
                    ...
                ],
                'boxes': [(x1, y1, x2, y2), ...]  # Bounding box koordinatları (piksel)
            }
        """
        try:
            # Model ile inference yap (device parametresi ile)
            results = self.model(image_path, verbose=True, device=self.device,retina_masks=True)
            
            masks_data = []
            bounding_boxes = []
            
            # Segmentasyon sonuçlarını işle
            if results and len(results) > 0:
                result = results[0]
                
                # Maske varsa işle
                if result.masks is not None:
                    masks = result.masks
                    boxes = result.boxes
                    
                    # Her maske için polygon koordinatlarını al
                    for i in range(len(masks)):
                        polygon = []
                        
                        # Maske polygon koordinatlarını al (piksel cinsinden)
                        mask = masks[i]
                        
                        # YOLOv8 maske formatı: mask.xy veya mask.data
                        if hasattr(mask, 'xy'):
                            # xy formatı: numpy array veya list, shape: (n_points, 2)
                            xy_data = mask.xy
                            
                            # numpy array ise
                            if isinstance(xy_data, np.ndarray):
                                if len(xy_data) > 0:
                                    # Eğer 3D array ise (n_contours, n_points, 2), ilk contour'u al
                                    if len(xy_data.shape) == 3:
                                        contour = xy_data[0]
                                    # Eğer 2D array ise (n_points, 2), direkt kullan
                                    elif len(xy_data.shape) == 2:
                                        contour = xy_data
                                    else:
                                        contour = xy_data.flatten().reshape(-1, 2)
                                    
                                    polygon_points = contour.tolist()
                                    polygon = [(float(x), float(y)) for x, y in polygon_points]
                            # list ise
                            elif isinstance(xy_data, list):
                                if len(xy_data) > 0:
                                    # İlk eleman numpy array olabilir
                                    if isinstance(xy_data[0], np.ndarray):
                                        contour = xy_data[0]
                                        polygon_points = contour.tolist()
                                    # Direkt list of lists
                                    elif isinstance(xy_data[0], (list, tuple)):
                                        polygon_points = xy_data
                                    else:
                                        # Tek boyutlu list, reshape et
                                        polygon_points = np.array(xy_data).reshape(-1, 2).tolist()
                                    
                                    polygon = [(float(x), float(y)) for x, y in polygon_points]
                            else:
                                # Diğer formatlar için dene
                                try:
                                    polygon_points = np.array(xy_data).reshape(-1, 2).tolist()
                                    polygon = [(float(x), float(y)) for x, y in polygon_points]
                                except:
                                    polygon = []
                        elif hasattr(mask, 'data'):
                            # Alternatif format
                            try:
                                import cv2
                                # Binary mask'dan contour çıkar
                                mask_array = mask.data.cpu().numpy() if hasattr(mask.data, 'cpu') else mask.data
                                if len(mask_array.shape) > 2:
                                    mask_array = mask_array[0]
                                contours, _ = cv2.findContours(
                                    (mask_array * 255).astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE
                                )
                                if len(contours) > 0:
                                    # En büyük contour'u al
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
                            except:
                                polygon = []
                        
                        # Polygon'u basitleştir (nokta sayısını azalt)
                        original_count = len(polygon)
                        if original_count > 10:  # Sadece 10'dan fazla nokta varsa basitleştir
                            polygon = simplify_polygon(polygon, epsilon_factor=epsilon_factor)
                            if len(polygon) < original_count:
                                print(f"[BİLGİ] Polygon basitleştirildi: {original_count} -> {len(polygon)} nokta (epsilon={epsilon_factor:.3f})")
                        
                        # Sınıf ID ve güven skoru
                        class_id = int(boxes.cls[i]) if boxes is not None and len(boxes.cls) > i else 0
                        confidence = float(boxes.conf[i]) if boxes is not None and len(boxes.conf) > i else 0.0
                        
                        if len(polygon) >= 3:  # En az 3 nokta gerekli
                            masks_data.append({
                                'polygon': polygon,
                                'class_id': class_id,
                                'confidence': confidence
                            })
                    
                    # Bounding box'ları al
                    if boxes is not None:
                        for i in range(len(boxes)):
                            # Bounding box koordinatları (xyxy formatı)
                            box = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else boxes.xyxy[i]
                            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                            bounding_boxes.append((x1, y1, x2, y2))
            
            print(f"[BİLGİ] Görsel işlendi: {image_path}, {len(masks_data)} maske, {len(bounding_boxes)} bounding box bulundu")
            return {
                'masks': masks_data,
                'boxes': bounding_boxes
            }
            
        except Exception as e:
            error_msg = f"Görsel işlenirken hata oluştu: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            print(f"[HATA] Görsel: {image_path}", file=sys.stderr)
            traceback.print_exc()
            raise Exception(error_msg)
    
    def get_model_info(self) -> dict:
        """
        Model bilgilerini döndürür.
        
        Returns:
            Model bilgilerini içeren dict
        """
        return {
            'path': self.model_path,
            'classes': self.model.names if hasattr(self.model, 'names') else {}
        }
