"""
Görsel işleme utility fonksiyonları
"""

from PIL import Image, ImageDraw
import os


def crop_image_by_bbox(image_path: str, bbox: tuple) -> tuple:
    """
    Bounding box'a göre görseli kırpar (padding ile).
    
    Args:
        image_path: Görsel dosyasının yolu
        bbox: (x1, y1, x2, y2) bounding box koordinatları
        
    Returns:
        (cropped_image, crop_box) tuple
        crop_box: (x1, y1, x2, y2) kırpma koordinatları (orijinal görsele göre)
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    x1, y1, x2, y2 = bbox
    
    # Box boyutları
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Padding hesapla ve hemen int'e çevir
    # Her kenar için kendi boyutunun %10'u kadar padding
    padding_left_right = int(round(box_width * 0.1))  # Sol ve sağ için genişliğin %10'u
    padding_top_bottom = int(round(box_height * 0.1))  # Üst ve alt için yüksekliğin %10'u
    
    # Yeni koordinatlar (int olarak hesapla)
    x1_new = max(0, int(round(x1 - padding_left_right)))  # Sol
    y1_new = max(0, int(round(y1 - padding_top_bottom)))  # Üst
    x2_new = min(img_width, int(round(x2 + padding_left_right)))  # Sağ
    y2_new = min(img_height, int(round(y2 + padding_top_bottom)))  # Alt
    
    # Kırp
    crop_box = (x1_new, y1_new, x2_new, y2_new)
    cropped_img = img.crop(crop_box)
    
    print(f"[BİLGİ] Görsel kırpıldı: {crop_box}, Boyut: {cropped_img.size}")
    
    return cropped_img, crop_box


def adjust_polygons_to_crop(masks_data: list, crop_box: tuple) -> list:
    """
    Polygon koordinatlarını kırpılmış görsele göre ayarlar.
    
    Args:
        masks_data: Orijinal görsele göre polygon'lar
        crop_box: (x1, y1, x2, y2) kırpma koordinatları
        
    Returns:
        Kırpılmış görsele göre ayarlanmış polygon'lar
    """
    x1_crop, y1_crop, x2_crop, y2_crop = crop_box
    adjusted_masks = []
    
    for mask_data in masks_data:
        polygon = mask_data['polygon']
        adjusted_polygon = []
        
        crop_width = x2_crop - x1_crop
        crop_height = y2_crop - y1_crop
        
        for x, y in polygon:
            # Kırpma offset'ini çıkar
            new_x = x - x1_crop
            new_y = y - y1_crop
            
            # Noktaları clamp et (crop sınırları içinde tut)
            # Crop dışındaki noktaları atma, clamp et
            new_x = max(0.0, min(float(crop_width), new_x))
            new_y = max(0.0, min(float(crop_height), new_y))
            adjusted_polygon.append((new_x, new_y))
        
        # En az 3 nokta varsa ekle
        if len(adjusted_polygon) >= 3:
            adjusted_masks.append({
                'polygon': adjusted_polygon,
                'class_id': mask_data.get('class_id', 0),
                'confidence': mask_data.get('confidence', 0.0)
            })
    
    return adjusted_masks


def adjust_polygons_from_crop(masks_data: list, crop_box: tuple) -> list:
    """
    Polygon koordinatlarını kırpılmış görselden orijinal görsele göre ayarlar.
    
    Args:
        masks_data: Kırpılmış görsele göre polygon'lar
        crop_box: (x1, y1, x2, y2) kırpma koordinatları
        
    Returns:
        Orijinal görsele göre ayarlanmış polygon'lar
    """
    x1_crop, y1_crop, x2_crop, y2_crop = crop_box
    adjusted_masks = []
    
    for mask_data in masks_data:
        polygon = mask_data['polygon']
        adjusted_polygon = []
        
        for x, y in polygon:
            # Kırpma offset'ini ekle
            new_x = x + x1_crop
            new_y = y + y1_crop
            adjusted_polygon.append((new_x, new_y))
        
        # En az 3 nokta varsa ekle
        if len(adjusted_polygon) >= 3:
            adjusted_masks.append({
                'polygon': adjusted_polygon,
                'class_id': mask_data.get('class_id', 0),
                'confidence': mask_data.get('confidence', 0.0)
            })
    
    return adjusted_masks


def letterbox_resize(image: Image.Image, target_size: tuple = (320, 320)) -> tuple:
    """
    Görseli letterbox ile resize eder (aspect ratio korunur).
    
    Args:
        image: PIL Image
        target_size: (width, height) hedef boyut
        
    Returns:
        (resized_image, offset_x, offset_y, scale, new_width, new_height) tuple
        offset: Görselin target_size içindeki konumu
        scale: Uygulanan ölçek faktörü
        new_width, new_height: Resize edilmiş görselin boyutları
    """
    target_width, target_height = target_size
    img_width, img_height = image.size
    
    # Aspect ratio'yu koru
    scale = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    # Resize et - IMAGE için BILINEAR kullan (LINEAR interpolation)
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Yeni görsel oluştur (siyah arka plan)
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    
    # Merkeze yerleştir
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    new_image.paste(resized, (offset_x, offset_y))
    
    return new_image, offset_x, offset_y, scale, new_width, new_height


def create_binary_mask(masks_data: list, width: int, height: int) -> Image.Image:
    """
    Polygon'lardan binary mask oluşturur.
    
    Args:
        masks_data: [{'polygon': [(x, y), ...], ...}, ...] maske verileri
        width: Mask genişliği
        height: Mask yüksekliği
        
    Returns:
        Binary mask (PIL Image, grayscale, 0=siyah, 255=beyaz)
    """
    # Siyah mask oluştur
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Her polygon için
    for mask_data in masks_data:
        polygon = mask_data.get('polygon', [])
        
        if len(polygon) < 3:
            continue
        
        # Polygon'u çiz (beyaz - 255)
        draw.polygon(polygon, fill=255)
    
    return mask


def resize_mask(mask: Image.Image, target_size: tuple) -> Image.Image:
    """
    Mask'ı resize eder (NEAREST interpolation kullanır - mask bozulmasın diye).
    
    Args:
        mask: PIL Image (grayscale mask)
        target_size: (width, height) hedef boyut
        
    Returns:
        Resize edilmiş mask
    """
    return mask.resize(target_size, Image.Resampling.NEAREST)


def create_valid_mask(
    target_size: tuple,
    offset_x: int,
    offset_y: int,
    roi_width: int,
    roi_height: int
) -> Image.Image:
    """
    Valid mask oluşturur (padding alanlarını ignore etmek için).
    
    Valid mask: ROI'den gelen alan = beyaz (255), padding alanları = siyah (0)
    Bu maske, loss hesaplamasında padding alanlarını ignore etmek için kullanılır.
    
    Args:
        target_size: (width, height) hedef boyut (örn: 320, 320)
        offset_x: ROI'nin target_size içindeki x konumu
        offset_y: ROI'nin target_size içindeki y konumu
        roi_width: Resize edilmiş ROI genişliği
        roi_height: Resize edilmiş ROI yüksekliği
        
    Returns:
        Binary valid mask (PIL Image, grayscale, 0=siyah, 255=beyaz)
    """
    target_width, target_height = target_size
    
    # Siyah mask oluştur
    valid_mask = Image.new('L', (target_width, target_height), 0)
    draw = ImageDraw.Draw(valid_mask)
    
    # ROI alanını beyaz yap (dikdörtgen)
    # PIL'in rectangle çizimi inclusive'dir: [x1, y1, x2, y2] şeklinde
    # x2 ve y2 dahildir, bu yüzden x2 = offset_x + roi_width - 1, y2 = offset_y + roi_height - 1
    x1 = offset_x
    y1 = offset_y
    x2 = offset_x + roi_width - 1  # PIL'in inclusive çizimi için -1
    y2 = offset_y + roi_height - 1  # PIL'in inclusive çizimi için -1
    
    # Sınırları kontrol et (target_size içinde olmalı)
    x1 = max(0, min(target_width - 1, x1))
    y1 = max(0, min(target_height - 1, y1))
    x2 = max(0, min(target_width - 1, x2))
    y2 = max(0, min(target_height - 1, y2))
    
    # ROI alanını beyaz yap
    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    return valid_mask
