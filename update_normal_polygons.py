"""
Normal mod verilerindeki polygon koordinatlarını içeri çekme scripti
Güncellenmiş verileri yeni bir klasöre kaydeder (orijinal veriler korunur)
"""

import os
import shutil
import random
import glob
from PIL import Image
import numpy as np


def shrink_polygon(polygon, shrink_factor=0.05):
    """
    Polygon'u merkeze doğru küçültür (shrink).
    
    Args:
        polygon: [(x, y), ...] nokta listesi
        shrink_factor: Küçültme faktörü (0.05 = %5 içeri çek, 0.1 = %10 içeri çek)
        
    Returns:
        Küçültülmüş polygon noktaları
    """
    if len(polygon) < 3:
        return polygon
    
    # Polygon'u numpy array'e çevir
    points = np.array(polygon, dtype=np.float64)
    
    # Centroid'i hesapla (ağırlık merkezi)
    centroid = np.mean(points, axis=0)
    
    # Her noktayı centroid'e doğru hareket ettir
    shrunk_points = []
    for point in points:
        # Centroid'den noktaya olan vektör
        vector = point - centroid
        # Vektörü shrink_factor kadar küçült
        new_point = centroid + vector * (1 - shrink_factor)
        shrunk_points.append((float(new_point[0]), float(new_point[1])))
    
    return shrunk_points


def split_train_valid(output_base, valid_ratio=0.1):
    """
    Güncellenmiş verileri train ve valid olarak ayırır.
    
    Args:
        output_base: Çıktı klasörü base yolu
        valid_ratio: Validation oranı (0.1 = %10)
        
    Returns:
        Validation'a taşınan dosya sayısı
    """
    images_train_dir = os.path.join(output_base, "images", "train")
    labels_train_dir = os.path.join(output_base, "labels", "train")
    images_valid_dir = os.path.join(output_base, "images", "valid")
    labels_valid_dir = os.path.join(output_base, "labels", "valid")
    
    if not os.path.exists(images_train_dir) or not os.path.exists(labels_train_dir):
        print(f"[UYARI] Train klasörleri bulunamadı")
        return 0
    
    # Valid klasörlerini oluştur
    os.makedirs(images_valid_dir, exist_ok=True)
    os.makedirs(labels_valid_dir, exist_ok=True)
    
    # Train klasöründeki görsel dosyalarını bul
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_train_dir, ext)))
    
    if len(image_files) == 0:
        print(f"[UYARI] Train klasöründe görsel bulunamadı")
        return 0
    
    # Valid için dosya sayısını hesapla (%10, en az 1 dosya)
    num_valid = max(1, int(len(image_files) * valid_ratio))
    
    # Rastgele karıştır
    random.shuffle(image_files)
    valid_images = image_files[:num_valid]
    
    print(f"[BİLGİ] Toplam görsel: {len(image_files)}")
    print(f"[BİLGİ] Train: {len(image_files) - num_valid} (%{(1-valid_ratio)*100:.1f})")
    print(f"[BİLGİ] Valid: {num_valid} (%{valid_ratio*100:.1f})")
    
    moved_count = 0
    for image_path in valid_images:
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        
        # Görseli valid klasörüne taşı
        dest_image = os.path.join(images_valid_dir, image_filename)
        try:
            shutil.move(image_path, dest_image)
        except Exception as e:
            print(f"[HATA] Görsel taşınamadı {image_filename}: {str(e)}")
            continue
        
        # Label dosyasını valid klasörüne taşı
        label_file = os.path.join(labels_train_dir, f"{image_name}.txt")
        if os.path.exists(label_file):
            dest_label = os.path.join(labels_valid_dir, f"{image_name}.txt")
            try:
                shutil.move(label_file, dest_label)
            except Exception as e:
                print(f"[HATA] Label taşınamadı {image_name}.txt: {str(e)}")
                # Görseli geri taşı
                try:
                    shutil.move(dest_image, image_path)
                except:
                    pass
                continue
        
        moved_count += 1
        
        if moved_count % 50 == 0:
            print(f"[İLERLEME] {moved_count}/{num_valid} dosya validation'a taşındı...")
    
    return moved_count


def update_normal_labels(shrink_factor=0.05, output_suffix="_shrunk"):
    """
    Normal mod label dosyalarını günceller - polygon'ları içeri çeker.
    Güncellenmiş verileri yeni bir klasöre kaydeder.
    
    Args:
        shrink_factor: Küçültme faktörü (0.05 = %5, 0.1 = %10)
        output_suffix: Çıktı klasörü için suffix (örn: "_shrunk")
    """
    base_dir = "/home/acomaster/Belgeler/AITrain/output/normal"
    images_dir = os.path.join(base_dir, "images", "train")
    labels_dir = os.path.join(base_dir, "labels", "train")
    
    # Yeni çıktı klasörleri
    output_base = "/home/acomaster/Belgeler/AITrain/output/normal" + output_suffix
    output_images_dir = os.path.join(output_base, "images", "train")
    output_labels_dir = os.path.join(output_base, "labels", "train")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"[HATA] Klasörler bulunamadı: {images_dir} veya {labels_dir}")
        return
    
    # Çıktı klasörlerini oluştur
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"[BİLGİ] Çıktı klasörü: {output_base}")
    
    # Tüm label dosyalarını bul
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    if not label_files:
        print("[UYARI] Label dosyası bulunamadı")
        return
    
    print(f"[BİLGİ] {len(label_files)} label dosyası bulundu")
    print(f"[BİLGİ] Shrink faktörü: {shrink_factor*100:.1f}%")
    
    updated_count = 0
    skipped_count = 0
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        image_name = os.path.splitext(label_file)[0]
        
        # İlgili görseli bul
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(images_dir, image_name + ext)
            if os.path.exists(potential_path):
                image_file = potential_path
                break
        
        if not image_file:
            print(f"[UYARI] Görsel bulunamadı: {image_name}")
            skipped_count += 1
            continue
        
        # Görseli yeni klasöre kopyala
        try:
            output_image_path = os.path.join(output_images_dir, os.path.basename(image_file))
            shutil.copy2(image_file, output_image_path)
        except Exception as e:
            print(f"[HATA] Görsel kopyalanamadı {image_file}: {str(e)}")
            skipped_count += 1
            continue
        
        # Görsel boyutlarını al
        try:
            img = Image.open(image_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"[HATA] Görsel açılamadı {image_file}: {str(e)}")
            skipped_count += 1
            continue
        
        # Label dosyasını oku
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[HATA] Label dosyası okunamadı {label_path}: {str(e)}")
            skipped_count += 1
            continue
        
        # Her satırı işle (her satır bir polygon)
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 7:  # class_id + en az 3 nokta (6 koordinat) = 7
                updated_lines.append(line)
                continue
            
            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]
            
            # Normalize koordinatları piksel koordinatlarına çevir
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * img_width
                    y = coords[i + 1] * img_height
                    polygon.append((x, y))
            
            if len(polygon) < 3:
                updated_lines.append(line)
                continue
            
            # Polygon'u küçült
            shrunk_polygon = shrink_polygon(polygon, shrink_factor)
            
            # Tekrar normalize et
            normalized_coords = []
            for x, y in shrunk_polygon:
                norm_x = x / img_width
                norm_y = y / img_height
                # Sınırları kontrol et
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))
                normalized_coords.extend([norm_x, norm_y])
            
            # Yeni satırı oluştur
            new_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
            updated_lines.append(new_line)
        
        # Yeni label dosyasını yaz
        output_label_path = os.path.join(output_labels_dir, label_file)
        try:
            with open(output_label_path, 'w') as f:
                for line in updated_lines:
                    f.write(line + "\n")
            updated_count += 1
            
            if updated_count % 100 == 0:
                print(f"[İLERLEME] {updated_count} dosya işlendi...")
        except Exception as e:
            print(f"[HATA] Dosya yazılamadı {output_label_path}: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"\n[BAŞARILI] {updated_count} dosya güncellendi ve '{output_base}' klasörüne kaydedildi")
    if skipped_count > 0:
        print(f"[UYARI] {skipped_count} dosya atlandı")
    
    # Train/Valid split yap (%90 train, %10 valid)
    print("\n" + "=" * 60)
    print("Train/Valid Split İşlemi Başlatılıyor...")
    print("=" * 60)
    split_count = split_train_valid(output_base, valid_ratio=0.1)
    print(f"[BAŞARILI] {split_count} dosya validation'a taşındı")


if __name__ == "__main__":
    # Shrink faktörünü buradan ayarlayabilirsiniz
    # 0.03 = %3 içeri çek (hafif)
    # 0.05 = %5 içeri çek (orta) - önerilen
    # 0.1 = %10 içeri çek (daha fazla)
    shrink_factor = 0.03  # %5 içeri çek
    
    # Çıktı klasörü için suffix
    output_suffix = "_shrunk"  # "normal_shrunk" klasörüne kaydedilecek
    
    print("=" * 60)
    print("Normal Mod Verilerini Güncelleme - Polygon Küçültme")
    print("=" * 60)
    print(f"Shrink faktörü: {shrink_factor*100:.1f}%")
    print(f"Çıktı klasörü: output/normal{output_suffix}")
    print("=" * 60)
    
    update_normal_labels(shrink_factor, output_suffix)
    
    print("=" * 60)
    print("İşlem tamamlandı!")
    print("=" * 60)
