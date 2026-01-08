"""
Görsel Düzenleme ve Nokta Yönetimi Modülü
Canvas üzerinde görsel gösterimi, polygon çizimi ve nokta düzenleme işlevleri.
"""

import tkinter as tk
from PIL import Image, ImageTk
import math
import sys
import traceback


class ImageEditor:
    """Canvas üzerinde görsel ve maske düzenleme sınıfı"""
    
    def __init__(self, canvas: tk.Canvas):
        """
        ImageEditor'ı başlatır.
        
        Args:
            canvas: Tkinter Canvas widget'ı
        """
        self.canvas = canvas
        self.image = None
        self.image_tk = None
        self.image_path = None
        self.image_width = 0
        self.image_height = 0
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Polygon ve nokta verileri
        self.polygons = []  # Her polygon için: {'points': [(x,y), ...], 'class_id': int}
        self.polygon_items = []  # Canvas item ID'leri
        self.point_items = []  # Nokta item ID'leri
        
        # Seçili nokta
        self.selected_point = None
        self.selected_polygon_idx = None
        self.selected_point_idx = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Son tıklama konumu (nokta ekleme için)
        self.last_click_x = None
        self.last_click_y = None
        
        # Pan (kaydırma) için
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_offset_x = 0
        self.pan_start_offset_y = 0
        
        # Mouse event'lerini bağla
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # Space tuşu event'leri (pan için)
        self.canvas.bind("<KeyPress>", self.on_key_press)
        self.canvas.bind("<KeyRelease>", self.on_key_release)
        
        # Mouse wheel zoom event'leri (Linux ve Windows/Mac desteği)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        
        # Canvas'a focus ver (mouse wheel event'lerini almak için)
        self.canvas.focus_set()
        
        # Nokta yakalama mesafesi (piksel)
        self.point_capture_distance = 10
        
        # Zoom ayarları
        self.min_scale = 0.1  # Minimum zoom (10%)
        self.max_scale = 5.0  # Maximum zoom (500%)
        self.zoom_factor = 1.1  # Her scroll'da %10 zoom
    
    def display_image(self, image_path: str, masks_data: list):
        """
        Görseli ve maskeleri canvas'a çizer.
        
        Args:
            image_path: Görsel dosyasının yolu
            masks_data: YOLOProcessor'dan gelen maske verileri
        """
        self.image_path = image_path
        
        # Canvas'ı temizle
        self.canvas.delete("all")
        self.polygon_items = []
        self.point_items = []
        
        # Görseli yükle
        try:
            self.image = Image.open(image_path)
            self.image_width, self.image_height = self.image.size
            
            # Canvas boyutlarını al
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas henüz render edilmemiş, varsayılan boyutlar kullan
                canvas_width = 800
                canvas_height = 600
            
            # Görseli canvas'a sığdır (aspect ratio korunarak)
            scale_w = canvas_width / self.image_width
            scale_h = canvas_height / self.image_height
            self.scale_factor = min(scale_w, scale_h, 1.0)  # Büyütme yapma, sadece küçült
            
            new_width = int(self.image_width * self.scale_factor)
            new_height = int(self.image_height * self.scale_factor)
            
            # Merkeze hizala
            self.offset_x = (canvas_width - new_width) // 2
            self.offset_y = (canvas_height - new_height) // 2
            
            # Görseli ölçekle
            resized_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(resized_image)
            
            # Canvas'a çiz
            self.canvas.create_image(
                self.offset_x + new_width // 2,
                self.offset_y + new_height // 2,
                image=self.image_tk,
                anchor=tk.CENTER
            )
            
            # Canvas'ı güncelle
            self.canvas.update_idletasks()
            
            # Polygon'ları çiz
            self.polygons = []
            for mask_data in masks_data:
                polygon_points = mask_data.get('polygon', [])
                if polygon_points:
                    # Piksel koordinatlarını canvas koordinatlarına çevir
                    canvas_points = self.pixel_to_canvas(polygon_points)
                    self.polygons.append({
                        'points': canvas_points,
                        'original_points': polygon_points,  # Orijinal piksel koordinatları
                        'class_id': mask_data.get('class_id', 0)
                    })
            
            self.draw_polygons()
            
        except Exception as e:
            error_msg = f"Görsel yüklenirken hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            print(f"[HATA] Görsel: {image_path}", file=sys.stderr)
            traceback.print_exc()
    
    def pixel_to_canvas(self, points: list) -> list:
        """
        Piksel koordinatlarını canvas koordinatlarına çevirir.
        
        Args:
            points: [(x, y), ...] piksel koordinatları
            
        Returns:
            Canvas koordinatları
        """
        canvas_points = []
        for x, y in points:
            canvas_x = self.offset_x + x * self.scale_factor
            canvas_y = self.offset_y + y * self.scale_factor
            canvas_points.append((canvas_x, canvas_y))
        return canvas_points
    
    def canvas_to_pixel(self, canvas_x: float, canvas_y: float) -> tuple:
        """
        Canvas koordinatlarını piksel koordinatlarına çevirir.
        
        Args:
            canvas_x: Canvas x koordinatı
            canvas_y: Canvas y koordinatı
            
        Returns:
            (pixel_x, pixel_y) tuple
        """
        pixel_x = (canvas_x - self.offset_x) / self.scale_factor
        pixel_y = (canvas_y - self.offset_y) / self.scale_factor
        return (pixel_x, pixel_y)
    
    def draw_polygons(self):
        """Tüm polygon'ları ve noktalarını çizer."""
        # Önce eski çizimleri temizle
        for item in self.polygon_items + self.point_items:
            self.canvas.delete(item)
        self.polygon_items = []
        self.point_items = []
        
        # Her polygon için
        for poly_idx, polygon in enumerate(self.polygons):
            points = polygon['points']
            if len(points) < 3:
                continue
            
            # Polygon çiz (yeşil renk)
            polygon_id = self.canvas.create_polygon(
                [coord for point in points for coord in point],
                outline='green',
                fill='',
                width=2,
                tags='polygon'
            )
            self.polygon_items.append(polygon_id)
            
            # Noktaları çiz
            for point_idx, (x, y) in enumerate(points):
                # Nokta rengi: seçili ise kırmızı, değilse mavi
                color = 'red' if (poly_idx == self.selected_polygon_idx and 
                                 point_idx == self.selected_point_idx) else 'blue'
                
                point_id = self.canvas.create_oval(
                    x - 5, y - 5, x + 5, y + 5,
                    fill=color,
                    outline='white',
                    width=2,
                    tags=('point', f'poly_{poly_idx}_point_{point_idx}')
                )
                self.point_items.append(point_id)
    
    def find_nearest_point(self, x: float, y: float) -> tuple:
        """
        Verilen koordinata en yakın noktayı bulur.
        
        Args:
            x: Canvas x koordinatı
            y: Canvas y koordinatı
            
        Returns:
            (polygon_idx, point_idx, distance) veya (None, None, None)
        """
        min_distance = self.point_capture_distance
        nearest_poly = None
        nearest_point = None
        
        for poly_idx, polygon in enumerate(self.polygons):
            for point_idx, (px, py) in enumerate(polygon['points']):
                distance = math.sqrt((x - px)**2 + (y - py)**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_poly = poly_idx
                    nearest_point = point_idx
        
        return (nearest_poly, nearest_point, min_distance if nearest_poly is not None else None)
    
    def on_mouse_click(self, event):
        """Mouse tıklama event'i"""
        # Canvas'a focus ver (space tuşu event'lerini almak için)
        self.canvas.focus_set()
        
        x, y = event.x, event.y
        
        # Space tuşu basılıysa pan moduna geç
        if self.is_panning:
            self.pan_start_x = x
            self.pan_start_y = y
            self.pan_start_offset_x = self.offset_x
            self.pan_start_offset_y = self.offset_y
            return
        
        # Son tıklama konumunu kaydet
        self.last_click_x = x
        self.last_click_y = y
        
        # En yakın noktayı bul
        poly_idx, point_idx, distance = self.find_nearest_point(x, y)
        
        if poly_idx is not None:
            # Bir nokta seçildi
            self.selected_polygon_idx = poly_idx
            self.selected_point_idx = point_idx
            self.drag_start_x = x
            self.drag_start_y = y
            self.draw_polygons()  # Seçili noktayı güncellemek için yeniden çiz
        else:
            # Nokta seçilmedi, seçimi temizle
            self.selected_polygon_idx = None
            self.selected_point_idx = None
    
    def on_mouse_drag(self, event):
        """Mouse sürükleme event'i"""
        x, y = event.x, event.y
        
        # Space tuşu basılıysa pan (kaydırma) yap
        if self.is_panning:
            # Pan offset'ini hesapla
            dx = x - self.pan_start_x
            dy = y - self.pan_start_y
            
            self.offset_x = self.pan_start_offset_x + dx
            self.offset_y = self.pan_start_offset_y + dy
            
            # Görseli ve polygon'ları yeniden çiz
            self.redraw_image()
            return
        
        # Normal nokta sürükleme
        if self.selected_polygon_idx is not None and self.selected_point_idx is not None:
            # Noktayı güncelle
            polygon = self.polygons[self.selected_polygon_idx]
            polygon['points'][self.selected_point_idx] = (x, y)
            
            # Orijinal piksel koordinatlarını da güncelle
            pixel_x, pixel_y = self.canvas_to_pixel(x, y)
            polygon['original_points'][self.selected_point_idx] = (pixel_x, pixel_y)
            
            # Yeniden çiz
            self.draw_polygons()
    
    def on_mouse_release(self, event):
        """Mouse bırakma event'i"""
        # Sürükleme bitti, seçimi koru
        # Pan modu space tuşu bırakıldığında temizlenecek
        pass
    
    def on_key_press(self, event):
        """Tuş basıldığında"""
        if event.keysym == "space":
            self.is_panning = True
            # Cursor'u değiştir (opsiyonel)
            self.canvas.config(cursor="hand2")
    
    def on_key_release(self, event):
        """Tuş bırakıldığında"""
        if event.keysym == "space":
            self.is_panning = False
            # Cursor'u normale döndür
            self.canvas.config(cursor="")
    
    def on_mouse_wheel(self, event):
        """Mouse wheel zoom event'i"""
        if self.image is None:
            return
        
        # Zoom yönünü belirle
        if event.num == 4 or event.delta > 0:
            # Zoom in (scroll up)
            new_scale = self.scale_factor * self.zoom_factor
        elif event.num == 5 or event.delta < 0:
            # Zoom out (scroll down)
            new_scale = self.scale_factor / self.zoom_factor
        else:
            return
        
        # Scale sınırlarını kontrol et
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        
        if new_scale == self.scale_factor:
            return  # Değişiklik yok
        
        # Mouse pozisyonunu al (zoom merkezi)
        mouse_x = event.x
        mouse_y = event.y
        
        # Mouse pozisyonunu piksel koordinatlarına çevir
        pixel_x, pixel_y = self.canvas_to_pixel(mouse_x, mouse_y)
        
        # Yeni scale'i uygula
        old_scale = self.scale_factor
        self.scale_factor = new_scale
        
        # Yeni görsel boyutları
        new_width = int(self.image_width * self.scale_factor)
        new_height = int(self.image_height * self.scale_factor)
        
        # Canvas boyutlarını al
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Mouse pozisyonunu yeni scale'e göre offset'i hesapla
        # Mouse pozisyonu sabit kalmalı, bu yüzden offset'i ayarla
        new_offset_x = mouse_x - pixel_x * self.scale_factor
        new_offset_y = mouse_y - pixel_y * self.scale_factor
        
        # Offset'i canvas sınırları içinde tut
        # Görsel canvas'tan taşmamalı (opsiyonel, taşmasına izin verebiliriz)
        self.offset_x = new_offset_x
        self.offset_y = new_offset_y
        
        # Görseli ve polygon'ları yeniden çiz
        self.redraw_image()
    
    def redraw_image(self):
        """Görseli ve polygon'ları yeniden çizer (zoom sonrası)"""
        if self.image is None:
            return
        
        # Canvas'ı temizle
        self.canvas.delete("all")
        self.polygon_items = []
        self.point_items = []
        
        # Yeni görsel boyutları
        new_width = int(self.image_width * self.scale_factor)
        new_height = int(self.image_height * self.scale_factor)
        
        # Görseli ölçekle
        resized_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(resized_image)
        
        # Canvas'a çiz
        self.canvas.create_image(
            self.offset_x + new_width // 2,
            self.offset_y + new_height // 2,
            image=self.image_tk,
            anchor=tk.CENTER
        )
        
        # Polygon'ları canvas koordinatlarına çevir ve çiz
        for polygon in self.polygons:
            # Orijinal piksel koordinatlarını canvas koordinatlarına çevir
            canvas_points = self.pixel_to_canvas(polygon['original_points'])
            polygon['points'] = canvas_points
        
        # Polygon'ları çiz
        self.draw_polygons()
    
    def add_point_at_last_click(self) -> bool:
        """
        Son tıklama konumuna yeni nokta ekler.
        
        Returns:
            True if point was added, False otherwise
        """
        if self.last_click_x is None or self.last_click_y is None:
            print("[UYARI] Önce canvas'a tıklayarak nokta eklemek istediğiniz yeri seçin")
            return False
        
        if len(self.polygons) == 0:
            print("[UYARI] Eklemek için önce bir polygon olmalı")
            return False
        
        # İlk polygon'a ekle (birden fazla polygon varsa)
        polygon = self.polygons[0]
        x, y = self.last_click_x, self.last_click_y
        
        # Piksel koordinatlarına çevir
        pixel_x, pixel_y = self.canvas_to_pixel(x, y)
        
        # En yakın kenarı bul ve o kenara ekle
        points = polygon['points']
        if len(points) < 2:
            # İlk noktaları ekle
            polygon['points'].append((x, y))
            polygon['original_points'].append((pixel_x, pixel_y))
        else:
            # En yakın kenarı bul
            min_dist = float('inf')
            insert_idx = len(points)
            
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                # Nokta-çizgi mesafesi hesapla
                dist = self.point_to_line_distance((x, y), p1, p2)
                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1
            
            # Noktayı ekle
            polygon['points'].insert(insert_idx, (x, y))
            polygon['original_points'].insert(insert_idx, (pixel_x, pixel_y))
        
        # Yeniden çiz
        self.draw_polygons()
        print(f"[BİLGİ] Nokta eklendi: ({pixel_x:.1f}, {pixel_y:.1f})")
        return True
    
    def point_to_line_distance(self, point: tuple, line_start: tuple, line_end: tuple) -> float:
        """
        Nokta ile çizgi arasındaki mesafeyi hesaplar.
        
        Args:
            point: (x, y) nokta
            line_start: (x, y) çizgi başlangıcı
            line_end: (x, y) çizgi sonu
            
        Returns:
            Mesafe
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Çizgi vektörü
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Nokta
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Nokta-çizgi mesafesi (projeksiyon ile)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def delete_selected_point(self) -> bool:
        """
        Seçili noktayı siler.
        
        Returns:
            True if point was deleted, False otherwise
        """
        if self.selected_polygon_idx is None or self.selected_point_idx is None:
            print("[UYARI] Silmek için önce bir nokta seçin")
            return False
        
        polygon = self.polygons[self.selected_polygon_idx]
        
        # En az 3 nokta olmalı
        if len(polygon['points']) <= 3:
            print("[UYARI] Polygon en az 3 nokta içermeli")
            return False
        
        # Noktayı sil
        deleted_point = polygon['points'].pop(self.selected_point_idx)
        deleted_original = polygon['original_points'].pop(self.selected_point_idx)
        
        # Seçimi temizle
        self.selected_polygon_idx = None
        self.selected_point_idx = None
        
        # Yeniden çiz
        self.draw_polygons()
        print(f"[BİLGİ] Nokta silindi: ({deleted_original[0]:.1f}, {deleted_original[1]:.1f})")
        return True
    
    def get_edited_points(self) -> list:
        """
        Düzenlenmiş noktaları piksel koordinatlarında döndürür.
        
        Returns:
            Her polygon için orijinal piksel koordinatlarını içeren liste
        """
        result = []
        for polygon in self.polygons:
            result.append({
                'polygon': polygon['original_points'],
                'class_id': polygon.get('class_id', 0)
            })
        return result
