"""
Görsel Düzenleme ve Nokta Yönetimi Modülü
Canvas üzerinde görsel gösterimi, polygon çizimi ve nokta düzenleme işlevleri.
"""

import tkinter as tk
from PIL import Image, ImageTk
import math
import sys
import traceback
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid


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
        self.base_shrink_points = []  # İçe çekme için base noktalar (orijinal canvas koordinatları)
        self.base_original_points = []  # İçe çekme için base noktalar (orijinal piksel koordinatları)
        self.current_shrink_factor = 0.0  # Mevcut shrink değeri (zoom sonrası yeniden uygulamak için)
        
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
        self.current_shrink_factor = 0.0  # Yeni görsel yüklendiğinde shrink sıfırla
        
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
            self.base_shrink_points = []  # Base noktaları sıfırla (canvas koordinatları)
            self.base_original_points = []  # Base noktaları sıfırla (piksel koordinatları)
            for mask_data in masks_data:
                polygon_points = mask_data.get('polygon', [])
                if polygon_points:
                    # Piksel koordinatlarını canvas koordinatlarına çevir
                    canvas_points = self.pixel_to_canvas(polygon_points)
                    # Orijinal piksel koordinatlarının kopyasını oluştur
                    original_points_copy = [(x, y) for x, y in polygon_points]
                    self.polygons.append({
                        'points': canvas_points,
                        'original_points': original_points_copy,  # Orijinal piksel koordinatları
                        'class_id': mask_data.get('class_id', 0)
                    })
                    # Base noktaları sakla (içe çekme için)
                    self.base_shrink_points.append([(x, y) for x, y in canvas_points])
                    self.base_original_points.append([(x, y) for x, y in polygon_points])
            
            # draw_polygons burada çağrılmayacak, main.py'de shrink kontrolü yapılacak
            
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
            # Pan modunda nokta seçimi yapma
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
            
            # Görseli ve polygon'ları yeniden çiz (noktalar takip edecek)
            self.redraw_image()
            return
        
        # Normal nokta sürükleme (pan modunda değilse)
        if not self.is_panning and self.selected_polygon_idx is not None and self.selected_point_idx is not None:
            # Noktayı güncelle
            polygon = self.polygons[self.selected_polygon_idx]
            polygon['points'][self.selected_point_idx] = (x, y)
            
            # Orijinal piksel koordinatlarını da güncelle
            pixel_x, pixel_y = self.canvas_to_pixel(x, y)
            polygon['original_points'][self.selected_point_idx] = (pixel_x, pixel_y)
            
            # Base noktaları da güncelle (içe çekme için)
            if self.selected_polygon_idx < len(self.base_shrink_points):
                if self.selected_point_idx < len(self.base_shrink_points[self.selected_polygon_idx]):
                    self.base_shrink_points[self.selected_polygon_idx][self.selected_point_idx] = (x, y)
            
            # Base original noktaları da güncelle
            if self.selected_polygon_idx < len(self.base_original_points):
                if self.selected_point_idx < len(self.base_original_points[self.selected_polygon_idx]):
                    self.base_original_points[self.selected_polygon_idx][self.selected_point_idx] = (pixel_x, pixel_y)
            
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
            # Seçili noktayı temizle
            self.selected_polygon_idx = None
            self.selected_point_idx = None
            # Seçimi kaldırmak için polygon'ları yeniden çiz
            self.draw_polygons()
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
        self.base_shrink_points = []  # Base noktaları yeniden hesapla
        self.base_original_points = []  # Base original noktaları yeniden hesapla
        for polygon in self.polygons:
            # Orijinal piksel koordinatlarını canvas koordinatlarına çevir
            canvas_points = self.pixel_to_canvas(polygon['original_points'])
            polygon['points'] = canvas_points
            # Base noktaları sakla (içe çekme için)
            self.base_shrink_points.append([(x, y) for x, y in canvas_points])
            self.base_original_points.append([(x, y) for x, y in polygon['original_points']])
        
        # Shrink uygulanmışsa yeniden uygula
        if self.current_shrink_factor > 0:
            self.shrink_polygons(self.current_shrink_factor)
        else:
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
            
            # Base noktaları da güncelle (içe çekme için)
            poly_idx = 0  # İlk polygon'a ekliyoruz
            if poly_idx < len(self.base_shrink_points):
                self.base_shrink_points[poly_idx].insert(insert_idx, (x, y))
            else:
                # Base noktalar yoksa oluştur
                self.base_shrink_points = []
                for p in self.polygons:
                    self.base_shrink_points.append([pt for pt in p['points']])
            
            # Base original noktaları da güncelle
            if poly_idx < len(self.base_original_points):
                self.base_original_points[poly_idx].insert(insert_idx, (pixel_x, pixel_y))
            else:
                # Base original noktalar yoksa oluştur
                self.base_original_points = []
                for p in self.polygons:
                    self.base_original_points.append([pt for pt in p['original_points']])
        
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
        
        # Base noktaları da güncelle (içe çekme için)
        if self.selected_polygon_idx < len(self.base_shrink_points):
            if self.selected_point_idx < len(self.base_shrink_points[self.selected_polygon_idx]):
                self.base_shrink_points[self.selected_polygon_idx].pop(self.selected_point_idx)
        
        # Base original noktaları da güncelle
        if self.selected_polygon_idx < len(self.base_original_points):
            if self.selected_point_idx < len(self.base_original_points[self.selected_polygon_idx]):
                self.base_original_points[self.selected_polygon_idx].pop(self.selected_point_idx)
        
        # Seçimi temizle
        self.selected_polygon_idx = None
        self.selected_point_idx = None
        
        # Yeniden çiz
        self.draw_polygons()
        print(f"[BİLGİ] Nokta silindi: ({deleted_original[0]:.1f}, {deleted_original[1]:.1f})")
        return True
    
    def shrink_polygons(self, shrink_factor: float):
        """
        Shapely kullanarak polygon'u gerçek anlamda içe doğru küçültür (buffer/offset).
        Bu yöntem kenarları paralel şekilde içe kaydırır.
        
        Args:
            shrink_factor: 0.0 (değişiklik yok) ile 1.0 (maksimum küçültme) arası değer
        """
        # Mevcut shrink değerini sakla (zoom sonrası yeniden uygulamak için)
        self.current_shrink_factor = shrink_factor
        
        if shrink_factor <= 0.0:
            # Slider sıfıra döndüğünde base noktalara geri dön
            if len(self.base_shrink_points) == len(self.polygons):
                for i, polygon in enumerate(self.polygons):
                    base_points = self.base_shrink_points[i]
                    base_original = self.base_original_points[i] if i < len(self.base_original_points) else None
                    if len(base_points) == len(polygon['points']):
                        # Canvas koordinatlarını base'e geri döndür
                        polygon['points'] = [(x, y) for x, y in base_points]
                        # Orijinal piksel koordinatlarını base'e geri döndür
                        if base_original and len(base_original) == len(polygon['original_points']):
                            polygon['original_points'] = [(x, y) for x, y in base_original]
                self.draw_polygons()
            return
        
        shrink_factor = min(1.0, max(0.0, shrink_factor))  # 0-1 arasına sınırla
        
        # Base noktalar yoksa, mevcut noktaları base olarak kullan
        if len(self.base_shrink_points) != len(self.polygons):
            self.base_shrink_points = []
            self.base_original_points = []
            for polygon in self.polygons:
                self.base_shrink_points.append([(x, y) for x, y in polygon['points']])
                self.base_original_points.append([(x, y) for x, y in polygon['original_points']])
        
        for poly_idx, polygon in enumerate(self.polygons):
            base_points = self.base_shrink_points[poly_idx] if poly_idx < len(self.base_shrink_points) else polygon['points']
            points = polygon['points']
            original_points = polygon['original_points']
            
            if len(base_points) < 3:
                continue
            
            try:
                # Shapely polygon oluştur
                shapely_poly = ShapelyPolygon(base_points)
                
                # Polygon geçerli değilse düzelt
                if not shapely_poly.is_valid:
                    shapely_poly = make_valid(shapely_poly)
                    if shapely_poly.geom_type != 'Polygon':
                        continue
                
                # Polygon'un maksimum içe küçültme mesafesini hesapla
                # (merkeze olan en kısa mesafe)
                centroid = shapely_poly.centroid
                
                # Polygon'un alanına göre maksimum offset hesapla
                # Daha küçük bir değer kullanarak aşırı küçülmeyi önle
                area = shapely_poly.area
                perimeter = shapely_poly.length
                
                # Maksimum offset: yaklaşık olarak polygon'un yarıçapı
                # area = pi * r^2 -> r = sqrt(area / pi)
                max_offset = math.sqrt(area / math.pi) * 0.9  # %90'ı kadar küçült
                
                # Slider değerine göre offset hesapla
                offset = shrink_factor * max_offset
                
                # Negatif buffer = içe doğru küçültme
                shrunk_poly = shapely_poly.buffer(-offset, join_style=2)  # join_style=2: mitre
                
                # Sonuç boş veya geçersiz olabilir
                if shrunk_poly.is_empty or not shrunk_poly.is_valid:
                    continue
                
                # MultiPolygon olabilir, en büyüğünü al
                if shrunk_poly.geom_type == 'MultiPolygon':
                    shrunk_poly = max(shrunk_poly.geoms, key=lambda p: p.area)
                elif shrunk_poly.geom_type != 'Polygon':
                    continue
                
                # Yeni noktaları al
                new_coords = list(shrunk_poly.exterior.coords)[:-1]  # Son nokta tekrar, çıkar
                
                if len(new_coords) < 3:
                    continue
                
                # Nokta sayısı değişmiş olabilir, orijinal sayıya yakın tutmaya çalış
                # Basit interpolasyon ile orijinal nokta sayısına eşitle
                target_count = len(base_points)
                new_coords = self._resample_polygon(new_coords, target_count)
                
                # Tüm noktaları güncelle
                if len(new_coords) == len(points):
                    for i in range(len(new_coords)):
                        new_x, new_y = new_coords[i]
                        points[i] = (new_x, new_y)
                else:
                    # Nokta sayısı farklıysa, listeyi tamamen değiştir
                    polygon['points'] = list(new_coords)
                    
            except Exception as e:
                continue
        
        # Yeniden çiz
        self.draw_polygons()
    
    def _resample_polygon(self, coords: list, target_count: int) -> list:
        """
        Polygon noktalarını hedef sayıya yeniden örnekler.
        Kenar boyunca eşit aralıklı noktalar oluşturur.
        """
        if len(coords) == target_count:
            return coords
        
        if len(coords) < 3:
            return coords
        
        # Toplam çevre uzunluğunu hesapla
        total_length = 0
        lengths = []
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lengths.append(length)
            total_length += length
        
        if total_length < 0.001:
            return coords
        
        # Hedef noktalar arası mesafe
        step = total_length / target_count
        
        # Yeni noktaları oluştur
        new_coords = []
        current_dist = 0
        current_idx = 0
        accumulated = 0
        
        for _ in range(target_count):
            target_dist = current_dist
            
            # Hedef mesafeye ulaşana kadar kenarlar üzerinde ilerle
            while current_idx < len(coords):
                edge_start = coords[current_idx]
                edge_end = coords[(current_idx + 1) % len(coords)]
                edge_length = lengths[current_idx]
                
                remaining_in_edge = edge_length - (target_dist - accumulated)
                
                if remaining_in_edge >= 0 and (target_dist - accumulated) <= edge_length:
                    # Bu kenar üzerinde bir nokta
                    t = (target_dist - accumulated) / edge_length if edge_length > 0 else 0
                    t = max(0, min(1, t))
                    new_x = edge_start[0] + t * (edge_end[0] - edge_start[0])
                    new_y = edge_start[1] + t * (edge_end[1] - edge_start[1])
                    new_coords.append((new_x, new_y))
                    break
                else:
                    accumulated += edge_length
                    current_idx += 1
            
            current_dist += step
        
        # Eğer yeterli nokta oluşturulamadıysa, orijinal koordinatları döndür
        if len(new_coords) < target_count:
            # Basit interpolasyon
            new_coords = []
            for i in range(target_count):
                t = i / target_count
                idx = int(t * len(coords))
                idx = min(idx, len(coords) - 1)
                new_coords.append(coords[idx])
        
        return new_coords
    
    def get_edited_points(self) -> list:
        """
        Düzenlenmiş noktaları piksel koordinatlarında döndürür.
        Mevcut canvas koordinatlarından (shrink uygulanmış olabilir) piksel koordinatlarına çevirir.
        
        Returns:
            Her polygon için piksel koordinatlarını içeren liste
        """
        result = []
        for polygon in self.polygons:
            # Mevcut canvas koordinatlarından piksel koordinatlarına çevir
            pixel_points = []
            for x, y in polygon['points']:
                pixel_x, pixel_y = self.canvas_to_pixel(x, y)
                pixel_points.append((pixel_x, pixel_y))
            
            result.append({
                'polygon': pixel_points,
                'class_id': polygon.get('class_id', 0)
            })
        return result
