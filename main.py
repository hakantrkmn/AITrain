"""
YOLOv8 Segmentasyon Düzenleme GUI
Ana uygulama dosyası
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
from pathlib import Path
from PIL import Image
import glob
import traceback
import sys
import json
import random

from yolo_processor import YOLOProcessor
from image_editor import ImageEditor
from viewer import open_viewer
from mods import NormalMode, UNetMode
from utils import load_config as load_config_util, save_config as save_config_util, ensure_output_structure


class YOLOSegmentationEditor:
    """Ana GUI uygulaması"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Segmentasyon Düzenleme")
        self.root.geometry("1000x700")
        
        # Değişkenler
        self.selected_folder = None
        self.selected_model = None
        self.yolo_processor = None
        self.image_editor = None
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.current_masks_data = []
        self.mode = 'normal'  # 'normal' veya 'unet'
        self.mode_state = {}  # Mod'a özel state bilgisi
        self.epsilon_factor = 0.002  # Polygon basitleştirme faktörü (varsayılan)
        self.shrink_value = 0.0  # Shrink yüzdesi (varsayılan)
        
        # Mod sınıfları
        self.output_base = "output"
        self.mode_handlers = {
            'normal': NormalMode(self.output_base),
            'unet': UNetMode(self.output_base)
        }
        
        # Config dosyası yolu
        self.config_path = "config.json"
        
        # Son görsel indeksleri (mod'a göre)
        self.last_image_indices = {}
        
        # Config'den değerleri yükle
        self.load_config()
        
        # GUI oluştur
        self.create_widgets()
        
        # Çıktı klasör yapısını oluştur
        ensure_output_structure(self.output_base)
    
    def on_mode_change(self, event=None):
        """Mod değiştiğinde çağrılır"""
        new_mode = self.mode_var.get()
        if new_mode != self.mode:
            # Eski mod'un son görsel indeksini kaydet
            if self.current_image_index >= 0:
                self.last_image_indices[self.mode] = self.current_image_index
                self.save_config()
            
            self.mode = new_mode
            print(f"[BİLGİ] Mod değiştirildi: {self.mode}")
            # Mevcut düzenlemeleri sıfırla
            self.current_masks_data = []
            self.mode_state = {}
            if self.image_editor:
                self.canvas.delete("all")
            
            # Yeni mod'un son görseline geç (eğer görseller yüklüyse)
            if self.image_files:
                saved_index = self.last_image_indices.get(self.mode, 0)
                if saved_index >= 0 and saved_index < len(self.image_files):
                    self.current_image_index = saved_index
                    self.load_current_image()
    
    def on_epsilon_change(self, event=None):
        """Epsilon değeri değiştiğinde çağrılır"""
        try:
            new_epsilon = self.epsilon_var.get()
            if 0.0001 <= new_epsilon <= 0.1:
                self.epsilon_factor = new_epsilon
                print(f"[BİLGİ] Epsilon faktörü güncellendi: {self.epsilon_factor:.4f}")
                self.save_config()  # Config'e kaydet
            else:
                # Geçersiz değer, eski değere geri dön
                self.epsilon_var.set(self.epsilon_factor)
        except:
            # Hata durumunda eski değere geri dön
            self.epsilon_var.set(self.epsilon_factor)
    
    def on_shrink_change(self, value=None):
        """Noktaları içe doğru hareket ettirme slider'ı değiştiğinde çağrılır"""
        if not self.image_editor:
            return
        
        try:
            # Slider değeri 0-100 arası, bunu 0.0-1.0 arasına çevir
            shrink_percent = self.shrink_var.get()
            shrink_factor = shrink_percent / 100.0
            
            # Label'ı güncelle
            self.shrink_label.config(text=f"{int(shrink_percent)}%")
            
            # Noktaları içe doğru hareket ettir
            self.image_editor.shrink_polygons(shrink_factor)
            
            # Config'e kaydet
            self.save_config()
            
        except Exception as e:
            print(f"[HATA] İçe çekme hatası: {str(e)}", file=sys.stderr)
    
    def apply_shrink(self, event=None):
        """Mevcut shrink değerini polygon'lara uygular (E tuşu)"""
        if not self.image_editor:
            return
        
        shrink_percent = self.shrink_var.get()
        if shrink_percent <= 0:
            print("[BİLGİ] Shrink değeri 0, uygulanacak bir şey yok")
            return
        
        # Önce base noktalarına dön (0'a çek)
        self.image_editor.shrink_polygons(0.0)
        
        # Sonra shrink uygula
        shrink_factor = shrink_percent / 100.0
        self.image_editor.shrink_polygons(shrink_factor)
        print(f"[BİLGİ] Shrink uygulandı: {shrink_percent}%")
    
    def create_widgets(self):
        """GUI widget'larını oluşturur"""
        
        # Üst frame - Butonlar
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Mod seçimi
        mode_frame = tk.Frame(top_frame)
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(mode_frame, text="Mod:", font=("Arial", 10)).pack(side=tk.LEFT, padx=2)
        self.mode_var = tk.StringVar(value="normal")
        mode_dropdown = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["normal", "unet"],
            state="readonly",
            width=10
        )
        mode_dropdown.pack(side=tk.LEFT, padx=2)
        mode_dropdown.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        tk.Button(
            top_frame,
            text="Klasör Seç",
            command=self.select_folder,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Model Seç",
            command=self.select_model,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Başlat",
            command=self.start_processing,
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Doğrula",
            command=self.open_viewer,
            width=15,
            height=2,
            bg="#FF9800",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        # Epsilon ayarı
        epsilon_frame = tk.Frame(top_frame)
        epsilon_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(epsilon_frame, text="Epsilon:", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        self.epsilon_var = tk.DoubleVar(value=0.002)
        epsilon_spinbox = tk.Spinbox(
            epsilon_frame,
            from_=0.0001,
            to=0.1,
            increment=0.0001,
            textvariable=self.epsilon_var,
            width=8,
            format="%.4f",
            command=self.on_epsilon_change
        )
        epsilon_spinbox.pack(side=tk.LEFT, padx=2)
        epsilon_spinbox.bind("<Return>", self.on_epsilon_change)
        epsilon_spinbox.bind("<FocusOut>", self.on_epsilon_change)
        
        # Noktaları içe doğru hareket ettirme slider'ı
        shrink_frame = tk.Frame(top_frame)
        shrink_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(shrink_frame, text="İçe Çek:", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        self.shrink_var = tk.DoubleVar(value=self.shrink_value)
        shrink_scale = tk.Scale(
            shrink_frame,
            from_=0.0,
            to=30.0,  # Maksimum %30 (daha etkili değerler için)
            orient=tk.HORIZONTAL,
            variable=self.shrink_var,
            length=150,
            resolution=1.0,
            command=self.on_shrink_change
        )
        shrink_scale.pack(side=tk.LEFT, padx=2)
        
        # Değer gösterimi için label
        self.shrink_label = tk.Label(shrink_frame, text=f"{int(self.shrink_value)}%", font=("Arial", 9), width=4)
        self.shrink_label.pack(side=tk.LEFT, padx=2)
        
        # Klasör ve model bilgisi
        self.info_label = tk.Label(
            top_frame,
            text="Klasör ve model seçin",
            fg="gray"
        )
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Canvas frame
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="gray",
            width=800,
            height=500
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas'a focus ver (klavye kısayolları için)
        self.canvas.focus_set()
        
        # Klavye kısayolları
        self.canvas.bind('<KeyPress-a>', lambda e: self.add_point())
        self.canvas.bind('<KeyPress-d>', lambda e: self.delete_point())
        self.canvas.bind('<KeyPress-A>', lambda e: self.add_point())  # Büyük harf
        self.canvas.bind('<KeyPress-D>', lambda e: self.delete_point())  # Büyük harf
        self.canvas.bind('<Insert>', lambda e: self.add_point())  # Insert tuşu
        self.canvas.bind('<Delete>', lambda e: self.delete_point())  # Delete tuşu
        self.canvas.bind('<KeyPress-plus>', lambda e: self.add_point())  # + tuşu
        self.canvas.bind('<KeyPress-minus>', lambda e: self.delete_point())  # - tuşu
        self.canvas.bind('<KeyPress-s>', lambda e: self.save_to_all_modes())  # Tümüne Kaydet
        self.canvas.bind('<KeyPress-S>', lambda e: self.save_to_all_modes())  # Tümüne Kaydet (büyük harf)
        self.canvas.bind('<KeyPress-e>', lambda e: self.apply_shrink())  # Shrink uygula
        self.canvas.bind('<KeyPress-E>', lambda e: self.apply_shrink())  # Shrink uygula (büyük harf)
        self.canvas.bind('<Right>', lambda e: self.next_image())  # Sonraki görsel
        self.canvas.bind('<Left>', lambda e: self.prev_image())  # Önceki görsel
        self.canvas.bind('<Button-3>', lambda e: self.next_image())  # Sağ tık - sonraki görsel
        
        # Alt frame - Kontrol butonları
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Button(
            bottom_frame,
            text="Nokta Ekle",
            command=self.add_point,
            width=12,
            height=2,
            bg="#4CAF50",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Nokta Sil",
            command=self.delete_point,
            width=12,
            height=2,
            bg="#F44336",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Kaydet",
            command=self.save_current,
            width=12,
            height=2,
            bg="#2196F3",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Tümüne Kaydet",
            command=self.save_to_all_modes,
            width=15,
            height=2,
            bg="#9C27B0",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Split",
            command=self.split_train_valid,
            width=12,
            height=2,
            bg="#FF9800",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Önceki",
            command=self.prev_image,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Sonraki",
            command=self.next_image,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Görseli Sil",
            command=self.delete_current_image,
            width=15,
            height=2,
            bg="#D32F2F",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            bottom_frame,
            text="Son Görsele Git",
            command=self.go_to_last_image,
            width=18,
            height=2,
            bg="#FF5722",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        # Durum etiketi
        self.status_label = tk.Label(
            bottom_frame,
            text="Hazır",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Görsel işlendi bilgisi
        self.processed_label = tk.Label(
            bottom_frame,
            text="",
            font=("Arial", 9),
            fg="green"
        )
        self.processed_label.pack(side=tk.LEFT, padx=10)
        
        # Klavye kısayolları bilgisi
        help_label = tk.Label(
            bottom_frame,
            text="Kısayollar: A=Ekle, D=Sil, S=Tümüne Kaydet, E=Shrink Uygula, →/SağTık=Sonraki, ←=Önceki",
            font=("Arial", 8),
            fg="gray"
        )
        help_label.pack(side=tk.LEFT, padx=10)
        
        # Config'den yüklenen değerleri göster
        self.update_info_label()
    
    
    def load_config(self):
        """Config dosyasından değerleri yükler"""
        config = load_config_util(self.config_path)
        if config.get('folder'):
            self.selected_folder = config['folder']
        if config.get('model'):
            self.selected_model = config['model']
            # Modeli yükle
            try:
                self.yolo_processor = YOLOProcessor(self.selected_model, device='cuda:0')
                print(f"[BİLGİ] Config'den model yüklendi: {os.path.basename(self.selected_model)}")
            except Exception as e:
                print(f"[UYARI] Config'den model yüklenemedi: {str(e)}")
                self.selected_model = None
        # Son görsel indekslerini yükle
        if config.get('last_image_index'):
            self.last_image_indices = config['last_image_index']
        # Epsilon değerini yükle
        if 'epsilon' in config:
            self.epsilon_factor = config['epsilon']
            if hasattr(self, 'epsilon_var'):
                self.epsilon_var.set(self.epsilon_factor)
        # Shrink değerini yükle
        if 'shrink' in config:
            self.shrink_value = config['shrink']
            if hasattr(self, 'shrink_var'):
                self.shrink_var.set(self.shrink_value)
                self.shrink_label.config(text=f"{int(self.shrink_value)}%")
    
    def save_config(self):
        """Mevcut seçimleri config dosyasına kaydeder"""
        shrink_value = self.shrink_var.get() if hasattr(self, 'shrink_var') else 0.0
        save_config_util(
            folder=self.selected_folder,
            model=self.selected_model,
            last_image_index=self.last_image_indices,
            epsilon=self.epsilon_factor,
            shrink=shrink_value,
            config_path=self.config_path
        )
    
    def save_last_image_index(self):
        """Mevcut görsel indeksini mod'a göre config'e kaydeder"""
        if self.current_image_index >= 0 and self.mode:
            self.last_image_indices[self.mode] = self.current_image_index
            self.save_config()
    
    def select_folder(self):
        """Görsellerin bulunduğu klasörü seçer"""
        # Config'den son seçilen klasörü al (varsa)
        initial_dir = self.selected_folder if self.selected_folder else "."
        folder = filedialog.askdirectory(
            title="Görsellerin bulunduğu klasörü seçin",
            initialdir=initial_dir
        )
        if folder:
            self.selected_folder = folder
            self.save_config()  # Config'e kaydet
            self.update_info_label()
    
    def select_model(self):
        """YOLOv8 model dosyasını seçer"""
        # Config'den son seçilen modeli al (varsa)
        initial_dir = os.path.dirname(self.selected_model) if self.selected_model else "."
        model_file = filedialog.askopenfilename(
            title="YOLOv8 model dosyasını seçin",
            filetypes=[("PyTorch Model", "*.pt"), ("Tüm Dosyalar", "*.*")],
            initialdir=initial_dir
        )
        if model_file:
            self.selected_model = model_file
            try:
                # Modeli yükle (yalnızca GPU)
                self.yolo_processor = YOLOProcessor(model_file, device='cuda:0')
                self.save_config()  # Config'e kaydet
                self.update_info_label()
            except Exception as e:
                error_msg = f"Model yüklenemedi: {str(e)}"
                print(f"[HATA] {error_msg}", file=sys.stderr)
                traceback.print_exc()
                messagebox.showerror("Hata", error_msg)
                self.selected_model = None
                self.yolo_processor = None
    
    def start_processing(self):
        """İşlemeyi başlatır"""
        if not self.selected_folder:
            warning_msg = "Lütfen bir klasör seçin!"
            print(f"[UYARI] {warning_msg}")
            messagebox.showwarning("Uyarı", warning_msg)
            return
        
        if not self.selected_model or not self.yolo_processor:
            warning_msg = "Lütfen bir model seçin!"
            print(f"[UYARI] {warning_msg}")
            messagebox.showwarning("Uyarı", warning_msg)
            return
        
        # Görsel dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(self.selected_folder, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.selected_folder, ext.upper())))
        
        if not self.image_files:
            warning_msg = "Seçilen klasörde görsel dosyası bulunamadı!"
            print(f"[UYARI] {warning_msg}")
            messagebox.showwarning("Uyarı", warning_msg)
            return
        
        # ImageEditor'ı başlat
        if not self.image_editor:
            self.image_editor = ImageEditor(self.canvas)
        
        # Canvas'a tekrar focus ver (klavye kısayolları için)
        self.canvas.focus_set()
        
        # İlk görseli yükle - config'den son görsel indeksini kontrol et
        saved_index = self.last_image_indices.get(self.mode, 0)
        if saved_index >= 0 and saved_index < len(self.image_files):
            self.current_image_index = saved_index
        else:
            self.current_image_index = 0
        self.load_current_image()
    
    def check_saved_data(self, image_path: str, mode: str) -> tuple:
        """
        Output klasöründe kaydedilmiş veri var mı kontrol eder.
        
        Returns:
            (has_saved_data, saved_masks_data, display_path, state) tuple
        """
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        
        if mode == 'normal':
            label_path = os.path.join(self.output_base, "normal", "labels", "train", f"{image_name}.txt")
            image_path_saved = os.path.join(self.output_base, "normal", "images", "train", image_filename)
        elif mode == 'unet':
            mask_path = os.path.join(self.output_base, "unet", "masks", "train", f"{image_name}.png")
            image_path_saved = os.path.join(self.output_base, "unet", "images", "train", image_filename)
            # UNet için mask dosyası varsa kaydedilmiş demektir
            if os.path.exists(mask_path) and os.path.exists(image_path_saved):
                return (True, None, image_path_saved, {})  # UNet için mask'ı sonra yükleyeceğiz
            return (False, None, None, None)
        else:
            return (False, None, None, None)
        
        # Normal için label dosyası kontrolü
        if os.path.exists(label_path) and os.path.exists(image_path_saved):
            # Label dosyasını oku ve polygon verilerine çevir
            try:
                img = Image.open(image_path_saved)
                img_width, img_height = img.size
                
                masks_data = []
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) < 7:  # class_id + en az 3 nokta (6 koordinat)
                            continue
                        
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # Normalize koordinatları piksel koordinatlarına çevir
                        polygon = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x_norm = coords[i]
                                y_norm = coords[i + 1]
                                x_pixel = x_norm * img_width
                                y_pixel = y_norm * img_height
                                polygon.append((x_pixel, y_pixel))
                        
                        if len(polygon) >= 3:
                            masks_data.append({
                                'polygon': polygon,
                                'class_id': class_id
                            })
                
                if masks_data:
                    return (True, masks_data, image_path_saved, {})
            except Exception as e:
                print(f"[UYARI] Kaydedilmiş veri okunurken hata: {str(e)}")
        
        return (False, None, None, None)
    
    def load_current_image(self):
        """Mevcut görseli yükler ve işler"""
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_files):
            return
        
        self.current_image_path = self.image_files[self.current_image_index]
        
        # İşlendi bilgisini temizle
        self.processed_label.config(text="")
        
        try:
            # Önce output klasöründe kaydedilmiş veri var mı kontrol et
            has_saved, saved_masks, saved_image_path, saved_state = self.check_saved_data(
                self.current_image_path, self.mode
            )
            
            if has_saved and saved_masks is not None:
                # Kaydedilmiş verileri yükle (modele sokma)
                print(f"[BİLGİ] Kaydedilmiş veriler yüklendi: {os.path.basename(self.current_image_path)}")
                
                # Mod handler'ı kullan (sadece görseli yüklemek için)
                mode_handler = self.mode_handlers.get(self.mode)
                if not mode_handler:
                    mode_handler = self.mode_handlers['normal']
                
                # Görseli yükle (masks_data boş olarak, çünkü kaydedilmiş verileri kullanacağız)
                temp_dir = os.path.join(self.output_base, "temp")
                display_path, _, state = mode_handler.load_image(
                    saved_image_path,
                    [],  # Boş masks_data - kaydedilmiş verileri kullanacağız
                    [],  # Boş bounding_boxes
                    temp_dir
                )
                
                # Kaydedilmiş mask verilerini kullan
                self.current_masks_data = saved_masks
                self.mode_state = saved_state
                
                # Canvas'a göster
                self.image_editor.display_image(display_path, self.current_masks_data)
                
                # İşlendi bilgisini göster
                self.processed_label.config(text="✓ Bu görsel işlendi", fg="green")
                
            elif has_saved and self.mode == 'unet':
                # UNet için mask dosyası var, mask'ı yükle
                image_filename = os.path.basename(self.current_image_path)
                image_name = os.path.splitext(image_filename)[0]
                mask_path = os.path.join(self.output_base, "unet", "masks", "train", f"{image_name}.png")
                
                # Mask dosyasından polygon'ları oluştur
                print(f"[BİLGİ] Kaydedilmiş UNet mask yüklendi: {os.path.basename(self.current_image_path)}")
                
                try:
                    import cv2
                    import numpy as np
                    
                    # Mask dosyasını oku
                    mask_img = Image.open(mask_path)
                    mask_array = np.array(mask_img)
                    
                    # Binary mask'dan contour çıkar
                    contours, _ = cv2.findContours(
                        (mask_array == 255).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Contour'lardan polygon'ları oluştur
                    masks_data = []
                    for contour in contours:
                        if len(contour) >= 3:
                            # Contour'u polygon'a çevir
                            polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]
                            masks_data.append({
                                'polygon': polygon,
                                'class_id': 0
                            })
                    
                    # UNet için kaydedilmiş görsel zaten letterbox'a çevrilmiş (320x320)
                    # Mask'tan çıkarılan polygon'lar da 320x320 koordinatlarında
                    # Doğrudan kullanabiliriz, mode handler'a göndermeye gerek yok
                    self.current_masks_data = masks_data
                    self.mode_state = {}  # UNet için state gerekmez, görsel zaten hazır
                    
                    # Canvas'a göster (kaydedilmiş görseli doğrudan kullan)
                    self.image_editor.display_image(saved_image_path, self.current_masks_data)
                    
                except Exception as e:
                    print(f"[UYARI] UNet mask okunurken hata: {str(e)}")
                    # Hata durumunda sadece görseli göster
                    self.image_editor.display_image(saved_image_path, [])
                    self.current_masks_data = []
                    self.mode_state = {}
                
                # İşlendi bilgisini göster
                self.processed_label.config(text="✓ Bu görsel işlendi", fg="green")
                
            else:
                # Kaydedilmiş veri yok, modele sok
                if not self.yolo_processor:
                    messagebox.showwarning("Uyarı", "Model yüklenmemiş!")
                    return
                
                # YOLOv8 ile işle
                result = self.yolo_processor.process_image(self.current_image_path, epsilon_factor=self.epsilon_factor)
                masks_data = result['masks']
                bounding_boxes = result['boxes']
                
                # Mod handler'ı kullan
                mode_handler = self.mode_handlers.get(self.mode)
                if not mode_handler:
                    mode_handler = self.mode_handlers['normal']
                
                temp_dir = os.path.join(self.output_base, "temp")
                display_path, adjusted_masks, state = mode_handler.load_image(
                    self.current_image_path,
                    masks_data,
                    bounding_boxes,
                    temp_dir
                )
                
                # Fallback kontrolü
                if state.get('fallback'):
                    self.mode_var.set("normal")
                    self.mode = 'normal'
                    mode_handler = self.mode_handlers['normal']
                    display_path, adjusted_masks, state = mode_handler.load_image(
                        self.current_image_path,
                        masks_data,
                        bounding_boxes,
                        temp_dir
                    )
                
                self.current_masks_data = adjusted_masks
                self.mode_state = state
                
                # Canvas'a göster
                self.image_editor.display_image(display_path, self.current_masks_data)
            
            # Canvas'a focus ver (klavye kısayolları için)
            self.canvas.focus_set()
            
            # Polygon'ları çiz
            self.image_editor.draw_polygons()
            
            # Durum güncelle
            self.update_status()
            
            # Son görsel indeksini kaydet
            self.save_last_image_index()
            
        except Exception as e:
            error_msg = f"Görsel işlenirken hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            print(f"[HATA] Görsel: {self.current_image_path}", file=sys.stderr)
            traceback.print_exc()
            messagebox.showerror("Hata", error_msg)
    
    
    def update_info_label(self):
        """Bilgi etiketini günceller (config'den yüklenen değerleri gösterir)"""
        folder_text = f"Klasör: {os.path.basename(self.selected_folder)}" if self.selected_folder else "Klasör: Seçilmedi"
        model_text = f"Model: {os.path.basename(self.selected_model)}" if self.selected_model else "Model: Seçilmedi"
        
        if self.selected_model and self.yolo_processor:
            device_info = self.yolo_processor.device
            self.info_label.config(text=f"{folder_text} | {model_text} | Cihaz: {device_info}")
        else:
            self.info_label.config(text=f"{folder_text} | {model_text}")
    
    def update_status(self):
        """Durum etiketini günceller"""
        total = len(self.image_files)
        current = self.current_image_index + 1
        self.status_label.config(text=f"Durum: {current}/{total}")
    
    def save_current(self, show_message=True):
        """Mevcut görseli ve etiketleri kaydeder"""
        if not self.current_image_path:
            warning_msg = "Kaydedilecek görsel yok!"
            print(f"[UYARI] {warning_msg}")
            if show_message:
                messagebox.showwarning("Uyarı", warning_msg)
            return False
        
        try:
            # Düzenlenmiş noktaları al
            edited_data = self.image_editor.get_edited_points()
            
            if not edited_data:
                warning_msg = "Kaydedilecek maske yok!"
                print(f"[UYARI] {warning_msg}")
                if show_message:
                    messagebox.showwarning("Uyarı", warning_msg)
                return False
            
            # Mod handler'ı kullan
            mode_handler = self.mode_handlers.get(self.mode)
            if not mode_handler:
                mode_handler = self.mode_handlers['normal']
            
            success = mode_handler.save_image(
                self.current_image_path,
                edited_data,
                self.mode_state
            )
            
            if success:
                success_msg = f"Görsel ve etiket kaydedildi ({self.mode} modu)"
                print(f"[BAŞARILI] {success_msg}")
                if show_message:
                    messagebox.showinfo("Başarılı", success_msg)
                return True
            else:
                error_msg = f"Kaydetme başarısız ({self.mode} modu)"
                print(f"[HATA] {error_msg}")
                if show_message:
                    messagebox.showerror("Hata", error_msg)
                return False
            
        except Exception as e:
            error_msg = f"Kaydetme sırasında hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            print(f"[HATA] Görsel: {self.current_image_path}", file=sys.stderr)
            traceback.print_exc()
            if show_message:
                messagebox.showerror("Hata", error_msg)
            return False
    
    def save_to_all_modes(self):
        """Mevcut düzenlemeyi tüm modlara (normal, unet) kaydeder"""
        if not self.current_image_path:
            warning_msg = "Kaydedilecek görsel yok!"
            print(f"[UYARI] {warning_msg}")
            messagebox.showwarning("Uyarı", warning_msg)
            return
        
        try:
            # Önce mevcut mod'a kaydet (mesaj gösterme)
            current_success = self.save_current(show_message=False)
            if not current_success:
                return
            
            # Şimdi diğer modlara da kaydet
            other_modes_success = True
            if self.mode == 'normal':
                # Normal modda düzenlenmiş, UNet'e de kaydet
                self.save_to_unet()
            elif self.mode == 'unet':
                # UNet'te düzenlenmiş, Normal'e de kaydet
                self.save_to_normal()
            
            # Tüm modlar için aynı görsel indeksini kaydet
            if self.current_image_index >= 0:
                self.last_image_indices['normal'] = self.current_image_index
                self.last_image_indices['unet'] = self.current_image_index
                self.save_config()
            
            # Sadece bir kere başarılı mesajı göster
            success_msg = f"Görsel ve etiketler tüm modlara kaydedildi!"
            print(f"[BAŞARILI] {success_msg}")
            messagebox.showinfo("Başarılı", success_msg)
            
        except Exception as e:
            error_msg = f"Tümüne kaydetme sırasında hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            messagebox.showerror("Hata", error_msg)
    
    def save_to_normal(self):
        """UNet'te düzenlenen noktaları Normal mod'a kaydeder"""
        from utils.image_utils import adjust_polygons_from_crop
        
        try:
            # Düzenlenmiş noktaları al
            edited_data = self.image_editor.get_edited_points()
            
            if not edited_data:
                return
            
            # Mod'a göre dönüşüm yap
            if self.mode == 'unet':
                # UNet'ten: letterbox koordinatlarından önce kırpılmış görsele, sonra orijinal görsele
                letterbox_offset = self.mode_state.get('letterbox_offset')
                letterbox_scale = self.mode_state.get('letterbox_scale')
                crop_box = self.mode_state.get('crop_box')
                
                if not letterbox_offset or not letterbox_scale or not crop_box:
                    print("[UYARI] UNet'ten Normal'e kaydetmek için gerekli bilgiler eksik")
                    return
                
                # Letterbox koordinatlarından kırpılmış görsel koordinatlarına çevir
                offset_x, offset_y = letterbox_offset
                adjusted_masks_crop = []
                for mask_data in edited_data:
                    polygon = mask_data['polygon']
                    # Letterbox offset'ini çıkar, scale'i geri al
                    adjusted_polygon = [((x - offset_x) / letterbox_scale, (y - offset_y) / letterbox_scale) 
                                       for x, y in polygon]
                    adjusted_masks_crop.append({
                        'polygon': adjusted_polygon,
                        'class_id': mask_data.get('class_id', 0)
                    })
                
                # Kırpılmış görsel koordinatlarından orijinal görsel koordinatlarına çevir
                adjusted_data = adjust_polygons_from_crop(adjusted_masks_crop, crop_box)
            else:
                # Normal moddan: zaten orijinal görsel koordinatlarında
                adjusted_data = edited_data
            
            if not adjusted_data:
                print("[UYARI] Normal'e kaydedilecek geçerli polygon yok")
                return
            
            # Normal klasörlerine kaydet
            output_images_dir = os.path.join(self.output_base, "normal", "images", "train")
            output_labels_dir = os.path.join(self.output_base, "normal", "labels", "train")
            
            # Klasörleri oluştur
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)
            
            image_filename = os.path.basename(self.current_image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Orijinal görseli kaydet
            output_image_path = os.path.join(output_images_dir, image_filename)
            shutil.copy2(self.current_image_path, output_image_path)
            
            # Orijinal görsel boyutlarını al
            img = Image.open(self.current_image_path)
            img_width, img_height = img.size
            
            # Etiket dosyasını oluştur
            output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
            
            with open(output_label_path, 'w') as f:
                for mask_data in adjusted_data:
                    polygon = mask_data['polygon']
                    class_id = mask_data.get('class_id', 0)
                    
                    if len(polygon) < 3:
                        continue
                    
                    # Normalize koordinatları hesapla (orijinal görsele göre)
                    normalized_coords = []
                    for x, y in polygon:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        normalized_coords.extend([norm_x, norm_y])
                    
                    line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
                    f.write(line + "\n")
            
            print(f"[BAŞARILI] Normal'e kaydedildi: {output_image_path}")
            
        except Exception as e:
            print(f"[HATA] Normal'e kaydetme hatası: {str(e)}", file=sys.stderr)
            traceback.print_exc()
    
    def save_to_unet(self):
        """Normal'da düzenlenen noktaları UNet formatına kaydeder"""
        from utils.image_utils import (
            crop_image_by_bbox,
            adjust_polygons_to_crop,
            letterbox_resize,
            create_binary_mask,
            create_valid_mask
        )
        
        try:
            # Düzenlenmiş noktaları al
            edited_data = self.image_editor.get_edited_points()
            
            if not edited_data:
                return
            
            # YOLO'dan bounding box'ları al
            result = self.yolo_processor.process_image(self.current_image_path, epsilon_factor=self.epsilon_factor)
            bounding_boxes = result['boxes']
            
            if len(bounding_boxes) == 0:
                print("[UYARI] UNet'e kaydetmek için bounding box bulunamadı")
                return
            
            # En büyük bounding box'ı kullan
            largest_box = max(bounding_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            cropped_image, crop_box = crop_image_by_bbox(self.current_image_path, largest_box)
            
            # Polygon koordinatlarını kırpılmış görsele göre ayarla
            if self.mode == 'normal':
                # Normal moddan geliyorsa, önce kırpılmış görsele göre ayarla
                adjusted_masks_crop = adjust_polygons_to_crop(edited_data, crop_box)
            else:
                # UNet'ten geliyorsa, letterbox koordinatlarından kırpılmış görsele çevir
                letterbox_offset = self.mode_state.get('letterbox_offset')
                letterbox_scale = self.mode_state.get('letterbox_scale')
                
                if not letterbox_offset or not letterbox_scale:
                    print("[UYARI] UNet'ten UNet'e kaydetmek için gerekli bilgiler eksik")
                    return
                
                offset_x, offset_y = letterbox_offset
                adjusted_masks_crop = []
                for mask_data in edited_data:
                    polygon = mask_data['polygon']
                    # Letterbox offset'ini çıkar, scale'i geri al
                    adjusted_polygon = [((x - offset_x) / letterbox_scale, (y - offset_y) / letterbox_scale) 
                                       for x, y in polygon]
                    adjusted_masks_crop.append({
                        'polygon': adjusted_polygon,
                        'class_id': mask_data.get('class_id', 0),
                        'confidence': mask_data.get('confidence', 0.0)
                    })
            
            # 576x320'e letterbox resize (aspect ratio korunarak)
            target_size = (576, 320)
            letterbox_img, offset_x, offset_y, scale, new_width, new_height = letterbox_resize(cropped_image, target_size)
            
            # Polygon koordinatlarını letterbox resize'e göre ayarla
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
            
            # Çıktı klasörleri
            output_images_dir = os.path.join(self.output_base, "unet", "images", "train")
            output_masks_dir = os.path.join(self.output_base, "unet", "masks", "train")
            output_valid_dir = os.path.join(self.output_base, "unet", "valid", "train")
            
            # Klasörleri oluştur
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_masks_dir, exist_ok=True)
            os.makedirs(output_valid_dir, exist_ok=True)
            
            # Dosya ismini al
            image_filename = os.path.basename(self.current_image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # 576x320 letterbox görseli kaydet
            output_image_path = os.path.join(output_images_dir, image_filename)
            letterbox_img.save(output_image_path)
            
            # Binary mask (object mask) oluştur ve kaydet (dosya adı aynı, sadece uzantı .png)
            binary_mask = create_binary_mask(adjusted_masks, target_size[0], target_size[1])
            mask_filename = f"{image_name}.png"
            output_mask_path = os.path.join(output_masks_dir, mask_filename)
            binary_mask.save(output_mask_path)
            
            # Valid mask oluştur ve kaydet - tamamen beyaz (576x320)
            valid_mask = Image.new('L', target_size, 255)  # Tamamen beyaz
            valid_filename = f"{image_name}.png"
            output_valid_path = os.path.join(output_valid_dir, valid_filename)
            valid_mask.save(output_valid_path)
            
            print(f"[BAŞARILI] UNet'e kaydedildi: {output_image_path}, {output_mask_path}, {output_valid_path}")
            
        except Exception as e:
            print(f"[HATA] UNet'e kaydetme hatası: {str(e)}", file=sys.stderr)
            traceback.print_exc()
    
    def add_point(self, event=None):
        """Nokta ekleme butonu/klavye kısayolu"""
        if not self.image_editor:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin!")
            return
        
        success = self.image_editor.add_point_at_last_click()
        if not success:
            # Sadece buton tıklamasında mesaj göster (klavye kısayolunda gösterme)
            if event is None:
                messagebox.showinfo("Bilgi", "Nokta eklemek için:\n1. Canvas'a tıklayarak eklemek istediğiniz yeri seçin\n2. 'A' tuşuna basın veya 'Nokta Ekle' butonuna tıklayın")
    
    def delete_point(self, event=None):
        """Nokta silme butonu/klavye kısayolu"""
        if not self.image_editor:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin!")
            return
        
        success = self.image_editor.delete_selected_point()
        if not success:
            # Sadece buton tıklamasında mesaj göster (klavye kısayolunda gösterme)
            if event is None:
                messagebox.showinfo("Bilgi", "Nokta silmek için:\n1. Silmek istediğiniz noktaya tıklayın (mavi/kırmızı nokta)\n2. 'D' tuşuna basın veya 'Nokta Sil' butonuna tıklayın")
    
    def open_viewer(self):
        """Kaydedilmiş görselleri doğrulama penceresini açar"""
        try:
            # Mevcut mod'u kullan
            current_mode = self.mode_var.get() if hasattr(self, 'mode_var') else self.mode
            open_viewer(self.output_base, current_mode)
        except Exception as e:
            error_msg = f"Doğrulama penceresi açılırken hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            messagebox.showerror("Hata", error_msg)
    
    def prev_image(self):
        """Önceki görsele geçer"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        else:
            info_msg = "İlk görsel!"
            print(f"[BİLGİ] {info_msg}")
            messagebox.showinfo("Bilgi", info_msg)
    
    def next_image(self):
        """Sonraki görsele geçer"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        else:
            info_msg = "Tüm görseller işlendi!"
            print(f"[BİLGİ] {info_msg}")
            messagebox.showinfo("Bilgi", info_msg)
    
    def delete_current_image(self):
        """Mevcut görseli siler ve deletedimages klasörüne taşır"""
        if not self.current_image_path or not self.image_files:
            messagebox.showwarning("Uyarı", "Silinecek görsel yok!")
            return
        
        # Onay al
        result = messagebox.askyesno(
            "Görsel Sil",
            f"Bu görseli silmek istediğinizden emin misiniz?\n\n{os.path.basename(self.current_image_path)}"
        )
        if not result:
            return
        
        try:
            # Deletedimages klasörünü oluştur
            deleted_dir = os.path.join(self.selected_folder, "deletedimages")
            os.makedirs(deleted_dir, exist_ok=True)
            
            # Görseli taşı
            image_filename = os.path.basename(self.current_image_path)
            dest_path = os.path.join(deleted_dir, image_filename)
            
            # Eğer hedef dosya varsa, üzerine yazma - benzersiz isim oluştur
            counter = 1
            base_name, ext = os.path.splitext(image_filename)
            while os.path.exists(dest_path):
                new_filename = f"{base_name}_{counter}{ext}"
                dest_path = os.path.join(deleted_dir, new_filename)
                counter += 1
            
            shutil.move(self.current_image_path, dest_path)
            print(f"[BİLGİ] Görsel silindi: {dest_path}")
            
            # İlişkili dosyaları da taşı (eğer output klasörlerinde varsa)
            image_name = os.path.splitext(image_filename)[0]
            
            # Normal mod dosyaları
            normal_image = os.path.join(self.output_base, "normal", "images", "train", image_filename)
            normal_label = os.path.join(self.output_base, "normal", "labels", "train", f"{image_name}.txt")
            if os.path.exists(normal_image):
                shutil.move(normal_image, os.path.join(deleted_dir, f"normal_{image_filename}"))
            if os.path.exists(normal_label):
                shutil.move(normal_label, os.path.join(deleted_dir, f"normal_{image_name}.txt"))
            
            # UNet dosyaları
            unet_image = os.path.join(self.output_base, "unet", "images", "train", image_filename)
            unet_mask = os.path.join(self.output_base, "unet", "masks", "train", f"{image_name}.png")
            unet_valid = os.path.join(self.output_base, "unet", "valid", "train", f"{image_name}.png")
            if os.path.exists(unet_image):
                shutil.move(unet_image, os.path.join(deleted_dir, f"unet_{image_filename}"))
            if os.path.exists(unet_mask):
                shutil.move(unet_mask, os.path.join(deleted_dir, f"unet_{image_name}.png"))
            if os.path.exists(unet_valid):
                shutil.move(unet_valid, os.path.join(deleted_dir, f"unet_valid_{image_name}.png"))
            
            # Listedeki görseli kaldır
            self.image_files.remove(self.current_image_path)
            
            # Eğer son görselse, bir öncekine geç
            if self.current_image_index >= len(self.image_files):
                self.current_image_index = len(self.image_files) - 1
            
            # Eğer liste boşaldıysa
            if len(self.image_files) == 0:
                messagebox.showinfo("Bilgi", "Tüm görseller silindi!")
                self.current_image_path = None
                self.canvas.delete("all")
                self.update_status()
                return
            
            # Bir sonraki görsele geç
            if self.current_image_index < len(self.image_files):
                self.load_current_image()
            else:
                self.current_image_index = len(self.image_files) - 1
                if self.current_image_index >= 0:
                    self.load_current_image()
            
            # Son görsel indeksini güncelle
            self.save_last_image_index()
            
            success_msg = f"Görsel silindi ve deletedimages klasörüne taşındı."
            print(f"[BAŞARILI] {success_msg}")
            messagebox.showinfo("Başarılı", success_msg)
            
        except Exception as e:
            error_msg = f"Görsel silinirken hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            messagebox.showerror("Hata", error_msg)
    
    def go_to_last_image(self):
        """Config'den kaydedilmiş son görsele gider"""
        if not self.image_files:
            messagebox.showwarning("Uyarı", "Yüklü görsel yok!")
            return
        
        saved_index = self.last_image_indices.get(self.mode)
        if saved_index is None:
            messagebox.showinfo("Bilgi", f"Bu mod ({self.mode}) için kaydedilmiş son görsel yok.")
            return
        
        if saved_index < 0 or saved_index >= len(self.image_files):
            messagebox.showwarning("Uyarı", f"Kaydedilmiş indeks ({saved_index + 1}) geçersiz. Toplam görsel sayısı: {len(self.image_files)}")
            return
        
        self.current_image_index = saved_index
        self.load_current_image()
        messagebox.showinfo("Bilgi", f"Son görsele gidildi: {saved_index + 1}/{len(self.image_files)}")
    
    def split_train_valid(self):
        """Tüm modlarda train verilerinin %10'unu validation'a ayırır"""
        try:
            # Kullanıcıya onay sor
            result = messagebox.askyesno(
                "Split Onayı",
                "Tüm modlarda train verilerinin %10'u validation'a ayrılacak.\n"
                "Bu işlem dosyaları train klasöründen valid klasörüne taşıyacak.\n\n"
                "Devam etmek istiyor musunuz?"
            )
            if not result:
                return
            
            total_moved = 0
            
            # Normal mod split
            normal_images_train = os.path.join(self.output_base, "normal", "images", "train")
            normal_images_valid = os.path.join(self.output_base, "normal", "images", "valid")
            normal_labels_train = os.path.join(self.output_base, "normal", "labels", "train")
            normal_labels_valid = os.path.join(self.output_base, "normal", "labels", "valid")
            
            moved = self._split_mode(
                normal_images_train, normal_images_valid,
                normal_labels_train, normal_labels_valid,
                "normal"
            )
            total_moved += moved
            
            # UNet split
            unet_images_train = os.path.join(self.output_base, "unet", "images", "train")
            unet_images_valid = os.path.join(self.output_base, "unet", "images", "valid")
            unet_masks_train = os.path.join(self.output_base, "unet", "masks", "train")
            unet_masks_valid = os.path.join(self.output_base, "unet", "masks", "valid")
            unet_valid_train = os.path.join(self.output_base, "unet", "valid", "train")
            unet_valid_valid = os.path.join(self.output_base, "unet", "valid", "valid")
            
            moved = self._split_unet_mode(
                unet_images_train, unet_images_valid,
                unet_masks_train, unet_masks_valid,
                unet_valid_train, unet_valid_valid
            )
            total_moved += moved
            
            success_msg = f"Split tamamlandı! Toplam {total_moved} görsel validation'a taşındı."
            print(f"[BAŞARILI] {success_msg}")
            messagebox.showinfo("Başarılı", success_msg)
            
        except Exception as e:
            error_msg = f"Split sırasında hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            messagebox.showerror("Hata", error_msg)
    
    def _split_mode(self, images_train_dir, images_valid_dir, labels_train_dir, labels_valid_dir, mode_name):
        """Bir mod için train/valid split yapar (Normal için)"""
        if not os.path.exists(images_train_dir):
            print(f"[UYARI] {mode_name} mod için train klasörü bulunamadı: {images_train_dir}")
            return 0
        
        # Valid klasörlerini oluştur
        os.makedirs(images_valid_dir, exist_ok=True)
        os.makedirs(labels_valid_dir, exist_ok=True)
        
        # Train klasöründeki görsel dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_train_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_train_dir, ext.upper())))
        
        if len(image_files) == 0:
            print(f"[UYARI] {mode_name} mod için train klasöründe görsel bulunamadı")
            return 0
        
        # %10'unu seç (en az 1 dosya)
        num_valid = max(1, int(len(image_files) * 0.2))
        random.shuffle(image_files)
        valid_images = image_files[:num_valid]
        
        moved_count = 0
        for image_path in valid_images:
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Görseli taşı
            dest_image = os.path.join(images_valid_dir, image_filename)
            shutil.move(image_path, dest_image)
            
            # Etiket dosyasını taşı
            label_file = os.path.join(labels_train_dir, f"{image_name}.txt")
            if os.path.exists(label_file):
                dest_label = os.path.join(labels_valid_dir, f"{image_name}.txt")
                shutil.move(label_file, dest_label)
            
            moved_count += 1
            print(f"[BİLGİ] {mode_name}: {image_filename} validation'a taşındı")
        
        print(f"[BAŞARILI] {mode_name} mod: {moved_count} görsel validation'a taşındı")
        return moved_count
    
    def _split_unet_mode(self, images_train_dir, images_valid_dir, masks_train_dir, masks_valid_dir, valid_train_dir, valid_valid_dir):
        """UNet mod için train/valid split yapar"""
        if not os.path.exists(images_train_dir):
            print(f"[UYARI] UNet mod için train klasörü bulunamadı: {images_train_dir}")
            return 0
        
        # Valid klasörlerini oluştur
        os.makedirs(images_valid_dir, exist_ok=True)
        os.makedirs(masks_valid_dir, exist_ok=True)
        os.makedirs(valid_valid_dir, exist_ok=True)
        
        # Train klasöründeki görsel dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_train_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_train_dir, ext.upper())))
        
        if len(image_files) == 0:
            print(f"[UYARI] UNet mod için train klasöründe görsel bulunamadı")
            return 0
        
        # %10'unu seç (en az 1 dosya)
        num_valid = max(1, int(len(image_files) * 0.2))
        random.shuffle(image_files)
        valid_images = image_files[:num_valid]
        
        moved_count = 0
        for image_path in valid_images:
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Görseli taşı
            dest_image = os.path.join(images_valid_dir, image_filename)
            shutil.move(image_path, dest_image)
            
            # Mask dosyasını taşı
            mask_file = os.path.join(masks_train_dir, f"{image_name}.png")
            if os.path.exists(mask_file):
                dest_mask = os.path.join(masks_valid_dir, f"{image_name}.png")
                shutil.move(mask_file, dest_mask)
            
            # Valid mask dosyasını taşı
            valid_file = os.path.join(valid_train_dir, f"{image_name}.png")
            if os.path.exists(valid_file):
                dest_valid = os.path.join(valid_valid_dir, f"{image_name}.png")
                shutil.move(valid_file, dest_valid)
            
            moved_count += 1
            print(f"[BİLGİ] UNet: {image_filename} validation'a taşındı")
        
        print(f"[BAŞARILI] UNet mod: {moved_count} görsel validation'a taşındı")
        return moved_count


def main():
    """Ana fonksiyon"""
    root = tk.Tk()
    app = YOLOSegmentationEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
