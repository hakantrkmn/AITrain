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
from mods import NormalMode, BBModMode, UNetMode
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
        self.mode = 'normal'  # 'normal', 'bbmod' veya 'unet'
        self.mode_state = {}  # Mod'a özel state bilgisi
        self.epsilon_factor = 0.002  # Polygon basitleştirme faktörü (varsayılan)
        
        # Mod sınıfları
        self.output_base = "output"
        self.mode_handlers = {
            'normal': NormalMode(self.output_base),
            'bbmod': BBModMode(self.output_base),
            'unet': UNetMode(self.output_base)
        }
        
        # Config dosyası yolu
        self.config_path = "config.json"
        
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
            self.mode = new_mode
            print(f"[BİLGİ] Mod değiştirildi: {self.mode}")
            # Mevcut düzenlemeleri sıfırla
            self.current_masks_data = []
            self.mode_state = {}
            if self.image_editor:
                self.canvas.delete("all")
    
    def on_epsilon_change(self, event=None):
        """Epsilon değeri değiştiğinde çağrılır"""
        try:
            new_epsilon = self.epsilon_var.get()
            if 0.001 <= new_epsilon <= 0.1:
                self.epsilon_factor = new_epsilon
                print(f"[BİLGİ] Epsilon faktörü güncellendi: {self.epsilon_factor:.3f}")
            else:
                # Geçersiz değer, eski değere geri dön
                self.epsilon_var.set(self.epsilon_factor)
        except:
            # Hata durumunda eski değere geri dön
            self.epsilon_var.set(self.epsilon_factor)
    
    def on_epsilon_change(self, event=None):
        """Epsilon değeri değiştiğinde çağrılır"""
        try:
            new_epsilon = self.epsilon_var.get()
            if 0.001 <= new_epsilon <= 0.1:
                self.epsilon_factor = new_epsilon
                print(f"[BİLGİ] Epsilon faktörü güncellendi: {self.epsilon_factor:.3f}")
            else:
                # Geçersiz değer, eski değere geri dön
                self.epsilon_var.set(self.epsilon_factor)
        except:
            # Hata durumunda eski değere geri dön
            self.epsilon_var.set(self.epsilon_factor)
    
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
            values=["normal", "bbmod", "unet"],
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
            from_=0.001,
            to=0.1,
            increment=0.001,
            textvariable=self.epsilon_var,
            width=8,
            format="%.3f",
            command=self.on_epsilon_change
        )
        epsilon_spinbox.pack(side=tk.LEFT, padx=2)
        epsilon_spinbox.bind("<Return>", self.on_epsilon_change)
        epsilon_spinbox.bind("<FocusOut>", self.on_epsilon_change)
        
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
        self.canvas.bind('<Right>', lambda e: self.next_image())  # Sonraki görsel
        self.canvas.bind('<Left>', lambda e: self.prev_image())  # Önceki görsel
        
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
        
        # Durum etiketi
        self.status_label = tk.Label(
            bottom_frame,
            text="Hazır",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Klavye kısayolları bilgisi
        help_label = tk.Label(
            bottom_frame,
            text="Kısayollar: A=Ekle, D=Sil, S=Tümüne Kaydet, →=Sonraki, ←=Önceki",
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
    
    def save_config(self):
        """Mevcut seçimleri config dosyasına kaydeder"""
        save_config_util(
            folder=self.selected_folder,
            model=self.selected_model,
            config_path=self.config_path
        )
    
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
        
        # İlk görseli yükle
        self.current_image_index = 0
        self.load_current_image()
    
    def load_current_image(self):
        """Mevcut görseli yükler ve işler"""
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_files):
            return
        
        self.current_image_path = self.image_files[self.current_image_index]
        
        try:
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
            
            # Durum güncelle
            self.update_status()
            
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
        """Mevcut düzenlemeyi tüm modlara (normal, bbmod, unet) kaydeder"""
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
                # Normal modda düzenlenmiş, BBMod ve UNet'e de kaydet
                self.save_to_bbmod()
                self.save_to_unet()
            elif self.mode == 'bbmod':
                # BBMod'da düzenlenmiş, Normal ve UNet'e de kaydet
                self.save_to_normal()
                self.save_to_unet()
            elif self.mode == 'unet':
                # UNet'te düzenlenmiş, Normal ve BBMod'a da kaydet
                self.save_to_normal()
                self.save_to_bbmod()
            
            # Sadece bir kere başarılı mesajı göster
            success_msg = f"Görsel ve etiketler tüm modlara kaydedildi!"
            print(f"[BAŞARILI] {success_msg}")
            messagebox.showinfo("Başarılı", success_msg)
            
        except Exception as e:
            error_msg = f"Tümüne kaydetme sırasında hata: {str(e)}"
            print(f"[HATA] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            messagebox.showerror("Hata", error_msg)
    
    def save_to_bbmod(self):
        """Normal veya UNet modunda düzenlenen noktaları BBMod'a kaydeder"""
        from utils.image_utils import crop_image_by_bbox, adjust_polygons_to_crop
        
        try:
            # Düzenlenmiş noktaları al
            edited_data = self.image_editor.get_edited_points()
            
            if not edited_data:
                return
            
            # Mod'a göre dönüşüm yap
            if self.mode == 'unet':
                # UNet'ten: letterbox koordinatlarından kırpılmış görsel koordinatlarına
                letterbox_offset = self.mode_state.get('letterbox_offset')
                letterbox_scale = self.mode_state.get('letterbox_scale')
                
                if not letterbox_offset or not letterbox_scale:
                    print("[UYARI] UNet'ten BBMod'a kaydetmek için gerekli bilgiler eksik")
                    return
                
                # Letterbox koordinatlarından kırpılmış görsel koordinatlarına çevir
                offset_x, offset_y = letterbox_offset
                edited_data = []
                for mask_data in self.image_editor.get_edited_points():
                    polygon = mask_data['polygon']
                    # Letterbox offset'ini çıkar, scale'i geri al
                    adjusted_polygon = [((x - offset_x) / letterbox_scale, (y - offset_y) / letterbox_scale) 
                                       for x, y in polygon]
                    edited_data.append({
                        'polygon': adjusted_polygon,
                        'class_id': mask_data.get('class_id', 0)
                    })
            
            # YOLO'dan bounding box'ları al (UNet'ten geliyorsa zaten crop_box var, ama yine de kontrol edelim)
            if self.mode == 'normal':
                result = self.yolo_processor.process_image(self.current_image_path, epsilon_factor=self.epsilon_factor)
                bounding_boxes = result['boxes']
                
                if len(bounding_boxes) == 0:
                    print("[UYARI] BBMod'a kaydetmek için bounding box bulunamadı")
                    return
                
                # En büyük bounding box'ı kullan
                largest_box = max(bounding_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cropped_image, crop_box = crop_image_by_bbox(self.current_image_path, largest_box)
                
                # Polygon koordinatlarını kırpılmış görsele göre ayarla
                adjusted_data = adjust_polygons_to_crop(edited_data, crop_box)
            else:
                # UNet'ten geliyorsa, zaten kırpılmış görsel koordinatlarında
                crop_box = self.mode_state.get('crop_box')
                if not crop_box:
                    print("[UYARI] BBMod'a kaydetmek için crop_box bilgisi yok")
                    return
                
                # Görseli yeniden kırp (kaydetmek için)
                from PIL import Image
                img = Image.open(self.current_image_path)
                cropped_image = img.crop(crop_box)
                adjusted_data = edited_data
            
            if not adjusted_data:
                print("[UYARI] BBMod'a kaydedilecek geçerli polygon yok")
                return
            
            # BBMod klasörlerine kaydet
            output_images_dir = os.path.join(self.output_base, "bbmod", "images", "train")
            output_labels_dir = os.path.join(self.output_base, "bbmod", "labels", "train")
            
            image_filename = os.path.basename(self.current_image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Kırpılmış görseli kaydet
            output_image_path = os.path.join(output_images_dir, image_filename)
            cropped_image.save(output_image_path)
            
            # Kırpılmış görsel boyutlarını al
            img_width, img_height = cropped_image.size
            
            # Etiket dosyasını oluştur
            output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
            
            with open(output_label_path, 'w') as f:
                for mask_data in adjusted_data:
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
            
        except Exception as e:
            print(f"[HATA] BBMod'a kaydetme hatası: {str(e)}", file=sys.stderr)
            traceback.print_exc()
    
    def save_to_normal(self):
        """BBMod veya UNet'te düzenlenen noktaları Normal mod'a kaydeder"""
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
                
            elif self.mode == 'bbmod':
                # BBMod'dan: kırpılmış görsel koordinatlarından orijinal görsel koordinatlarına
                crop_box = self.mode_state.get('crop_box')
                if not crop_box:
                    print("[UYARI] Normal'e kaydetmek için crop_box bilgisi yok")
                    return
                
                adjusted_data = adjust_polygons_from_crop(edited_data, crop_box)
            else:
                # Normal moddan: zaten orijinal görsel koordinatlarında
                adjusted_data = edited_data
            
            if not adjusted_data:
                print("[UYARI] Normal'e kaydedilecek geçerli polygon yok")
                return
            
            # Normal klasörlerine kaydet
            output_images_dir = os.path.join(self.output_base, "normal", "images", "train")
            output_labels_dir = os.path.join(self.output_base, "normal", "labels", "train")
            
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
        """Normal veya BBMod'da düzenlenen noktaları UNet formatına kaydeder"""
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
            elif self.mode == 'bbmod':
                # BBMod'dan geliyorsa, zaten kırpılmış görsele göre
                adjusted_masks_crop = edited_data
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
            
            # 320x320'e letterbox resize
            letterbox_img, offset_x, offset_y, scale, new_width, new_height = letterbox_resize(cropped_image, (320, 320))
            
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
            
            # 320x320 letterbox görseli kaydet
            output_image_path = os.path.join(output_images_dir, image_filename)
            letterbox_img.save(output_image_path)
            
            # Binary mask (object mask) oluştur ve kaydet (dosya adı aynı, sadece uzantı .png)
            binary_mask = create_binary_mask(adjusted_masks, 320, 320)
            mask_filename = f"{image_name}.png"
            output_mask_path = os.path.join(output_masks_dir, mask_filename)
            binary_mask.save(output_mask_path)
            
            # Valid mask oluştur ve kaydet (valid klasörüne, dosya adı aynı, sadece uzantı .png)
            valid_mask = create_valid_mask((320, 320), offset_x, offset_y, new_width, new_height)
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
            
            # BBMod split
            bbmod_images_train = os.path.join(self.output_base, "bbmod", "images", "train")
            bbmod_images_valid = os.path.join(self.output_base, "bbmod", "images", "valid")
            bbmod_labels_train = os.path.join(self.output_base, "bbmod", "labels", "train")
            bbmod_labels_valid = os.path.join(self.output_base, "bbmod", "labels", "valid")
            
            moved = self._split_mode(
                bbmod_images_train, bbmod_images_valid,
                bbmod_labels_train, bbmod_labels_valid,
                "bbmod"
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
        """Bir mod için train/valid split yapar (Normal ve BBMod için)"""
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
        num_valid = max(1, int(len(image_files) * 0.1))
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
        num_valid = max(1, int(len(image_files) * 0.1))
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
