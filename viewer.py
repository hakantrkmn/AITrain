"""
Kaydedilmiş Görselleri ve Maske Noktalarını Görüntüleme Modülü
Doğrulama için kaydedilmiş görselleri ve etiketleri gösterir.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk
import glob


class LabelViewer:
    """Kaydedilmiş görselleri ve maske noktalarını görüntüleme penceresi"""
    
    def __init__(self, root, output_base="output", mode="normal"):
        self.root = root
        self.root.title(f"Kaydedilmiş Görselleri Doğrula - {mode.upper()} Mod")
        self.root.geometry("1000x700")
        
        self.output_base = output_base
        self.mode = mode
        
        # Mod'a göre klasör yapısı
        if mode == "unet":
            self.images_dir = os.path.join(output_base, "unet", "images", "train")
            self.masks_dir = os.path.join(output_base, "unet", "masks", "train")
            self.labels_dir = None
        else:
            self.images_dir = os.path.join(output_base, "normal", "images", "train")
            self.labels_dir = os.path.join(output_base, "normal", "labels", "train")
            self.masks_dir = None
        
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.current_label_path = None
        self.current_mask_path = None
        self.polygon_points = []
        
        self.create_widgets()
        self.load_images()
    
    def create_widgets(self):
        """GUI widget'larını oluşturur"""
        
        # Üst frame
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Mod bilgisi
        mode_label = tk.Label(
            top_frame,
            text=f"Mod: {self.mode.upper()}",
            font=("Arial", 10, "bold"),
            fg="blue"
        )
        mode_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Klasör Seç",
            command=self.select_folder,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Önceki",
            command=self.prev_image,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Sonraki",
            command=self.next_image,
            width=15,
            height=2
        ).pack(side=tk.LEFT, padx=5)
        
        # Durum etiketi
        self.status_label = tk.Label(
            top_frame,
            text="Hazır",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
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
        
        # Alt frame - Bilgi
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.info_label = tk.Label(
            bottom_frame,
            text="Kaydedilmiş görselleri görüntülemek için klasör seçin",
            font=("Arial", 9),
            fg="gray"
        )
        self.info_label.pack()
    
    def select_folder(self):
        """Çıktı klasörünü seçer"""
        folder = filedialog.askdirectory(
            title=f"Kaydedilmiş görsellerin bulunduğu klasörü seçin ({self.mode.upper()} mod)",
            initialdir=self.images_dir if os.path.exists(self.images_dir) else "."
        )
        if folder:
            self.images_dir = folder
            # Labels/Masks klasörünü bul (aynı seviyede olmalı)
            parent_dir = os.path.dirname(os.path.dirname(folder))
            if os.path.basename(folder) == "train":
                # Mod'a göre labels/masks klasörünü bul
                if self.mode == "unet":
                    self.masks_dir = os.path.join(parent_dir, "unet", "masks", "train")
                    self.labels_dir = None
                else:
                    self.labels_dir = os.path.join(parent_dir, "normal", "labels", "train")
                    self.masks_dir = None
            else:
                # Aynı klasörde labels/masks klasörü ara
                if self.mode == "unet":
                    masks_candidate = os.path.join(os.path.dirname(folder), "masks", "train")
                    if os.path.exists(masks_candidate):
                        self.masks_dir = masks_candidate
                    else:
                        self.masks_dir = os.path.join(os.path.dirname(os.path.dirname(folder)), "unet", "masks", "train")
                    self.labels_dir = None
                else:
                    labels_candidate = os.path.join(os.path.dirname(folder), "labels", "train")
                    if os.path.exists(labels_candidate):
                        self.labels_dir = labels_candidate
                    else:
                        self.labels_dir = os.path.join(os.path.dirname(os.path.dirname(folder)), "normal", "labels", "train")
                    self.masks_dir = None
            
            self.load_images()
    
    def load_images(self):
        """Görsel dosyalarını yükler"""
        if not os.path.exists(self.images_dir):
            messagebox.showwarning("Uyarı", f"Klasör bulunamadı: {self.images_dir}")
            return
        
        # Görsel dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext.upper())))
        
        if not self.image_files:
            messagebox.showwarning("Uyarı", "Seçilen klasörde görsel dosyası bulunamadı!")
            return
        
        # İlk görseli yükle
        self.current_image_index = 0
        self.load_current_image()
    
    def load_current_image(self):
        """Mevcut görseli ve etiketini yükler"""
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_files):
            return
        
        self.current_image_path = self.image_files[self.current_image_index]
        
        # Etiket/Mask dosyasını bul
        image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        if self.mode == "unet":
            # UNet: Mask dosyasını bul (dosya adı aynı, sadece uzantı .png)
            mask_filename = f"{image_name}.png"
            self.current_mask_path = os.path.join(self.masks_dir, mask_filename)
            self.current_label_path = None
        else:
                # Normal: Label dosyasını bul
            self.current_label_path = os.path.join(self.labels_dir, f"{image_name}.txt")
            self.current_mask_path = None
        
        try:
            # Görseli yükle
            img = Image.open(self.current_image_path)
            img_width, img_height = img.size
            
            # Canvas boyutlarına göre ölçekle
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            
            # Görseli canvas'a sığdır
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale_factor = min(scale_w, scale_h, 1.0)
            
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Merkeze hizala
            offset_x = (canvas_width - new_width) // 2
            offset_y = (canvas_height - new_height) // 2
            
            # Görseli ölçekle
            resized_image = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(resized_image)
            
            # Canvas'ı temizle
            self.canvas.delete("all")
            
            # Görseli çiz
            self.canvas.create_image(
                offset_x + new_width // 2,
                offset_y + new_height // 2,
                image=self.image_tk,
                anchor=tk.CENTER
            )
            
            # Etiket/Mask dosyasını oku ve çiz
            self.polygon_points = []
            if self.mode == "unet":
                # UNet: Binary mask'ı göster
                if self.current_mask_path and os.path.exists(self.current_mask_path):
                    self.load_and_draw_mask(offset_x, offset_y, scale_factor, img_width, img_height)
                else:
                    self.info_label.config(text=f"Mask dosyası bulunamadı: {os.path.basename(self.current_mask_path) if self.current_mask_path else 'N/A'}")
            else:
                # Normal: YOLO label dosyasını oku
                if self.current_label_path and os.path.exists(self.current_label_path):
                    self.load_and_draw_labels(offset_x, offset_y, scale_factor, img_width, img_height)
                else:
                    self.info_label.config(text=f"Etiket dosyası bulunamadı: {os.path.basename(self.current_label_path) if self.current_label_path else 'N/A'}")
            
            # Durum güncelle
            self.update_status()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Görsel yüklenirken hata: {str(e)}")
    
    def load_and_draw_labels(self, offset_x, offset_y, scale_factor, img_width, img_height):
        """Etiket dosyasını oku ve polygon'ları çiz"""
        try:
            with open(self.current_label_path, 'r') as f:
                lines = f.readlines()
            
            polygon_count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 7:  # class_id + en az 3 nokta (6 koordinat)
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Normalize koordinatları piksel koordinatlarına çevir
                pixel_points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        x_norm = coords[i]
                        y_norm = coords[i + 1]
                        x_pixel = x_norm * img_width
                        y_pixel = y_norm * img_height
                        pixel_points.append((x_pixel, y_pixel))
                
                if len(pixel_points) >= 3:
                    # Canvas koordinatlarına çevir
                    canvas_points = []
                    for x, y in pixel_points:
                        canvas_x = offset_x + x * scale_factor
                        canvas_y = offset_y + y * scale_factor
                        canvas_points.append((canvas_x, canvas_y))
                    
                    # Polygon çiz (yeşil)
                    if len(canvas_points) >= 3:
                        flat_points = [coord for point in canvas_points for coord in point]
                        self.canvas.create_polygon(
                            flat_points,
                            outline='green',
                            fill='',
                            width=2,
                            tags='polygon'
                        )
                        
                        # Noktaları çiz (mavi)
                        for x, y in canvas_points:
                            self.canvas.create_oval(
                                x - 4, y - 4, x + 4, y + 4,
                                fill='blue',
                                outline='white',
                                width=1,
                                tags='point'
                            )
                        
                        polygon_count += 1
            
            self.info_label.config(
                text=f"Görsel: {os.path.basename(self.current_image_path)} | "
                     f"Etiket: {os.path.basename(self.current_label_path)} | "
                     f"{polygon_count} polygon bulundu"
            )
            
        except Exception as e:
            self.info_label.config(text=f"Etiket okuma hatası: {str(e)}")
    
    def load_and_draw_mask(self, offset_x, offset_y, scale_factor, img_width, img_height):
        """UNet binary mask'ı yükle ve görselin üzerine overlay olarak çiz"""
        try:
            # Mask'ı yükle
            mask_img = Image.open(self.current_mask_path)
            mask_width, mask_height = mask_img.size
            
            # UNet mask'ları 768x768 olmalı, görsel de 768x768
            # Mask'ı görsel boyutuna göre ölçekle (genelde aynı boyutta olacaklar)
            if mask_width != img_width or mask_height != img_height:
                # Boyutlar farklıysa resize et
                resized_mask = mask_img.resize((img_width, img_height), Image.Resampling.NEAREST)
            else:
                resized_mask = mask_img
            
            # Mask'ı RGBA'ya çevir (yeşil overlay için)
            import numpy as np
            mask_array = np.array(resized_mask)
            mask_rgba_array = np.zeros((img_height, img_width, 4), dtype=np.uint8)
            
            # Beyaz pikselleri (255) yeşil yarı saydam yap
            white_pixels = mask_array == 255
            mask_rgba_array[white_pixels] = [0, 255, 0, 128]  # Yeşil, %50 saydam
            
            mask_rgba = Image.fromarray(mask_rgba_array, 'RGBA')
            
            # Canvas boyutlarına göre ölçekle
            canvas_mask_width = int(img_width * scale_factor)
            canvas_mask_height = int(img_height * scale_factor)
            mask_rgba_scaled = mask_rgba.resize((canvas_mask_width, canvas_mask_height), Image.Resampling.NEAREST)
            
            # Mask'ı canvas'a çiz (overlay)
            mask_tk = ImageTk.PhotoImage(mask_rgba_scaled)
            self.mask_tk = mask_tk  # Referansı sakla
            
            # Canvas'a mask overlay ekle (görselin üzerine)
            self.canvas.create_image(
                offset_x + canvas_mask_width // 2,
                offset_y + canvas_mask_height // 2,
                image=mask_tk,
                anchor=tk.CENTER
            )
            
            self.info_label.config(
                text=f"Görsel: {os.path.basename(self.current_image_path)} | "
                     f"Mask: {os.path.basename(self.current_mask_path)}"
            )
            
        except Exception as e:
            self.info_label.config(text=f"Mask okuma hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_status(self):
        """Durum etiketini günceller"""
        total = len(self.image_files)
        current = self.current_image_index + 1
        self.status_label.config(text=f"Durum: {current}/{total}")
    
    def prev_image(self):
        """Önceki görsele geçer"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        else:
            messagebox.showinfo("Bilgi", "İlk görsel!")
    
    def next_image(self):
        """Sonraki görsele geçer"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        else:
            messagebox.showinfo("Bilgi", "Son görsel!")


def open_viewer(output_base="output", mode="normal"):
    """Viewer penceresini açar"""
    viewer_window = tk.Toplevel()
    app = LabelViewer(viewer_window, output_base, mode)
    return app
