import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
import time
import json

# PyTorch imports (fallback için)
import torch
import torch.nn as nn
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("WARNING: segmentation_models_pytorch not installed. PyTorch fallback disabled.")

# --- SABİTLER ---
DEFAULT_UNET_MODEL_PATH = "/home/acomaster/Belgeler/AITrain/Modeller/claudunet/unet_model_legacy.onnx"
DEFAULT_YOLO_MODEL_PATH = "/home/acomaster/Belgeler/AITrain/Modeller/800yolo11finetune/best.pt"
DEFAULT_JSON_PATH = "/home/acomaster/Belgeler/AITrain/Modeller/unet5762k/best_threshold.json"
DEFAULT_IMG_DIR = "/home/acomaster/Belgeler/AITrain/unettestimages"
TARGET_W, TARGET_H = 576, 320


def crop_image_by_bbox(img: np.ndarray, bbox: tuple) -> tuple:
    """
    Bounding box'a göre görseli kırpar (padding ile).
    """
    img_height, img_width = img.shape[:2]
    
    x1, y1, x2, y2 = bbox
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    padding_left_right = int(round(box_width * 0.1))
    padding_top_bottom = int(round(box_height * 0.1))
    
    x1_new = max(0, int(round(x1 - padding_left_right)))
    y1_new = max(0, int(round(y1 - padding_top_bottom)))
    x2_new = min(img_width, int(round(x2 + padding_left_right)))
    y2_new = min(img_height, int(round(y2 + padding_top_bottom)))
    
    crop_box = (x1_new, y1_new, x2_new, y2_new)
    cropped_img = img[y1_new:y2_new, x1_new:x2_new]
    
    return cropped_img, crop_box


def letterbox_resize(img: np.ndarray, target_size: tuple = (576, 320)) -> tuple:
    """
    Görseli letterbox ile resize eder (aspect ratio korunur).
    """
    target_width, target_height = target_size
    img_height, img_width = img.shape[:2]
    
    scale = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    new_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    new_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    
    return new_image, offset_x, offset_y, scale, new_width, new_height


class ZoomableCanvas(tk.Canvas):
    """Mouse ile zoom ve pan yapılabilen canvas"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.image = None
        self.photo = None
        self.image_id = None
        
        # Zoom ve pan değişkenleri
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Mouse drag için
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Event bindings
        self.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.bind("<Button-4>", self.on_mousewheel)    # Linux scroll up
        self.bind("<Button-5>", self.on_mousewheel)    # Linux scroll down
        self.bind("<ButtonPress-1>", self.on_drag_start)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Double-Button-1>", self.reset_view)
        
        # Linux'ta scroll için focus gerekli - mouse girdiğinde focus al
        self.bind("<Enter>", self._on_enter)
        
    def _on_enter(self, event):
        """Mouse canvas'a girdiğinde focus al (Linux scroll için gerekli)"""
        self.focus_set()
        
    def set_image(self, pil_image):
        """Görseli ayarla"""
        self.image = pil_image
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()
        
    def update_display(self):
        """Görseli zoom ve pan'a göre güncelle"""
        if self.image is None:
            return
        
        # Canvas boyutları
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400
        
        # Zoom uygula
        new_width = int(self.image.width * self.zoom_level)
        new_height = int(self.image.height * self.zoom_level)
        
        if new_width > 0 and new_height > 0:
            resized = self.image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(resized)
            
            # Merkez koordinatları + pan offset
            x = canvas_width // 2 + self.pan_x
            y = canvas_height // 2 + self.pan_y
            
            # Önceki görseli sil ve yenisini ekle
            self.delete("all")
            self.image_id = self.create_image(x, y, image=self.photo, anchor=tk.CENTER)
    
    def on_mousewheel(self, event):
        """Mouse wheel ile zoom"""
        # Zoom faktörü
        if event.num == 4 or event.delta > 0:  # Scroll up - zoom in
            factor = 1.2
        elif event.num == 5 or event.delta < 0:  # Scroll down - zoom out
            factor = 0.8
        else:
            return "break"
        
        new_zoom = self.zoom_level * factor
        
        # Zoom sınırları
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_level = new_zoom
            self.update_display()
        
        return "break"  # Event'in başka yere gitmesini engelle
    
    def on_drag_start(self, event):
        """Drag başlangıcı"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    
    def on_drag(self, event):
        """Drag ile pan"""
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        self.pan_x += dx
        self.pan_y += dy
        
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        self.update_display()
    
    def reset_view(self, event=None):
        """Çift tıkla görünümü sıfırla"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()


class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UNet Segmentasyon Test Aracı (ONNX/PyTorch + YOLO Detect)")
        self.root.geometry("1600x900")

        # --- Değişkenler ---
        self.image_paths = []
        self.current_index = 0
        self.unet_session = None
        self.pytorch_model = None  # PyTorch fallback için
        self.model_type = None     # 'onnx' veya 'pytorch'
        self.device = None         # PyTorch device
        self.yolo_model = None
        self.current_logits = None
        self.current_original_img = None
        self.current_cropped_img = None
        self.current_letterbox_img = None
        self.current_crop_box = None
        self.current_letterbox_params = None  # (offset_x, offset_y, scale, new_w, new_h)
        self.current_yolo_mask = None  # YOLO segmentasyon maskesi
        self.current_onnx_path = None  # Mevcut ONNX model yolu (mod değişikliğinde kullanılır)
        self.current_full_probs = None  # Orijinal boyutta UNet olasılık haritası (piksel değeri için)
        self.start_threshold = 0.5
        
        self.load_initial_settings()
        self.setup_ui()
        
        if os.path.exists(DEFAULT_UNET_MODEL_PATH):
            self.load_unet_model(DEFAULT_UNET_MODEL_PATH)
        else:
            self.status_label.config(text="UNet modeli bulunamadı, lütfen manuel seçin.", fg="red")

        if os.path.exists(DEFAULT_YOLO_MODEL_PATH):
            self.load_yolo_model(DEFAULT_YOLO_MODEL_PATH)
        else:
            self.yolo_status_label.config(text="YOLO modeli bulunamadı", fg="red")

        if os.path.exists(DEFAULT_IMG_DIR):
            self.load_images_from_folder(DEFAULT_IMG_DIR)

    def load_initial_settings(self):
        try:
            if os.path.exists(DEFAULT_JSON_PATH):
                with open(DEFAULT_JSON_PATH, "r") as f:
                    data = json.load(f)
                    self.start_threshold = data.get("threshold", 0.5)
                    print(f"JSON Threshold yüklendi: {self.start_threshold}")
        except Exception as e:
            print(f"JSON okuma hatası: {e}")

    def setup_ui(self):
        # 1. ÜST PANEL (Kontroller)
        control_frame = tk.Frame(self.root, pady=10, bg="#f0f0f0")
        control_frame.pack(fill=tk.X)

        btn_folder = tk.Button(control_frame, text="📁 Klasör Seç", command=self.select_folder, bg="#2196F3", fg="white")
        btn_folder.pack(side=tk.LEFT, padx=10)
        
        btn_yolo = tk.Button(control_frame, text="🎯 YOLO Model (.pt)", command=self.select_yolo_model, bg="#FF5722", fg="white")
        btn_yolo.pack(side=tk.LEFT, padx=5)
        
        self.yolo_status_label = tk.Label(control_frame, text="YOLO: -", bg="#f0f0f0", font=("Arial", 8))
        self.yolo_status_label.pack(side=tk.LEFT, padx=5)
        
        btn_unet = tk.Button(control_frame, text="🧠 UNet Model (.onnx/.pth)", command=self.select_unet_model, bg="#9C27B0", fg="white")
        btn_unet.pack(side=tk.LEFT, padx=5)
        
        # ONNX Mod Seçimi (CPU/GPU)
        tk.Label(control_frame, text="ONNX:", bg="#f0f0f0", font=("Arial", 8)).pack(side=tk.LEFT, padx=(10, 2))
        self.onnx_mode_var = tk.StringVar(value="GPU")
        self.onnx_mode_combo = ttk.Combobox(control_frame, textvariable=self.onnx_mode_var, 
                                            values=["GPU", "CPU"], width=5, state="readonly")
        self.onnx_mode_combo.pack(side=tk.LEFT, padx=2)
        self.onnx_mode_combo.bind("<<ComboboxSelected>>", self.on_onnx_mode_change)

        tk.Label(control_frame, text="Threshold:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(20, 5))
        
        self.thresh_var = tk.DoubleVar(value=self.start_threshold)
        self.slider = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.001, 
                               orient=tk.HORIZONTAL, variable=self.thresh_var, length=250, 
                               command=self.on_slider_change)
        self.slider.pack(side=tk.LEFT, padx=5)
        
        # Piksel olasılık değeri gösterici
        self.pixel_prob_label = tk.Label(control_frame, text="Piksel: -", bg="#f0f0f0", 
                                         font=("Consolas", 9), fg="#9C27B0", width=20)
        self.pixel_prob_label.pack(side=tk.LEFT, padx=10)

        self.file_label = tk.Label(control_frame, text="Dosya: -", bg="#f0f0f0", font=("Arial", 10, "bold"))
        self.file_label.pack(side=tk.RIGHT, padx=20)

        # 2. ORTA PANEL (Görseller) - 2 satır
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Üst satır: 4 küçük panel
        top_row = tk.Frame(self.image_frame)
        top_row.pack(fill=tk.X, pady=5)
        
        for i in range(4):
            top_row.columnconfigure(i, weight=1)

        self.panel_orig = self.create_image_panel(top_row, "Orijinal + YOLO Detect", 0)
        self.panel_cropped = self.create_image_panel(top_row, "Kırpılmış (Letterbox 576x320)", 1)
        self.panel_heat = self.create_image_panel(top_row, "Olasılık (Isı) Haritası", 2)
        self.panel_mask = self.create_image_panel(top_row, "Final Maske (Threshold)", 3)

        # Alt satır: İki büyük zoomable overlay panel (YOLO ve UNet)
        bottom_row = tk.Frame(self.image_frame)
        bottom_row.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # İki sütun için grid ayarla
        bottom_row.columnconfigure(0, weight=1)
        bottom_row.columnconfigure(1, weight=1)
        bottom_row.rowconfigure(0, weight=1)
        
        # Sol panel: YOLO Segmentasyon Maskesi
        yolo_overlay_frame = tk.Frame(bottom_row, bd=2, relief=tk.GROOVE)
        yolo_overlay_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        
        yolo_title_frame = tk.Frame(yolo_overlay_frame)
        yolo_title_frame.pack(fill=tk.X)
        
        tk.Label(yolo_title_frame, text="🎯 YOLO Segmentasyon Maskesi (Zoom: Scroll, Pan: Drag)", 
                 font=("Arial", 10, "bold"), fg="#FF5722").pack(side=tk.LEFT, padx=5)
        
        self.yolo_zoom_label = tk.Label(yolo_title_frame, text="Zoom: 100%", font=("Arial", 9), fg="blue")
        self.yolo_zoom_label.pack(side=tk.RIGHT, padx=10)
        
        self.yolo_overlay_canvas = ZoomableCanvas(yolo_overlay_frame, bg="gray30", highlightthickness=0)
        self.yolo_overlay_canvas.pack(fill=tk.BOTH, expand=True)
        
        # YOLO canvas zoom eventi (add="+" ile mevcut bind'lere eklenir, override etmez)
        self.yolo_overlay_canvas.bind("<MouseWheel>", self.update_yolo_zoom_label, add="+")
        self.yolo_overlay_canvas.bind("<Button-4>", self.update_yolo_zoom_label, add="+")
        self.yolo_overlay_canvas.bind("<Button-5>", self.update_yolo_zoom_label, add="+")
        self.yolo_overlay_canvas.bind("<Double-Button-1>", self.update_yolo_zoom_label, add="+")
        
        # Sağ panel: UNet Segmentasyon Maskesi
        unet_overlay_frame = tk.Frame(bottom_row, bd=2, relief=tk.GROOVE)
        unet_overlay_frame.grid(row=0, column=1, padx=5, sticky="nsew")
        
        unet_title_frame = tk.Frame(unet_overlay_frame)
        unet_title_frame.pack(fill=tk.X)
        
        tk.Label(unet_title_frame, text="🧠 UNet Segmentasyon Maskesi (Zoom: Scroll, Pan: Drag)", 
                 font=("Arial", 10, "bold"), fg="#9C27B0").pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = tk.Label(unet_title_frame, text="Zoom: 100%", font=("Arial", 9), fg="blue")
        self.zoom_label.pack(side=tk.RIGHT, padx=10)
        
        self.overlay_canvas = ZoomableCanvas(unet_overlay_frame, bg="gray30", highlightthickness=0)
        self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
        
        # UNet canvas zoom eventi (add="+" ile mevcut bind'lere eklenir, override etmez)
        self.overlay_canvas.bind("<MouseWheel>", self.update_zoom_label, add="+")
        self.overlay_canvas.bind("<Button-4>", self.update_zoom_label, add="+")
        self.overlay_canvas.bind("<Button-5>", self.update_zoom_label, add="+")
        self.overlay_canvas.bind("<Double-Button-1>", self.update_zoom_label, add="+")
        
        # Mouse hareket - piksel olasılık değerini göster
        self.overlay_canvas.bind("<Motion>", self.on_unet_canvas_motion)

        # 3. ALT PANEL (Navigasyon ve Durum)
        bottom_frame = tk.Frame(self.root, pady=10, bg="#e0e0e0")
        bottom_frame.pack(fill=tk.X)

        btn_prev = tk.Button(bottom_frame, text="<< Geri", command=self.prev_image, width=15)
        btn_prev.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(bottom_frame, text="Hazır", bg="#e0e0e0", font=("Consolas", 12))
        self.status_label.pack(side=tk.LEFT, expand=True)

        btn_next = tk.Button(bottom_frame, text="İleri >>", command=self.next_image, width=15)
        btn_next.pack(side=tk.RIGHT, padx=20)

    def update_zoom_label(self, event=None):
        """UNet Zoom seviyesini güncelle"""
        self.root.after(50, lambda: self.zoom_label.config(
            text=f"Zoom: {int(self.overlay_canvas.zoom_level * 100)}%"
        ))
    
    def update_yolo_zoom_label(self, event=None):
        """YOLO Zoom seviyesini güncelle"""
        self.root.after(50, lambda: self.yolo_zoom_label.config(
            text=f"Zoom: {int(self.yolo_overlay_canvas.zoom_level * 100)}%"
        ))

    def on_unet_canvas_motion(self, event):
        """UNet canvas üzerinde mouse hareket ettiğinde piksel olasılık değerini göster"""
        if self.current_full_probs is None or self.overlay_canvas.image is None:
            self.pixel_prob_label.config(text="Piksel: -")
            return
        
        # Canvas koordinatlarından görsel koordinatlarına dönüştür
        canvas = self.overlay_canvas
        
        # Canvas merkezi ve pan offset
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        center_x = canvas_width // 2 + canvas.pan_x
        center_y = canvas_height // 2 + canvas.pan_y
        
        # Görsel boyutları (zoom'lu)
        img_width = int(canvas.image.width * canvas.zoom_level)
        img_height = int(canvas.image.height * canvas.zoom_level)
        
        # Görselin sol üst köşesi
        img_left = center_x - img_width // 2
        img_top = center_y - img_height // 2
        
        # Mouse pozisyonu görsel içinde mi?
        rel_x = event.x - img_left
        rel_y = event.y - img_top
        
        if 0 <= rel_x < img_width and 0 <= rel_y < img_height:
            # Zoom'u geri al - orijinal görsel koordinatları
            orig_x = int(rel_x / canvas.zoom_level)
            orig_y = int(rel_y / canvas.zoom_level)
            
            # Probs haritasındaki koordinatlar
            prob_h, prob_w = self.current_full_probs.shape
            
            if 0 <= orig_x < prob_w and 0 <= orig_y < prob_h:
                prob_value = self.current_full_probs[orig_y, orig_x]
                threshold = self.thresh_var.get()
                
                # Threshold'a göre renk
                if prob_value > threshold:
                    color = "#4CAF50"  # Yeşil - threshold üstü
                else:
                    color = "#F44336"  # Kırmızı - threshold altı
                
                self.pixel_prob_label.config(
                    text=f"P: {prob_value:.4f} ({orig_x},{orig_y})",
                    fg=color
                )
                return
        
        self.pixel_prob_label.config(text="Piksel: -", fg="#9C27B0")

    def create_image_panel(self, parent, title, col):
        frame = tk.Frame(parent, bd=2, relief=tk.GROOVE)
        frame.grid(row=0, column=col, padx=5, sticky="nsew")
        
        lbl_title = tk.Label(frame, text=title, font=("Arial", 9, "bold"))
        lbl_title.pack(side=tk.TOP, fill=tk.X)
        
        lbl_img = tk.Label(frame, text="Görüntü Yok")
        lbl_img.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return lbl_img

    def on_onnx_mode_change(self, event=None):
        """ONNX mod değiştiğinde modeli yeniden yükle"""
        if hasattr(self, 'current_onnx_path') and self.current_onnx_path:
            mode = self.onnx_mode_var.get()
            print(f"ONNX modu değiştirildi: {mode}")
            self.load_unet_model(self.current_onnx_path)

    def load_unet_model(self, path):
        """
        Model yükleme - ONNX için seçilen moda göre (CPU/GPU), PyTorch için otomatik
        """
        self.unet_session = None
        self.pytorch_model = None
        self.model_type = None  # 'onnx' veya 'pytorch'
        self.device = None

        # Eğer .pth dosyası seçildiyse direkt PyTorch'a git
        if path.endswith('.pth'):
            self.current_onnx_path = None
            self._load_pytorch_model(path)
            return

        # ONNX path'i sakla (mod değişikliğinde kullanmak için)
        self.current_onnx_path = path
        
        # Seçilen moda göre ONNX yükle
        onnx_mode = self.onnx_mode_var.get()
        onnx_success = self._try_load_onnx(path, use_gpu=(onnx_mode == "GPU"))

        if onnx_success:
            # ONNX yüklendi, test inference yap
            test_success = self._test_onnx_inference()
            if test_success:
                print(f"ONNX model {onnx_mode}'da çalışıyor!")
                return
            else:
                print(f"ONNX {onnx_mode} inference başarısız!")
                self.unet_session = None
                self.status_label.config(text=f"ONNX {onnx_mode} başarısız!", fg="red")
                return

        # ONNX başarısız
        self.status_label.config(text=f"ONNX {onnx_mode} yüklenemedi!", fg="red")

    def _try_load_onnx(self, path, use_gpu=True):
        """ONNX model yüklemeyi dene - use_gpu parametresine göre CPU veya GPU"""
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

            if use_gpu:
                # GPU modu
                provider_options = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                    }),
                    ('CPUExecutionProvider', {})
                ]
            else:
                # CPU modu - sadece CPU kullan
                provider_options = [
                    ('CPUExecutionProvider', {})
                ]

            self.unet_session = ort.InferenceSession(
                path,
                sess_options=sess_options,
                providers=[p[0] for p in provider_options],
                provider_options=[p[1] for p in provider_options]
            )
            self.input_name = self.unet_session.get_inputs()[0].name
            self.output_name = self.unet_session.get_outputs()[0].name
            self.model_type = 'onnx'
            active_provider = self.unet_session.get_providers()[0]
            print(f"ONNX Model Yüklendi: {path}")
            print(f"Active Provider: {active_provider}")
            self.status_label.config(text=f"UNet: {os.path.basename(path)} (ONNX-{active_provider})", fg="green")
            return True
        except Exception as e:
            print(f"ONNX yükleme hatası: {e}")
            return False

    def _test_onnx_inference(self):
        """ONNX model ile test inference yap"""
        try:
            dummy_input = np.random.randn(1, 3, TARGET_H, TARGET_W).astype(np.float32)
            _ = self.unet_session.run([self.output_name], {self.input_name: dummy_input})
            return True
        except Exception as e:
            print(f"ONNX test inference hatası: {e}")
            return False

    def _load_pytorch_model(self, path):
        """PyTorch .pth model yükle"""
        if not SMP_AVAILABLE:
            err_msg = "segmentation_models_pytorch kurulu değil!\npip install segmentation-models-pytorch"
            print(f"[HATA] {err_msg}")
            messagebox.showerror("Hata", err_msg)
            return

        try:
            # Device seç
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"PyTorch Device: {self.device}")

            # Model oluştur (eğitimde kullanılan aynı mimari)
            self.pytorch_model = smp.Unet(
                encoder_name="timm-efficientnet-b0",
                encoder_weights=None,  # Ağırlıkları .pth'den yükleyeceğiz
                in_channels=3,
                classes=1,
            )

            # Ağırlıkları yükle (checkpoint veya direkt state_dict olabilir)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Checkpoint dictionary mi yoksa direkt state_dict mi kontrol et
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint formatı
                state_dict = checkpoint['model_state_dict']
                print(f"Checkpoint yüklendi - Epoch: {checkpoint.get('epoch', '?')}, Val IoU: {checkpoint.get('val_iou', '?'):.4f}")
            else:
                # Direkt state_dict
                state_dict = checkpoint

            self.pytorch_model.load_state_dict(state_dict)
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()

            self.model_type = 'pytorch'
            device_name = 'GPU' if self.device.type == 'cuda' else 'CPU'
            print(f"PyTorch Model Yüklendi: {path}")
            self.status_label.config(text=f"UNet: {os.path.basename(path)} (PyTorch-{device_name})", fg="green")
        except Exception as e:
            err_msg = f"PyTorch model yüklenirken hata oluştu:\n{e}"
            print(f"[HATA] {err_msg}")
            messagebox.showerror("Hata", err_msg)
            self.status_label.config(text="Model yüklenemedi!", fg="red")

    def load_yolo_model(self, path):
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(path)
            print(f"YOLO Model Yüklendi: {path}")
            self.yolo_status_label.config(text=f"YOLO: {os.path.basename(path)}", fg="green")
        except Exception as e:
            err_msg = f"YOLO model yüklenirken hata oluştu:\n{e}"
            print(f"[HATA] {err_msg}")
            messagebox.showerror("Hata", err_msg)
            self.yolo_status_label.config(text="YOLO: Hata!", fg="red")

    def select_unet_model(self):
        path = filedialog.askopenfilename(filetypes=[
            ("All Models", "*.onnx *.pth"),
            ("ONNX Models", "*.onnx"),
            ("PyTorch Models", "*.pth")
        ])
        if path:
            self.load_unet_model(path)

    def select_yolo_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Models", "*.pt")])
        if path:
            self.load_yolo_model(path)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.load_images_from_folder(path)

    def load_images_from_folder(self, folder_path):
        exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        self.image_paths.sort()
        
        if self.image_paths:
            self.current_index = 0
            self.run_inference_current()
        else:
            self.status_label.config(text="Klasörde resim bulunamadı.", fg="red")

    def detect_and_crop(self, img_path):
        """YOLO ile detect edip en büyük bounding box'a göre kırpar ve segmentasyon maskesini alır."""
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]
        
        # YOLO maskesini sıfırla
        self.current_yolo_mask = None
        
        if self.yolo_model is None:
            letterbox_img, offset_x, offset_y, scale, new_w, new_h = letterbox_resize(img_rgb, (TARGET_W, TARGET_H))
            self.current_letterbox_params = (offset_x, offset_y, scale, new_w, new_h)
            return img_rgb, img_rgb, letterbox_img, None, img_rgb
        
        results = self.yolo_model(img_path, verbose=False, retina_masks=True)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            print(f"[UYARI] Detection bulunamadı: {os.path.basename(img_path)}")
            letterbox_img, offset_x, offset_y, scale, new_w, new_h = letterbox_resize(img_rgb, (TARGET_W, TARGET_H))
            self.current_letterbox_params = (offset_x, offset_y, scale, new_w, new_h)
            return img_rgb, img_rgb, letterbox_img, None, img_rgb
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # YOLO segmentasyon maskesini al (eğer varsa)
        yolo_combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        has_masks = hasattr(results[0], 'masks') and results[0].masks is not None
        
        if has_masks:
            masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
            # Tüm maskeleri birleştir
            for i, mask in enumerate(masks):
                # Maske boyutunu orijinal görsel boyutuna çevir
                mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                yolo_combined_mask = np.maximum(yolo_combined_mask, (mask_resized > 0.5).astype(np.uint8) * 255)
            self.current_yolo_mask = yolo_combined_mask
            print(f"[INFO] YOLO segmentasyon maskesi alındı: {len(masks)} nesne")
        else:
            print(f"[INFO] YOLO modelinde segmentasyon maskesi yok (sadece detection)")
        
        largest_box = None
        largest_area = 0
        largest_idx = -1
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)
                largest_idx = i
        
        if largest_box is None:
            letterbox_img, offset_x, offset_y, scale, new_w, new_h = letterbox_resize(img_rgb, (TARGET_W, TARGET_H))
            self.current_letterbox_params = (offset_x, offset_y, scale, new_w, new_h)
            return img_rgb, img_rgb, letterbox_img, None, img_rgb
        
        # Sadece en büyük nesnenin maskesini al (eğer varsa)
        if has_masks and largest_idx >= 0 and largest_idx < len(masks):
            single_mask = masks[largest_idx]
            single_mask_resized = cv2.resize(single_mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            self.current_yolo_mask = (single_mask_resized > 0.5).astype(np.uint8) * 255
        
        cropped_img, crop_box = crop_image_by_bbox(img_rgb, largest_box)
        letterbox_img, offset_x, offset_y, scale, new_w, new_h = letterbox_resize(cropped_img, (TARGET_W, TARGET_H))
        self.current_letterbox_params = (offset_x, offset_y, scale, new_w, new_h)
        
        detection_img = img_rgb.copy()
        x1, y1, x2, y2 = [int(v) for v in largest_box]
        cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if crop_box:
            cx1, cy1, cx2, cy2 = crop_box
            cv2.rectangle(detection_img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
        
        return img_rgb, cropped_img, letterbox_img, crop_box, detection_img

    def preprocess_for_unet(self, img: np.ndarray) -> np.ndarray:
        img_norm = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
        
        img_input = img_norm.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        
        return img_input

    def run_inference_current(self):
        if not self.image_paths:
            return

        img_path = self.image_paths[self.current_index]
        self.file_label.config(text=f"{os.path.basename(img_path)} ({self.current_index+1}/{len(self.image_paths)})")

        start_t = time.time()
        result = self.detect_and_crop(img_path)
        if result is None:
            self.status_label.config(text="Görsel işlenemedi!", fg="red")
            return

        self.current_original_img, self.current_cropped_img, self.current_letterbox_img, self.current_crop_box, detection_img = result
        detect_time = (time.time() - start_t) * 1000

        # Model kontrolü
        if self.unet_session is None and self.pytorch_model is None:
            self.status_label.config(text="UNet modeli yüklenmedi!", fg="red")
            self.display_detection_only(detection_img)
            return

        input_tensor = self.preprocess_for_unet(self.current_letterbox_img)

        start_t = time.time()

        # Model tipine göre inference
        if self.model_type == 'onnx' and self.unet_session is not None:
            logits = self.unet_session.run([self.output_name], {self.input_name: input_tensor})[0]
            backend = "ONNX"
        elif self.model_type == 'pytorch' and self.pytorch_model is not None:
            logits = self._pytorch_inference(input_tensor)
            backend = "PyTorch"
        else:
            self.status_label.config(text="Model hatası!", fg="red")
            return

        unet_time = (time.time() - start_t) * 1000

        self.status_label.config(text=f"YOLO: {detect_time:.1f}ms | UNet ({backend}): {unet_time:.1f}ms | Toplam: {detect_time+unet_time:.1f}ms", fg="blue")

        self.current_logits = logits
        self.current_detection_img = detection_img

        self.update_visuals()

    def _pytorch_inference(self, input_tensor):
        """PyTorch model ile inference"""
        with torch.no_grad():
            # numpy -> torch tensor
            tensor = torch.from_numpy(input_tensor).to(self.device)
            output = self.pytorch_model(tensor)
            # torch tensor -> numpy
            return output.cpu().numpy()

    def display_detection_only(self, detection_img):
        img_pil = Image.fromarray(detection_img)
        self.display_on_label(self.panel_orig, img_pil)
        
        if self.current_letterbox_img is not None:
            cropped_pil = Image.fromarray(self.current_letterbox_img)
            self.display_on_label(self.panel_cropped, cropped_pil)
        
        self.panel_heat.config(image='', text='UNet modeli yok')
        self.panel_mask.config(image='', text='UNet modeli yok')
        
        # YOLO maskesi varsa göster
        yolo_overlay_pil = self.create_yolo_overlay_image()
        if yolo_overlay_pil:
            self.yolo_overlay_canvas.set_image(yolo_overlay_pil)
            self.update_yolo_zoom_label()

    def on_slider_change(self, val):
        if self.current_logits is not None:
            self.update_visuals()

    def create_yolo_overlay_image(self) -> Image.Image:
        """
        YOLO segmentasyon maskesini orijinal görsel üzerine turuncu overlay olarak koyar.
        Yumuşatma yok - YOLO'dan gelen raw maske direkt kullanılıyor.
        """
        if self.current_original_img is None:
            return None
        
        overlay_img = self.current_original_img.copy().astype(np.float32)
        
        if self.current_yolo_mask is None:
            # Maske yoksa sadece orijinal görseli döndür
            return Image.fromarray(self.current_original_img)
        
        # Turuncu renk overlay (YOLO için)
        orange_color = np.array([255, 140, 0], dtype=np.float32)
        
        # Maskeyi float'a çevir (0-1 arası) - yumuşatma yok, raw maske
        mask_float = self.current_yolo_mask.astype(np.float32) / 255.0
        
        # Alpha mask (3 kanala genişlet)
        alpha_mask = mask_float[:, :, np.newaxis]
        
        # Alpha blending
        blend_strength = 0.4
        overlay_img = overlay_img * (1 - alpha_mask * blend_strength) + orange_color * alpha_mask * blend_strength
        overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
        
        # Kontur çiz - raw kontur, yumuşatma yok
        contours, _ = cv2.findContours(self.current_yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            cv2.drawContours(overlay_img, [contour], -1, (255, 100, 0), 2)
        
        return Image.fromarray(overlay_img)

    def create_overlay_image(self, probs: np.ndarray, threshold: float) -> Image.Image:
        """
        Maskeyi orijinal görsel üzerine yeşil overlay olarak koyar.
        Olasılık değerlerini (probs) scale edip, threshold son anda uygulanır.
        Bu sayede kenarlar çok daha smooth olur.
        """
        if self.current_letterbox_params is None:
            return Image.fromarray(self.current_original_img)
        
        offset_x, offset_y, scale, new_w, new_h = self.current_letterbox_params
        
        # 1. Letterbox'tan crop koordinatlarına dönüştür (float olarak - 0.0-1.0)
        # Letterbox padding'ini çıkar
        probs_cropped = probs[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
        
        # 2. Crop boyutuna scale et - INTER_CUBIC ile smooth interpolation
        if self.current_cropped_img is not None:
            crop_h, crop_w = self.current_cropped_img.shape[:2]
            # Float olarak scale et - bu çok önemli!
            probs_at_crop_size = cv2.resize(probs_cropped.astype(np.float32), (crop_w, crop_h), 
                                            interpolation=cv2.INTER_CUBIC)
        else:
            probs_at_crop_size = probs_cropped.astype(np.float32)
        
        # 3. Orijinal görsel boyutunda boş olasılık matrisi oluştur
        orig_h, orig_w = self.current_original_img.shape[:2]
        full_probs = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        # 4. Crop box varsa, olasılıkları doğru konuma yerleştir
        if self.current_crop_box is not None:
            cx1, cy1, cx2, cy2 = self.current_crop_box
            target_h = cy2 - cy1
            target_w = cx2 - cx1
            
            # Boyut farkı varsa smooth resize et
            probs_h, probs_w = probs_at_crop_size.shape[:2]
            if probs_h != target_h or probs_w != target_w:
                probs_at_crop_size = cv2.resize(probs_at_crop_size, (target_w, target_h), 
                                                interpolation=cv2.INTER_CUBIC)
            
            full_probs[cy1:cy2, cx1:cx2] = probs_at_crop_size
        else:
            # Crop yok, tüm görsel kullanılmış
            if probs_at_crop_size.shape != (orig_h, orig_w):
                probs_at_crop_size = cv2.resize(probs_at_crop_size, (orig_w, orig_h), 
                                                interpolation=cv2.INTER_CUBIC)
            full_probs = probs_at_crop_size
        
        # Piksel değeri göstermek için sakla
        self.current_full_probs = full_probs
        
        # 5. Threshold uygula ve binary mask oluştur
        full_mask = (full_probs > threshold).astype(np.uint8) * 255
        
        # 6. Kenarları smooth yapmak için hafif Gaussian blur
        # Önce float'a çevir, blur uygula, sonra tekrar threshold
        smooth_probs = cv2.GaussianBlur(full_probs, (5, 5), 0)
        smooth_mask_float = np.clip((smooth_probs - threshold + 0.1) * 5, 0, 1)  # Soft threshold
        
        # 7. Orijinal görsel üzerine yeşil overlay
        overlay_img = self.current_original_img.copy().astype(np.float32)
        
        # Yeşil renk overlay
        green_color = np.array([0, 255, 0], dtype=np.float32)
        
        # Smooth alpha mask (3 kanala genişlet)
        alpha_mask = smooth_mask_float[:, :, np.newaxis]
        
        # Alpha blending: overlay = original * (1 - alpha*0.4) + green * alpha*0.4
        blend_strength = 0.4
        overlay_img = overlay_img * (1 - alpha_mask * blend_strength) + green_color * alpha_mask * blend_strength
        overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
        
        # 8. Smooth kontur çiz
        # Kontur için binary mask kullan ama çizgiyi anti-aliased yap
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Her kontur için smooth çizgi
        for contour in contours:
            # Konturu smooth et (Douglas-Peucker ile basitleştir, sonra spline ile smooth)
            epsilon = 0.001 * cv2.arcLength(contour, True)
            smoothed = cv2.approxPolyDP(contour, epsilon, True)
            
            # Anti-aliased çizgi için LINE_AA kullan
            cv2.drawContours(overlay_img, [smoothed], -1, (0, 200, 0), 2, cv2.LINE_AA)
        
        return Image.fromarray(overlay_img)

    def update_visuals(self):
        if self.current_logits is None:
            return

        threshold = self.thresh_var.get()
        
        # Sigmoid ile olasılıklara çevir
        probs = 1 / (1 + np.exp(-self.current_logits))
        probs = probs.squeeze()  # (H, W)
        
        # Binary mask (küçük paneller için)
        mask = (probs > threshold).astype(np.uint8) * 255

        # 1. Orijinal + Detection
        detection_pil = Image.fromarray(self.current_detection_img)
        self.display_on_label(self.panel_orig, detection_pil)

        # 2. Kırpılmış (Letterbox)
        letterbox_pil = Image.fromarray(self.current_letterbox_img)
        self.display_on_label(self.panel_cropped, letterbox_pil)

        # 3. Heatmap
        prob_uint8 = (probs * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heat_pil = Image.fromarray(heatmap_color)
        self.display_on_label(self.panel_heat, heat_pil)

        # 4. Maske
        mask_pil = Image.fromarray(mask)
        self.display_on_label(self.panel_mask, mask_pil)
        
        # 5. YOLO Overlay (Sol panel)
        yolo_overlay_pil = self.create_yolo_overlay_image()
        if yolo_overlay_pil:
            self.yolo_overlay_canvas.set_image(yolo_overlay_pil)
            self.update_yolo_zoom_label()
        
        # 6. UNet Overlay (Sağ panel) - probs ve threshold ile smooth scale
        overlay_pil = self.create_overlay_image(probs, threshold)
        self.overlay_canvas.set_image(overlay_pil)
        self.update_zoom_label()

    def display_on_label(self, label, img_pil):
        disp_w, disp_h = 280, 160
        img_pil = img_pil.resize((disp_w, disp_h), Image.Resampling.BILINEAR)

        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        label.image = img_tk

    def next_image(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.run_inference_current()

    def prev_image(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.run_inference_current()


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
