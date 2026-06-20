import sys
import os
import time
import subprocess

# 1. IMPORT LIGHTWEIGHT PYQT5 MODULES FIRST
try:
    # Added QMessageBox to the import list
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt
except ImportError:
    print("ERROR: PyQt5 is not installed.")
    print("Will try to install.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])

# 1. IMPORT LIGHTWEIGHT PYQT5 MODULES FIRST
try:
    # Added QMessageBox to the import list
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt
except ImportError:
    print("ERROR: PyQt5 is not installed.")
    print("could not be installed will shutdown.")
    time.sleep(5)
    sys.exit(1)
# 2. INITIALIZE APP & SPLASH SCREEN IMMEDIATELY
app = QApplication(sys.argv)

# Load your banner image
splash_pixmap = QPixmap("banner.jpg") 
splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
splash.show()
app.processEvents() # Force UI to immediately draw the splash screen

# --- NEW: DEPENDENCY SAFEGUARD ---
def check_and_install_dependencies():
    # Dictionary mapping internal module names to their PyPI package names
    dependencies = {
        'pydicom': 'pydicom',
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'torch': 'torch',
        'monai': 'monai',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'einops': 'einops'

    }
    
    missing_packages = []
    for module, pip_name in dependencies.items():
        try:
            splash.showMessage(f"Loading {module}...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
            app.processEvents()
            __import__(module)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Missing Dependencies Detected")
        msg.setText("The application is missing the following required packages:\n\n" + "\n".join(missing_packages))
        msg.setInformativeText("Would you like to automatically install them now?\n\n(Note: This may take a few minutes. The application will pause during installation.)")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        msg.setWindowFlags(Qt.WindowStaysOnTopHint) # Keep dialog above splash screen
        
        if msg.exec_() == QMessageBox.Yes:
            
            # --- NEW: PYTORCH CUDA SAFEGUARD ---
            if 'torch' in missing_packages:
                torch_msg = QMessageBox()
                torch_msg.setIcon(QMessageBox.Warning)
                torch_msg.setWindowTitle("CUDA / GPU Acceleration Warning")
                torch_msg.setText("PyTorch is required, but automatic installation will only download the CPU version.")
                torch_msg.setInformativeText(
                    "If you have an NVIDIA GPU, the CPU-only version will be slower.\n\n"
                    "It is highly recommended to click 'Cancel' and install PyTorch manually using the specific command for your system from:\n"
                    "https://pytorch.org/\n\n"
                    "Are you sure you want to proceed with the CPU-only installation?"
                )
                torch_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
                torch_msg.setDefaultButton(QMessageBox.Cancel) # Default to Cancel to protect the user
                torch_msg.setWindowFlags(Qt.WindowStaysOnTopHint)
                
                if torch_msg.exec_() != QMessageBox.Yes:
                    splash.showMessage("Installation cancelled by user. Exiting...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
                    app.processEvents()
                    time.sleep(2)
                    sys.exit(0)
            # -----------------------------------

            splash.showMessage(f"Installing missing packages... Please wait.", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
            app.processEvents()
            
            try:
                # Use sys.executable to ensure pip installs to the correct active Python environment
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
                splash.showMessage("Installation complete! Resuming startup...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
                app.processEvents()
                time.sleep(2) # Brief pause so the user sees the success message
            except subprocess.CalledProcessError as e:
                err = QMessageBox()
                err.setIcon(QMessageBox.Critical)
                err.setWindowTitle("Installation Failed")
                err.setText(f"Failed to install packages automatically.\n\nPlease manually run:\npip install {' '.join(missing_packages)}")
                err.exec_()
                sys.exit(1)
        else:
            # User chose not to install
            sys.exit(0)

# Run the dependency check before any heavy imports proceed
splash.showMessage("Checking dependencies...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
app.processEvents()
check_and_install_dependencies()
# ---------------------------------

# Proceed with your original sequenced loading
splash.showMessage("Loading pydicom and numpy...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
app.processEvents()

import pydicom
import numpy as np
import cv2

splash.showMessage("Loading Torch...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
app.processEvents()

import torch
import torch.nn as nn
import torch.nn.functional as F

splash.showMessage("Loading MONAI...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
app.processEvents()

from monai.networks.nets import SwinUNETR, UNet, AttentionUnet, SegResNet, DynUNet, DenseNet, resnet18

splash.showMessage("Loading PyQt...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
app.processEvents()


from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QComboBox, QTextEdit, QSlider,QCheckBox)
from PyQt5.QtCore import Qt

splash.showMessage("Loading Matplotlib and scipy...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
app.processEvents()

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy.ndimage as ndi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # <-- Add this
from matplotlib.figure import Figure

# --- MONAI Segmentation Architectures ---

class SwinUNETRR(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, feature_size=24, heads=(3,6,12,24), depths=(2,2,2,2), do_rate=0.1):
        super(SwinUNETRR, self).__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,     
            feature_size=feature_size,
            num_heads=heads,
            depths=depths,
            spatial_dims=2,                
            drop_rate=do_rate,
            use_checkpoint=True
        )
    def forward(self, x):
        return self.model(x)

class UNett(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, channels=(64, 128, 256, 512, 1024), num_res_units=2, strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3):
        super(UNett, self).__init__()
        self.unet = UNet(
            spatial_dims=2,               
            in_channels=in_channels,
            out_channels=out_channels,     
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size
        )
    def forward(self, x):
        return self.unet(x)

class AttentionUNett(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2)):
        super(AttentionUNett, self).__init__()
        self.model = AttentionUnet(
            spatial_dims=2,               
            in_channels=in_channels,
            out_channels=out_channels,     
            channels=channels,
            strides=strides,
        )
    def forward(self, x):
        return self.model(x)

class SegResNett(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_filters=32, blocks_down=(1, 2, 2, 4)):
        super(SegResNett, self).__init__()
        self.model = SegResNet(
            spatial_dims=2,               
            in_channels=in_channels,
            out_channels=out_channels,     
            init_filters=init_filters,
            blocks_down=blocks_down,
        )
    def forward(self, x):
        return self.model(x)

class DynUNett(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(DynUNett, self).__init__()
        kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]
        upsample_kernels = [[2, 2], [2, 2], [2, 2], [2, 2]]
        self.model = DynUNet(
            spatial_dims=2,               
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=upsample_kernels,
            res_block=True
        )
    def forward(self, x):
        return self.model(x)


# --- Main Application ---

class DMSAAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aether - Standalone DMSA Analyzer")
        self.setGeometry(100, 100, 1900, 1000)
        self.setAcceptDrops(True) 

        # Data placeholders
        self.ant_image = None
        self.post_image = None
        self.post_mask = None
        self.ant_mask = None
        self.post_probs = None 

        self.display_srf_l = 0.0  # <-- Add this
        self.display_srf_r = 0.0  # <-- Add this

        # New Oblique placeholders
        self.lao_image = None
        self.rpo_image = None
        self.lpo_image = None
        self.rao_image = None


        # Raw data caches for dynamic preprocessing
        self.raw_images = {'ant': None, 'post': None, 'lao': None, 'rpo': None, 'lpo': None, 'rao': None}
        self.spacings = {'ant': None, 'post': None, 'lao': None, 'rpo': None, 'lpo': None, 'rao': None}

        
        self.bg_masks_post = {1: None, 2: None}
        self.bg_masks_ant = {1: None, 2: None}
        
        # Hardcoded model directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.colormap = 'gray_r' 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hardcoded probability cutoffs
        self.thresholds = {
            "Anomaly": {"left": 0.677, "right": 0.353},
            "Normal":  {"left": 0.713, "right": 0.810},
            "Afunc":   {"left": 0.620, "right": 0.580},
            "Horseshoe": 0.710  # <--- Add your specific threshold here
        }
        
        self.initUI()
        self.log(f"System initialized. Compute device: {self.device}")


    def toggle_pan(self):
            # Turn off zoom if active
        if "zoom" in str(self.toolbar_main.mode).lower():
            self.toolbar_main.zoom()
        if "zoom" in str(self.toolbar_oblique.mode).lower():
            self.toolbar_oblique.zoom()
 
        if self.btn_pan.isChecked():
            self.btn_zoom.setChecked(False)
            # If panning is NOT active, toggle it on
            if "pan" not in str(self.toolbar_main.mode).lower():
                self.toolbar_main.pan()
            if "pan" not in str(self.toolbar_oblique.mode).lower():
                self.toolbar_oblique.pan()
        else:
            # Turn off pan if it is active
            if "pan" in str(self.toolbar_main.mode).lower():
                self.toolbar_main.pan()
            if "pan" in str(self.toolbar_oblique.mode).lower():
                self.toolbar_oblique.pan()

    def toggle_zoom(self):
            # Turn off pan if it is active
        if "pan" in str(self.toolbar_main.mode).lower():
            self.toolbar_main.pan()
        if "pan" in str(self.toolbar_oblique.mode).lower():
            self.toolbar_oblique.pan()
        
        
        if self.btn_zoom.isChecked():
            self.btn_pan.setChecked(False)
            
            # If zooming is NOT active, toggle it on
            if "zoom" not in str(self.toolbar_main.mode).lower():
                self.toolbar_main.zoom()
            if "zoom" not in str(self.toolbar_oblique.mode).lower():
                self.toolbar_oblique.zoom()
        else:
            # Turn off zoom if active
            if "zoom" in str(self.toolbar_main.mode).lower():
                self.toolbar_main.zoom()
            if "zoom" in str(self.toolbar_oblique.mode).lower():
                self.toolbar_oblique.zoom()

    def reset_views(self):
        # Return axes to their default states
        self.toolbar_main.home()
        self.toolbar_oblique.home()
        
        # Reset UI button states
        self.btn_pan.setChecked(False)
        self.btn_zoom.setChecked(False)
        
        # Clear interaction modes from toolbars safely
        if "pan" in str(self.toolbar_main.mode).lower():
            self.toolbar_main.pan()
        if "zoom" in str(self.toolbar_main.mode).lower():
            self.toolbar_main.zoom()
            
        if "pan" in str(self.toolbar_oblique.mode).lower():
            self.toolbar_oblique.pan()
        if "zoom" in str(self.toolbar_oblique.mode).lower():
            self.toolbar_oblique.zoom()
















    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(380)

        # Patient Info Display
        self.lbl_patient_info = QLabel("Patient Name: N/A\nPatient ID: N/A")
        self.lbl_patient_info.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; background-color: #e0e0e0; border-radius: 5px;")
        control_layout.addWidget(self.lbl_patient_info)

        self.btn_load = QPushButton("Load DICOM File(s)")
        self.btn_load.clicked.connect(self.load_dicom_dialog)
        control_layout.addWidget(self.btn_load)
        
        drag_drop_label = QLabel("(Or drag and drop files here)")
        drag_drop_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(drag_drop_label)

        control_layout.addWidget(QLabel("Select Segmentation Architecture:"))
        self.combo_arch = QComboBox()
        self.combo_arch.addItems([
            "SwinUNETR (2D)",  
            "SegResNet (2D)", "DynUNet (2D)"
        ])
        control_layout.addWidget(self.combo_arch)


        # --- NEW: Preprocessing Controls ---
        
        self.cb_normalize = QCheckBox("Normalize total counts to 200k")
        self.cb_normalize.stateChanged.connect(self.apply_preprocessing)
        control_layout.addWidget(self.cb_normalize)
        
        self.cb_resample = QCheckBox("Resample to 2x2mm pixel size")
        self.cb_resample.stateChanged.connect(self.apply_preprocessing)
        control_layout.addWidget(self.cb_resample)
        # -----------------------------------











        self.btn_segment = QPushButton("Segment & Quantify SRF")
        self.btn_segment.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold; padding: 10px;")
        self.btn_segment.clicked.connect(self.run_segmentation)
        self.btn_segment.setEnabled(False)
        control_layout.addWidget(self.btn_segment)

        # ROI Threshold Controls
        control_layout.addWidget(QLabel("Left Kidney ROI Threshold (%):"))
        self.slider_thresh_l = QSlider(Qt.Horizontal)
        self.slider_thresh_l.setMinimum(1)
        self.slider_thresh_l.setMaximum(99)
        self.slider_thresh_l.setValue(50)
        self.slider_thresh_l.valueChanged.connect(self.recalc_masks)
        control_layout.addWidget(self.slider_thresh_l)

        control_layout.addWidget(QLabel("Right Kidney ROI Threshold (%):"))
        self.slider_thresh_r = QSlider(Qt.Horizontal)
        self.slider_thresh_r.setMinimum(1)
        self.slider_thresh_r.setMaximum(99)
        self.slider_thresh_r.setValue(50)
        self.slider_thresh_r.valueChanged.connect(self.recalc_masks)
        control_layout.addWidget(self.slider_thresh_r)

        # Background ROI Selection
        control_layout.addWidget(QLabel("Background ROI Method:"))
        self.combo_bg_method = QComboBox()
        self.combo_bg_method.addItems(["Circumferential", "Center Rectangle", "Inferolateral Crescent"])
        self.combo_bg_method.currentTextChanged.connect(self.recalc_masks)
        control_layout.addWidget(self.combo_bg_method)

        # Classification Module UI
        control_layout.addWidget(QLabel("--- Kidney Evaluation Module ---"))
        self.combo_class_arch = QComboBox()
        self.combo_class_arch.addItems(["DenseNet121"])
        control_layout.addWidget(self.combo_class_arch)

        self.btn_classify = QPushButton("Run Evaluation")
        self.btn_classify.setStyleSheet("background-color: #4682B4; color: white; font-weight: bold; padding: 10px;")
        self.btn_classify.clicked.connect(self.run_anomaly_classification)
        self.btn_classify.setEnabled(False)
        control_layout.addWidget(self.btn_classify)

        control_layout.addWidget(QLabel("Select Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['gray_r', 'gray', 'nipy_spectral', 'gist_heat','gnuplot2', 'CMRmap','inferno', 'afmhot', 'gnuplot', 'jet', 'turbo', 'cubehelix', 'gist_ncar' ])
        self.combo_cmap.currentTextChanged.connect(self.update_plots)
        control_layout.addWidget(self.combo_cmap)

        control_layout.addWidget(QLabel("Window Level (%):"))
        self.slider_window = QSlider(Qt.Horizontal)
        self.slider_window.setMinimum(5)
        self.slider_window.setMaximum(100)
        self.slider_window.setValue(100)
        self.slider_window.valueChanged.connect(self.update_plots)
        control_layout.addWidget(self.slider_window)

        # --- NEW: Navigation Controls ---
        nav_layout = QHBoxLayout()
        
        self.btn_pan = QPushButton("Pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(self.toggle_pan)
        
        self.btn_zoom = QPushButton("Zoom")
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.clicked.connect(self.toggle_zoom)
        
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_views)
        
        nav_layout.addWidget(self.btn_pan)
        nav_layout.addWidget(self.btn_zoom)
        nav_layout.addWidget(self.btn_reset)
        control_layout.addLayout(nav_layout)
        # --------------------------------

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        control_layout.addWidget(QLabel("Execution Log:"))
        control_layout.addWidget(self.log_box)

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        control_layout.addWidget(QLabel("Quantification & Evaluation Results:"))
        control_layout.addWidget(self.results_box)

        # Right Panel: Primary 2x2 Grid via Matplotlib
        self.canvas = FigureCanvas(Figure(figsize=(10, 10)))
        self.axes = self.canvas.figure.subplots(2, 2, sharex=True, sharey=True)
        self.canvas.figure.tight_layout()
        
        # --- NEW: Hidden Main Toolbar ---
        self.toolbar_main = NavigationToolbar(self.canvas, self)
        self.toolbar_main.hide()

        # Far Right Panel: Secondary 2x2 Grid for Oblique Views
        self.canvas_oblique = FigureCanvas(Figure(figsize=(10, 10)))
        self.axes_oblique = self.canvas_oblique.figure.subplots(2, 2, sharex=True, sharey=True)
        self.canvas_oblique.figure.tight_layout()
        
        # --- NEW: Hidden Oblique Toolbar ---
        self.toolbar_oblique = NavigationToolbar(self.canvas_oblique, self)
        self.toolbar_oblique.hide()

        # ADD THESE THREE LINES TO PREVENT GARBAGE COLLECTION:
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.canvas_oblique)


    def log(self, message):
        self.log_box.append(message)
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


    def extract_all_file_paths(self, paths):
        """Recursively scans folders to extract all files."""
        all_files = []
        for path in paths:
            if os.path.isdir(path):
                # Walk through all subdirectories
                for root, _, files in os.walk(path):
                    for file in files:
                        # Skip hidden system files (like .DS_Store on Mac)
                        if not file.startswith('.'):
                            all_files.append(os.path.join(root, file))
            elif os.path.isfile(path):
                all_files.append(path)
        return all_files









    # --- Drag and Drop Events ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            
            # Get raw paths from the drop event
            raw_paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
            
            # Use our new helper to scan subfolders
            expanded_file_paths = self.extract_all_file_paths(raw_paths)
            
            # Pass the expanded list to the processor
            self.process_files(expanded_file_paths)
        else:
            event.ignore()

    # --- File Loading Logic ---
    def load_dicom_dialog(self):
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open DMSA DICOM", "", "DICOM Files (*.dcm *.ima);;All Files (*)", options=options)
        if file_names:
            self.process_files(file_names)

    def determine_view(self, ds, index=0):
        angle = None
        name_hint = None
        
        try:
            if hasattr(ds, 'DetectorInformationSequence') and len(ds.DetectorInformationSequence) > index:
                seq = ds.DetectorInformationSequence[index]
                if hasattr(seq, 'StartAngle'):
                    angle = float(seq.StartAngle)
            elif hasattr(ds, 'StartAngle'):
                angle = float(ds.StartAngle)
        except Exception:
            pass

        desc = ""
        desc += str(getattr(ds, 'DatasetName', '')).lower() + " "
        desc += str(getattr(ds, 'SeriesDescription', '')).lower() + " "
        desc += str(getattr(ds, 'ProtocolName', '')).lower()

        desc = desc.lower()


        if 'ant' in desc and 'post' not in desc: name_hint = 'ant'
        elif 'post' in desc and 'ant' not in desc: name_hint = 'post'
        elif 'lao' in desc and 'rpo' not in desc: name_hint = 'lao'
        elif 'rpo' in desc and 'lao' not in desc: name_hint = 'rpo'
        elif 'lpo' in desc and 'rao' not in desc: name_hint = 'lpo'
        elif 'rao' in desc and 'lpo' not in desc: name_hint = 'rao'

        angle_hint = None
        if angle is not None:
            if abs(angle - 180) < 10: angle_hint = 'ant'
            elif abs(angle - 360) < 10 or abs(angle - 0) < 10: angle_hint = 'post'
            elif abs(angle - 135) < 10: angle_hint = 'lao'
            elif abs(angle - 315) < 10: angle_hint = 'rpo'
            elif abs(angle - 45) < 10: angle_hint = 'lpo'
            elif abs(angle - 225) < 10: angle_hint = 'rao'

        if angle_hint and name_hint and angle_hint != name_hint:
            self.log(f"WARNING: Metadata conflict! Angle ({angle}) suggests {angle_hint.upper()}, but text suggests {name_hint.upper()}. Defaulting to Angle calculation.")
            return angle_hint
        
        return angle_hint or name_hint or 'unknown'

    def apply_preprocessing(self):
        # Reset masks since image dimensions/counts may change
        self.ant_mask = None
        self.post_mask = None
        self.post_probs = None

        def process_view(view_key):
            img = self.raw_images[view_key]
            if img is None:
                return None
            
            img = img.astype(np.float32)
            
            # 1. Resample to 2x2mm
            if self.cb_resample.isChecked():
                spacing = self.spacings[view_key]
                if spacing is not None:
                    # factor = current_spacing / target_spacing
                    factor_y = spacing[0] / 2.0
                    factor_x = spacing[1] / 2.0
                    
                    if factor_y != 1.0 or factor_x != 1.0:
                        new_w = int(img.shape[1] * factor_x)
                        new_h = int(img.shape[0] * factor_y)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    self.log(f"Warning: No PixelSpacing metadata for {view_key.upper()}. Skipping resample.")

            # 2. Normalize to 200k Counts
            if self.cb_normalize.isChecked():
                total_counts = np.sum(img)
                if total_counts > 0:
                    img = img * (200000.0 / total_counts)
            
            # 3. Always Crop & Pad to 256x256
            th, tw = 256, 256
            h, w = img.shape
            
            # Crop center if larger than 256x256
            if h > th or w > tw:
                y_start = max((h - th) // 2, 0)
                y_end = min(y_start + th, h)
                x_start = max((w - tw) // 2, 0)
                x_end = min(x_start + tw, w)
                img = img[y_start:y_end, x_start:x_end]
                h, w = img.shape # Update shapes post-crop
                
            # Pad with zeros if smaller than 256x256
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)
            
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                
            return img

        # Apply to all buffers
        self.ant_image = process_view('ant')
        self.post_image = process_view('post')
        self.lao_image = process_view('lao')
        self.rpo_image = process_view('rpo')
        self.lpo_image = process_view('lpo')
        self.rao_image = process_view('rao')

        self.check_segment_ready()
        self.update_plots()






















    def process_files(self, file_paths):
        # Clear raw caches
        self.raw_images = {k: None for k in self.raw_images}
        self.spacings = {k: None for k in self.spacings}
        
        ref_patient_id = None
        ref_patient_name = None

        self.log(f"Scanning {len(file_paths)} potential file(s)...")

        for path in file_paths:
            try:
                # Read the DICOM header. Using defer_size speeds up scanning 
                # by avoiding reading massive pixel arrays until necessary.
                ds = pydicom.dcmread(path, force=True)
            except Exception as e:
                # If it fails to read, it's a system file or non-dicom. Quietly skip.
                continue

            # --- NEW: DICOM GUARDRAILS ---
            # Extract Modality and SOP Class UID safely
            modality = getattr(ds, 'Modality', '')
            sop_class = getattr(ds, 'SOPClassUID', '')
            
            # 1.2.840.10008.5.1.4.1.1.20 is the official UID for Nuclear Medicine Image Storage
            is_nm = (modality == 'NM') or ('1.2.840.10008.5.1.4.1.1.20' in sop_class)
            
            if not is_nm:
                # It's a valid DICOM, but not a Nuclear Medicine scan (e.g., Structured Report)
                # Comment out the log below if it gets too spammy for large folders
                self.log(f"Ignored {os.path.basename(path)}: Not an NM scan (Modality: {modality}).")
                continue
            
            # Verify it actually has pixel data
            if not hasattr(ds, 'pixel_array'):
                self.log(f"Ignored {os.path.basename(path)}: No pixel data found.")
                continue
            # -----------------------------

            # Continue with existing patient validation
            pid = str(getattr(ds, 'PatientID', 'Unknown_ID'))
            pname = str(getattr(ds, 'PatientName', 'Unknown_Name'))

            if ref_patient_id is None:
                ref_patient_id = pid
                ref_patient_name = pname
                self.lbl_patient_info.setText(f"Patient Name: {ref_patient_name}\nPatient ID: {ref_patient_id}")
            elif pid != ref_patient_id:
                self.log(f"Ignored {os.path.basename(path)}: Patient ID mismatch.")
                continue

            # Extract Pixel Spacing for resampling
            try:
                spacing = ds.PixelSpacing
                pixel_spacing = (float(spacing[0]), float(spacing[1]))
            except AttributeError:
                pixel_spacing = None

            arr = ds.pixel_array

            if arr.ndim == 3 and arr.shape[0] >= 2:
                num_frames = arr.shape[0]
                extracted_views = []
                
                for i in range(num_frames):
                    view = self.determine_view(ds, index=i)
                    if view in self.raw_images and self.raw_images[view] is None:
                        self.raw_images[view] = arr[i]
                        self.spacings[view] = pixel_spacing
                        extracted_views.append(view.upper())
                
                if extracted_views:
                    self.log(f"Extracted frames ({', '.join(extracted_views)}) from multi-frame file.")

            elif arr.ndim == 2:
                view = self.determine_view(ds)
                if view in self.raw_images and self.raw_images[view] is None:
                    self.raw_images[view] = arr
                    self.spacings[view] = pixel_spacing
                    self.log(f"Assigned {os.path.basename(path)} to {view.upper()}.")
                else:
                    self.log(f"Ignored {os.path.basename(path)}: Unrecognized view or slot already filled.")

        if self.raw_images['post'] is None:
            self.log("Warning: No valid posterior image detected. Segmentation requires a posterior image.")

        # Run everything through the preprocessing pipeline to push to display buffers
        self.apply_preprocessing()
        self.results_box.setText("")



    def check_segment_ready(self):
        if self.post_image is not None:
            self.btn_segment.setEnabled(True)
        else:
            self.btn_segment.setEnabled(False)
            self.btn_classify.setEnabled(False)

    def update_plots(self):
        self.colormap = self.combo_cmap.currentText()
        window_multiplier = self.slider_window.value() / 100.0
        
        for ax in self.axes.flatten():
            ax.clear()
            ax.axis('off')

        if self.ant_image is not None:
            vmax_ant = np.max(self.ant_image) * window_multiplier
            self.axes[0, 0].imshow(self.ant_image, cmap=self.colormap, vmin=0, vmax=vmax_ant)
            self.axes[0, 0].set_title("Anterior View")
            self.axes[1, 0].imshow(self.ant_image, cmap=self.colormap, vmin=0, vmax=vmax_ant)
            self.axes[1, 0].set_title("Anterior + Seg Overlay")
            if self.ant_mask is not None:
                self.overlay_mask(self.axes[1, 0], self.ant_mask, self.bg_masks_ant, self.display_srf_l, self.display_srf_r)
        else:
            self.axes[0, 0].set_title("Anterior View (Not Loaded)")
            self.axes[1, 0].set_title("Anterior (Not Loaded)")

        if self.post_image is not None:
            vmax_post = np.max(self.post_image) * window_multiplier
            self.axes[0, 1].imshow(self.post_image, cmap=self.colormap, vmin=0, vmax=vmax_post)
            self.axes[0, 1].set_title("Posterior View")
            self.axes[1, 1].imshow(self.post_image, cmap=self.colormap, vmin=0, vmax=vmax_post)
            self.axes[1, 1].set_title("Posterior + Seg Overlay")
        if self.post_mask is not None:
                self.overlay_mask(self.axes[1, 1], self.post_mask, self.bg_masks_post, self.display_srf_l, self.display_srf_r)

        self.canvas.draw()

        # --- NEW: Oblique views rendering ---
        for ax in self.axes_oblique.flatten():
            ax.clear()
            ax.axis('off')

        def plot_oblique(ax, img, title):
            if img is not None:
                vmax = np.max(img) * window_multiplier
                ax.imshow(img, cmap=self.colormap, vmin=0, vmax=vmax)
                ax.set_title(title)
            else:
                ax.set_title(f"{title} (Not Loaded)")

        # Map them intuitively: Left obliques on the left, Right on the right
        plot_oblique(self.axes_oblique[0, 0], self.lao_image, "LAO View")
        plot_oblique(self.axes_oblique[0, 1], self.rao_image, "RAO View")
        plot_oblique(self.axes_oblique[1, 0], self.lpo_image, "LPO View")
        plot_oblique(self.axes_oblique[1, 1], self.rpo_image, "RPO View")

        self.canvas_oblique.draw()






    def overlay_mask(self, ax, mask, bg_masks=None, srf_l=None, srf_r=None):
        if np.any(mask == 1):
            ax.contour(mask == 1, levels=[0.5], colors=['red'], linewidths=2)
        if np.any(mask == 2):
            ax.contour(mask == 2, levels=[0.5], colors=['blue'], linewidths=2)
            
        if bg_masks:
            if bg_masks.get(1) is not None and np.any(bg_masks[1]):
                ax.contour(bg_masks[1], levels=[0.5], colors=['orange'], linewidths=1.5, linestyles='dashed')
            if bg_masks.get(2) is not None and np.any(bg_masks[2]):
                ax.contour(bg_masks[2], levels=[0.5], colors=['cyan'], linewidths=1.5, linestyles='dashed')

        # --- New Text Annotation Logic ---
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)

        if srf_l is not None and srf_l > 0.0:
            coords = np.argwhere(mask == 1) # Label 1 is Left
            if len(coords) > 0:
                max_y = coords[:, 0].max()
                mean_x = coords[:, 1].mean()
                ax.text(mean_x, max_y + 8, f"{srf_l:.1f}%", color='red', fontsize=11, 
                        fontweight='bold', ha='center', va='top', bbox=bbox_props)

        if srf_r is not None and srf_r > 0.0:
            coords = np.argwhere(mask == 2) # Label 2 is Right
            if len(coords) > 0:
                max_y = coords[:, 0].max()
                mean_x = coords[:, 1].mean()
                ax.text(mean_x, max_y + 8, f"{srf_r:.1f}%", color='blue', fontsize=11, 
                        fontweight='bold', ha='center', va='top', bbox=bbox_props)









    def run_segmentation(self):
        if self.post_image is None:
            return

        self.log("Initializing Segmentation Network...")
        arch_choice = self.combo_arch.currentText()
        
        model_filename = ""
        try:
            if "SwinUNETR" in arch_choice:
                model_filename = "swinunetr_seg.pth"
                model = SwinUNETRR().to(self.device)
            elif "UNet" in arch_choice and "Attention" not in arch_choice and "Dyn" not in arch_choice:
                model_filename = "unet_seg.pth"
                model = UNett().to(self.device)
            elif "Attention" in arch_choice:
                model_filename = "attention_unet_seg.pth"
                model = AttentionUNett().to(self.device)
            elif "SegResNet" in arch_choice:
                model_filename = "segresnet_seg.pth"
                model = SegResNett().to(self.device)
            elif "DynUNet" in arch_choice:
                model_filename = "dynunet_seg.pth"
                model = DynUNett().to(self.device)
                
            pth_path = os.path.join(self.base_dir, model_filename)
            
            if not os.path.exists(pth_path):
                self.log(f"Error: Could not find required segmentation weights at {pth_path}")
                return
                
            model.load_state_dict(torch.load(pth_path, map_location=self.device))
            model.eval()
            self.log(f"Loaded weights from {model_filename} successfully.")
        except Exception as e:
            self.log(f"Error loading segmentation structure: {str(e)}")
            return

        self.log("Running AI inference on Posterior image...")
        
        try:
            with torch.no_grad():
                img_data = self.post_image.astype(np.float32)
                tensor_slice = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0).to(self.device)
                
                output_logits = model(tensor_slice)
                probabilities = F.softmax(output_logits, dim=1)
                
                self.post_probs = probabilities.cpu().numpy()[0]
            
            self.log("Segmentation complete. Calculating functional parameters...")
            self.recalc_masks()
            self.btn_classify.setEnabled(True)
            
        except Exception as e:
            self.log(f"Inference error: {str(e)}")

    def recalc_masks(self):
        if self.post_probs is None:
            return
            
        thresh_l = self.slider_thresh_l.value() / 100.0
        thresh_r = self.slider_thresh_r.value() / 100.0
        
        mask = np.zeros_like(self.post_probs[0], dtype=np.uint8)
        
        mask[self.post_probs[1] >= thresh_l] = 1
        mask[self.post_probs[2] >= thresh_r] = 2
        
        overlap = (self.post_probs[1] >= thresh_l) & (self.post_probs[2] >= thresh_r)
        if np.any(overlap):
            left_higher = self.post_probs[1] > self.post_probs[2]
            mask[overlap & left_higher] = 1
            mask[overlap & ~left_higher] = 2
            
        self.post_mask = mask
        
        if self.ant_image is not None:
            self.ant_mask = np.fliplr(self.post_mask)
            
        self.calculate_srf()
        self.update_plots()

    def get_background_roi(self, full_mask, image, label_idx, method):
        kidney_mask = (full_mask == label_idx)
        if not np.any(kidney_mask):
            return 0.0, None

        bg_mask = np.zeros_like(full_mask, dtype=bool)

        if method == "Circumferential":
            dilated = ndi.binary_dilation(kidney_mask, iterations=6)
            gap = ndi.binary_dilation(kidney_mask, iterations=2)
            bg_mask = dilated ^ gap

        elif method == "Inferolateral Crescent":
            dilated = ndi.binary_dilation(kidney_mask, iterations=8)
            gap = ndi.binary_dilation(kidney_mask, iterations=3)
            ring_mask = dilated ^ gap
            
            coords = np.argwhere(full_mask > 0)
            if len(coords) > 0:
                overall_cx = (coords[:, 1].min() + coords[:, 1].max()) / 2
            else:
                overall_cx = image.shape[1] / 2
                
            cy, cx = ndi.center_of_mass(kidney_mask)
            y_coords, x_coords = np.indices(image.shape)
            
            if cx < overall_cx:
                directional = (y_coords > cy) & (x_coords < cx)
            else:
                directional = (y_coords > cy) & (x_coords > cx)
                
            bg_mask = ring_mask & directional
            
            if not np.any(bg_mask):
                bg_mask = ring_mask
                
        elif method == "Center Rectangle":
            coords = np.argwhere(full_mask > 0)
            if len(coords) > 0:
                max_y = coords[:, 0].max()
                min_x = coords[:, 1].min()
                max_x = coords[:, 1].max()
                center_x = (min_x + max_x) // 2
                
                rect_h, rect_w = 10, 40
                y_start = min(max_y + 10, image.shape[0] - rect_h - 1)
                y_start = max(y_start, 0) 
                y_end = min(y_start + rect_h, image.shape[0])
                
                x_start = max(center_x - rect_w // 2, 0)
                x_end = min(x_start + rect_w, image.shape[1])
                
                if y_start < y_end and x_start < x_end:
                    bg_mask[y_start:y_end, x_start:x_end] = True
                else:
                    dilated = ndi.binary_dilation(kidney_mask, iterations=8)
                    gap = ndi.binary_dilation(kidney_mask, iterations=2)
                    bg_mask = dilated ^ gap
            else:
                bg_mask = np.zeros_like(full_mask, dtype=bool)

        bg_pixels = image[bg_mask]
        return (np.mean(bg_pixels) if len(bg_pixels) > 0 else 0.0), bg_mask

    def calculate_srf(self):
        labels = {1: "Left", 2: "Right"}
        net_counts = {'Ant': {}, 'Post': {}}
        bg_method = self.combo_bg_method.currentText()
        
        self.bg_masks_post = {1: None, 2: None}
        self.bg_masks_ant = {1: None, 2: None}
        
        for label_idx in labels.keys():
            post_kidney_mask = (self.post_mask == label_idx)
            post_bg_mean, post_bg_mask = self.get_background_roi(self.post_mask, self.post_image, label_idx, bg_method)
            self.bg_masks_post[label_idx] = post_bg_mask
            
            post_gross = np.sum(self.post_image[post_kidney_mask])
            post_area = np.sum(post_kidney_mask)
            post_net = post_gross - (post_bg_mean * post_area)
            net_counts['Post'][label_idx] = max(0, post_net)
            
            if self.ant_image is not None and self.ant_mask is not None:
                ant_kidney_mask = (self.ant_mask == label_idx)
                ant_bg_mean, ant_bg_mask = self.get_background_roi(self.ant_mask, self.ant_image, label_idx, bg_method)
                self.bg_masks_ant[label_idx] = ant_bg_mask
                
                ant_gross = np.sum(self.ant_image[ant_kidney_mask])
                ant_area = np.sum(ant_kidney_mask)
                ant_net = ant_gross - (ant_bg_mean * ant_area)
                net_counts['Ant'][label_idx] = max(0, ant_net)

        total_post = net_counts['Post'][1] + net_counts['Post'][2]
        post_srf_l = (net_counts['Post'][1] / total_post * 100) if total_post > 0 else 0
        post_srf_r = (net_counts['Post'][2] / total_post * 100) if total_post > 0 else 0

        total_counts_ant = np.sum(self.ant_image) if self.ant_image is not None else None
        total_counts_post = np.sum(self.post_image) if self.post_image is not None else None

        self.log("\n=== Total Image Counts ===")
        if total_counts_ant is not None:
            self.log(f"Anterior total counts: {total_counts_ant:,.0f}")
        else:
            self.log("Anterior total counts: N/A")
        if total_counts_post is not None:
            self.log(f"Posterior total counts: {total_counts_post:,.0f}")
        else:
            self.log("Posterior total counts: N/A")


        results_text = "=== Split Renal Function (SRF) ===\n\n"
        results_text += f"Posterior Only Method:\n  Left Kidney:  {post_srf_l:.1f} %\n  Right Kidney: {post_srf_r:.1f} %\n\n"

        if self.ant_image is not None:
            gm_l = np.sqrt(net_counts['Ant'][1] * net_counts['Post'][1])
            gm_r = np.sqrt(net_counts['Ant'][2] * net_counts['Post'][2])
            total_gm = gm_l + gm_r
            gm_srf_l = (gm_l / total_gm * 100) if total_gm > 0 else 0
            gm_srf_r = (gm_r / total_gm * 100) if total_gm > 0 else 0
            
            # Store GM for annotation
            self.display_srf_l = gm_srf_l
            self.display_srf_r = gm_srf_r
            
            results_text += f"Geometric Mean Method:\n  Left Kidney:  {gm_srf_l:.1f} %\n  Right Kidney: {gm_srf_r:.1f} %\n\n"
        else:
            # Store Posterior-only for annotation
            self.display_srf_l = post_srf_l
            self.display_srf_r = post_srf_r
            
            results_text += "Geometric Mean Method:\n  N/A (Anterior view not provided)\n\n"

        self.results_box.append(results_text)

    # --- Classification Logic ---
    def get_kidney_crop(self, image, mask, label_idx, size=(256, 256)):
        coords = np.argwhere(mask == label_idx)
        if len(coords) == 0:
            return None
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        margin = 10
        y0 = max(0, y0 - margin)
        y1 = min(image.shape[0], y1 + margin)
        x0 = max(0, x0 - margin)
        x1 = min(image.shape[1], x1 + margin)
        
        crop = image[y0:y1, x0:x1]
        
        crop_min, crop_max = crop.min(), crop.max()
        if crop_max > crop_min:
            crop = (crop - crop_min) / (crop_max - crop_min)
            
        return cv2.resize(crop.astype(np.float32), size)

    def run_anomaly_classification(self):
        if self.post_image is None:
            self.log("Error: Need a  posterior image to run classification.")
            return

        models = {
            "Anomaly": {"left": "cortical_anomaly_left.pth", "right": "cortical_anomaly_right.pth"},
            "Normal":  {"left": "normal_kidney_left.pth",   "right": "normal_kidney_right.pth"},
            "Afunc":   {"left": "afunctional_left.pth",     "right": "afunctional_right.pth"}
        }

        self.log("Running multi-model clinical evaluation...")
        
        arch_choice = self.combo_class_arch.currentText()
        results_str = "=== Clinical Evaluation Results ===\n\n"
        
        try:
            # --- NEW: Global Horseshoe Kidney Evaluation ---
            self.log("Evaluating for Horseshoe Morphology...")
            
            # Prepare the full posterior image tensor
            hk_tensor = torch.from_numpy(self.post_image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

            if arch_choice == "DenseNet121":
                hk_model = DenseNet(
                    spatial_dims=2,               
                    in_channels=1,         
                    out_channels=2,        
                    init_features=32,
                    growth_rate=16,
                    block_config=(6, 12, 24, 16),  
                    dropout_prob=0.1,  
                ).to(self.device)
            else:
                hk_model = resnet18(spatial_dims=2, n_input_channels=1, num_classes=2).to(self.device)
            
            # Ensure your model file is named 'horseshoe_kidney.pth' in the base directory
            hk_model_path = os.path.join(self.base_dir, "horseshoe.pth")
            
            if os.path.exists(hk_model_path):
                hk_model.load_state_dict(torch.load(hk_model_path, map_location=self.device))
                hk_model.eval()
                
                with torch.no_grad():
                    out_hk = hk_model(hk_tensor)
                    probs_hk = F.softmax(out_hk, dim=1)
                    prob_positive_hk = probs_hk[0, 1].item()
                
                hk_threshold = self.thresholds.get("Horseshoe", 0.710)
                if prob_positive_hk >= hk_threshold:
                    results_str += f"Global Morphology: Probable Horseshoe Kidney Anomaly (conf: {prob_positive_hk:.2f})\n\n"



            # --- Existing Left/Right Side Evaluation ---
            for side in ['left', 'right']:
                label_idx = 1 if side == 'left' else 2
                crop = self.post_image.astype(np.float32)
                
                if crop is None:
                    results_str += f"{side.capitalize()} Kidney: Afunctional / Not Visualized\n"
                    continue

                tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(self.device)
                
                findings = []
                for type_name, files in models.items():
                    threshold = self.thresholds[type_name][side]
                    
                    if arch_choice == "DenseNet121":
                        model = DenseNet(
                            spatial_dims=2,               
                            in_channels=1,         
                            out_channels=2,        
                            init_features=32,
                            growth_rate=16,
                            block_config=(6, 12, 24, 16),  
                            dropout_prob=0.1,  
                        ).to(self.device)
                    else:
                        model = resnet18(spatial_dims=2, n_input_channels=1, num_classes=2).to(self.device)
                    
                    model.load_state_dict(torch.load(os.path.join(self.base_dir, files[side]), map_location=self.device))
                    model.eval()
                    
                    with torch.no_grad():
                        out = model(tensor)
                        probs = F.softmax(out, dim=1)
                        prob_positive = probs[0, 1].item()
                    
                    if prob_positive >= threshold:
                        if type_name == "Anomaly":
                            findings.append(f"Probable Abnormal Cortical Uptake (conf: {prob_positive:.2f}) ")
                        elif type_name == "Normal":
                            findings.append(f"Probably Normal Kidney (conf: {prob_positive:.2f})")
                        elif type_name == "Afunc":
                            findings.append(f"Probable Afunctional/Agenetic Kidney (conf: {prob_positive:.2f})")
                    else:
                        if type_name == "Anomaly":
                            findings.append(f"Probably Normal Cortical Uptake (conf: {1 - prob_positive:.2f})")
                        elif type_name == "Normal":
                            findings.append(f"Probably Abnormal Kidney (conf: {1 - prob_positive:.2f})")

                if not findings:
                    results_str += f"{side.capitalize()} Kidney: Indeterminate\n"
                else:
                    results_str += f"{side.capitalize()} Kidney: {', '.join(findings)}\n"

            self.results_box.append(results_str)
            self.log("Clinical evaluation complete.")

        except Exception as e:
            self.log(f"Evaluation error: {str(e)}")


# --- HIDE CONSOLE FUNCTION ---
def hide_console():
    """Hides the Windows console window dynamically."""
    if os.name == 'nt':  # Check if operating system is Windows
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd != 0:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # 0 represents SW_HIDE

if __name__ == '__main__':
    # Note: 'app = QApplication(sys.argv)' was already executed at the very top!
    
    # Initialize your main application
    ex = DMSAAnalyzerApp()
    ex.show()
    
    # Remove the splash screen smoothly by linking its closure to the main window
    splash.finish(ex)
    
    # Hide the background console
    hide_console()
    
    # Execute the PyQt loop
    sys.exit(app.exec_())