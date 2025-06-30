import sys, os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QListWidget, QListWidgetItem, QLabel, QStatusBar, QComboBox,
                             QPushButton, QGroupBox, QCheckBox, QStyle, QMessageBox, QSpinBox, QFormLayout, QDoubleSpinBox)
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pyvistaqt import QtInteractor
import trimesh

from src import config
from src.app_controller import AppController
from src.processing.local_processor import LocalProcessor
from src.processing.remote_processor import RemoteProcessor

from src.config import DEFAULT_ENGINE, PIPELINE_OPTIONS
from PIL.ImageQt import ImageQt
from PIL import Image as PILImage
import numpy as np

class MainWindow(QMainWindow):
    # Ce signal est maintenant agnostique : il demande juste un traitement.
    processing_request = pyqtSignal(str, str, dict)
    thumbnail_request = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Plateforme de Visualisation 3D - Mode: {config.PROCESSING_MODE.upper()}")
        self.setGeometry(100, 100, 1800, 1000)

        self.controller = AppController()
        self.option_widgets = {}

        # --- INSTANCIATION DU PROCESSEUR SELON LA CONFIGURATION ---
        self.thread = QThread()
        
        if config.PROCESSING_MODE == "local":
            print("INFO: Initialisation en mode de traitement LOCAL.")
            self.processor = LocalProcessor(self.controller)
        elif config.PROCESSING_MODE == "remote":
            print("INFO: Initialisation en mode de traitement REMOTE.")
            api_key = os.environ.get("RUNPOD_API_KEY")
            if not api_key:
                QMessageBox.critical(self, "Erreur de Configuration",
                                     "La variable d'environnement RUNPOD_API_KEY n'est pas définie.\n"
                                     "Le mode 'remote' est indisponible.")
                # Fallback ou sortie gracieuse
                sys.exit("Clé API RunPod non trouvée.")
            self.processor = RemoteProcessor(api_key, config.RUNPOD_ENDPOINT_ID)
        else:
            raise ValueError(f"Mode de traitement inconnu : {config.PROCESSING_MODE}")

        self.processor.moveToThread(self.thread)

        self.processing_request.connect(self.processor.process)
        self.processor.finished.connect(self.on_processing_finished)
        self.processor.error.connect(self.on_error)
        
        # Le chargement des miniatures reste local et rapide
        self.local_thumb_worker = LocalProcessor(self.controller)
        self.thumb_thread = QThread()
        self.local_thumb_worker.moveToThread(self.thumb_thread)
        self.thumbnail_request.connect(self.local_thumb_worker.load_thumbnails)
        self.local_thumb_worker.thumbnail_data_ready.connect(self.on_thumbnail_data_ready)
        self.thumb_thread.start()
        # --- FIN DE L'INSTANCIATION ---

        self.thread.started.connect(self.start_background_tasks)

        self._setup_ui()
        if self.controller.items:
            self.item_browser.setCurrentRow(0)
            self.on_item_selected(self.item_browser.currentItem())

        self.thread.start()

    def _setup_ui(self):
        self.setStatusBar(QStatusBar(self))
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        left_panel_layout = QVBoxLayout()
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)
        left_panel_widget.setFixedWidth(450)
        left_panel_layout.setSpacing(10)
        left_panel_layout.addWidget(QLabel("<b>1. Sélectionnez une Image ou une Scène:</b>"))
        self.item_browser = QListWidget()
        self.item_browser.setSpacing(5)
        for path in self.controller.items:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            icon_type = QStyle.StandardPixmap.SP_DirIcon if os.path.isdir(path) else QStyle.StandardPixmap.SP_FileIcon
            item.setIcon(self.style().standardIcon(icon_type))
            self.item_browser.addItem(item)
        self.item_browser.itemClicked.connect(self.on_item_selected)
        left_panel_layout.addWidget(self.item_browser, 3)
        left_panel_layout.addWidget(QLabel("<b>Prévisualisation :</b>"))
        self.preview_label = QLabel("Cliquez sur une image pour voir un aperçu.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(int(self.controller.PREVIEW_SIZE[1] * 0.8))
        self.preview_label.setStyleSheet("background-color: #2E2E2E; border-radius: 5px;")
        left_panel_layout.addWidget(self.preview_label, 2)
        left_panel_layout.addWidget(QLabel("<b>2. Choisissez un Moteur de Reconstruction:</b>"))
        self.engine_selector = QComboBox()
        self.engine_selector.addItems(self.controller.engines.keys())
        self.engine_selector.setCurrentText(DEFAULT_ENGINE)
        left_panel_layout.addWidget(self.engine_selector)
        self.pipeline_options_group = QGroupBox("3. Options de Pré-traitement")
        self.pipeline_options_layout = QFormLayout()
        self.pipeline_options_group.setLayout(self.pipeline_options_layout)
        left_panel_layout.addWidget(self.pipeline_options_group)
        self.engine_options_group = QGroupBox("4. Options du Moteur")
        self.engine_options_layout = QFormLayout()
        self.engine_options_group.setLayout(self.engine_options_layout)
        left_panel_layout.addWidget(self.engine_options_group)
        left_panel_layout.addStretch()
        process_button = QPushButton("Lancer le Traitement")
        process_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;")
        process_button.clicked.connect(self.on_process_clicked)
        left_panel_layout.addWidget(process_button)
        main_layout.addWidget(left_panel_widget)
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor, 4)
        self.engine_selector.currentTextChanged.connect(self.on_engine_changed)
        self.on_engine_changed(self.engine_selector.currentText())

    def on_engine_changed(self, engine_name):
        self.option_widgets.clear()
        for layout in [self.pipeline_options_layout, self.engine_options_layout]:
            while layout.count():
                child = layout.takeAt(0)
                if widget := child.widget(): widget.deleteLater()
        for key, params in PIPELINE_OPTIONS.items():
            widget = None
            if params['type'] == 'bool': widget = QCheckBox(); widget.setChecked(params.get('default', False))
            elif params['type'] == 'choice': widget = QComboBox(); widget.addItems(params.get('choices', [])); widget.setCurrentText(params.get('default', ''))
            elif params['type'] == 'float': widget = QDoubleSpinBox(); widget.setRange(params.get('min', 0.0), params.get('max', 100.0)); widget.setSingleStep(params.get('step', 0.1)); widget.setValue(params.get('default', 1.0))
            if widget: self.pipeline_options_layout.addRow(params['label'], widget); self.option_widgets[key] = widget
        engine = self.controller.get_engine(engine_name)
        if engine and 'options' in engine.config:
            for key, params in engine.config['options'].items():
                widget = None
                if params['type'] == 'bool': widget = QCheckBox(); widget.setChecked(params.get('default', False))
                elif params['type'] == 'int': widget = QSpinBox(); widget.setRange(params.get('min', 0), params.get('max', 100)); widget.setValue(params.get('default', 0))
                elif params['type'] == 'choice': widget = QComboBox(); widget.addItems(params.get('choices', [])); widget.setCurrentText(str(params.get('default', '')))
                if widget: self.engine_options_layout.addRow(params['label'], widget); self.option_widgets[key] = widget
        self.engine_options_group.setVisible(self.engine_options_layout.rowCount() > 0)


    def start_background_tasks(self):
        if self.controller.items:
            self.thumbnail_request.emit()

    def on_thumbnail_data_ready(self, index: int, raw_data: bytes, width: int, height: int):
        if item := self.item_browser.item(index):
            if raw_data:
                qimage = QImage(raw_data, width, height, QImage.Format.Format_RGBA8888)
                if not qimage.isNull(): item.setIcon(QIcon(QPixmap.fromImage(qimage.copy())))


    def on_item_selected(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.update_preview_panel(path)


    def update_preview_panel(self, path):
        preview_pil = self.controller.get_preview_image(path)
        if preview_pil:
            pixmap = QPixmap.fromImage(ImageQt(preview_pil))
            self.preview_label.setPixmap(pixmap.scaled(self.preview_label.width(), self.preview_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else: self.preview_label.setText("Pas d'aperçu pour les scènes.")


    def on_process_clicked(self):
        if not (current_item := self.item_browser.currentItem()):
            self.statusBar().showMessage("Veuillez sélectionner une image.", 5000)
            return

        path = current_item.data(Qt.ItemDataRole.UserRole)
        engine_name = self.engine_selector.currentText()
        options = {k: w.isChecked() if isinstance(w, QCheckBox) else w.value() if isinstance(w, (QSpinBox, QDoubleSpinBox)) else w.currentText() for k, w in self.option_widgets.items()}
        
        # Le cache n'est pertinent qu'en mode local pour l'instant.
        # Le worker distant pourrait avoir son propre cache, mais c'est transparent pour nous.
        if config.PROCESSING_MODE == "local":
             mesh_cache_key = self.controller.get_mesh_cache_key(path, engine_name, options)
             if mesh_cache_key in self.controller.mesh_cache:
                 print(f"Cache HIT (Maillage final) pour {os.path.basename(path)}")
                 self.update_3d_view(self.controller.mesh_cache[mesh_cache_key])
                 return

        self.statusBar().showMessage(f"Lancement du traitement avec {engine_name} en mode {config.PROCESSING_MODE}...")
        self.processing_request.emit(path, engine_name, options)

    def on_processing_finished(self, mesh: trimesh.Trimesh):
        """
        Ce slot reçoit le maillage final, que le traitement ait été local ou distant.
        Le processeur distant est chargé de télécharger le .glb et de le charger en objet Trimesh.
        """
        self.statusBar().showMessage("Traitement terminé avec succès.", 5000)
        self.update_3d_view(mesh, reset_camera=True)
        
        if config.PROCESSING_MODE == 'local' and self.item_browser.currentItem():
            path = self.item_browser.currentItem().data(Qt.ItemDataRole.UserRole)
            engine_name = self.engine_selector.currentText()
            options = {k: w.isChecked() if isinstance(w, QCheckBox) else w.value() if isinstance(w, (QSpinBox, QDoubleSpinBox)) else w.currentText() for k, w in self.option_widgets.items()}
            mesh_cache_key = self.controller.get_mesh_cache_key(path, engine_name, options)
            self.controller.mesh_cache[mesh_cache_key] = mesh


    def on_error(self, message):
        self.statusBar().showMessage(f"Erreur: {message}", 10000)
        QMessageBox.critical(self, "Erreur Critique", message)
    
    def update_3d_view(self, mesh, reset_camera: bool = True):
        self.plotter.clear()
        if mesh and isinstance(mesh, trimesh.Trimesh):
            pv_mesh = self.controller.trimesh_to_polydata(mesh)
            self.plotter.add_mesh(pv_mesh, scalars='colors', rgb=True, smooth_shading=True, specular=0.3)
        
        if reset_camera:
            self.plotter.reset_camera()

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait()
        self.thumb_thread.quit()
        self.thumb_thread.wait()
        super().closeEvent(event)