from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PIL import Image
import numpy as np
import math
import trimesh

from src.engines.preprocessor import RMBGPreprocessor
from src.geometry_builder import GeometryBuilder
from src import config as app_config

def resize_and_pad(img: Image.Image, target_size: int, divisor: int = 64) -> Image.Image:
    """Redimensionne et ajoute un rembourrage pour que les dimensions soient divisibles."""
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    new_width = int(math.ceil(img.width / divisor)) * divisor
    new_height = int(math.ceil(img.height / divisor)) * divisor
    
    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    paste_x = (new_width - img.width) // 2
    paste_y = (new_height - img.height) // 2
    padded_image.paste(img, (paste_x, paste_y))
    
    print(f"Image redimensionnée à: {img.size}, puis rembourrée à: {padded_image.size}")
    return padded_image

class LocalProcessor(QObject):
    """
    Gère le pipeline de traitement de reconstruction 3D sur la machine locale.
    """
    # Le signal finished émet maintenant directement l'objet trimesh final.
    finished = pyqtSignal(object) 
    error = pyqtSignal(str)
    thumbnail_data_ready = pyqtSignal(int, bytes, int, int)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.preprocessor = RMBGPreprocessor(app_config.DEVICE)
        self.builder = GeometryBuilder()

    @pyqtSlot(str, str, dict)
    def process(self, path: str, engine_name: str, options: dict):
        """
        Slot qui exécute le pipeline de reconstruction complet.
        """
        try:
            print(f"\n--- Démarrage du pipeline de traitement LOCAL pour {engine_name} ---")
            img = Image.open(path).convert("RGB")
            
            # --- Pré-traitement de l'image ---
            resize_target = options.get('resize_to', 'Original')
            if resize_target != 'Original':
                img = resize_and_pad(img, int(resize_target))

            fg_mask = None
            if options.get('bg_removal', False):
                print("Application de la suppression d'arrière-plan...")
                preproc_data = self.preprocessor.process(img)
                img = preproc_data['image']
                fg_mask = preproc_data['mask']
            
            # --- Inférence du moteur IA ---
            engine = self.controller.get_engine(engine_name)
            engine.load_model_if_needed()
            raw_data = engine.process(img, options)
            if raw_data is None: 
                raise ValueError("Le moteur n'a retourné aucune donnée.")

            # --- Construction de la géométrie ---
            mesh = self.builder.build(raw_data, np.array(img), fg_mask, options)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("La construction du maillage a échoué ou a retourné un type incorrect.")

            # --- Mise en cache (logique simple) ---
            mesh_cache_key = self.controller.get_mesh_cache_key(path, engine_name, options)
            self.controller.mesh_cache[mesh_cache_key] = mesh

            self.finished.emit(mesh)

        except Exception as e:
            import traceback
            error_message = f"Erreur dans le processeur local: {traceback.format_exc()}"
            print(error_message)
            self.error.emit(error_message)

    @pyqtSlot()
    def load_thumbnails(self):
        """Charge les miniatures en arrière-plan (cette partie reste toujours locale)."""
        print("Démarrage du chargement des miniatures en arrière-plan...")
        for index, path in enumerate(self.controller.items):
            try:
                thumb_pil = self.controller.get_thumbnail(path)
                if thumb_pil:
                    thumb_rgba = thumb_pil.convert("RGBA").tobytes()
                    width, height = thumb_pil.size
                    self.thumbnail_data_ready.emit(index, thumb_rgba, width, height)
            except Exception as e:
                print(f"Erreur miniature {index}: {e}")
        print("Chargement des miniatures terminé.")