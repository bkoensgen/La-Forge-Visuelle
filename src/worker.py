from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PIL import Image
import numpy as np
from src.engines.preprocessor import RMBGPreprocessor
from src.geometry_builder import GeometryBuilder
from src import config
import math

def resize_and_pad(img: Image.Image, target_size: int, divisor: int = 64) -> Image.Image:
    original_image = img.copy()
    original_image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    new_width = int(math.ceil(original_image.width / divisor)) * divisor
    new_height = int(math.ceil(original_image.height / divisor)) * divisor
    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    paste_x = (new_width - original_image.width) // 2
    paste_y = (new_height - original_image.height) // 2
    padded_image.paste(original_image, (paste_x, paste_y))
    print(f"Image redimensionnée à: {original_image.size}, puis rembourrée à: {padded_image.size}")
    return padded_image

class Worker(QObject):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    thumbnail_data_ready = pyqtSignal(int, bytes, int, int)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.preprocessor = RMBGPreprocessor(config.DEVICE)
        self.builder = GeometryBuilder()

    @pyqtSlot(str, str, dict)
    def process(self, path, engine_name, options):
        try:
            print(f"\n--- Démarrage du pipeline de traitement pour {engine_name} ---")
            img = Image.open(path).convert("RGB")
            
            resize_target = options.get('resize_to', 'Original')
            if resize_target != 'Original':
                img = resize_and_pad(img, int(resize_target))

            fg_mask = None
            if options.get('bg_removal', False):
                print("Application de la suppression d'arrière-plan...")
                preproc_data = self.preprocessor.process(img)
                img = preproc_data['image']
                fg_mask = preproc_data['mask']

            engine = self.controller.get_engine(engine_name)

            engine.load_model_if_needed()
            
            raw_data = engine.process(img, options)
            if raw_data is None: raise ValueError("Le moteur n'a retourné aucune donnée.")

            raw_data_cache_key = self.controller.get_raw_data_cache_key(path, engine_name, options)
            self.controller.raw_data_cache[raw_data_cache_key] = raw_data

            mesh = self.builder.build(raw_data, np.array(img), fg_mask, options)
            
            mesh_cache_key = self.controller.get_mesh_cache_key(path, engine_name, options)
            self.controller.mesh_cache[mesh_cache_key] = mesh

            self.finished.emit(raw_data_cache_key, mesh)

        except Exception as e:
            import traceback
            self.error.emit(f"Erreur: {traceback.format_exc()}")

    @pyqtSlot()
    def load_thumbnails(self):
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