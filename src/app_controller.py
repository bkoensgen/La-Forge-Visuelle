import os
import importlib
import numpy as np
import pyvista as pv
from PIL import Image
from src import config

class AppController:
    """
    Le cerveau. Gère la logique, les données, les caches et les moteurs.
    """
    def __init__(self):
        self.items = []
        self.engines = {}
        self.raw_data_cache = {} # Pour les résultats lents de l'IA
        self.mesh_cache = {}       # Pour le résultat 3D final
        self.thumbnail_cache = {}
        self.preview_cache = {}

        self.THUMB_SIZE = (128, 128)
        self.PREVIEW_SIZE = (400, 400)

        self._discover_items()
        self._load_engines()

    def _discover_items(self):
        folder = config.INPUT_FOLDER
        try:
            dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.items = sorted(dirs) + sorted(files)
            print(f"{len(self.items)} items (images/scènes) trouvés.")
        except FileNotFoundError:
            print(f"ERREUR: Dossier d'entrée '{folder}' non trouvé.")
    
    def _load_engines(self):
        print("Chargement des moteurs de reconstruction...")
        for name, cfg in config.ENGINES_CONFIG.items():
            try:
                module = importlib.import_module(cfg['module'])
                EngineClass = getattr(module, cfg['class'])
                self.engines[name] = EngineClass(cfg, config.DEVICE)
                print(f"  - Moteur '{name}' initialisé.")
            except Exception as e:
                print(f"Erreur lors de l'initialisation du moteur '{name}': {e}")
        print("Moteurs chargés.")

    def get_engine(self, name):
        return self.engines.get(name)

    def get_mesh_cache_key(self, path, engine_name, options):
        """Génère une clé de cache pour le maillage final, incluant TOUTES les options."""
        options_tuple = tuple(sorted(options.items()))
        return (path, engine_name, options_tuple)

    def get_raw_data_cache_key(self, path, engine_name, options):
        """
        NOUVEAU : Génère une clé de cache pour les données brutes de l'IA.
        N'inclut QUE les options qui affectent le modèle IA lui-même, pas le post-traitement.
        """
        engine = self.get_engine(engine_name)
        engine_option_keys = engine.config.get('options', {}).keys()
        
        # On filtre les options pour ne garder que celles du moteur
        engine_specific_options = {k: v for k, v in options.items() if k in engine_option_keys}
        
        options_tuple = tuple(sorted(engine_specific_options.items()))
        return (path, engine_name, options_tuple)

    def get_thumbnail(self, path):
        if path in self.thumbnail_cache: return self.thumbnail_cache[path]
        if os.path.isdir(path): return None
        try:
            with Image.open(path) as img:
                img.thumbnail(self.THUMB_SIZE, Image.Resampling.LANCZOS)
                thumb = img.copy()
                self.thumbnail_cache[path] = thumb
                return thumb
        except Exception: return None

    def get_preview_image(self, path: str):
        if path in self.preview_cache: return self.preview_cache[path]
        if os.path.isdir(path): return None
        try:
            with Image.open(path) as img:
                img.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
                preview = img.copy()
                self.preview_cache[path] = preview
                return preview
        except Exception as e:
            print(f"Erreur de création de la prévisualisation pour {path}: {e}")
            return None

    def trimesh_to_polydata(self, mesh):
        if not mesh: return None
        polydata = pv.PolyData(mesh.vertices)
        if hasattr(mesh.visual, 'vertex_colors'):
            polydata['colors'] = mesh.visual.vertex_colors[:, :3]
        if len(mesh.faces) > 0:
            polydata.faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        return polydata