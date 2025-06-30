from abc import ABC, abstractmethod
from PIL import Image

class BaseEngine(ABC):
    CAPABILITIES = {'single_image': False, 'scene_folder': False}

    def __init__(self, engine_config, device):
        self.config = engine_config
        self.device = device
        self.model = None
        self.is_loaded = False

    def load_model_if_needed(self):
        if not self.is_loaded:
            self._load_model()
            self.is_loaded = True

    @abstractmethod
    def _load_model(self): pass

    @abstractmethod
    def process(self, image: Image.Image, options: dict):
        """
        Traite un objet image PIL et retourne un maillage Trimesh.
        Note: La méthode prend maintenant un objet Image, pas un chemin.
        """
        pass