import torch
import numpy as np
from PIL import Image
from moge.model.v2 import MoGeModel
from .base_engine import BaseEngine

class MogeEngine(BaseEngine):
    CAPABILITIES = {'single_image': True, 'scene_folder': False}

    def _load_model(self):
        print(f"Chargement du modèle MoGe '{self.config['model_name']}'...")
        self.model = MoGeModel.from_pretrained(self.config['model_name']).to(self.device).eval()

    def process(self, image: Image.Image, options: dict) -> dict:
        """
        Ne fait que l'inférence et retourne les données brutes.
        """
        print("Lancement de l'inférence MoGe...")
        tensor = torch.tensor(np.array(image)/255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        
        with torch.no_grad():
             # Convertit directement les tenseurs de sortie en numpy
             raw_data = {k: v.cpu().numpy() for k, v in self.model.infer(tensor).items()}
        
        print("Inférence MoGe terminée.")
        return raw_data