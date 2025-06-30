import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .base_engine import BaseEngine

from depthfm.dfm import DepthFM as DepthFMModel

class DepthFMEngine(BaseEngine):
    CAPABILITIES = {'single_image': True, 'scene_folder': False}

    def _load_model(self):
        ckpt_path = self.config['model_name']
        print(f"Chargement du modèle DepthFM depuis '{ckpt_path}'...")
        self.model = DepthFMModel(ckpt_path=ckpt_path).to(self.device).eval()

    def process(self, image: Image.Image, options: dict) -> dict:
        """
        Fait l'inférence et retourne la carte de profondeur brute.
        """
        print("Lancement de l'inférence DepthFM...")
        img_tensor = transforms.ToTensor()(image) * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        num_steps = options.get('num_steps', 2)
        ensemble_size = options.get('ensemble_size', 4)

        with torch.no_grad(), torch.cuda.amp.autocast():
            depth = self.model.predict_depth(img_tensor, num_steps=num_steps, ensemble_size=ensemble_size)

        depth_map_numpy = (depth.squeeze().cpu().numpy() + 1.0) / 2.0
        
        print("Inférence DepthFM terminée.")
        return {'depth_map': depth_map_numpy}