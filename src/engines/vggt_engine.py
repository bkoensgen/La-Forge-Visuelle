import torch
import numpy as np
import trimesh
import os
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from .base_engine import BaseEngine

class VGGTEngine(BaseEngine):
    CAPABILITIES = {'single_image': True, 'scene_folder': True} # Laissons scene_folder pour l'instant

    def __init__(self, engine_config, device):
        super().__init__(engine_config, device)

    def _load_model(self):
        print(f"Chargement du modèle VGGT '{self.config['model_name']}'...")
        self.model = VGGT.from_pretrained(self.config['model_name']).to(self.device)

    def process(self, image: Image.Image, options: dict):
        # VGGT attend des chemins de fichiers, nous devons donc sauvegarder temporairement l'image traitée
        temp_path = "temp_vggt_input.png"
        image.save(temp_path)

        images_to_load = [temp_path]
        
        images = load_and_preprocess_images(images_to_load).to(self.device)
        os.remove(temp_path) # Nettoyage immédiat

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            predictions = self.model(images)

        if "point_map" not in predictions or predictions["point_map"] is None: return None
        points = predictions["point_map"].reshape(-1, 3).cpu().numpy()
        
        z_coords = points[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        colors = np.zeros_like(points, dtype=np.uint8)
        if z_max > z_min:
            z_norm = (z_coords - z_min) / (z_max - z_min)
            colors[:, 0] = 255 * z_norm
            colors[:, 2] = 255 * (1 - z_norm)

        print("Reconstruction VGGT terminée.")
        return trimesh.Trimesh(vertices=points, vertex_colors=colors)