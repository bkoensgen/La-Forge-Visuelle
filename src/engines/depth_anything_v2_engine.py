import torch
import numpy as np
from PIL import Image
import os
import cv2

# Importation depuis le code du vendor que nous avons cloné
from depth_anything_v2.dpt import DepthAnythingV2

from .base_engine import BaseEngine

class DepthAnythingV2Engine(BaseEngine):
    """
    Moteur pour Depth Anything V2, utilisant le code et les poids officiels.
    """
    CAPABILITIES = {'single_image': True, 'scene_folder': False}

    # Map des choix UI aux paramètres du constructeur du modèle et aux fichiers de poids
    MODEL_CONFIG = {
        'Small': {'encoder': 'vits', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'weight_file': 'depth_anything_v2_vits.pth'},
        'Base':  {'encoder': 'vitb', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'weight_file': 'depth_anything_v2_vitb.pth'},
        'Large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'weight_file': 'depth_anything_v2_vitl.pth'},
    }

    def __init__(self, engine_config, device):
        super().__init__(engine_config, device)
        self.loaded_variant = None

    def _load_model(self):
        """ Méthode vide pour satisfaire le contrat de la classe de base. """
        pass

    def _load_specific_variant(self, variant: str):
        """ Charge une variante spécifique du modèle si elle n'est pas déjà chargée. """
        if variant == self.loaded_variant and self.model is not None:
            return

        config = self.MODEL_CONFIG.get(variant)
        if not config:
            raise ValueError(f"Variante de modèle inconnue : {variant}")

        weight_path = os.path.join('checkpoints', config['weight_file'])
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Fichier de poids non trouvé : {weight_path}. Veuillez exécuter 'python install_helper.py' pour le télécharger.")

        print(f"Chargement du modèle Depth Anything V2 (variante: {variant}) depuis '{weight_path}'...")
        
        model_params = {k: v for k, v in config.items() if k != 'weight_file'}
        self.model = DepthAnythingV2(**model_params).to(self.device).eval()

        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        
        self.loaded_variant = variant
        self.is_loaded = True
        print("Chargement terminé.")

    def process(self, image: Image.Image, options: dict) -> dict:
        """ Effectue l'inférence pour obtenir une carte de profondeur. """
        selected_variant = options.get('model_variant', 'Large')
        self._load_specific_variant(selected_variant)

        if not self.is_loaded:
            raise RuntimeError("Le modèle n'a pas pu être chargé.")

        print("Lancement de l'inférence Depth Anything V2...")
        
        # Le modèle attend une image BGR (format OpenCV), on convertit depuis PIL (RGB)
        raw_img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Utilisation de la méthode d'inférence personnalisée
        depth = self.model.infer_image(raw_img_bgr)
        
        # Le résultat est déjà un tableau NumPy, on le normalise simplement
        min_val, max_val = np.min(depth), np.max(depth)
        if max_val > min_val:
            normalized_depth = (depth - min_val) / (max_val - min_val)
        else:
            normalized_depth = np.zeros_like(depth)
            
        print("Inférence Depth Anything V2 terminée.")
        return {'depth_map': normalized_depth}