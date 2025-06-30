import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from src import config
import numpy as np

class RMBGPreprocessor:
    def __init__(self, device):
        self.device = device
        self.model = None

    def load_model_if_needed(self):
        if self.model is None:
            cfg = config.RMBG_CONFIG
            print(f"Chargement du pré-processeur BG Removal '{cfg['model_name']}'...")
            self.model = AutoModelForImageSegmentation.from_pretrained(cfg['model_name'], trust_remote_code=True).to(self.device).eval()

    def process(self, image: Image.Image) -> dict:
        """
        Traite une image pour en supprimer le fond.
        Retourne un dictionnaire contenant l'image nettoyée et le masque.
        """
        self.load_model_if_needed()
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        with torch.no_grad():
            input_tensor = transform(image.resize((1024, 1024))).unsqueeze(0).to(self.device)
            result = self.model(input_tensor)
            mask_tensor = result[0][0].squeeze()
        
        mask = transforms.ToPILImage()(mask_tensor).resize(image.size, Image.Resampling.LANCZOS)
        
        cleaned_image = Image.new("RGBA", image.size)
        cleaned_image.paste(image, mask=mask)
        
        return {'image': cleaned_image.convert("RGB"), 'mask': np.array(mask)}