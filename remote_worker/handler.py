import os
import sys
import base64
import trimesh
import tempfile
from PIL import Image
import numpy as np
import io
import runpod
import traceback

# --- Initialisation (ne change pas) ---
project_root = os.path.dirname(os.path.abspath(__file__))
depthfm_repo_path = os.path.join(project_root, 'vendor', 'depthfm_repo')
da_repo_path = os.path.join(project_root, 'vendor', 'depth_anything_v2_repo')
sys.path.insert(0, depthfm_repo_path)
sys.path.insert(0, da_repo_path)
sys.path.insert(0, project_root)

from src.app_controller import AppController
from src.geometry_builder import GeometryBuilder
from src.engines.preprocessor import RMBGPreprocessor
from src import config as app_config

print("--- Initialisation du Worker RunPod (démarrage à froid) ---")
controller = AppController()
builder = GeometryBuilder()
preprocessor = RMBGPreprocessor(app_config.DEVICE)
print("--- Worker prêt à recevoir des tâches ---")
# ---------------------------------------------


def handler(job):
    try:
        job_input = job.get('input', {}) # Utilise .get() pour la sécurité

        # --- VÉRIFICATION DE L'ENTRÉE ---
        # Si c'est une requête de test ou une entrée invalide, on s'arrête poliment.
        image_b64 = job_input.get('image_b64')
        engine_name = job_input.get('engine_name')

        if not all([image_b64, engine_name]):
            error_msg = "Entrée invalide. Les clés 'image_b64' et 'engine_name' sont requises."
            print(error_msg)
            # On retourne une erreur propre au lieu de crasher.
            # RunPod verra ça comme une tâche terminée (avec erreur), pas comme un worker planté.
            return {"error": error_msg}
        # ------------------------------------

        options = job_input.get('options', {}) # .get() pour les options aussi
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # --- Traitement (ne change pas) ---
        def resize_and_pad(img_in: Image.Image, target_size: int, divisor: int = 64) -> Image.Image:
            import math
            img_in.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            new_width = int(math.ceil(img_in.width / divisor)) * divisor
            new_height = int(math.ceil(img_in.height / divisor)) * divisor
            padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
            padded_image.paste(img_in, ((new_width - img_in.width) // 2, (new_height - img_in.height) // 2))
            return padded_image

        resize_target = options.get('resize_to', 'Original')
        if resize_target != 'Original':
            img = resize_and_pad(img, int(resize_target))
        
        fg_mask = None
        if options.get('bg_removal', False):
            preproc_data = preprocessor.process(img)
            img = preproc_data['image']
            fg_mask = preproc_data['mask']
        
        engine = controller.get_engine(engine_name)
        engine.load_model_if_needed()
        raw_data = engine.process(img, options)
        mesh = builder.build(raw_data, np.array(img), fg_mask, options)
        
        if not mesh:
            raise ValueError("La construction de la géométrie a échoué.")
        # --- Fin du traitement ---

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
            mesh.export(file_obj=tmp_file.name, file_type='glb')
            temp_path = tmp_file.name
        
        print(f"Maillage exporté vers {temp_path}. RunPod va l'uploader.")
        
        return temp_path

    except Exception as e:
        error_message = f"Erreur dans le handler: {traceback.format_exc()}"
        print(error_message)
        return {"error": error_message}

runpod.serverless.start({"handler": handler})