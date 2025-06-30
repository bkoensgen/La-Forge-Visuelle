import torch

# Configuration du mode de traitement ---
# Choisir le mode d'exécution du pipeline de reconstruction.
# "local": Utilise le GPU de votre machine. Nécessite une configuration locale complète.
# "remote": Dédorte le calcul sur un worker RunPod Serverless. Nécessite une clé API.
PROCESSING_MODE = "local" # Options: "local", "remote"

# ID de votre endpoint RunPod Serverless (à créer dans le tableau de bord RunPod)
RUNPOD_ENDPOINT_ID = "VOTRE_ENDPOINT_ID_ICI"
# ----------------------------------------------------


# --- Options de pré-traitement (pipeline) applicables à plusieurs moteurs ---
PIPELINE_OPTIONS = {
    'bg_removal': {'label': "Supprimer l'arrière-plan (RMBG)", 'default': True, 'type': 'bool'},
    'resize_to': {
        'label': "Réduire l'image à (max)",
        'default': "Original",
        'type': 'choice',
        'choices': ["Original", "1024", "768", "512"]
    },
    'depth_scale': {
        'label': "Échelle de Profondeur",
        'default': 1.0,
        'type': 'float',
        'min': 0.1,
        'max': 50.0,
        'step': 0.5
    }
}

# --- Dictionnaire central pour la configuration des moteurs de reconstruction ---
ENGINES_CONFIG = {
    'MoGe': {
        'class': 'MogeEngine',
        'module': 'src.engines.moge_engine',
        'model_name': "Ruicheng/moge-2-vitl-normal",
        'options': {
            'render_mode': {'label': "Nuage de Points (rapide)", 'default': False, 'type': 'bool'},
            'quality_filters': {'label': "Filtres Qualité (lent)", 'default': True, 'type': 'bool'},
        }
    },
    'DepthAnythingV2': {
        'class': 'DepthAnythingV2Engine',
        'module': 'src.engines.depth_anything_v2_engine',
        'options': {
            'model_variant': {
                'label': "Variante du Modèle",
                'default': 'Large',
                'type': 'choice',
                'choices': ['Small', 'Base', 'Large']
            }
        }
    },
    'DepthFM': {
        'class': 'DepthFMEngine',
        'module': 'src.engines.depthfm_engine',
        'model_name': 'checkpoints/depthfm-v1.ckpt',
        'options': {
            'num_steps': {'label': "Nombre d'étapes", 'default': 2, 'min': 1, 'max': 10, 'type': 'int'},
            'ensemble_size': {'label': "Taille de l'ensemble", 'default': 4, 'min': 1, 'max': 8, 'type': 'int'},
        }
    },
    'VGGT': {
        'class': 'VGGTEngine',
        'module': 'src.engines.vggt_engine',
        'model_name': "facebook/VGGT-1B",
        'options': {}
    }
}

# Configuration partagée
RMBG_CONFIG = {'model_name': "briaai/RMBG-1.4"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ENGINE = 'MoGe'
INPUT_FOLDER = "images"

# Paramètres de reconstruction pour MoGe
POISSON_DEPTH = 9
ENABLE_NORMAL_ESTIMATION = True
ENABLE_DENSITY_FILTER = True
DENSITY_FILTER_QUANTILE = 0.01
ENABLE_SMOOTHING = True
SMOOTHING_ITERATIONS = 15
ENABLE_DECIMATION = True
DECIMATION_REDUCTION_FACTOR = 3