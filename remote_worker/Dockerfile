# Étape 1: Image "Builder" pour pré-compiler les dépendances
# On utilise une image "devel" qui contient les outils de compilation nécessaires.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

# Installer Python et pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*
# Créer un lien symbolique pour que la commande 'python' fonctionne
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Préparer le répertoire de travail
WORKDIR /app

# Copier les requirements et les pré-compiler en "roues" (wheels) pour une installation rapide plus tard
COPY remote_worker/requirements-remote.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements-remote.txt


# Étape 2: Image Finale de Production
# On part d'une image "runtime", beaucoup plus légère car elle ne contient pas les outils de compilation.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Variables d'environnement pour une exécution non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC

# Installation des dépendances système MINIMALES requises pour l'exécution
# On ajoute git, wget, et curl, qui sont maintenant nécessaires pour télécharger nos modèles et dépôts.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python pré-compilées depuis l'étape "builder"
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# --- CORRECTION MAJEURE : TÉLÉCHARGEMENT DES CHECKPOINTS ---
# Au lieu de COPIER les gros fichiers, on les télécharge directement dans l'image.
# Cette couche sera mise en cache, donc le téléchargement ne se fera qu'une seule fois.
RUN mkdir -p /app/checkpoints
RUN echo "Téléchargement des checkpoints..."
RUN curl -L -o /app/checkpoints/depthfm-v1.ckpt "https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt"
RUN curl -L -o /app/checkpoints/depth_anything_v2_vits.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
RUN curl -L -o /app/checkpoints/depth_anything_v2_vitb.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
RUN curl -L -o /app/checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"

# --- CORRECTION MAJEURE : TÉLÉCHARGEMENT DES DÉPÔTS VENDOR ---
# On clone les dépôts au lieu de les copier. C'est plus propre et évite d'inclure l'historique .git lourd.
# L'option --depth 1 est une optimisation qui ne télécharge que la dernière version, rendant le clone plus rapide.
RUN mkdir -p /app/vendor
RUN echo "Clonage des dépôts vendor..."
RUN git clone --depth 1 https://github.com/CompVis/depth-fm.git /app/vendor/depthfm_repo
RUN git clone --depth 1 https://huggingface.co/spaces/depth-anything/Depth-Anything-V2 /app/vendor/depth_anything_v2_repo

# --- Copie du code source de votre application ---
# On ne copie que les fichiers strictement nécessaires au fonctionnement du worker.
COPY src/config.py ./src/
COPY src/engines/ ./src/engines/
COPY src/geometry_builder.py ./src/
COPY src/__init__.py ./src/
COPY src/engines/__init__.py ./src/engines/

# Copier le script handler qui est le point d'entrée pour RunPod
COPY remote_worker/handler.py .

# Commande par défaut pour exécuter le worker RunPod.
CMD ["python", "-u", "handler.py"]