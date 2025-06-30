# Étape 1: Image de base
# On utilise une image officielle NVIDIA avec CUDA 12.1 et Python 3.10 sur Ubuntu 22.04.
# C'est une base solide et compatible avec PyTorch et les GPU modernes.
FROM python:3.10-slim as builder

# Installer les dépendances système nécessaires pour PyTorch et OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python dans une roue
COPY requirements-remote.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements-remote.txt


# Étape 2: Image finale
# On part d'une image plus légère, mais qui contient les drivers CUDA.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Variables d'environnement pour éviter les questions interactives
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Créer un lien symbolique pour que 'python' pointe vers 'python3.10'
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances Python pré-compilées de l'étape de build
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copier les fichiers sources NÉCESSAIRES depuis le projet principal.
# C'est la clé de l'architecture unifiée : on ne duplique pas le code.
COPY ../src/config.py ./src/
COPY ../src/engines/ ./src/engines/
COPY ../src/geometry_builder.py ./src/
COPY ../src/__init__.py ./src/
COPY ../src/engines/__init__.py ./src/engines/

# Copier les fichiers spécifiques au worker distant
COPY ./handler.py .

# Copier les checkpoints des modèles
# Cette commande suppose que vous avez lancé le script d'installation localement
# et que le dossier 'checkpoints' existe à la racine.
COPY ../checkpoints/ ./checkpoints/

# Copier les dépôts clonés (vendor)
# Cela évite de devoir cloner les dépôts à chaque démarrage du conteneur.
COPY ../vendor/ ./vendor/

# Commande par défaut pour exécuter le worker RunPod.
# RunPod s'attend à ce que le handler soit défini dans la configuration de l'Endpoint.
# Le script rp_fastapi_async.py est fourni par l'environnement RunPod.
CMD ["python", "-u", "/runpod-python-utils/rp_fastapi_async.py"]