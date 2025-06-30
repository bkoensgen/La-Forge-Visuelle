# La-Forge-Visuelle

> Un visualiseur et outil de reconstruction 3D qui transforme n'importe quelle image 2D en un modèle 3D détaillé, en utilisant la puissance de multiples moteurs d'IA et une architecture flexible locale ou cloud.

---

## ✨ Fonctionnalités Principales

*   **Interface Intuitive** : Une application de bureau simple et claire construite avec PyQt6 pour visualiser et interagir avec les modèles 3D générés.
*   **Moteurs d'IA Multiples** : Intègre plusieurs modèles de reconstruction de pointe pour choisir le meilleur résultat selon vos besoins :
    *   Microsoft **MoGe**
    *   **Depth Anything V2**
    *   **DepthFM**
    *   **VGGT**
*   **Architecture de Traitement Hybride** :
    *   **Mode `local`** : Utilisez la puissance de votre propre GPU pour des tests et des reconstructions rapides sans coût externe.
    *   **Mode `remote`** : Déportez les calculs intensifs sur une infrastructure GPU **on-demand** via [RunPod](https://runpod.io), ne payant que pour les secondes de calcul utilisées. Idéal pour libérer vos ressources locales ou pour les machines moins puissantes.
*   **Options de Pré-traitement** : Améliorez la qualité de la reconstruction avec des outils intégrés comme la suppression automatique de l'arrière-plan.
*   **Installation Simplifiée** : Un script unique (`install_helper.py`) gère le téléchargement des dépendances, des modèles et des dépôts externes.

## 🏛️ Architecture

Le projet est conçu autour d'une architecture flexible qui permet de séparer l'interface utilisateur du moteur de calcul.

    +---------------------------+
    |      Application GUI      |
    |      (main_window.py)     |
    +-------------+-------------+
                  |
    +-------------v-------------+      +---------------------------+
    |    Choix du Processeur    |----->|      src/config.py        |
    |  (Basé sur le config)     |      |  (PROCESSING_MODE)        |
    +-------------+-------------+      +---------------------------+
                  |
      +---------------------------+    +---------------------------+
      | Si "local"                |    | Si "remote"               |
      v                           v    v                           v
    +---------------------------+    +---------------------------+
    |     LocalProcessor        |    |     RemoteProcessor       |
    | (utilise le GPU local)    |    | (appelle l'API RunPod)    |
    +---------------------------+    +---------------------------+

Cette approche permet de tester et développer rapidement en local, tout en ayant la capacité de passer à une puissance de calcul quasi illimitée dans le cloud en changeant une seule ligne de configuration.

## 🚀 Démarrage Rapide

Suivez ces étapes pour installer et lancer l'application sur votre machine.

### 1. Prérequis

*   [Python 3.8+](https://www.python.org/)
*   [Git](https://git-scm.com/)
*   (Optionnel mais recommandé) Un GPU NVIDIA avec les drivers CUDA installés pour le traitement en mode `local`.

### 2. Installation

1.  **Clonez le dépôt :**

        git clone https://github.com/bkoensgen/La-Forge-Visuelle.git
        cd La-Forge-Visuelle

2.  **Créez un environnement virtuel :**

        python -m venv venv
    
    *   Sur Windows :
    
            venv\Scripts\activate
    
    *   Sur macOS/Linux :
    
            source venv/bin/activate

3.  **Lancez le script d'installation :**
    Ce script va télécharger et installer toutes les dépendances Python, cloner les dépôts externes requis et télécharger les poids des modèles d'IA dans le dossier `checkpoints/`.

        python install_helper.py

### 3. Lancement de l'application

Une fois l'installation terminée, vous pouvez lancer l'application :

    python main.py

## ⚙️ Configuration

Toute la configuration se fait dans le fichier `src/config.py`.

### Basculer entre le mode Local et Distant

Ouvrez `src/config.py` et modifiez la variable `PROCESSING_MODE` :

    # Choisir 'local' pour utiliser votre machine, ou 'remote' pour utiliser RunPod.
    PROCESSING_MODE = "local"

### Configurer le Mode Distant

Pour utiliser le mode `remote`, vous devez :
1.  Avoir un compte [RunPod](https://runpod.io).
2.  Créer un "Endpoint Serverless" (voir la section ci-dessous).
3.  Mettre à jour `src/config.py` avec l'ID de votre endpoint :

        RUNPOD_ENDPOINT_ID = "VOTRE_ID_ENDPOINT_ICI"

4.  Définir votre clé API RunPod comme une **variable d'environnement** pour des raisons de sécurité.
    *   **Windows (PowerShell)** :

            $env:RUNPOD_API_KEY="VOTRE_CLE_API_RUNPOD"

    *   **macOS/Linux** :

            export RUNPOD_API_KEY="VOTRE_CLE_API_RUNPOD"

## 🐳 Utilisation Avancée : Créer le Worker Docker pour RunPod

Pour utiliser le mode `remote`, vous devez construire et pousser une image Docker contenant le code de reconstruction.

1.  **Installez Docker** sur votre machine.

2.  **Construisez l'image :**
    À la racine du projet, lancez :

        # Remplacez 'votre_nom_dockerhub' par votre identifiant
        docker build -t votre_nom_dockerhub/relief-worker:v1 -f remote_worker/Dockerfile .

3.  **Poussez l'image vers un registre (Docker Hub, GHCR...) :**

        docker login
        docker push votre_nom_dockerhub/relief-worker:v1

4.  **Créez l'Endpoint sur RunPod** en utilisant le nom de l'image que vous venez de pousser.

## 🙏 Remerciements

Ce projet n'existerait pas sans les travaux incroyables des équipes derrière les modèles de reconstruction. Merci à :
*   **Microsoft** pour [MoGe](https://github.com/microsoft/MoGe)
*   Les auteurs de [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
*   **CompVis Group** pour [DepthFM](https://github.com/CompVis/depth-fm)
*   **Facebook Research** pour [VGGT](https://github.com/facebookresearch/vggt)

## 📜 Licence

Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.