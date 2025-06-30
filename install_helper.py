import sys
import subprocess
import os
import urllib.request
import shutil

def run_command(command):
    """Exécute une commande shell et gère les erreurs."""
    try:
        print(f"--- Exécution: {' '.join(command)} ---")
        subprocess.check_call(command)
        print("--- Succès ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERREUR: La commande a échoué avec le code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERREUR: Commande non trouvée. Assurez-vous que le programme est dans votre PATH.")
        return False

def download_file(url, dest_path):
    """Télécharge un fichier avec une barre de progression."""
    if os.path.exists(dest_path):
        print(f"Le fichier '{os.path.basename(dest_path)}' existe déjà. Téléchargement ignoré.")
        return True
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        print(f"Téléchargement de {os.path.basename(dest_path)} depuis {url}...")
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            bytes_so_far, chunk_size = 0, 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk: break
                out_file.write(chunk)
                bytes_so_far += len(chunk)
                if total_size > 0:
                    percent = bytes_so_far * 100 / total_size
                    sys.stdout.write(f"\r -> {percent:.1f}% [{bytes_so_far/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB]")
                    sys.stdout.flush()
        print("\n--- Téléchargement terminé ---")
        return True
    except Exception as e:
        print(f"\nERREUR lors du téléchargement de {url}: {e}")
        return False

def main():
    """
    Script d'installation complet pour l'ENVIRONNEMENT DE DÉVELOPPEMENT LOCAL.
    Installe les dépendances pour la GUI, le traitement local et le client distant.
    """
    pip_executable = [sys.executable, '-m', 'pip']
    vendor_dir = 'vendor'
    checkpoints_dir = 'checkpoints'
    os.makedirs(vendor_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Installation des dépendances Python de base pour le développement local
    print("\n--- Installation des dépendances Python depuis requirements.txt ---")
    # Ce fichier contient maintenant tout le nécessaire pour le mode local ET distant (client)
    if not run_command(pip_executable + ['install', '-r', 'requirements.txt']): return

    print("\n--- Installation de PyTorch (avec support CUDA si possible) ---")
    # Laisser l'utilisateur choisir ou forcer une version spécifique est souvent une bonne pratique
    # Note : Si 'torch' est déjà dans requirements.txt, cette commande peut le mettre à jour.
    if not run_command(pip_executable + ['install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu121']): return

    print("\n--- Installation des bibliothèques depuis des dépôts Git ---")
    # Ces installations sont maintenant gérées directement par pip dans requirements.txt si possible,
    # mais les laisser ici peut être utile pour la clarté.
    # if not run_command(pip_executable + ['install', 'git+https://github.com/microsoft/MoGe.git']): return
    # if not run_command(pip_executable + ['install', 'git+https://github.com/facebookresearch/vggt.git']): return

    print("\n--- Clonage des dépôts externes dans le dossier 'vendor' ---")
    
    # Clonage de DepthFM
    depthfm_repo_dir = os.path.join(vendor_dir, 'depthfm_repo')
    if not os.path.exists(depthfm_repo_dir):
        if not run_command(['git', 'clone', 'https://github.com/CompVis/depth-fm.git', depthfm_repo_dir]):
            print("Échec du clonage de DepthFM. Assurez-vous que 'git' est installé. Arrêt.")
            return
    else:
        print(f"Le dossier '{depthfm_repo_dir}' existe déjà. Clonage ignoré.")
        
    # Clonage de Depth Anything V2
    da2_repo_dir = os.path.join(vendor_dir, 'depth_anything_v2_repo')
    if not os.path.exists(da2_repo_dir):
        if not run_command(['git', 'clone', 'https://huggingface.co/spaces/depth-anything/Depth-Anything-V2', da2_repo_dir]):
            print("Échec du clonage de Depth Anything V2. Arrêt.")
            return
    else:
        print(f"Le dossier '{da2_repo_dir}' existe déjà. Clonage ignoré.")

    print("\n--- Téléchargement des checkpoints des modèles ---")

    checkpoints_to_download = {
        "depthfm-v1.ckpt": "https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt",
        "depth_anything_v2_vits.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "depth_anything_v2_vitb.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "depth_anything_v2_vitl.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    }

    for filename, url in checkpoints_to_download.items():
        dest_path = os.path.join(checkpoints_dir, filename)
        if not download_file(url, dest_path):
            print(f"AVERTISSEMENT: Échec du téléchargement du checkpoint {filename}.")

    print("\n\n---------------------------------------------------------")
    print("--- Installation de l'environnement local terminée ! ---")
    print("---------------------------------------------------------")
    print("Vous pouvez maintenant lancer l'application avec 'python src/main.py'.")
    print("N'oubliez pas de configurer 'src/config.py' et vos variables d'environnement.")

if __name__ == "__main__":
    main()