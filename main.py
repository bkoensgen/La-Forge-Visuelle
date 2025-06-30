import sys
import os

def add_vendor_to_path():
    """Ajoute les dépôts clonés au chemin d'importation de Python."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Chemin vers le repo DepthFM
    depthfm_repo_path = os.path.join(project_root, 'vendor', 'depthfm_repo')
    if os.path.isdir(depthfm_repo_path):
        print(f"INFO: Ajout de '{depthfm_repo_path}' au chemin d'importation de Python.")
        sys.path.insert(0, depthfm_repo_path)
    else:
        print(f"AVERTISSEMENT: Dossier DepthFM '{depthfm_repo_path}' non trouvé.")
        print("INFO: Le moteur DepthFM ne sera pas disponible.")
        print("Veuillez exécuter 'python install_helper.py' pour le cloner.")

    da_repo_path = os.path.join(project_root, 'vendor', 'depth_anything_v2_repo')
    if os.path.isdir(da_repo_path):
        print(f"INFO: Ajout de '{da_repo_path}' au chemin d'importation de Python.")
        sys.path.insert(0, da_repo_path)
    else:
        print(f"AVERTISSEMENT: Dossier Depth Anything V2 '{da_repo_path}' non trouvé.")
        print("INFO: Le moteur Depth Anything V2 ne sera pas disponible.")

add_vendor_to_path()

from PyQt6.QtWidgets import QApplication
from src.main_window import MainWindow

def main():
    """Point d'entrée de l'application PyQt."""
    print("--- Lancement de l'application ---")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()