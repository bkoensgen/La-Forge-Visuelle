from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from .runpod_client import RunPodClient
import trimesh

class RemoteProcessor(QObject):
    """
    Gère le pipeline de traitement en déléguant le calcul à un worker RunPod distant.
    Il a la même interface (signaux) que LocalProcessor pour être interchangeable.
    """
    finished = pyqtSignal(object)  # Émet un objet trimesh
    error = pyqtSignal(str)

    def __init__(self, api_key: str, endpoint_id: str):
        super().__init__()
        self.client = RunPodClient(api_key, endpoint_id)

    @pyqtSlot(str, str, dict)
    def process(self, path: str, engine_name: str, options: dict):
        """
        Slot qui envoie une tâche de reconstruction au worker distant et attend le résultat.
        """
        try:
            print(f"\n--- Démarrage du pipeline de traitement REMOTE pour {engine_name} ---")
            
            # La méthode process_remote est bloquante, c'est pourquoi ce worker
            # doit s'exécuter dans un QThread pour ne pas geler la GUI.
            mesh = self.client.process_remote(
                image_path=path,
                engine_name=engine_name,
                options=options
            )

            if not isinstance(mesh, trimesh.Trimesh):
                 raise TypeError(f"Le client distant a retourné un objet de type inattendu: {type(mesh)}")

            print("--- Tâche distante terminée et résultat récupéré. ---")
            self.finished.emit(mesh)

        except Exception as e:
            import traceback
            error_message = f"Erreur dans le processeur distant: {traceback.format_exc()}"
            print(error_message)
            self.error.emit(error_message)