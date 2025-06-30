import base64
import requests
import trimesh
import os
import time

class RunPodClient:
    """
    Classe pour communiquer avec l'API REST de RunPod Serverless sans utiliser le SDK.
    Cela garantit la compatibilité avec Windows.
    """
    def __init__(self, api_key: str, endpoint_id: str):
        if not api_key:
            raise ValueError("La clé API RunPod ne peut pas être vide.")
        if not endpoint_id or endpoint_id == "VOTRE_ENDPOINT_ID_ICI":
            raise ValueError("L'ID de l'endpoint RunPod n'est pas configuré.")
        
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        # Les URL de l'API RunPod pour lancer une tâche et vérifier son statut. [1]
        self.run_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        self.status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status"

    def _get_headers(self):
        """Prépare les en-têtes d'authentification pour la requête API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def process_remote(self, image_path: str, engine_name: str, options: dict) -> trimesh.Trimesh:
        """
        Fonction bloquante qui gère le cycle de vie complet d'une tâche RunPod via l'API REST.
        """
        print(f"Préparation de la tâche pour l'endpoint '{self.endpoint_id}' via l'API REST...")

        # 1. Lire et encoder l'image en base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        # 2. Préparer le corps de la requête (payload)
        payload = {
            "input": {
                "image_b64": image_b64,
                "engine_name": engine_name,
                "options": options
            }
        }

        # 3. Soumettre la tâche à l'endpoint
        print("Envoi de la tâche à RunPod...")
        response = requests.post(self.run_url, headers=self._get_headers(), json=payload)
        
        if response.status_code != 200:
            raise ConnectionError(f"Échec de la soumission de la tâche. Statut: {response.status_code}, Réponse: {response.text}")

        job_data = response.json()
        job_id = job_data.get("id")
        print(f"Tâche soumise avec l'ID: {job_id}. En attente du résultat...")

        # 4. Attendre le résultat en interrogeant l'URL de statut (polling)
        status_url_with_id = f"{self.status_url}/{job_id}"
        timeout_seconds = 600  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            status_response = requests.get(status_url_with_id, headers=self._get_headers())
            status_data = status_response.json()

            status = status_data.get("status")
            if status == "COMPLETED":
                print("Tâche terminée avec succès.")
                output = status_data.get("output")
                if output is None:
                    raise ValueError("La tâche est terminée mais n'a retourné aucune sortie ('output').")
                
                # Le 'output' est l'URL du fichier GLB uploadé par le worker
                mesh_url = output
                break
            elif status in ["FAILED", "CANCELLED"]:
                error_detail = status_data.get("output", "Aucun détail d'erreur fourni.")
                raise RuntimeError(f"La tâche RunPod a échoué avec le statut '{status}'. Détails: {error_detail}")
            
            # Attendre avant de vérifier à nouveau pour ne pas surcharger l'API
            time.sleep(2)
        else:
            raise TimeoutError("Le délai d'attente pour la tâche RunPod a été dépassé.")

        # 5. Télécharger le fichier de maillage depuis l'URL retournée
        print(f"Téléchargement du maillage depuis : {mesh_url}")
        mesh_response = requests.get(mesh_url)
        if mesh_response.status_code != 200:
            raise IOError(f"Impossible de télécharger le fichier de maillage. Statut: {mesh_response.status_code}")

        # 6. Charger le maillage en objet Trimesh depuis les données binaires en mémoire
        try:
            # trimesh peut charger directement depuis un objet fichier en mémoire
            with requests.get(mesh_url, stream=True) as r:
                r.raise_for_status()
                # On utilise un "file-like object" en mémoire
                mesh = trimesh.load(r.raw, file_type='glb')
        except Exception as e:
            raise IOError(f"Impossible de charger le maillage depuis les données téléchargées. Erreur: {e}")

        print("Maillage chargé avec succès.")
        return mesh