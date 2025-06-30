import trimesh
import numpy as np
from PIL import Image
import open3d as o3d
from scipy.spatial import KDTree
from src import config

class GeometryBuilder:
    """
    Centralise la logique de construction de maillages 3D à partir
    de différentes formes de données brutes issues des moteurs IA.
    """
    def build(self, raw_data: dict, processed_image: np.ndarray, fg_mask: np.ndarray = None, options: dict = None):
        """
        Aiguille vers la bonne méthode de construction en fonction des données et des options.
        """
        print("--- Démarrage de la construction de la géométrie ---")
        if options is None:
            options = {}

        if options.get('render_mode') is True and 'points' in raw_data:
            print("Option 'Nuage de Points' sélectionnée. Construction simplifiée.")
            return self._build_point_cloud_from_moge_data(raw_data, processed_image, fg_mask)

        if 'depth_map' in raw_data:
            return self._build_from_depth_map(raw_data['depth_map'], processed_image, options)
        elif 'points' in raw_data and 'normal' in raw_data:
            return self._build_from_points_and_normals(raw_data, processed_image, fg_mask, options)
        elif 'points' in raw_data:
            return self._build_from_points_only(raw_data)
        else:
            print("ERREUR: Données brutes non reconnues pour la construction du maillage.")
            return None
    
    def _apply_fg_mask(self, model_mask, fg_mask):
        """Applique le masque de premier plan au masque du modèle."""
        if fg_mask is None:
            return model_mask
            
        if fg_mask.shape != model_mask.shape:
            h, w = model_mask.shape
            fg_mask_pil = Image.fromarray(fg_mask)
            fg_mask_resized = fg_mask_pil.resize((w, h), Image.Resampling.NEAREST)
            fg_mask = np.array(fg_mask_resized)
        
        return model_mask & (fg_mask > 128)

    def _build_point_cloud_from_moge_data(self, data, img_rgb, fg_mask):
        """Construit un simple nuage de points coloré pour MoGe."""
        final_mask = self._apply_fg_mask(data['mask'], fg_mask)
        pts = data['points'][final_mask]
        colors = img_rgb[final_mask]
        return trimesh.Trimesh(vertices=pts, vertex_colors=colors)

    def _build_from_points_only(self, data):
        """Pour les moteurs simples comme VGGT."""
        print("Construction à partir d'un nuage de points simple.")
        return trimesh.Trimesh(vertices=data['points'], vertex_colors=data.get('vertex_colors'))

    def _build_from_depth_map(self, depth_map, rgb_image, options: dict):
        """Pour les moteurs comme DepthFM et Depth Anything V2."""
        print("Construction à partir d'une carte de profondeur.")
        h, w = depth_map.shape
        fx = fy = w * 1.2
        cx, cy = w / 2, h / 2
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        
        # --- CORRECTION FINALE DE LA LOGIQUE GÉOMÉTRIQUE ---

        # ÉTAPE 1: On définit une "profondeur de base" CONSTANTE pour calculer la silhouette X/Y.
        # Cette valeur ne change pas avec le slider et fixe la largeur/hauteur de l'objet.
        # Une valeur de 1.0 est un bon point de départ neutre.
        base_z_for_xy = depth_map * 1.0 
        
        x = (jj - cx) * base_z_for_xy / fx
        y = (ii - cy) * base_z_for_xy / fy

        # ÉTAPE 2: On récupère l'échelle de l'UI pour modifier UNIQUEMENT la profondeur finale.
        depth_scale = options.get('depth_scale', 10.0)
        print(f"Utilisation de l'échelle de profondeur : {depth_scale}")
        
        # On applique l'échelle à la carte de profondeur pour la coordonnée Z.
        final_z = depth_map * depth_scale

        # ÉTAPE 3: On combine la silhouette X/Y fixe avec la profondeur Z variable.
        # Le signe négatif sur Z est pour que la profondeur s'éloigne de la caméra.
        points = np.stack([x, y, -final_z], axis=-1).reshape(-1, 3)
        
        colors = rgb_image.reshape(-1, 3)
        return trimesh.Trimesh(vertices=points, vertex_colors=colors)

    def _build_from_points_and_normals(self, data, img_rgb, fg_mask, options: dict):
        """Logique avancée pour MoGe, utilisant la reconstruction de surface."""
        final_mask = self._apply_fg_mask(data['mask'], fg_mask)
        pts, norms, colors = data['points'][final_mask], data['normal'][final_mask], img_rgb[final_mask]
        if len(pts) < 100:
            print("AVERTISSEMENT: Pas assez de points valides, retour à un simple nuage de points.")
            return trimesh.Trimesh(vertices=pts, vertex_colors=colors)
        print("Lancement de la reconstruction de surface Poisson...")
        try:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            pcd.normals = o3d.utility.Vector3dVector(norms)
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=config.POISSON_DEPTH)
            if not mesh_o3d: raise ValueError("Échec de la reconstruction Poisson.")
            if options.get('quality_filters', True):
                print("Application des filtres de qualité...")
                dens = np.asarray(densities)
                keep_mask = dens > np.quantile(dens, config.DENSITY_FILTER_QUANTILE)
                mesh_o3d.remove_vertices_by_mask(~keep_mask)
            mesh = trimesh.Trimesh(np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles), process=False)
            if comps := mesh.split(only_watertight=False): mesh = max(comps, key=lambda c: len(c.faces))
            _, idx = KDTree(pts).query(mesh.vertices, k=1)
            mesh.visual.vertex_colors = colors[idx]
            print("Maillage de surface de haute qualité construit.")
            return mesh
        except Exception as e:
            print(f"ERREUR durant la construction avancée: {e}. Retour à un simple nuage de points.")
            return trimesh.Trimesh(vertices=pts, vertex_colors=colors)