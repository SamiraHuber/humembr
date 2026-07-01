import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from humembr.eval.person.clustering_method import ClusteringMethod, save_images
from humembr.eval.person.face_utils import (
    Face,
    extract_faces,
    save_cropped_face,
)
from humembr.eval.person.reid_utils import get_kpr_reid_embeddings


class GEFFClustering(ClusteringMethod):
    """
    GEFF: Gallery Enrichment with Face Features.

    Implementation of the approach described in "GEFF: Improving Any Clothes-Changing
    Person ReID Model using Gallery Enrichment with Face Features".

    This class adapts the method for unsupervised clustering:
    1.  **Pose-Verified Face Extraction**: Ensures detected faces match the person body
        (using pose keypoints) as per Section 3.1.1.
    2.  **Gallery Enrichment**: Clusters faces to link different appearances of the same
        identity (Section 3.1).
    3.  **Score Vector Combination**: Combines ReID and Face scores/distances using alpha
        weighting (Equation 1).
    """

    def __init__(
        self,
        name: str,
        person_data: list[Any],
        output_folder: str,
        alpha: float = 0.2,  # Weight for ReID score (paper/repo default)
        face_eps: float = 0.61,  # EPS for face clustering (approx 0.8 cos similarity)
        final_eps: float = 0.62,  # EPS for final clustering
        min_samples: int = 2,
    ):
        super().__init__(name, person_data, output_folder)
        self.alpha = alpha
        self.face_eps = face_eps
        self.final_eps = final_eps
        self.min_samples = min_samples

    def run(
        self,
        face_app: Any = None,
        kpr_extractor: Any = None,
        osnet_extractor: Any = None,
    ) -> None:
        if not face_app or not kpr_extractor:
            print(f"Face app or KPR extractor not provided. Skipping {self.name}.")
            return

        # 1. Extract Face Embeddings with Pose Verification (Paper Sec 3.1.1)
        face_map = self._extract_verified_faces(face_app)

        # 2. Extract ReID Embeddings
        reid_map = get_kpr_reid_embeddings(
            self.image_paths, kpr_extractor, self.person_data
        )

        if not reid_map:
            print("No ReID embeddings extracted. Aborting.")
            return

        # Filter to valid images (must have ReID, face is optional)
        valid_paths = list(reid_map.keys())
        if not valid_paths:
            print("No valid images found.")
            return

        # 3. Face Clustering / Gallery Enrichment (Paper Sec 3.1)
        # We cluster faces to find "Enriched Gallery" connections.
        face_labels = self._cluster_faces(valid_paths, face_map)

        # 4. Compute Base Distance Matrices
        # Note: Paper combines Score Vectors (Similarities).
        # Here we work with Distances (1 - Similarity).
        # Linear combination holds: D_comb = alpha * D_reid + (1-alpha) * D_face

        reid_dist = self._compute_reid_distance(valid_paths, reid_map)
        face_dist = self._compute_face_distance(valid_paths, face_map)

        # 5. Enrich ReID Matrix based on Face Clusters
        # If faces match, we link the samples in ReID space (Enrichment).
        enriched_reid_dist = self._enrich_reid_matrix(reid_dist, face_labels)

        # 6. Combine Scores (Paper Eq 1)
        final_dist = self.alpha * enriched_reid_dist + (1.0 - self.alpha) * face_dist

        # 7. Final Clustering using DBSCAN
        print(f"Clustering with DBSCAN (eps={self.final_eps})...")
        dbscan = DBSCAN(
            eps=self.final_eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(final_dist)

        # 8. Evaluate and Save
        self._evaluate_and_save(valid_paths, labels, face_map)

    def _extract_verified_faces(self, face_app: Any) -> Dict[str, Face]:
        """
        Extracts faces and verifies them using pose keypoints (Paper Sec 3.1.1).
        "Matching Face to Pose Estimation ... verify that the detected face indeed
        belongs to the targeted individual."

        Simplified: Uses extract_faces utility and filters by keypoint overlap.
        """
        print("Extracting and verifying faces...")
        verified_faces: Dict[str, Face] = {}

        # Use utility to get all faces (cached)
        faces, _ = extract_faces(self.image_paths, face_app)

        # Map faces by image path for easier lookup
        # Since extract_faces might return multiple faces per image, we need to handle that.
        # However, following the instruction to assume one face per crop (or pick the best one),
        # we will filter them here.
        faces_by_path = defaultdict(list)
        for face in faces:
            faces_by_path[face.image_path].append(face)

        # Map crop_name to PersonItem for easy access to keypoints
        name_to_item = {Path(p.crop_path).name: p for p in self.person_data}
        face_counter = 0

        for image_path in tqdm(self.image_paths, desc="Verifying faces"):
            img_name = Path(image_path).name
            if img_name not in name_to_item:
                continue

            item = name_to_item[img_name]

            # Get target keypoints (from KeypointRCNN or similar)
            target_kpts_samples = [s for s in item.kpr_keypoints_sample if s.is_target]
            if not target_kpts_samples:
                continue

            # Use the first target sample's keypoints
            kpts = target_kpts_samples[0].keypoints

            # Get candidate faces for this image
            detected_faces = faces_by_path.get(image_path, [])
            if not detected_faces:
                continue

            best_face = None
            max_score = -1.0

            for face_obj in detected_faces:
                bbox = face_obj.bounding_box  # [x1, y1, x2, y2]

                # Check overlap with Nose (0) and Eyes (1, 2)
                points_in_box = 0
                indices_to_check = [0, 1, 2]
                for idx in indices_to_check:
                    if idx < len(kpts):
                        x, y, c = kpts[idx]
                        # Check if point is inside bbox
                        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                            points_in_box += 1

                if points_in_box > 0:
                    if points_in_box > max_score:
                        max_score = float(points_in_box)
                        best_face = face_obj
                    elif points_in_box == max_score:
                        pass  # Tie break

            if best_face is not None:
                # Re-assign ID to ensure continuity in this subset if needed,
                # though unique IDs from extract_faces are also fine.
                verified_faces[str(image_path)] = best_face
                face_counter += 1

        print(
            f"Found {len(verified_faces)} verified faces out of {len(self.image_paths)} images."
        )
        return verified_faces

    def _cluster_faces(
        self, paths: List[str], face_map: Dict[str, Face]
    ) -> npt.NDArray[Any]:
        """
        Clusters faces to identify groups for Gallery Enrichment.
        Returns labels array where -1 indicates noise or no face.
        """
        print("Clustering faces for Enrichment...")
        n = len(paths)
        face_feats = []
        indices_with_faces = []

        for i, p in enumerate(paths):
            if p in face_map:
                face_feats.append(face_map[p].embedding)
                indices_with_faces.append(i)

        if not face_feats:
            return np.full(n, -1)

        face_feats_np = np.array(face_feats)
        # DBSCAN on faces
        db = DBSCAN(eps=self.face_eps, min_samples=2, metric="cosine", n_jobs=-1)
        labels_sub = db.fit_predict(face_feats_np)

        full_labels = np.full(n, -1)
        full_labels[indices_with_faces] = labels_sub

        n_clusters = len(set(full_labels)) - (1 if -1 in full_labels else 0)
        print(f"Identified {n_clusters} face clusters for enrichment.")
        return full_labels

    def _compute_reid_distance(
        self, paths: List[str], reid_map: Dict[str, Any]
    ) -> npt.NDArray[Any]:
        feats = np.array([reid_map[p] for p in paths])
        return cosine_distances(feats)

    def _compute_face_distance(
        self, paths: List[str], face_map: Dict[str, Face]
    ) -> npt.NDArray[Any]:
        n = len(paths)
        # Default distance for missing faces: 2.0 (Max cosine distance)
        # This effectively removes the face component's positive influence for missing faces
        dist_mat = np.full((n, n), 2.0)

        indices_with_faces = [i for i, p in enumerate(paths) if p in face_map]

        if len(indices_with_faces) > 1:
            feats = np.array([face_map[paths[i]].embedding for i in indices_with_faces])
            sub_dist = cosine_distances(feats)

            ix_grid = np.ix_(indices_with_faces, indices_with_faces)
            dist_mat[ix_grid] = sub_dist

        return dist_mat

    def _enrich_reid_matrix(
        self, reid_dist: npt.NDArray[Any], face_labels: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """
        Updates ReID distance matrix based on Face Clusters.
        Implements the "Gallery Enrichment" concept:
        If samples share a Face Cluster, they are the "Same Identity".
        The distance from Sample X to Identity Y is min(Dist(X, member) for member in Y).
        """
        print("Enriching ReID matrix...")
        enriched_dist = reid_dist.copy()

        unique_labels = set(face_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            # Indices of images in this face cluster
            cluster_indices = np.where(face_labels == label)[0]

            if len(cluster_indices) < 2:
                continue

            # Min-pooling: For every sample (row), find the best match within the cluster (cols)
            # This "links" the cluster members together in ReID space relative to outside queries
            min_dists = np.min(reid_dist[:, cluster_indices], axis=1)

            # Broadcast this minimum distance to all members of the cluster
            for idx in cluster_indices:
                enriched_dist[:, idx] = np.minimum(enriched_dist[:, idx], min_dists)

        # Symmetrize to ensure valid distance matrix for clustering
        enriched_dist = np.minimum(enriched_dist, enriched_dist.T)
        return enriched_dist

    def _evaluate_and_save(
        self, paths: List[str], labels: npt.NDArray[Any], face_map: Dict[str, Face]
    ) -> None:
        # Prepare labels for evaluation
        true_labels = []
        pred_labels = []

        for i, path_str in enumerate(paths):
            path_obj = Path(path_str)
            true_l = self.image_name_to_label.get(path_obj.name)
            if true_l is not None:
                true_labels.append(true_l)
                pred_labels.append(labels[i])

        self._evaluate_and_print(
            true_labels, pred_labels, f"GEFF Evaluation (alpha={self.alpha})"
        )

        # Save images
        print("Saving clusters...")
        cluster_map = {paths[i]: labels[i] for i in range(len(paths))}
        save_images(paths, self.output_folder, cluster_map)

        # Save verified face crops for inspection
        faces_dir = os.path.join(self.output_folder, "face_crops")
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)

        for path_str, face_obj in face_map.items():
            if path_str in cluster_map:
                cid = cluster_map[path_str]
                c_name = "noise" if cid == -1 else f"person_{cid + 1}"
                c_dir = os.path.join(faces_dir, c_name)
                if not os.path.exists(c_dir):
                    os.makedirs(c_dir)

                img_name = Path(path_str).stem
                out_path = os.path.join(c_dir, f"{img_name}_face.jpg")
                save_cropped_face(path_str, face_obj.bounding_box, out_path)
