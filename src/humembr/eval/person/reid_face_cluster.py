from abc import abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cosine

from humembr.eval.person.clustering_method import (
    ClusteringMethod,
    PersonItem,
    save_images,
)
from humembr.eval.person.face_cluster import FaceClusteringMethod
from humembr.eval.person.face_utils import Face, extract_faces
from humembr.eval.person.reid_utils import (
    get_kpr_reid_embeddings,
    get_osnet_reid_embeddings,
)

KPR_REID_DISTANCE_THRESHOLD = 0.079
OSNET_REID_DISTANCE_THRESHOLD = 0.167


class FacePlusReidBase(ClusteringMethod):
    """Base class for clustering faces and using ReID to match no-face images."""

    def __init__(
        self,
        name: str,
        person_data: list[PersonItem],
        output_folder: str,
        reid_model: Literal["kpr", "osnet"],
        save_outputs: bool = True,
    ):
        super().__init__(name, person_data, output_folder)
        self.reid_model = reid_model
        self.save_outputs = save_outputs

    def run(
        self,
        face_app: Any = None,
        kpr_extractor: Any = None,
        osnet_extractor: Any = None,
    ) -> None:
        if not face_app or not kpr_extractor:
            print(f"Face app or KPR extractor not provided. Skipping {self.name}.")
            return

        # 1. Face clustering part
        # We still need the instance for clustering and saving utilities
        face_clusterer = FaceClusteringMethod(
            self.name,
            self.person_data,
            self.base_output_folder,
        )

        # Use utility function to extract faces
        faces, images_without_faces = extract_faces(self.image_paths, face_app)

        if not faces:
            print(f"No faces detected. Skipping {self.name}.")
            if images_without_faces:
                face_clusterer._save_images_without_faces(images_without_faces)
            return

        face_embeddings = np.array([f.embedding for f in faces])
        face_pred_labels = face_clusterer._cluster_embeddings(face_embeddings)

        # --- Create initial clusters from faces ---
        face_clusters: dict[int, list[Face]] = defaultdict(list)
        for i, face in enumerate(faces):
            face_clusters[face_pred_labels[i]].append(face)

        if self.save_outputs:
            face_clusterer._save_face_clusters(faces, face_pred_labels)

        # 2. ReID matching part
        images_with_faces_paths = {Path(f.image_path) for f in faces}
        all_reid_paths = list(
            {str(p) for p in images_with_faces_paths} | set(images_without_faces)
        )

        if self.reid_model == "kpr":
            reid_embeddings = get_kpr_reid_embeddings(
                all_reid_paths, kpr_extractor, self.person_data
            )
        else:
            reid_embeddings = get_osnet_reid_embeddings(all_reid_paths, osnet_extractor)

        unmatched_images, matched_map = self._assign_missing_images(
            images_without_faces, reid_embeddings, face_clusters
        )

        # --- Final Evaluation ---
        self._evaluate_combined_results(faces, face_pred_labels, matched_map)

        # --- Save Matched and Unmatched ---
        if self.save_outputs:
            print("Saving matched ReID images...")
            save_images(list(matched_map.keys()), self.output_folder, matched_map)
            face_clusterer._save_images_without_faces(unmatched_images)

    @abstractmethod
    def _assign_missing_images(
        self,
        images_without_faces: list[str],
        reid_embeddings: dict[str, npt.NDArray[Any]],
        face_clusters: dict[int, list[Face]],
    ) -> tuple[list[str], dict[str, int]]:
        """Assigns missing-face images to face clusters.

        Returns:
            unmatched: list of image paths that couldn't be assigned
            matched_map: dict mapping image path -> cluster_id
        """
        pass

    def _evaluate_combined_results(
        self,
        faces: list[Face],
        face_pred_labels: npt.NDArray[Any],
        matched_map: dict[str, int],
    ) -> None:
        # Construct true and pred labels for all images that were clustered
        true_labels: list[str] = []
        pred_labels: list[int] = []

        # Labels for images with faces
        face_map = {
            Path(f.image_path).name: face_pred_labels[i] for i, f in enumerate(faces)
        }

        processed_images = set()

        for img_path in self.image_paths:
            img_name = Path(img_path).name
            if img_name in processed_images:
                continue

            true_label = self.image_name_to_label.get(img_name)
            if true_label is None:
                continue

            pred_label = -1
            if img_path in matched_map:
                pred_label = matched_map[img_path]
            elif img_name in face_map:
                pred_label = face_map[img_name]

            true_labels.append(true_label)
            pred_labels.append(pred_label)
            processed_images.add(img_name)

        self._evaluate_and_print(
            true_labels,
            pred_labels,
            title=f"Combined Face + ReID Evaluation ({self.name})",
        )


class FacePlusReidNearestNeighbor(FacePlusReidBase):
    """Assigns using 1-NN (Single Nearest Neighbor)."""

    def __init__(
        self,
        name: str,
        person_data: list[PersonItem],
        output_folder: str,
        reid_model: Literal["kpr", "osnet"],
        threshold: float = KPR_REID_DISTANCE_THRESHOLD,
        save_outputs: bool = True,
    ) -> None:
        super().__init__(
            name,
            person_data,
            output_folder,
            reid_model,
            save_outputs=save_outputs,
        )
        self.threshold = threshold

    def _assign_missing_images(
        self,
        images_without_faces: list[str],
        reid_embeddings: dict[str, npt.NDArray[Any]],
        face_clusters: dict[int, list[Face]],
    ) -> tuple[list[str], dict[str, int]]:
        unmatched = list(images_without_faces)
        matched_map: dict[str, int] = {}

        gallery_embeddings: list[tuple[npt.NDArray, int, str]] = []
        for cid, faces_in_cluster in face_clusters.items():
            if cid == -1:  # Ignore noise cluster
                continue
            cluster_img_paths = {Path(f.image_path) for f in faces_in_cluster}
            for img_path in cluster_img_paths:
                img_path_str = str(img_path)
                if img_path_str in reid_embeddings:
                    gallery_embeddings.append(
                        (reid_embeddings[img_path_str], cid, img_path_str)
                    )

        if not gallery_embeddings:
            return unmatched, matched_map

        for img_path in images_without_faces:
            if img_path not in reid_embeddings:
                continue

            no_face_emb = reid_embeddings[img_path]
            distances = [cosine(no_face_emb, emb) for emb, _, _ in gallery_embeddings]

            if not distances:
                continue

            best_match_idx = np.argmin(distances)
            min_dist = distances[best_match_idx]

            if min_dist < self.threshold:
                best_cid, match_path = gallery_embeddings[best_match_idx][1:]
                # print(
                #     f"{img_path} matched with {match_path} with distance of {min_dist:.2f}"
                # )
                matched_map[img_path] = best_cid
                if img_path in unmatched:
                    unmatched.remove(img_path)

        return unmatched, matched_map


class FacePlusReidCentroid(FacePlusReidBase):
    """Assigns using Centroid matching (closest Cluster Mean)."""

    def __init__(
        self,
        name: str,
        person_data: list[PersonItem],
        output_folder: str,
        reid_model: Literal["kpr", "osnet"],
        threshold: float = KPR_REID_DISTANCE_THRESHOLD,
        save_outputs: bool = True,
    ) -> None:
        super().__init__(
            name,
            person_data,
            output_folder,
            reid_model,
            save_outputs=save_outputs,
        )
        self.threshold = threshold

    def _assign_missing_images(
        self,
        images_without_faces: list[str],
        reid_embeddings: dict[str, npt.NDArray[Any]],
        face_clusters: dict[int, list[Face]],
    ) -> tuple[list[str], dict[str, int]]:
        unmatched = list(images_without_faces)
        matched_map: dict[str, int] = {}

        # 1. Compute Centroids for each cluster
        cluster_centroids = {}  # cid -> mean_embedding
        for cid, faces_in_cluster in face_clusters.items():
            if cid == -1:
                continue

            embeddings = []
            cluster_img_paths = {Path(f.image_path) for f in faces_in_cluster}
            for img_path in cluster_img_paths:
                img_path_str = str(img_path)
                if img_path_str in reid_embeddings:
                    embeddings.append(reid_embeddings[img_path_str])

            if embeddings:
                # Compute mean
                mean_emb = np.mean(embeddings, axis=0)
                # Normalize (optional but recommended for cosine distance)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                cluster_centroids[cid] = mean_emb

        if not cluster_centroids:
            return unmatched, matched_map

        # 2. Assign images
        for img_path in images_without_faces:
            if img_path not in reid_embeddings:
                continue

            no_face_emb = reid_embeddings[img_path]

            # Find closest centroid
            best_cid = None
            min_dist = float("inf")

            for cid, centroid in cluster_centroids.items():
                dist = cosine(no_face_emb, centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cid = cid

            if min_dist < self.threshold and best_cid is not None:
                matched_map[img_path] = best_cid
                if img_path in unmatched:
                    unmatched.remove(img_path)

        return unmatched, matched_map


class FacePlusReidKNN(FacePlusReidBase):
    """Assigns using k-Nearest Neighbors (k-NN) with majority voting."""

    def __init__(
        self,
        name: str,
        person_data: list[PersonItem],
        output_folder: str,
        reid_model: Literal["kpr", "osnet"],
        k: int = 5,
        threshold: float = KPR_REID_DISTANCE_THRESHOLD,
        save_outputs: bool = True,
    ) -> None:
        super().__init__(
            name,
            person_data,
            output_folder,
            reid_model,
            save_outputs=save_outputs,
        )
        self.k = k
        self.threshold = threshold

    def _assign_missing_images(
        self,
        images_without_faces: list[str],
        reid_embeddings: dict[str, npt.NDArray[Any]],
        face_clusters: dict[int, list[Face]],
    ) -> tuple[list[str], dict[str, int]]:
        unmatched = list(images_without_faces)
        matched_map: dict[str, int] = {}

        # Build Gallery
        gallery_embeddings = []
        for cid, faces_in_cluster in face_clusters.items():
            if cid == -1:
                continue
            cluster_img_paths = {Path(f.image_path) for f in faces_in_cluster}
            for img_path in cluster_img_paths:
                img_path_str = str(img_path)
                if img_path_str in reid_embeddings:
                    gallery_embeddings.append((reid_embeddings[img_path_str], cid))

        if not gallery_embeddings:
            return unmatched, matched_map

        for img_path in images_without_faces:
            if img_path not in reid_embeddings:
                continue

            no_face_emb = reid_embeddings[img_path]

            # Calculate all distances
            distances = []  # (distance, cid)
            for emb, cid in gallery_embeddings:
                dist = cosine(no_face_emb, emb)
                distances.append((dist, cid))

            # Sort by distance
            distances.sort(key=lambda x: x[0])

            # Get top k
            top_k = distances[: self.k]

            # Filter by threshold (optional: require at least one neighbor to be close)
            valid_neighbors = [d for d in top_k if d[0] < self.threshold]

            if not valid_neighbors:
                continue

            # Majority Vote
            # Count the cluster IDs in the valid neighbors
            cids = [d[1] for d in valid_neighbors]
            vote_counts = Counter(cids)

            # Get the most common cluster ID
            best_cid, count = vote_counts.most_common(1)[0]

            # (Optional) You could add a tie-breaking rule or minimum vote count here

            matched_map[img_path] = best_cid
            if img_path in unmatched:
                unmatched.remove(img_path)

        return unmatched, matched_map
