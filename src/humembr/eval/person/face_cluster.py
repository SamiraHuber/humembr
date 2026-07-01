import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from humembr.eval.person.clustering_method import ClusteringMethod
from humembr.eval.person.face_utils import Face, extract_faces, save_cropped_face
from humembr.util.config import load_config

cfg = load_config()


class FaceClusteringMethod(ClusteringMethod):
    """Clusters based on detected face embeddings."""

    def __init__(
        self,
        name: str,
        person_data: list[Any],
        output_folder: str,
        dbscan_eps: float = 0.579,
        dbscan_min_samples: int = 4,
        save_outputs: bool = True,
    ) -> None:
        super().__init__(name, person_data, output_folder)
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.save_outputs = save_outputs

    def run(
        self,
        face_app: Any = None,
        kpr_extractor: Any = None,
        osnet_extractor: Any = None,
    ) -> None:
        if not face_app:
            print("Face app not provided. Skipping face-only clustering.")
            return

        faces, images_without_faces = extract_faces(self.image_paths, face_app)

        if not faces:
            print("No faces detected. Skipping face-only clustering.")
            return

        face_embeddings = np.array([face.embedding for face in faces])
        pred_labels = self._cluster_embeddings(face_embeddings)

        # --- Evaluation ---
        true_labels = [
            self.image_name_to_label.get(Path(face.image_path).name) for face in faces
        ]
        valid_indices = [i for i, label in enumerate(true_labels) if label is not None]

        if not valid_indices:
            print(
                "Could not find ground truth for any detected faces. Skipping evaluation."
            )
        else:
            true_labels_eval = [str(true_labels[i]) for i in valid_indices]
            pred_labels_eval = [pred_labels[i] for i in valid_indices]
            self._evaluate_and_print(
                true_labels_eval,
                pred_labels_eval,
                f"Evaluation on {len(valid_indices)} of {len(faces)} detected faces",
            )

        # --- Save Results ---
        if self.save_outputs:
            self._save_face_clusters(faces, pred_labels)
            self._save_images_without_faces(images_without_faces)

    def _cluster_embeddings(self, embeddings: npt.NDArray[Any]) -> npt.NDArray[Any]:
        print("Clustering face embeddings with DBSCAN...")
        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
            n_jobs=-1,
        )
        dbscan.fit(embeddings)
        return dbscan.labels_

    def _save_face_clusters(self, faces: list[Face], labels: npt.NDArray[Any]) -> None:
        print("Saving clustered faces...")
        clusters: dict[int, list[Face]] = defaultdict(list)
        for i, face in enumerate(faces):
            clusters[labels[i]].append(face)

        for cluster_id, faces_in_cluster in tqdm(
            clusters.items(), desc="Saving face clusters"
        ):
            cluster_name = "noise" if cluster_id == -1 else f"person_{cluster_id + 1}"
            person_dir = os.path.join(self.output_folder, cluster_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            for i, face in enumerate(faces_in_cluster):
                img_name = os.path.splitext(os.path.basename(face.image_path))[0]
                output_path = os.path.join(person_dir, f"{img_name}_face_{i}.jpg")
                save_cropped_face(face.image_path, face.bounding_box, output_path)

    def _save_images_without_faces(self, image_paths: list[str]) -> None:
        no_faces_dir = os.path.join(self.output_folder, "no_faces")
        if not os.path.exists(no_faces_dir):
            os.makedirs(no_faces_dir)
        print(f"Saving {len(image_paths)} images with no faces...")
        for image_path in image_paths:
            shutil.copy(str(image_path), no_faces_dir)
