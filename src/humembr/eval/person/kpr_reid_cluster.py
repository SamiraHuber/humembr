from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN

from humembr.eval.person.clustering_method import ClusteringMethod, save_images
from humembr.eval.person.reid_utils import get_kpr_reid_embeddings


class KprReidClusteringMethod(ClusteringMethod):
    """Clusters based on ReID embeddings."""

    def __init__(
        self,
        name: str,
        person_data: list[Any],
        output_folder: str,
        dbscan_eps: float = 0.005,
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
        if not kpr_extractor:
            print("KPR extractor not provided. Skipping ReID-only clustering.")
            return

        reid_embeddings_map = get_kpr_reid_embeddings(
            self.image_paths, kpr_extractor, self.person_data
        )

        if not reid_embeddings_map:
            print("No ReID embeddings extracted. Skipping ReID-only clustering.")
            return

        image_order = list(reid_embeddings_map.keys())
        embeddings = np.array([reid_embeddings_map[path] for path in image_order])

        pred_labels = self._cluster_embeddings(embeddings)

        # --- Evaluation ---
        true_labels = [
            str(self.image_name_to_label.get(Path(p).name)) for p in image_order
        ]
        self._evaluate_and_print(true_labels, pred_labels)

        # --- Save Results ---
        if self.save_outputs:
            print("Saving ReID clusters...")
            cluster_map = {
                image_order[i]: pred_labels[i] for i in range(len(image_order))
            }
            save_images(self.image_paths, self.output_folder, cluster_map)

    def _cluster_embeddings(self, embeddings: npt.NDArray[Any]) -> npt.NDArray[Any]:
        print("Clustering ReID embeddings with DBSCAN...")
        print(embeddings.shape)
        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
            n_jobs=-1,
        )  # Different eps may be needed
        dbscan.fit(embeddings)
        return dbscan.labels_
