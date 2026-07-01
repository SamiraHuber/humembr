import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)


def save_images(
    image_paths: list[str], output_dir: str, cluster_map: dict[str, int]
) -> None:
    """Saves images into cluster-named subdirectories."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path, cluster_id in cluster_map.items():
        if cluster_id == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"person_{cluster_id + 1}"

        person_dir = os.path.join(output_dir, cluster_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        try:
            shutil.copy(str(image_path), person_dir)
        except Exception as e:
            print(f"Error copying {image_path}: {e}")


@dataclass
class KeypointSample:
    keypoints: list[tuple[float, float, float]]
    is_target: bool


@dataclass
class PersonItem:
    id: int
    label: str
    crop_path: str
    kpr_keypoints_sample: list[KeypointSample]


class ClusteringMethod(ABC):
    """Abstract base class for a clustering strategy."""

    def __init__(
        self,
        name: str,
        person_data: list[PersonItem],
        output_folder: str,
    ):
        self.name = name
        self.person_data = person_data
        self.image_paths = [p.crop_path for p in person_data]
        self.base_output_folder = output_folder

        self.output_folder = os.path.join(self.base_output_folder, self.name)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.image_name_to_label = {
            Path(p.crop_path).name: p.label for p in self.person_data
        }
        self.metrics = {}

    @abstractmethod
    def run(
        self,
        face_app: Any = None,
        kpr_extractor: Any = None,
        osnet_extractor: Any = None,
    ) -> None:
        """Executes the clustering and evaluation for this method."""
        pass

    def _evaluate_and_print(
        self,
        true_labels: list[str],
        pred_labels: npt.NDArray[Any] | list[int],
        title: str = "Clustering Evaluation Metrics",
    ) -> None:
        # filter noise label out
        label_indices = np.array(pred_labels) != -1
        assigned_ratio = label_indices.sum() / len(self.image_paths)
        print(
            f"{assigned_ratio * 100:.2f}% persons were assigned to clusters (rest is noise)"
        )

        true_labels_no_noise = np.array(true_labels)[label_indices]
        pred_labels_no_noise = np.array(pred_labels)[label_indices]
        homogeneity = homogeneity_score(true_labels_no_noise, pred_labels_no_noise)
        completeness = completeness_score(true_labels_no_noise, pred_labels_no_noise)
        v_measure = v_measure_score(true_labels_no_noise, pred_labels_no_noise)

        self.metrics = {
            "Homogeneity": homogeneity,
            "Completeness": completeness,
            "V-measure": v_measure,
            "Assign-Ratio": assigned_ratio,
        }

        print(f"\n--- {title} ({self.name}) ---")
        print(f"Homogeneity: {homogeneity:.4f}")
        print(f"Completeness: {completeness:.4f}")
        print(f"V-measure: {v_measure:.4f}")
