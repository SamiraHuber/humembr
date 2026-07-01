import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from osnet.torchreid.utils import FeatureExtractor

from humembr.eval.person.clustering_method import KeypointSample, PersonItem
from humembr.eval.person.face_cluster import FaceClusteringMethod
from humembr.eval.person.kpr_reid_cluster import KprReidClusteringMethod
from humembr.eval.person.osnet_reid_cluster import OsnetReidClusteringMethod
from humembr.eval.person.reid_face_cluster import (
    OSNET_REID_DISTANCE_THRESHOLD,
    FacePlusReidCentroid,
    FacePlusReidKNN,
    FacePlusReidNearestNeighbor,
)
from humembr.processing.reid import HiddenPrints, get_reid_kpr_model
from humembr.util.config import load_config

cfg = load_config()


def initialize_face_app() -> Any:
    """Initializes the InsightFace face analysis model."""
    print("Initializing face analysis model...")

    with HiddenPrints():
        face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        face_app.prepare(
            ctx_id=0, det_size=(640, 640), det_thresh=cfg.perception.face_conf
        )
        return face_app


def load_person_data(args: argparse.Namespace) -> list[PersonItem]:
    data_path = Path(args.image_folder).parent
    meta_data_path = str(data_path / "person.json")
    with open(meta_data_path, "r") as f:
        persons = []
        for p in json.load(f):
            person = PersonItem(**p)
            person.kpr_keypoints_sample = [
                KeypointSample(**k) for k in p["kpr_keypoints_sample"]
            ]
            persons.append(person)

    for p in persons:
        p.crop_path = str(data_path / p.crop_path)
    return persons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster person images using different methods."
    )
    parser.add_argument(
        "image_folder", help="Path to the folder containing JPG images."
    )
    args = parser.parse_args()

    # Clear output folder to ensure no old results remain
    output_dir = cfg.eval_dir / "eval/clustered_results"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_app = initialize_face_app()
    kpr_extractor = get_reid_kpr_model()
    osnet_extractor = FeatureExtractor(
        model_name="osnet_x1_0", device="cpu", verbose=False
    )

    persons = load_person_data(args)

    methods = [
        FaceClusteringMethod("face_only", persons, str(output_dir)),
        OsnetReidClusteringMethod("osnet_reid_only", persons, str(output_dir)),
        KprReidClusteringMethod("kpr_reid_only", persons, str(output_dir)),
        FacePlusReidNearestNeighbor(
            "kpr_reid_nn", persons, str(output_dir), reid_model="kpr"
        ),
        FacePlusReidNearestNeighbor(
            "osnet_reid_nn",
            persons,
            str(output_dir),
            reid_model="osnet",
            threshold=OSNET_REID_DISTANCE_THRESHOLD,
        ),
        FacePlusReidCentroid(
            "kpr_reid_centroid",
            persons,
            str(output_dir),
            reid_model="kpr",
            threshold=0.095,
        ),
        FacePlusReidCentroid(
            "osnet_reid_centroid",
            persons,
            str(output_dir),
            reid_model="osnet",
            threshold=0.15,
        ),
        FacePlusReidKNN(
            "kpr_reid_3nn", persons, str(output_dir), k=3, reid_model="kpr"
        ),
        FacePlusReidKNN(
            "kpr_reid_5nn", persons, str(output_dir), k=5, reid_model="kpr"
        ),
        FacePlusReidKNN(
            "kpr_reid_10nn", persons, str(output_dir), k=10, reid_model="kpr"
        ),
        FacePlusReidKNN(
            "osnet_reid_3nn",
            persons,
            str(output_dir),
            k=3,
            reid_model="osnet",
            threshold=OSNET_REID_DISTANCE_THRESHOLD,
        ),
        FacePlusReidKNN(
            "osnet_reid_5nn",
            persons,
            str(output_dir),
            k=5,
            reid_model="osnet",
            threshold=OSNET_REID_DISTANCE_THRESHOLD,
        ),
        FacePlusReidKNN(
            "osnet_reid_10nn",
            persons,
            str(output_dir),
            k=10,
            reid_model="osnet",
            threshold=OSNET_REID_DISTANCE_THRESHOLD,
        ),
    ]

    for method in methods:
        print(f"\n{'=' * 20} Running Method: {method.name} {'=' * 20}")
        method.run(
            face_app=face_app,
            kpr_extractor=kpr_extractor,
            osnet_extractor=osnet_extractor,
        )

    # Collect results
    results = []
    for method in methods:
        if method.metrics:
            res: Dict[str, Any] = method.metrics.copy()
            res["Method"] = method.name
            results.append(res)

    if not results:
        print("No results to report.")
        return

    # Save to CSV
    csv_path = output_dir / "clustering_metrics.csv"
    fieldnames = ["Method", "Homogeneity", "Completeness", "V-measure", "Assign-Ratio"]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"\nMetrics saved to {csv_path}")

    # Plot results
    plot_path = output_dir / "clustering_comparison.png"
    metric_names = ["Homogeneity", "Completeness", "V-measure"]

    x = np.arange(len(metric_names))  # label locations
    width = 0.8 / len(results)  # width of bars

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, res in enumerate(results):
        values = [res[m] for m in metric_names]
        offset = width * i
        # Calculate offset so group is centered
        # Start at x - total_width/2
        # But simpler: just offset from x
        ax.bar(
            x + offset - (len(results) * width) / 2 + width / 2,
            values,
            width,
            label=res["Method"],
        )

    ax.set_ylabel("Score")
    ax.set_title("Clustering Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

    print("\nAll clustering methods complete.")
    print(f"You can find the results in the '{str(output_dir)}' directory.")


if __name__ == "__main__":
    main()
