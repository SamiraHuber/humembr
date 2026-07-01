import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import matplotlib.pyplot as plt
from osnet.torchreid.utils import FeatureExtractor

from humembr.eval.person.clustering_method import KeypointSample, PersonItem
from humembr.eval.person.eval import initialize_face_app
from humembr.eval.person.face_cluster import FaceClusteringMethod
from humembr.eval.person.kpr_reid_cluster import KprReidClusteringMethod
from humembr.eval.person.osnet_reid_cluster import OsnetReidClusteringMethod
from humembr.eval.person.reid_face_cluster import FacePlusReidNearestNeighbor
from humembr.processing.reid import get_reid_kpr_model
from humembr.util.config import load_config

cfg = load_config()


# In-file configuration only. Edit these values directly.
IMAGE_FOLDER = str(cfg.eval_dir / "datasets/person-data/images")
OUTPUT_FOLDER = str(cfg.eval_dir / "eval/clustered_results/dbscan_eps_plot")

FACE_EPS_VALUES = [0.45, 0.5, 0.55, 0.58, 0.61, 0.64, 0.68]
OSNET_EPS_VALUES = [0.08, 0.1, 0.125, 0.15, 0.175, 0.2]
KPR_EPS_VALUES = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
KPR_FACE_NN_DISTANCE_VALUES = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15]

FACE_MIN_SAMPLES = 4
OSNET_MIN_SAMPLES = 4
KPR_MIN_SAMPLES = 4

# Plot styling for paper readability.
FIG_SIZE = (12, 7)
FIG_DPI = 300
TITLE_FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 18
LINE_WIDTH = 4
MARKER_SIZE = 8


@dataclass
class EpsResult:
    method: str
    eps: float
    min_samples: int
    homogeneity: float
    completeness: float
    v_measure: float
    assign_ratio: float


class SweepMethod(Protocol):
    metrics: dict[str, float]

    def run(
        self,
        face_app: object | None = None,
        kpr_extractor: object | None = None,
        osnet_extractor: object | None = None,
    ) -> None: ...


def _validate_config() -> None:
    for name, values in (
        ("FACE_EPS_VALUES", FACE_EPS_VALUES),
        ("OSNET_EPS_VALUES", OSNET_EPS_VALUES),
        ("KPR_EPS_VALUES", KPR_EPS_VALUES),
        ("KPR_FACE_NN_DISTANCE_VALUES", KPR_FACE_NN_DISTANCE_VALUES),
    ):
        if not values:
            raise ValueError(f"{name} must contain at least one epsilon value.")

    for name, value in (
        ("FACE_MIN_SAMPLES", FACE_MIN_SAMPLES),
        ("OSNET_MIN_SAMPLES", OSNET_MIN_SAMPLES),
        ("KPR_MIN_SAMPLES", KPR_MIN_SAMPLES),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1.")


def _load_person_data(image_folder: str) -> list[PersonItem]:
    data_path = Path(image_folder).parent
    meta_data_path = data_path / "person.json"

    with open(meta_data_path, "r") as f:
        raw_persons = json.load(f)

    persons: list[PersonItem] = []
    for p in raw_persons:
        person = PersonItem(**p)
        person.kpr_keypoints_sample = [
            KeypointSample(**k) for k in p["kpr_keypoints_sample"]
        ]
        person.crop_path = str(data_path / person.crop_path)
        persons.append(person)

    return persons


def _run_method_sweep(
    method_name: str,
    method_factory: Callable[[float], SweepMethod],
    eps_values: list[float],
    face_app: object | None = None,
    kpr_extractor: object | None = None,
    osnet_extractor: object | None = None,
) -> list[EpsResult]:
    results: list[EpsResult] = []

    for eps in eps_values:
        method = method_factory(eps)
        method.run(
            face_app=face_app,
            kpr_extractor=kpr_extractor,
            osnet_extractor=osnet_extractor,
        )

        metrics = getattr(method, "metrics", {})
        if not metrics:
            continue

        results.append(
            EpsResult(
                method=method_name,
                eps=eps,
                min_samples=int(getattr(method, "dbscan_min_samples", 0)),
                homogeneity=float(metrics.get("Homogeneity", 0.0)),
                completeness=float(metrics.get("Completeness", 0.0)),
                v_measure=float(metrics.get("V-measure", 0.0)),
                assign_ratio=float(metrics.get("Assign-Ratio", 0.0)),
            )
        )

    return results


def _write_results(path: Path, results: list[EpsResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Method",
                "eps",
                "min_samples",
                "Homogeneity",
                "Completeness",
                "V-measure",
                "Assign-Ratio",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Method": r.method,
                    "eps": r.eps,
                    "min_samples": r.min_samples,
                    "Homogeneity": r.homogeneity,
                    "Completeness": r.completeness,
                    "V-measure": r.v_measure,
                    "Assign-Ratio": r.assign_ratio,
                }
            )


def _plot_method_results(
    path: Path, method_name: str, results: list[EpsResult]
) -> None:
    method_results = sorted(
        [r for r in results if r.method == method_name], key=lambda r: r.eps
    )
    if not method_results:
        return

    x = [r.eps for r in method_results]
    y_v_measure = [r.v_measure for r in method_results]
    y_assign_ratio = [r.assign_ratio for r in method_results]

    plt.figure(figsize=FIG_SIZE)
    plt.plot(
        x,
        y_v_measure,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="V-measure",
        color="#4C78A8",
    )
    plt.plot(
        x,
        y_assign_ratio,
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Assign-Ratio",
        color="#E45756",
    )

    plt.xlabel("Distance (DBSCAN eps, cosine)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Score", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(
        f"{method_name}: V-measure and Assign-Ratio over DBSCAN Distance",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI)
    plt.close()


def main() -> None:
    _validate_config()

    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    persons = _load_person_data(IMAGE_FOLDER)
    all_results: list[EpsResult] = []

    print("Initializing extractors...")
    face_app = initialize_face_app()

    osnet_extractor = FeatureExtractor(
        model_name="osnet_x1_0", device="cpu", verbose=False
    )
    kpr_extractor = get_reid_kpr_model()

    print("Running FaceCluster sweep using FaceClusteringMethod...")
    all_results.extend(
        _run_method_sweep(
            method_name="FaceCluster",
            eps_values=FACE_EPS_VALUES,
            face_app=face_app,
            method_factory=lambda eps: FaceClusteringMethod(
                "face_only",
                persons,
                str(output_dir),
                dbscan_eps=eps,
                dbscan_min_samples=FACE_MIN_SAMPLES,
                save_outputs=False,
            ),
        )
    )

    print("Running OsNet sweep using OsnetReidClusteringMethod...")
    all_results.extend(
        _run_method_sweep(
            method_name="OsNet",
            eps_values=OSNET_EPS_VALUES,
            osnet_extractor=osnet_extractor,
            method_factory=lambda eps: OsnetReidClusteringMethod(
                "osnet_reid_only",
                persons,
                str(output_dir),
                dbscan_eps=eps,
                dbscan_min_samples=OSNET_MIN_SAMPLES,
                save_outputs=False,
            ),
        )
    )

    print("Running KPR sweep using KprReidClusteringMethod...")
    all_results.extend(
        _run_method_sweep(
            method_name="KPR",
            eps_values=KPR_EPS_VALUES,
            kpr_extractor=kpr_extractor,
            method_factory=lambda eps: KprReidClusteringMethod(
                "kpr_reid_only",
                persons,
                str(output_dir),
                dbscan_eps=eps,
                dbscan_min_samples=KPR_MIN_SAMPLES,
                save_outputs=False,
            ),
        )
    )

    print("Running KPR Face+ReID NN sweep using FacePlusReidNearestNeighbor...")
    all_results.extend(
        _run_method_sweep(
            method_name="KPR+Face NN",
            eps_values=KPR_FACE_NN_DISTANCE_VALUES,
            face_app=face_app,
            kpr_extractor=kpr_extractor,
            method_factory=lambda distance: FacePlusReidNearestNeighbor(
                "kpr_reid_nn",
                persons,
                str(output_dir),
                reid_model="kpr",
                threshold=distance,
                save_outputs=False,
            ),
        )
    )

    if not all_results:
        print("No results generated. Nothing to write or plot.")
        return

    csv_path = output_dir / "dbscan_eps_comparison.csv"
    face_plot_path = output_dir / "dbscan_eps_facecluster.png"
    osnet_plot_path = output_dir / "dbscan_eps_osnet.png"
    kpr_plot_path = output_dir / "dbscan_eps_kpr.png"
    kpr_face_nn_plot_path = output_dir / "dbscan_eps_kpr_face_nn.png"

    _write_results(csv_path, all_results)
    _plot_method_results(face_plot_path, "FaceCluster", all_results)
    _plot_method_results(osnet_plot_path, "OsNet", all_results)
    _plot_method_results(kpr_plot_path, "KPR", all_results)
    _plot_method_results(kpr_face_nn_plot_path, "KPR+Face NN", all_results)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {face_plot_path}")
    print(f"Saved plot: {osnet_plot_path}")
    print(f"Saved plot: {kpr_plot_path}")
    print(f"Saved plot: {kpr_face_nn_plot_path}")


if __name__ == "__main__":
    main()
