import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from osnet.torchreid.utils import FeatureExtractor
from sklearn.cluster import DBSCAN
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

from humembr.eval.person.clustering_method import KeypointSample, PersonItem
from humembr.eval.person.reid_utils import (
    get_kpr_reid_embeddings,
    get_osnet_reid_embeddings,
)
from humembr.processing.reid import get_reid_kpr_model


@dataclass
class GridResult:
    method: str
    eps: float
    min_samples: int
    homogeneity: float
    completeness: float
    v_measure: float
    assign_ratio: float


def _parse_float_list(raw: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(v) for v in values]


def _parse_int_list(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(v) for v in values]


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


def _evaluate(
    true_labels: list[str], pred_labels: np.ndarray[Any, Any]
) -> dict[str, float]:
    assigned_mask = pred_labels != -1
    assign_ratio = float(np.mean(assigned_mask)) if len(pred_labels) > 0 else 0.0

    if assigned_mask.sum() == 0:
        return {
            "Homogeneity": 0.0,
            "Completeness": 0.0,
            "V-measure": 0.0,
            "Assign-Ratio": assign_ratio,
        }

    true_labels_no_noise = np.array(true_labels)[assigned_mask]
    pred_labels_no_noise = pred_labels[assigned_mask]

    return {
        "Homogeneity": float(
            homogeneity_score(true_labels_no_noise, pred_labels_no_noise)
        ),
        "Completeness": float(
            completeness_score(true_labels_no_noise, pred_labels_no_noise)
        ),
        "V-measure": float(v_measure_score(true_labels_no_noise, pred_labels_no_noise)),
        "Assign-Ratio": assign_ratio,
    }


def _run_grid_search(
    method_name: str,
    embeddings_map: dict[str, np.ndarray[Any, Any]],
    image_name_to_label: dict[str, str],
    eps_values: list[float],
    min_samples_values: list[int],
) -> list[GridResult]:
    image_order = list(embeddings_map.keys())
    embeddings = np.array([embeddings_map[p] for p in image_order])
    true_labels = [str(image_name_to_label[Path(p).name]) for p in image_order]

    results: list[GridResult] = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="cosine",
                n_jobs=-1,
            )
            pred_labels = dbscan.fit_predict(embeddings)
            metrics = _evaluate(true_labels, pred_labels)
            results.append(
                GridResult(
                    method=method_name,
                    eps=eps,
                    min_samples=min_samples,
                    homogeneity=metrics["Homogeneity"],
                    completeness=metrics["Completeness"],
                    v_measure=metrics["V-measure"],
                    assign_ratio=metrics["Assign-Ratio"],
                )
            )

    return results


def _write_results(path: Path, results: list[GridResult]) -> None:
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


def _best_result(results: list[GridResult]) -> GridResult:
    return max(
        results,
        key=lambda r: (r.v_measure, r.assign_ratio, r.homogeneity, r.completeness),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search DBSCAN parameters for OSNet/KPR ReID clustering."
    )
    parser.add_argument(
        "image_folder",
        help="Path to folder containing person crop images (expects sibling person.json).",
    )
    parser.add_argument(
        "--output_folder",
        default="../../eval/eval/clustered_results/dbscan_grid_search",
        help="Output folder for grid-search CSV files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["osnet", "kpr"],
        default=["osnet", "kpr"],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--osnet_eps",
        default="0.08,0.1,0.125,0.15,0.175",
        help="Comma-separated eps values for OSNet DBSCAN.",
    )
    parser.add_argument(
        "--kpr_eps",
        default="0.03,0.04,0.05,0.06,0.07",
        help="Comma-separated eps values for KPR DBSCAN.",
    )
    parser.add_argument(
        "--min_samples",
        default="2,3,4,5,6",
        help="Comma-separated DBSCAN min_samples values for both methods.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    persons = _load_person_data(args.image_folder)
    image_paths = [p.crop_path for p in persons]
    image_name_to_label = {Path(p.crop_path).name: p.label for p in persons}

    min_samples_values = _parse_int_list(args.min_samples)
    all_best: list[GridResult] = []

    if "osnet" in args.methods:
        osnet_eps_values = _parse_float_list(args.osnet_eps)
        print(
            f"Running OSNet grid search for {len(osnet_eps_values) * len(min_samples_values)} combinations..."
        )
        osnet_extractor = FeatureExtractor(
            model_name="osnet_x1_0", device="cpu", verbose=False
        )
        osnet_embeddings_map = get_osnet_reid_embeddings(
            image_paths, osnet_extractor, model_name="osnet_x1_0"
        )
        if osnet_embeddings_map:
            osnet_results = _run_grid_search(
                method_name="osnet_reid_only",
                embeddings_map=osnet_embeddings_map,
                image_name_to_label=image_name_to_label,
                eps_values=osnet_eps_values,
                min_samples_values=min_samples_values,
            )
            _write_results(output_dir / "osnet_dbscan_grid_search.csv", osnet_results)
            best_osnet = _best_result(osnet_results)
            all_best.append(best_osnet)
            print(
                "Best OSNet: "
                f"eps={best_osnet.eps}, min_samples={best_osnet.min_samples}, "
                f"V-measure={best_osnet.v_measure:.4f}, Assign-Ratio={best_osnet.assign_ratio:.4f}"
            )
        else:
            print("No OSNet embeddings extracted. Skipping OSNet grid search.")

    if "kpr" in args.methods:
        kpr_eps_values = _parse_float_list(args.kpr_eps)
        print(
            f"Running KPR grid search for {len(kpr_eps_values) * len(min_samples_values)} combinations..."
        )
        kpr_extractor = get_reid_kpr_model()
        kpr_embeddings_map = get_kpr_reid_embeddings(
            image_paths, kpr_extractor, persons
        )
        if kpr_embeddings_map:
            kpr_results = _run_grid_search(
                method_name="reid_only",
                embeddings_map=kpr_embeddings_map,
                image_name_to_label=image_name_to_label,
                eps_values=kpr_eps_values,
                min_samples_values=min_samples_values,
            )
            _write_results(output_dir / "kpr_dbscan_grid_search.csv", kpr_results)
            best_kpr = _best_result(kpr_results)
            all_best.append(best_kpr)
            print(
                "Best KPR: "
                f"eps={best_kpr.eps}, min_samples={best_kpr.min_samples}, "
                f"V-measure={best_kpr.v_measure:.4f}, Assign-Ratio={best_kpr.assign_ratio:.4f}"
            )
        else:
            print("No KPR embeddings extracted. Skipping KPR grid search.")

    if all_best:
        _write_results(output_dir / "best_dbscan_params.csv", all_best)
        print(
            f"Saved best parameter summary to {output_dir / 'best_dbscan_params.csv'}"
        )
    else:
        print("No results were generated.")


if __name__ == "__main__":
    main()
