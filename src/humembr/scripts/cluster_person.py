import json
from calendar import c
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchreid.utils.visualization.display_kpr_samples import (
    display_distance_matrix,
    display_kpr_reid_samples_grid,
)

from humembr.util.config import load_config


def main():
    cfg = load_config()
    display_mode = "save"  # 'plot' or 'save'
    kpr_cfg = build_config(
        config_path=cfg.root_dir / "src/humembr/processing/kpr-config.yaml"
    )
    kpr_cfg.use_gpu = torch.cuda.is_available()
    # kpr_cfg.model.kpr.test_embeddings = ["foreg"]

    extractor = KPRFeatureExtractor(kpr_cfg)
    person_dataset_path = cfg.eval_dir / "datasets/person-data"
    data_path = Path(person_dataset_path)
    json_path = data_path / "person.json"

    with open(str(json_path), "r") as f:
        persons = json.load(f)

    samples = []
    count_per_name = defaultdict(int)

    for p in persons:
        name = p["label"]
        if count_per_name[name] > 3:
            continue
        count_per_name[name] += 1
        img = cv2.imread(data_path / p["crop_path"])
        keypoints_xyc = []
        negative_kps = []
        for entry in p["kpr_keypoints_sample"]:
            if entry["is_target"]:
                keypoints_xyc.append(entry["keypoints"])
            else:
                negative_kps.append(entry["keypoints"])

        assert len(keypoints_xyc) == 1, (
            "Only one target keypoint set is supported for now."
        )

        keypoints_xyc = np.array(keypoints_xyc[0])
        negative_kps = np.array(negative_kps)

        sample = {
            "image": img,
            "keypoints_xyc": keypoints_xyc,
            "negative_kps": negative_kps,
        }
        samples.append(sample)
    print(f"total {len(samples)} samples")

    samples_grp_1, embeddings_grp_1, visibility_scores_grp_1, parts_masks_grp_1 = (
        extractor(samples)
    )

    dis_mat_img = cfg.eval_dir / "demo/results/distance_matrix.png"
    Path(dis_mat_img).parent.mkdir(parents=True, exist_ok=True)
    display_kpr_reid_samples_grid(
        samples_grp_1,
        display_mode=display_mode,
        save_path=cfg.eval_dir / "demo/results/samples_grid.png",
    )

    distance_matrix, _ = compute_distance_matrix_using_bp_features(
        embeddings_grp_1,
        embeddings_grp_1,
        visibility_scores_grp_1,
        visibility_scores_grp_1,
        use_gpu=False,
        use_logger=False,
    )

    distances = distance_matrix.cpu().detach().numpy() / 2

    display_distance_matrix(
        distances,
        samples_grp_1,
        samples_grp_1,
        display_mode=display_mode,
        save_path=dis_mat_img,
    )


if __name__ == "__main__":
    main()
