import json
import os

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
    display_mode = "save"  # 'plot' or 'save'
    cfg = load_config()
    kpr_cfg = build_config(
        config_path=str(cfg.root_dir / "src/humembr/processing/kpr-config.yaml")
    )
    kpr_cfg.use_gpu = torch.cuda.is_available()
    kpr_cfg.model.load_weights = str(
        cfg.root_dir
        / "src/humembr/processing/pretrained/kpr_occ_pt_SOLIDER_81.24_90.59_42326409.pth.tar"
    )
    # test_embeddings = ["foreg"]
    test_embeddings = ["bn_foreg"]
    # test_embeddings = ["bn_foreg", "parts"]

    kpr_cfg.model.kpr.test_embeddings = test_embeddings

    extractor = KPRFeatureExtractor(kpr_cfg)

    def load_kpr_samples(images_folder, keypoints_folder):
        image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
        samples = []
        for img_name in image_files:
            img_path = os.path.join(images_folder, img_name)
            json_path = os.path.join(
                keypoints_folder, img_name.replace(".jpg", ".json")
            )

            img = cv2.imread(img_path)

            with open(json_path, "r") as json_file:
                keypoints_data = json.load(json_file)

            keypoints_xyc = []
            negative_kps = []

            for entry in keypoints_data:
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
                "keypoints_xyc": keypoints_xyc,  # the positive prompts indicating the re-identification target
                # "keypoints_xyc": None,
                "negative_kps": negative_kps,  # the negative keypoints indicating other pedestrians
            }

            samples.append(sample)
        return samples

    base_folder = cfg.root_dir / "datasets"
    group1_folder = base_folder / "group1"
    group2_folder = base_folder / "group2"

    samples_grp_1 = load_kpr_samples(
        group1_folder / "images", group1_folder / "keypoints"
    )[:30]
    samples_grp_2 = load_kpr_samples(
        group2_folder / "images", group2_folder / "keypoints"
    )[:30]
    samples_grp_1, embeddings_grp_1, visibility_scores_grp_1, parts_masks_grp_1 = (
        extractor(samples_grp_1)
    )
    print(visibility_scores_grp_1)
    print(embeddings_grp_1.shape)

    samples_grp_2, embeddings_grp_2, visibility_scores_grp_2, parts_masks_grp_2 = (
        extractor(samples_grp_2)
    )

    dis_mat_img = (
        cfg.eval_dir / f"demo/results/distance_matrix_{'_'.join(test_embeddings)}.png"
    )
    dis_mat_img.parent.mkdir(parents=True, exist_ok=True)

    display_kpr_reid_samples_grid(
        samples_grp_1 + samples_grp_2,
        display_mode=display_mode,
        save_path=str(
            cfg.eval_dir
            / f"demo/results/samples_grid_{'_'.join(test_embeddings)}.png"
        ),
    )

    distance_matrix, _ = compute_distance_matrix_using_bp_features(
        embeddings_grp_1,
        embeddings_grp_2,
        visibility_scores_grp_1,
        visibility_scores_grp_2,
        use_gpu=False,
        use_logger=False,
    )

    distances = distance_matrix.cpu().detach().numpy() / 2

    display_distance_matrix(
        distances,
        samples_grp_1,
        samples_grp_2,
        display_mode=display_mode,
        save_path=str(dis_mat_img),
    )
    print(distances.shape)
    print(distances.mean())
    print(distances.mean(axis=1))


if __name__ == "__main__":
    main()
