import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from humembr.db.repositories.person_repository import get_all_person_observations
from humembr.util.config import load_config


def build_group(group_dir: str, person_kpts: list[tuple[str, Any, Any]]):
    dir = Path(group_dir)
    shutil.rmtree(dir, ignore_errors=True)  # Clean up previous group
    img_dir = dir / "images"
    kps_dir = dir / "keypoints"
    img_dir.mkdir(parents=True, exist_ok=True)
    kps_dir.mkdir(parents=True, exist_ok=True)

    for img_origin, pos_kps_origin, neg_kps_origin in person_kpts:
        img_name = Path(img_origin).name
        img_path = img_dir / img_name
        shutil.copy(img_origin, img_path)

        # Create keypoint file
        kps_name = img_name.replace(".jpg", ".json")
        kps_path = kps_dir / kps_name
        kpts_sample = [{"keypoints": pos_kps_origin, "is_target": True}]
        kpts_sample.extend(
            [{"keypoints": kpts, "is_target": False} for kpts in neg_kps_origin]
        )
        with open(kps_path, "w") as f:
            json.dump(kpts_sample, f, indent=4)


def main():
    cfg = load_config()
    all = get_all_person_observations()
    all_persons = list(
        filter(
            lambda p: p.id
            in [
                5505,
                5507,
                5551,
                5555,
                5572,
                5640,
                5823,
                5895,
                5908,
                5936,
                5987,
                5986,
                6056,
                6076,
                6079,
                5664,
                5625,
                5544,
            ],
            all,
        )
    )

    visible = []

    for person in tqdm(all_persons):
        # The third element in each keypoint tuple is the visibility/confidence score
        visible.append((person.crop_path, person.keypoints, person.neg_keypoints))

    print(f"found {len(visible)} visible images")
    firstpart = visible[::2]
    secondpart = visible[1::2]
    print(firstpart, secondpart)
    print(f"firstpart: {len(firstpart)}, secondpart: {len(secondpart)}")

    build_group(str(cfg.eval_dir / "datasets/group1"), firstpart)
    build_group(str(cfg.eval_dir / "datasets/group2"), secondpart)


if __name__ == "__main__":
    main()
