import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from humembr.db.repositories.person_repository import get_all_person_observations


def main(data_dir: str):
    data_folder = Path(data_dir)
    image_folder = data_folder / "images/"
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    all_persons = get_all_person_observations()
    person_by_identity = defaultdict(list)
    for p in all_persons:
        if p.person_identity_id is None:
            continue

        kpts_kpr = get_kpr_kpts(p)

        img_path = image_folder / Path(p.crop_path).name
        shutil.copy(p.crop_path, img_path)
        rel_img_path = Path(*img_path.parts[-2:])

        entity = {
            "id": p.id,
            "label": str(p.person_identity_id),
            "crop_path": rel_img_path,
            "kpr_keypoints_sample": kpts_kpr,
        }
        person_by_identity[p.person_identity_id].append(entity)

    keys = list(person_by_identity.keys())
    for k in keys:
        if len(person_by_identity[k]) == 1:
            del person_by_identity[k]
        else:
            print(len(person_by_identity[k]))
    with open(data_folder / "person.json", "w") as f:
        persons = [p for ps in person_by_identity.values() for p in ps]
        json.dump(persons, f, default=str)


def get_kpr_kpts(p):
    kpts_sample = [{"keypoints": p.keypoints, "is_target": True}]
    kpts_sample.extend(
        [{"keypoints": kpts, "is_target": False} for kpts in p.neg_keypoints]
    )

    return kpts_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../datasets/person-data")
    args = parser.parse_args()
    main(args.data_path)
