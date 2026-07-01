import argparse
import json
import shutil
from pathlib import Path

from humembr.db.repositories.image_repository import get_all_images
from humembr.util.graph import get_spot_graph, get_waypoint_by_id


def main(data_path: str):
    images = get_all_images()
    graph = get_spot_graph()

    data_folder = Path(data_path)
    image_folder = data_folder / "images/"
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    outputs = []
    for i in images:
        if i.waypoint_id is None:
            continue
        if i.camera_id != "front_fisheye_image":
            continue
        waypoint = get_waypoint_by_id(graph, i.waypoint_id)
        if not waypoint:
            print(
                f"could not find waypoint for {i.waypoint_id} at {i.creation_timestamp}"
            )
            continue
        position = waypoint.waypoint_tform_ko.position  # type: ignore
        x, y, z = position.x, position.y, position.z
        yaw = i.rotation

        creation_time_str = i.creation_timestamp.strftime("%Y%m%d%H%M%S%f")[:-3]
        new_image_path = image_folder / f"{i.id}-{creation_time_str}.jpg"

        shutil.copy(i.image_path, new_image_path)
        entity = {
            "id": i.id,
            "position": [x, y, z],
            "theta": yaw,
            "time": i.creation_timestamp,
            "caption": i.caption,
            "image_path": new_image_path,
        }
        outputs.append(entity)

    with open(data_folder / "dataset.json", "w") as f:
        json.dump(outputs, f, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../datasets/spot-data")
    args = parser.parse_args()
    main(args.data_path)
