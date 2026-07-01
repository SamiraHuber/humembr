import json
import os

import cv2
from tqdm import tqdm

from humembr.db.repositories import image_repository
from humembr.processing.yolo import get_yolo_model
from humembr.util.config import load_config

confidenc = 0.75


def save_person_crops_and_keypoints(
    image_path: str, result, output_dir, confidence=0.75
):
    img = cv2.imread(image_path)

    for person_idx, box in enumerate(result.boxes):
        x1, y1, cropped_img, cropped_img_path = save_crop_image(
            image_path, output_dir, img, person_idx, box
        )

        person_kpts_tensor = result.keypoints.data[person_idx]
        kpts_list = person_kpts_tensor.tolist()

        formatted_kpts = transform_kpts_to_crop(
            confidence, x1, y1, cropped_img, kpts_list
        )

        kpts_sample = [{"keypoints": formatted_kpts, "is_target": True}]

        save_kpts(cropped_img_path, kpts_sample)


def transform_kpts_to_crop(min_conf, x1, y1, cropped_img, kpts_list):
    formatted_kpts = []
    for kp in kpts_list:
        kpts = transform_kp(min_conf, x1, y1, cropped_img, kp)
        formatted_kpts.append(kpts)
    return formatted_kpts


def save_kpts(cropped_img_path, kpts_sample):
    keypoints_filename = cropped_img_path.replace("images", "keypoints").replace(
        ".jpg", ".json"
    )
    keypoints_dir = os.path.dirname(keypoints_filename)
    os.makedirs(keypoints_dir, exist_ok=True)

    with open(keypoints_filename, "w") as f:
        json.dump(kpts_sample, f, indent=4)


def transform_kp(min_conf, x1, y1, cropped_img, kp):
    x, y, conf = kp

    # check if keypoint is in image, reference:
    # https://community.ultralytics.com/t/pose-estimation-key-points-outside-bounding-box/686/7
    rel_x = x - x1
    rel_y = y - y1
    coords_positive = rel_x >= 0 and rel_y >= 0
    coords_in_image = rel_x < cropped_img.shape[1] and rel_y < cropped_img.shape[0]

    if conf < min_conf or not coords_positive or not coords_in_image:
        kpts = [0.0, 0.0, 0.0]
    else:
        # Storing relative keypoint coordinates
        assert_keypoints_in_crop(cropped_img, rel_x, rel_y)
        kpts = [rel_x, rel_y, 1.0]
    return kpts


def assert_keypoints_in_crop(cropped_img, rel_x, rel_y):
    assert rel_x >= 0 and rel_y >= 0, (
        f"Relative coordinates should be positive: {rel_x}, {rel_y}, shape: {cropped_img.shape}"
    )
    assert rel_x < cropped_img.shape[1] and rel_y < cropped_img.shape[0], (
        f"Relative coordinates should be within image bounds: {rel_x}, {rel_y}, shape: {cropped_img.shape}"
    )


def save_crop_image(image_path, output_dir, img, person_idx, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_img = img[y1:y2, x1:x2]

    base_name = os.path.basename(image_path)
    original_img_name = os.path.splitext(base_name)[0]
    cropped_img_name = f"{original_img_name}_person_{person_idx}.jpg"
    cropped_img_path = os.path.join(output_dir, cropped_img_name)
    cv2.imwrite(cropped_img_path, cropped_img)
    return x1, y1, cropped_img, cropped_img_path


def main():
    yolo_model = get_yolo_model()
    cfg = load_config()
    all_images = image_repository.get_all_images()
    output_dir = str(cfg.eval_dir / "dataset/images")
    os.makedirs(output_dir, exist_ok=True)

    for image in tqdm(all_images):
        results = yolo_model(image.image_path, conf=confidenc)
        if not results:
            continue

        result = results[0]
        if result.boxes and len(result.boxes) > 0:
            save_person_crops_and_keypoints(
                image.image_path, result, output_dir, confidence=confidenc
            )


if __name__ == "__main__":
    main()
