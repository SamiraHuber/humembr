import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from humembr.db.repositories.person_repository import (
    get_nearest_observation_with_face,
    get_observations_without_face_and_identity,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot nearest neighbors for observations without face."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.083,
        help="Distance threshold for successful match.",
    )
    args = parser.parse_args()
    # simple extractor: model_name from model zoo, or supply path to weights
    threshold = args.threshold

    output_dir = "eval/no_face_nn_plots"
    success_dir = os.path.join(output_dir, "successful_match")
    fail_dir = os.path.join(output_dir, "over_threshold")

    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    print("Fetching observations without face and identity...")
    observations = get_observations_without_face_and_identity()
    print(f"Found {len(observations)} observations.")

    if not observations:
        return

    count_below_threshold = 0
    print(f"Processing observations with threshold {threshold}...")

    for obs in tqdm(observations):
        # Ensure reid_vector is a numpy array for the query
        reid_vec = (
            np.array(obs.reid_vector)
            if not isinstance(obs.reid_vector, np.ndarray)
            else obs.reid_vector
        )
        nn = get_nearest_observation_with_face(reid_vec)

        if not nn:
            continue

        is_match = nn.distance <= threshold
        if is_match:
            count_below_threshold += 1

        # Load images
        img1_path = obs.crop_path
        img2_path = nn.crop_path

        # Check if paths exist
        if not os.path.exists(img1_path):
            # Try prepending project root if relative path fails (though usually running from root)
            pass

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None:
            # print(f"Could not load target image: {img1_path}")
            continue
        if img2 is None:
            # print(f"Could not load NN image: {img2_path}")
            continue

        # Resize images to a fixed height for side-by-side display
        target_h = 300

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        scale1 = target_h / h1
        scale2 = target_h / h2

        img1_resized = cv2.resize(img1, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(img2, (int(w2 * scale2), target_h))

        # Create canvas
        gap = 20
        # Extra space at bottom for text
        text_h = 60
        combined_w = img1_resized.shape[1] + img2_resized.shape[1] + gap
        combined_h = target_h + text_h

        combined = (
            np.zeros((combined_h, combined_w, 3), dtype=np.uint8) + 255
        )  # White background

        # Place images
        combined[0:target_h, : img1_resized.shape[1]] = img1_resized
        combined[0:target_h, img1_resized.shape[1] + gap :] = img2_resized

        # Add text
        # Distance
        text_dist = f"Dist: {nn.distance:.4f}"
        color = (0, 0, 255) if nn.distance > threshold else (0, 255, 0)
        cv2.putText(
            combined,
            text_dist,
            (10, target_h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # Labels
        cv2.putText(
            combined,
            f"Target (ID: {obs.id})",
            (10, target_h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            combined,
            f"NN w/ Face (ID: {nn.id})",
            (img1_resized.shape[1] + gap, target_h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        # Save to appropriate folder
        filename = f"obs_{obs.id}_nn_{nn.id}.jpg"
        if is_match:
            save_path = os.path.join(success_dir, filename)
        else:
            save_path = os.path.join(fail_dir, filename)

        cv2.imwrite(save_path, combined)

    print(f"Plots saved to {output_dir} (subfolders: successful_match, over_threshold)")
    print(f"Number of matches below {threshold} distance: {count_below_threshold}")


if __name__ == "__main__":
    main()
