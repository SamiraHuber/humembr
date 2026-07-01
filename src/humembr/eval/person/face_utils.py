import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm


class Face:
    def __init__(
        self,
        image_path: str,
        bounding_box: npt.NDArray[Any] | list[float],
        embedding: npt.NDArray[Any],
        face_id: int,
    ) -> None:
        self.image_path = image_path
        self.bounding_box = [int(val) for val in bounding_box]
        self.embedding = embedding
        self.face_id = face_id


def save_cropped_face(
    original_image_path: str,
    bounding_box: list[float] | list[int] | npt.NDArray[Any],
    output_path: str,
) -> None:
    """Crops a face from an image and saves it."""
    try:
        with Image.open(original_image_path) as img:
            # Convert to tuple[float, float, float, float] expected by PIL
            box = tuple(float(x) for x in bounding_box)
            if len(box) >= 4:
                cropped_img = img.crop(box[:4])  # type: ignore
                cropped_img.save(output_path)
    except Exception as e:
        print(f"Error saving cropped face from {original_image_path}: {e}")


def extract_faces(
    image_paths: list[str], face_app: Any
) -> tuple[list[Face], list[str]]:
    print("Extracting faces and embeddings...")
    all_faces: list[Face] = []
    images_with_faces = set()
    face_counter = 0

    cache_dir = Path("embeddings_cache") / "faces"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Processing images for faces"):
        img_name = Path(image_path).name
        cache_path = cache_dir / f"{img_name}.pkl"
        faces_data = None

        # Try loading from cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    faces_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading face cache for {img_name}: {e}")

        # If not in cache, compute
        if faces_data is None:
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    continue

                detected_faces = face_app.get(img)
                faces_data = []
                if detected_faces:
                    for face in detected_faces:
                        faces_data.append(
                            {"bbox": face.bbox, "embedding": face.embedding}
                        )

                # Save to cache (even if empty, to avoid re-reading image)
                with open(cache_path, "wb") as f:
                    pickle.dump(faces_data, f)

            except Exception as e:
                print(f"Error processing {image_path} for faces: {e}")
                continue

        # Process faces data
        if faces_data:
            images_with_faces.add(image_path)
            for face_dat in faces_data:
                all_faces.append(
                    Face(
                        str(image_path),
                        face_dat["bbox"],
                        face_dat["embedding"],
                        face_counter,
                    )
                )
                face_counter += 1

    images_without_faces = [p for p in image_paths if p not in images_with_faces]
    print(f"Found {len(all_faces)} faces in {len(image_paths)} images.")
    return all_faces, images_without_faces
