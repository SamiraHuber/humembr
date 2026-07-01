import hashlib
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from humembr.eval.person.clustering_method import PersonItem

CACHE_DIR = Path("embeddings_cache")


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stats = path.stat()
        return stats.st_mtime_ns, stats.st_size
    except OSError:
        return None


def _cache_key_for_path(img_path: str) -> str:
    resolved = str(Path(img_path).resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()


def _as_float32_numpy(value: Any) -> npt.NDArray[np.float32]:
    # Handle torch tensors without importing torch as a hard dependency.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)

    return np.asarray(value, dtype=np.float32)


def get_kpr_reid_embeddings(
    image_paths: list[str], kpr_extractor: Any, person_data: list[PersonItem]
) -> dict[str, npt.NDArray[np.float32]]:
    # This is a simplified version; in a real scenario, you'd batch this.
    print("Extracting ReID embeddings...")
    image_name_to_kpts = {
        Path(p.crop_path).name: p.kpr_keypoints_sample for p in person_data
    }
    embeddings: dict[str, npt.NDArray[np.float32]] = {}

    reid_cache_dir = CACHE_DIR / "reid"
    if not reid_cache_dir.exists():
        reid_cache_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Extracting ReID embeddings"):
        img_name = Path(img_path).name

        # Check cache
        cache_path = reid_cache_dir / f"{img_name}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embeddings[img_path] = pickle.load(f)
                continue
            except Exception as e:
                print(f"Error loading cache for {img_name}: {e}")

        if img_name not in image_name_to_kpts:
            continue

        kpts_sample = image_name_to_kpts[img_name]
        keypoints_xyc = [s.keypoints for s in kpts_sample if s.is_target]
        negative_kps = [s.keypoints for s in kpts_sample if not s.is_target]

        assert len(keypoints_xyc) == 1, (
            "Only one target keypoint set is supported for now."
        )

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        sample = {
            "image": img,
            "keypoints_xyc": np.array(keypoints_xyc[0]),
            "negative_kps": np.array(negative_kps),
        }
        _, emb, _, _ = kpr_extractor([sample])
        embedding = emb[0].squeeze().cpu().detach().numpy()
        embeddings[img_path] = embedding

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"Error saving cache for {img_name}: {e}")

    return embeddings


def get_osnet_reid_embeddings(
    image_paths: list[str],
    extractor: Any,
    model_name: str = "osnet_x1_0",
) -> dict[str, npt.NDArray[np.float32]]:
    print("Extracting OSNet ReID embeddings...")
    embeddings: dict[str, npt.NDArray[np.float32]] = {}

    cache_dir = CACHE_DIR / "osnet_reid" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Extracting OSNet ReID embeddings"):
        img_file = Path(img_path)
        fingerprint = _file_fingerprint(img_file)
        if fingerprint is None:
            continue
        mtime_ns, size = fingerprint

        cache_path = cache_dir / f"{_cache_key_for_path(img_path)}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)

                if isinstance(payload, np.ndarray):
                    embeddings[img_path] = _as_float32_numpy(payload)
                    continue

                if (
                    isinstance(payload, dict)
                    and payload.get("model") == model_name
                    and payload.get("mtime_ns") == mtime_ns
                    and payload.get("size") == size
                ):
                    embeddings[img_path] = _as_float32_numpy(payload.get("embedding"))
                    continue
            except Exception as e:
                print(f"Error loading OSNet cache for {img_file.name}: {e}")

        try:
            raw_embedding = extractor(str(img_file))
            embedding = _as_float32_numpy(raw_embedding).flatten()
        except Exception as e:
            print(f"Error extracting OSNet embedding for {img_file.name}: {e}")
            continue

        embeddings[img_path] = embedding

        try:
            payload = {
                "embedding": embedding,
                "model": model_name,
                "mtime_ns": mtime_ns,
                "size": size,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"Error saving OSNet cache for {img_file.name}: {e}")

    return embeddings
