import logging
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from humembr.db.repositories import person_repository
from humembr.eval.person.face_utils import save_cropped_face
from humembr.processing import face_clustering_service
from humembr.processing.reid import get_kpr_embedding, get_reid_kpr_model
from humembr.processing.yolo import get_yolo_model
from humembr.scripts.build_cropped_dataset import save_crop_image, transform_kpts_to_crop
from humembr.util.config import load_config
from humembr.util.keypoints import keypoint_vis_score

logger = logging.getLogger(__name__)


class PersonProcessor:
    def __init__(self, device):
        self.device = device
        self.yolo_model = get_yolo_model()
        self.cfg = load_config()
        self.extractor = get_reid_kpr_model()
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(
            ctx_id=0, det_size=(640, 640), det_thresh=self.cfg.perception.face_conf
        )
        self.crop_output_dir = self.cfg.img_dir / "crop_persons"
        if not self.crop_output_dir.exists():
            self.crop_output_dir.mkdir(parents=True, exist_ok=True)

        self.face_folder = self.cfg.img_dir / "faces"
        if not self.face_folder.exists():
            self.face_folder.mkdir(exist_ok=True, parents=True)

    def process_person_in_image(self, img_path: Path, img_id: int):
        start = time.time()
        results = self.yolo_model(
            img_path, conf=self.cfg.perception.person_conf, verbose=False
        )
        if not results:
            return

        img = cv2.imread(str(img_path))
        result = results[0]
        if not result.boxes:
            return

        logger.info(f"found {len(result.boxes)} persons in {img_path}")
        for person_idx, box in enumerate(result.boxes):
            person_kpts_tensor = result.keypoints.data[person_idx]
            negative_kpts_tensor = [
                r.tolist()
                for i, r in enumerate(result.keypoints.data)
                if i != person_idx
            ]
            kpts_list = person_kpts_tensor.tolist()

            kpts_score = keypoint_vis_score(kpts_list)
            logger.debug("kpts score: %s", kpts_score)
            if kpts_score < 0.5:
                logger.info(
                    "dropped person due to less keypoints visible, kpts score: %s",
                    kpts_score,
                )
                continue

            x1, y1, cropped_img, cropped_img_path = save_crop_image(
                img_path, self.crop_output_dir, img, person_idx, box
            )

            pos_trans_kpts = transform_kpts_to_crop(
                self.cfg.perception.person_conf, x1, y1, cropped_img, kpts_list
            )

            vis_neg_trans_kpts = self.get_negative_keypoints(
                negative_kpts_tensor, x1, y1, cropped_img
            )

            sample = {
                "image": cropped_img,
                "keypoints_xyc": pos_trans_kpts,
                "negative_kps": vis_neg_trans_kpts,  # the negative keypoints indicating other pedestrians
                "is_target": True,
            }

            emb, _ = get_kpr_embedding(self.extractor, sample)

            face_kpts = pos_trans_kpts[
                :3
            ]  # the first 3 keypoints are the face keypoints
            face_visible_in_crop = any(kpt[2] > 0.5 for kpt in face_kpts)

            face_emb = None
            det_score = None
            faces = None
            if face_visible_in_crop:
                faces = self.face_app.get(cropped_img)
                if len(faces) == 1:
                    face_emb = faces[0].embedding
                    det_score: float | None = faces[0].det_score

            person_id = person_repository.add_person_observation(
                img_id,
                emb,
                face_emb,
                det_score,
                cropped_img_path,
                pos_trans_kpts,
                vis_neg_trans_kpts,
            )

            if faces:
                face_img = self.face_folder / f"{person_id}.jpg"
                save_cropped_face(cropped_img_path, faces[0].bbox, str(face_img))

            if face_emb is not None:
                self.assign_face_cluster(person_id, face_emb)
            else:
                self.assign_reid_nn(person_id)

        logger.info(f"person processing took {time.time() - start:.2f} seconds")

    def assign_reid_nn(self, person_id: int):
        person = person_repository.get_person_observation_by_id(person_id)
        if person is None:
            return
        neighbor = person_repository.get_nearest_observation_with_face(
            np.array(person.reid_vector)
        )

        if not neighbor:
            return

        if neighbor.distance < 0.083 and neighbor.person_identity_id is not None:
            identity = person_repository.get_person_identity_by_id(
                neighbor.person_identity_id
            )
            assert identity is not None
            person_repository.add_observation_to_identity(
                identity.name, neighbor.person_identity_id
            )
            logger.info(f"assigned person obs {person_id} to identity {identity.name}")
        else:
            logger.info(
                f"could not assign perons obs {person_id} to identiy with dist {neighbor.distance} and person id {neighbor.person_identity_id}"
            )

    def get_negative_keypoints(self, negative_kpts_tensor, x1, y1, cropped_img):
        vis_neg_trans_kpts = []
        for neg_kpts in negative_kpts_tensor:
            neg_trans_kpts = transform_kpts_to_crop(
                self.cfg.perception.person_conf, x1, y1, cropped_img, neg_kpts
            )
            has_visibile = any(kpt[2] > 0.0 for kpt in neg_trans_kpts)
            if has_visibile:
                vis_neg_trans_kpts.append(neg_trans_kpts)
        return vis_neg_trans_kpts

    def assign_face_cluster(self, person_id: int, face_emb: np.ndarray):
        try:
            face_clustering_service.asign_person_to_cluster(person_id, face_emb)

        except Exception as e:
            logger.error(f"Error during online clustering: {e}", exc_info=True)
