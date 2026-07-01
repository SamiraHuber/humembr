import concurrent.futures
import logging
import threading
import time
from pathlib import Path

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from humembr.db.repositories import image_repository
from humembr.processing.person_processor import PersonProcessor
from humembr.processing.resnet import get_resnet_embeddings, get_resnet_model
from humembr.processing.sentence import get_sentence_model
from humembr.robot.waypoint_actions.image_interface import Camera, ImageInterface
from humembr.util.config import load_config
from humembr.util.device import get_device

logger = logging.getLogger(__name__)


class BackgroundImageCapture(threading.Thread):
    def __init__(self, image_interface: ImageInterface):
        super().__init__(name="BackgroundImageCaptureThread", daemon=True)
        self._image_interface = image_interface
        self._stop_event = threading.Event()
        self._pause = False
        self.cfg = load_config()

        self.device = get_device()
        self.resnet_model, self.resnet_processor = get_resnet_model(self.device)
        self.sentence_model = get_sentence_model()

        self.person_processor = PersonProcessor(self.device)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="PersonProcessor"
        )

    def run(self):
        while not self._stop_event.is_set():
            if self._pause:
                print("Pausing image capture")
                time.sleep(0.15)
                continue

            start = time.time()
            try:
                img_per_camera, waypoint_id, creation_time, rotation = (
                    self._image_interface.get_images()
                )

                image_path = img_per_camera["front_fisheye_image"]
                fut = self.executor.submit(
                    self.process_img,
                    "front_fisheye_image",
                    waypoint_id,
                    image_path,
                    creation_time,
                    rotation,
                )
                fut.add_done_callback(
                    lambda f: logger.info("add img to db with id %s", f.result())
                )

            except Exception as e:
                print("Error during background image capture")
                print(e)
            self._stop_event.wait(0.8)
            duration = time.time() - start
            logger.info(f"image capture iteration took {duration:.2f}s.")
        print(f"[{self.name}] Shutting down person processing thread pool.")
        self.executor.shutdown(wait=True)

    def filter_duplicates(self, camera: Camera, waypoint: str, image_path: Path):
        image = image_repository.get_latest_observation_by_waypoint_and_camera(
            waypoint, camera
        )
        if image:
            new_img = Image.open(image_path).convert("RGB")
            old_img = Image.open(image.image_path).convert("RGB")
            embs = get_resnet_embeddings(
                self.resnet_processor,
                self.resnet_model,
                self.device,
                [new_img, old_img],
            )
            simalarity = cosine_similarity([embs[0]], [embs[1]])[0][0]  # type: ignore
            logger.debug(
                f"similarity to previous image for waypoint {waypoint} and camera {camera}: {simalarity}"
            )
            return simalarity > self.cfg.perception.filter_sim
        else:
            logger.warning(
                f"no previous image found for waypoint {waypoint} and camera {camera}"
            )
            return False

    def process_img(
        self,
        camera: Camera,
        waypoint_id: str,
        image_path: Path,
        creation_time_dt,
        rotation: float,
    ):
        if self.filter_duplicates(camera, waypoint_id, image_path):
            logger.debug(
                "discarded duplicate image due to similarity for waypoint %s and camera %s",
                waypoint_id,
                camera,
            )
            return

        assert waypoint_id, "waypoint id has to be provided"
        img_id = image_repository.add_image(
            creation_time_dt, str(image_path), waypoint_id, camera, rotation
        )

        self.person_processor.process_person_in_image(image_path, img_id)
        return img_id

    def pause(self):
        self._pause = True

    def unpause(self):
        self._pause = False

    def stop(self):
        print(f"[{self.name}] Signaling stop.")
        self._stop_event.set()
        print(f"[{self.name}] All person processing tasks have completed.")
