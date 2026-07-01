import logging
import multiprocessing
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from scipy import ndimage

from humembr.robot.stitching_worker import StitchingProcess
from humembr.robot.waypoint_actions.waypoint_action import WaypointAction
from humembr.util.config import Config
from humembr.util.graph import get_degree_of_rot
from humembr.util.types import Camera

logger = logging.getLogger(__name__)

ROTATION_ANGLE = {
    "frontleft_fisheye_image": -78,
    "frontright_fisheye_image": -102,
    "back_fisheye_image": 0,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180,
}


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)  # type: ignore


class ImageInterface(WaypointAction):
    def __init__(self, robot, cfg: Config, graph_nav_interface) -> None:
        self.robot = robot
        self.graph_nav_interface = graph_nav_interface
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.img_dir = cfg.img_dir
        self.job_queue = multiprocessing.Queue()
        self.manager = multiprocessing.Manager()
        self.stitching_worker = StitchingProcess(self.job_queue, cfg)
        self.config = cfg
        self.stitching_worker.start()

    def on_waypoint_do(self):
        pass

    def shutdown(self):
        print("Shutting down ImageInterface and StitchingWorker...")
        self.job_queue.put(None)  # Sentinel value to stop the worker
        self.stitching_worker.join(timeout=5)
        self.stitching_worker.terminate()
        self.manager.shutdown()

    def get_images(self) -> tuple[dict[Camera, Path], str, datetime, float]:
        image_request = [
            build_image_request(
                "frontleft_fisheye_image",
                pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,  # type: ignore
            ),
            build_image_request(
                "frontright_fisheye_image",
                pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,  # type: ignore
            ),
        ]

        state = self.graph_nav_interface.get_localization_state()
        waypoint_id = state.localization.waypoint_id
        assert waypoint_id, "waypoint id is required"
        rotation = get_degree_of_rot(state.localization.seed_tform_body.rotation) % 360

        image_responses = self.image_client.get_image(image_request)
        response_map = {r.source.name: r for r in image_responses}

        # stich front left and right
        left_resp = response_map.get("frontleft_fisheye_image")
        right_resp = response_map.get("frontright_fisheye_image")
        stitched_bgr = self.stich_image(left_resp, right_resp)

        # create folder if not exists
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        folder_nested = self.img_dir / date
        self.create_path(folder_nested)

        # image path
        creation_time = datetime.now(timezone.utc)
        creation_time_str = creation_time.strftime("%Y%m%d%H%M%S%f")[:-3]

        stiched_img_path = (
            folder_nested / f"{creation_time_str}_front_fisheye_image.jpg"
        )

        cv2.imwrite(str(stiched_img_path), stitched_bgr)

        images_per_cam = self.save_single_images(image_responses, stiched_img_path)
        images_per_cam.update({"front_fisheye_image": stiched_img_path})
        return images_per_cam, waypoint_id, creation_time, rotation

    def stich_image(self, left_resp, right_resp):
        assert left_resp and right_resp
        start = time.time()
        result_queue = self.manager.Queue(1)
        self.job_queue.put((result_queue, left_resp, right_resp))
        stitched_bgr = result_queue.get()
        logger.debug(f"Stitching took {time.time() - start}s")
        return stitched_bgr

    def create_path(
        self,
        path: Path,
    ):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def save_single_images(
        self, image_responses, front_img_path: Path
    ) -> dict[Camera, Path]:
        paths: dict[Camera, Path] = {}
        for image in image_responses:
            if image.source.name in [
                "frontleft_fisheye_image",
                "frontright_fisheye_image",
            ]:
                continue

            num_bytes = 3
            dtype = np.uint8
            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:  # type: ignore
                try:
                    img = img.reshape(
                        (image.shot.image.rows, image.shot.image.cols, num_bytes)
                    )
                except ValueError:
                    img = cv2.imdecode(img, -1)
            else:
                img = cv2.imdecode(img, -1)

            img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])
            img_path = str(front_img_path).replace(
                "front_fisheye_image", image.source.name
            )
            cv2.imwrite(img_path, img)
            paths[image.source.name] = Path(img_path)

        return paths
