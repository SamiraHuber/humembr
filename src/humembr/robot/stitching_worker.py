import multiprocessing

from humembr.robot.waypoint_actions.image_sticker import OffscreenStitcher
from humembr.util.config import Config


class StitchingProcess(multiprocessing.Process):
    def __init__(self, job_queue: multiprocessing.Queue, config: Config):
        super().__init__(name="StitchingWorker")
        self.job_queue = job_queue
        self.config = config

    def run(self):
        # Expensive, one-time initialization of Pygame and OpenGL context.
        stitcher = OffscreenStitcher(
            width=self.config.vision.width, height=self.config.vision.height
        )

        while True:
            # A 'None' job is a sentinel value to signal termination.
            job = self.job_queue.get()
            if job is None:
                print(f"[{self.name}] Received shutdown signal. Exiting.")
                break

            result_queue, left_resp, right_resp = job
            stitched_image = stitcher.render(left_resp, right_resp)
            result_queue.put(stitched_image)
