import logging
import math
import time
import traceback

from bosdyn.client import math_helpers

from humembr.db.repositories import cmd_repository, robot_repository
from humembr.db.repositories.cmd_repository import SpotCmd
from humembr.db.repositories.util import get_db_connection
from humembr.robot.background_image_capture import BackgroundImageCapture
from humembr.robot.navigation.graph_nav_interface import GraphNavInterface
from humembr.robot.navigation.pose_interface import PoseInterface
from humembr.robot.navigation.power_interface import PowerInterface
from humembr.robot.waypoint_actions.image_interface import ImageInterface
from humembr.robot.waypoint_actions.waypoint_action import WaypointAction
from humembr.scripts.upload_waypoint_coords import insert_waypoints
from humembr.util.config import Config
from humembr.util.graph import (
    get_spot_graph,
)

logger = logging.getLogger(__name__)


def get_waypoint_input():
    inputs = input("choose waypoint>")
    waypoint = str.split(inputs)[0]
    return waypoint


class RobotController:
    def __init__(self, robot, config: Config):
        self.graph_nav_interface = GraphNavInterface(robot, str(config.map_dir))
        self.power_interface = PowerInterface(robot)
        self.image_interface = ImageInterface(robot, config, self.graph_nav_interface)
        self.pose_interface = PoseInterface(robot, config)
        self.waypoint_actions: list[WaypointAction] = [
            self.pose_interface,
        ]
        self.conn = get_db_connection()
        self.config = config
        self.curr_cmd: None | SpotCmd = None
        self.graph = get_spot_graph()

        self.background_image_capturer = BackgroundImageCapture(self.image_interface)

    def run(self):
        try:
            robot_repository.up_status()
            cmd_repository.clear_cmds()
            insert_waypoints(self.config)

            self.graph_nav_interface.clear_graph()
            logger.debug("cleared graph")
            self.graph_nav_interface.upload_graph_and_snapshots()
            logger.debug("uploaded graph")

            if self.config.robot.enable_ctrl:
                self.power_interface.toggle_power(should_power_on=True)

            self.graph_nav_interface.list_graph_waypoint_and_edge_ids()

            self.set_start_waypoint()

            self.background_image_capturer.start()

            while True:
                time.sleep(0.5)

                if not self.config.robot.enable_ctrl:
                    continue

                waypoint = self.get_next_waypoint()
                if waypoint == "q":
                    self.commit_db_waypoint()
                    break
                if waypoint == "c":
                    continue
                if waypoint == "happy":
                    self.be_happy()
                    continue

                is_stuck = self.graph_nav_interface.navigate_to(waypoint)
                if is_stuck:
                    self.reject_waypoint()
                    continue

                if self.curr_cmd and self.curr_cmd.rotation is not None:
                    self.rotate()

                for action in self.waypoint_actions:
                    action.on_waypoint_do()

                self.commit_db_waypoint()

            self.power_interface.toggle_power(should_power_on=False)
            robot_repository.down_status()

        except Exception as e:
            robot_repository.down_status()
            logger.exception(e)
            logger.error(traceback.format_exc())
        finally:
            self.image_interface.shutdown()
            self.background_image_capturer.stop()
            self.background_image_capturer.join()

    def be_happy(self):
        self.background_image_capturer.pause()
        self.pose_interface.happy_dance()
        self.background_image_capturer.unpause()
        self.commit_db_waypoint()
        self.pose_interface.sit()

    def rotate(self):
        assert self.curr_cmd, "no cmd is selected at the moment"
        assert self.curr_cmd.rotation is not None, "no rotation is set on cmd"
        radian = math.radians(self.curr_cmd.rotation)
        quat = math_helpers.Quat.from_yaw(radian)
        self.graph_nav_interface.rotate(quat)

    def set_start_waypoint(self):
        print("Starting waypoint is required. Enter 'q' for ficidual.")
        init_waypoint = get_waypoint_input()
        if init_waypoint == "q":
            self.graph_nav_interface.set_initial_localization_fiducial()
        else:
            self.graph_nav_interface.set_initial_localization_waypoint(init_waypoint)

    def get_next_waypoint(self):
        spot_cmd = cmd_repository.get_next_cmd()
        if spot_cmd is not None:
            logger.info(
                "found pending cmd: '%s' -> '%s'",
                spot_cmd.prompt,
                spot_cmd.waypoint_name,
            )
            valid = self.confirm(spot_cmd.waypoint_name)

            if valid:
                cmd_repository.process_cmd(spot_cmd.id)
                self.curr_cmd = spot_cmd
                return spot_cmd.waypoint_name

            cmd_repository.reject_cmd(spot_cmd.id)
            logger.info("rejected cmd {}", spot_cmd.prompt)

        logger.debug("no pending cmd found")
        if self.config.robot.yolo_mode:
            time.sleep(1)
            return "c"

        return get_waypoint_input()

    def commit_db_waypoint(self):
        if self.curr_cmd is not None:
            cmd_repository.success_cmd(self.curr_cmd.id)
            self.curr_cmd = None

    def reject_waypoint(self):
        if self.curr_cmd is not None:
            cmd_repository.reject_cmd(self.curr_cmd.id)
            self.curr_cmd = None

    def confirm(self, waypoint: str) -> bool:
        if self.config.robot.yolo_mode:
            logger.info("skipping confirmation due yolo mode")
            return True

        confirm = input(f"next waypoint is {waypoint}, confirm with y/n")
        return confirm == "y"
