import logging
import os
import sys
import warnings

import bosdyn.client.util
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError

from humembr.robot.robot_controller import RobotController
from humembr.util.config import load_config
from humembr.util.context import ContextMock

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

warnings.filterwarnings("ignore")


def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s -%(levelname)s - %(message)s"
    )
    logging.getLogger("humembr.robot.robot_controller").setLevel(logging.DEBUG)

    cfg = load_config()
    enable_ctrl = cfg.robot.enable_ctrl
    logging.debug(cfg)

    sdk = bosdyn.client.create_standard_sdk("GraphNavClient")
    robot = sdk.create_robot(cfg.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()  # type: ignore
    robot_control = RobotController(robot, cfg)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    estop_client = robot.ensure_client(EstopClient.default_service_name)

    estop_keep_alive = get_estop(enable_ctrl, estop_client)
    lease_keep_alive = (
        LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
        if enable_ctrl
        else ContextMock()
    )

    with estop_keep_alive:
        try:
            with lease_keep_alive:
                try:
                    robot_control.run()
                    return True
                except Exception as exc:
                    logging.exception(exc)
                    logging.error("Graph nav command line client threw an error.")
                    return False
        except ResourceAlreadyClaimedError:
            logging.error("The robot's lease is currently in use.")
            return False


def get_estop(aquire_lease, estop_client):
    ep = None
    estop_keep_alive = None
    if aquire_lease:
        ep = EstopEndpoint(estop_client, None, 5)
        ep.force_simple_setup()
        estop_keep_alive = EstopKeepAlive(ep)
    else:
        estop_keep_alive = ContextMock()
    return estop_keep_alive


if __name__ == "__main__":
    if not main():
        sys.exit(1)
