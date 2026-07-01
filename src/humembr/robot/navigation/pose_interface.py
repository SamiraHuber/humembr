import time

from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_command,
)
from bosdyn.geometry import EulerZXY

from humembr.robot.waypoint_actions.waypoint_action import WaypointAction
from humembr.util.config import Config


class PoseInterface(WaypointAction):
    def __init__(self, robot, config: Config):
        self.command_client = robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self.pitch_sit = config.robot.pitch_sit

    def sit(self):
        footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=self.pitch_sit)
        cmd = RobotCommandBuilder.synchro_stand_command(
            footprint_R_body=footprint_R_body
        )

        def check_status(status):
            standing_status = status.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status
            return standing_status == 1

        blocking_command(self.command_client, cmd, check_status)

    def happy_dance(self):
        """
        Performs a sequence of movements to express happiness.
        Wiggles the robot's body.
        """
        print("Here I wiggle with joy!")
        # Roll, Pitch, Yaw
        moves = [
            (0.4, 0.3, 0),
            (-0.4, -0.3, 0),
            (0.4, 0.3, 0),
            (-0.4, -0.3, 0),
            (0, 0.3, 0),
            (0, -0.3, 0),
            (0, 0, 0),
        ]
        for roll, pitch, yaw in moves:
            footprint_R_body = EulerZXY(yaw=yaw, roll=roll, pitch=pitch)
            cmd = RobotCommandBuilder.synchro_stand_command(
                footprint_R_body=footprint_R_body
            )
            self.command_client.robot_command(cmd)
            time.sleep(0.4)

        # Return to a neutral stand
        cmd = RobotCommandBuilder.synchro_stand_command()
        self.command_client.robot_command(cmd)
        time.sleep(1)  # give it time to stabilize

    def on_waypoint_do(self):
        print("sitting down")
        self.sit()
