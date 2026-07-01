import logging
import math
import time

from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client import math_helpers
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.robot_state import RobotStateClient

import humembr.robot.navigation.graph_nav_util as graph_nav_util

logger = logging.getLogger(__name__)


class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path: str):
        self._robot = robot
        self._max_x_vel = 1.5  # m/s
        self._max_y_vel = 1.5
        self._max_ang_vel = 1.0  # rad/s

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        self._robot_state_client = self._robot.ensure_client(
            RobotStateClient.default_service_name
        )

        # Create the client for the Graph Nav main service.
        self._graph_nav_client: GraphNavClient = self._robot.ensure_client(
            GraphNavClient.default_service_name
        )

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  # maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

    def get_localization_state(self):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state(request_gps_state=False)
        logger.debug(f"Got localization: \n{state.localization}")  # type: ignore
        odom_tform_body = get_odom_tform_body(
            state.robot_kinematics.transforms_snapshot  # type: ignore
        )
        logger.debug(
            f"Got robot state in kinematic odometry frame: \n{odom_tform_body}"
        )
        return state

    def set_initial_localization_waypoint(self, waypoint):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if not waypoint or len(waypoint) < 1:
            # If no waypoint id is given as input, then return without initializing.
            logger.info("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            waypoint, self._current_graph, self._current_annotation_name_to_wp_id
        )
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        ).to_proto()  # type: ignore
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint  # type: ignore
        localization.waypoint_tform_body.rotation.w = 1.0  # type: ignore
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,  # type: ignore
            ko_tform_body=current_odom_tform_body,
        )

    def set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        ).to_proto()  # type: ignore
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            ko_tform_body=current_odom_tform_body,
        )

    def list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            logger.info("Empty graph.")
            return
        self._current_graph = graph

        localization_id = (
            self._graph_nav_client.get_localization_state().localization.waypoint_id  # type: ignore
        )

        # Update and logging.info waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = (
            graph_nav_util.update_waypoints_and_edges(
                graph, localization_id, do_print=False
            )
        )

    def upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        logger.info("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)  # type: ignore
            logger.info(
                f"Loaded graph has {len(self._current_graph.waypoints)} waypoints and {len(self._current_graph.edges)} edges"  # type: ignore
            )
        for waypoint in self._current_graph.waypoints:  # type: ignore
            # Load the waypoint snapshots from disk.
            with open(
                f"{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}",
                "rb",
            ) as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())  # type: ignore
                self._current_waypoint_snapshots[waypoint_snapshot.id] = (  # type: ignore
                    waypoint_snapshot
                )
        for edge in self._current_graph.edges:  # type: ignore
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(
                f"{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}", "rb"
            ) as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())  # type: ignore
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot  # type: ignore
        # Upload the graph to the robot.
        logger.info("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)  # type: ignore
        response = self._graph_nav_client.upload_graph(
            graph=self._current_graph, generate_new_anchoring=true_if_empty
        )
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:  # type: ignore
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            logger.info(f"Uploaded {waypoint_snapshot.id}")
        for snapshot_id in response.unknown_edge_snapshot_ids:  # type: ignore
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            logger.info(f"Uploaded {edge_snapshot.id}")

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:  # type: ignore
            # The robot is not localized to the newly uploaded graph.
            logger.info("\n")
            logger.info(
                "Upload complete! The robot is currently not localized to the map; please localize"
                " the robot using commands (2) or (3) before attempting a navigation command."
            )

    def navigate_to(self, waypoint) -> bool:
        if (not waypoint) or (len(waypoint) < 1):
            logger.info("No waypoint provided as a destination for navigate to.")
            return False

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            waypoint, self._current_graph, self._current_annotation_name_to_wp_id
        )
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return False

        speed_limit = SE2VelocityLimit(
            max_vel=SE2Velocity(  # type: ignore
                linear=Vec2(x=self._max_x_vel, y=self._max_y_vel),  # type: ignore
                angular=self._max_ang_vel,  # type: ignore
            )
        )
        travel_params = graph_nav_pb2.TravelParams(
            velocity_limit=speed_limit,  # type: ignore
            # ignore_final_yaw=True,  # type: ignore
        )
        nav_to_cmd_id: int | None = None

        is_finished = False
        while not is_finished:
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    destination_waypoint,
                    1.0,
                    command_id=nav_to_cmd_id,
                    travel_params=travel_params,
                )  # type: ignore
            except ResponseError as e:
                logger.info(f"Error while navigating {e}")
                break
            time.sleep(0.5)
            assert nav_to_cmd_id
            is_finished = self._check_success(nav_to_cmd_id)

        is_stuck = self._check_for_stuck(nav_to_cmd_id)
        return is_stuck

    def _check_for_stuck(self, nav_to_cmd_id):
        assert nav_to_cmd_id
        status = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)
        return status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK  # type: ignore

    def rotate(self, quat: math_helpers.Quat):
        state = self.get_localization_state()

        s3_pose = math_helpers.SE3Pose.from_proto(state.localization.seed_tform_body)  # type: ignore
        s3_pose.rot = quat  # type: ignore
        cmd_id: int = self._graph_nav_client.navigate_to_anchor(
            seed_tform_goal=s3_pose.to_proto(), cmd_duration=20
        )  # type: ignore

        finished = False
        while not finished:
            finished = self._check_success(cmd_id)

    def clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if (
            status.status  # type: ignore
            == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL  # type: ignore
        ):
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:  # type: ignore
            logger.info(
                "Robot got lost when navigating the route, the robot will now sit down."
            )
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:  # type: ignore
            logger.info(
                "Robot got stuck when navigating the route, the robot will now sit down."
            )
            return True
        elif (
            status.status  # type: ignore
            == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED  # type: ignore
        ):
            logger.info("Robot is impaired.")
            return True
        else:
            return False
