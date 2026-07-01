import math

import networkx as nx
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client import math_helpers

from humembr.util.config import load_config
from humembr.util.types import Camera, Direction

cfg = load_config()

CAMERA_TO_DIRECTION: dict[Camera, Direction] = {
    "front_fisheye_image": "front",
    "back_fisheye_image": "back",
    "left_fisheye_image": "left",
    "right_fisheye_image": "right",
}

CAMERA_TO_ROTATION: dict[Camera, float] = {
    "front_fisheye_image": 0.0,
    "back_fisheye_image": 180.0,
    "left_fisheye_image": 90.0,
    "right_fisheye_image": -90.0,
}


class GraphUtil:
    def __init__(self):
        self.spot_graph = get_spot_graph()
        self.nx_graph = build_nx_graph(self.spot_graph)

    def find_shortest_path(
        self, start_waypoint_name, end_waypoint_name
    ) -> tuple[float | None, list[str] | None]:
        try:
            path = nx.shortest_path(
                self.nx_graph, start_waypoint_name, end_waypoint_name, weight="weight"
            )
            path_length = nx.shortest_path_length(
                self.nx_graph, start_waypoint_name, end_waypoint_name, weight="weight"
            )
            return path_length, path
        except nx.NetworkXNoPath:
            return None, None


def build_nx_graph(graph):
    G = nx.Graph()
    name_to_id = get_id_to_name(graph)
    for waypoint in graph.waypoints:
        G.add_node(waypoint.annotations.name)
    for edge in graph.edges:
        G.add_edge(
            name_to_id[edge.id.from_waypoint],
            name_to_id[edge.id.to_waypoint],
            weight=edge.annotations.cost.value,
        )

    return G


def get_spot_graph():
    with open(
        cfg.map_dir / "graph",
        "rb",
    ) as graph_file:
        # Load the graph from disk.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)  # type: ignore
        return current_graph


def get_waypoint_by_id(graph, name_or_id):
    for w in graph.waypoints:
        if w.id == name_or_id or w.annotations.name == name_or_id:
            return w


def get_degree_of_rot(rot):
    return math.degrees(math_helpers.Quat.from_proto(rot).to_yaw())


def get_degree_of_waypoint(graph, name_or_id: str):
    waypoint = None
    for w in graph.waypoints:
        if w.id == name_or_id or w.annotations.name == name_or_id:
            waypoint = w
    assert waypoint, "waypoint was not found!"
    # no fucking clue why i have to invert this on waypoint tform
    return -get_degree_of_rot(waypoint.waypoint_tform_ko.rotation)


def get_id_to_name(graph):
    id_to_name = {}
    for w in graph.waypoints:
        id = w.id
        name = w.annotations.name
        id_to_name[id] = name
    return id_to_name


def waypoint_distance_m(graph, waypoint_id_a: str, waypoint_id_b: str) -> float:
    # Build lookup: waypoint_id -> seed_tform_waypoint (as math_helpers.SE3Pose)
    seed_tform_waypoint = {
        a.id: math_helpers.SE3Pose.from_proto(a.seed_tform_waypoint)
        for a in graph.anchoring.anchors
    }

    Ta = seed_tform_waypoint[waypoint_id_a]  # seed_tform_waypoint_a
    Tb = seed_tform_waypoint[waypoint_id_b]  # seed_tform_waypoint_b

    a_tform_b = Ta.inverse() * Tb
    return a_tform_b.translation_norm()  # type: ignore
