import numpy as np
from bosdyn.api.graph_nav import map_pb2

from humembr.db.repositories import waypoint_repository
from humembr.util.config import load_config


def main():
    cfg = load_config()
    insert_waypoints(cfg)


def insert_waypoints(cfg):
    with open(
        str(cfg.map_dir / "graph"),
        "rb",
    ) as graph_file:
        # Load the graph from disk.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)  # type: ignore
        waypoint_repository.clear_waypoints()
        room_id = waypoint_repository.instert_room("unknown room")
        for w in sorted(
            current_graph.waypoints,  # type: ignore
            key=lambda w: int(w.annotations.name.split("_")[1]),  # type: ignore
        ):
            name = w.annotations.name
            position = w.waypoint_tform_ko.position
            x, y, z = position.x, position.y, position.z
            waypoint_repository.insert_waypoint(
                w.id, name, np.array([x, y, z]), room_id
            )


if __name__ == "__main__":
    main()
