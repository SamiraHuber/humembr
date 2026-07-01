from abc import ABC, abstractmethod


class WaypointAction(ABC):
    @abstractmethod
    def on_waypoint_do(self):
        pass
