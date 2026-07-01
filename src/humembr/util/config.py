import os
from dataclasses import dataclass
from pathlib import Path

import tomli
from dotenv import load_dotenv


@dataclass
class Evaluation:
    question_time_diff: float  # the allowed difference between two timestamp in minutes
    question_duration_diff: (
        float  # the allowed difference between two duations in minutes
    )
    question_spatial_diff: (
        float  # the allowed distance between two waypoints in meters(?)
    )
    question_llm: str  # the backbone llm for langchain interviewer agent
    tool_call_limit: int


@dataclass
class Robot:
    enable_ctrl: bool = True  # flag if the robot client should require a lease for controlling the robot
    pitch_sit: float = 0.0  # pitch of robot during observation pose
    yolo_mode: bool = False  # if set, each waypoint cmd must be confirmed on the cli


@dataclass
class Vision:
    width: int = 1280  # resolution of recorded images
    height: int = 800
    interval: float = 1.0
    safe_single_images: bool = False  # weither the front left and right images should be saved seperatly or just as a stiched one


@dataclass
class Perception:
    filter_sim: (
        float  # resnet50 similarity threshold to pass for new images to be processed
    )
    face_cluster_eps: float  # DBSCAN eps paramenter (distance radius for core points)
    face_cluster_k: (
        int  # DBSCAN k parameter (how many neighbors neccessary to be a core point?)
    )
    face_conf: float  # confidence threshold for face detection
    person_conf: float  # confidence for person detection
    caption_url: str
    caption_model: str
    caption_prompt: str


@dataclass
class Agent:
    time_penalty_max_hours: int
    time_penalty_decay: float
    weight_cosine: float
    weight_time: float
    context_include_rooms: bool = False


@dataclass
class Config:
    hostname: str  # hostname or ip of spot robot
    map_dir: Path  # directory of the recorded graph nav map
    img_dir: Path  # directory where recorded images of the robot should be saved
    root_dir: Path  # will be set automatically based on config file path
    eval_dir: Path  # directory for evaluation data, datasets, plots, etc
    backup_dir: Path
    vision: Vision
    robot: Robot
    agent: Agent
    perception: Perception
    evaluation: Evaluation
    db_conn_str: str


def load_config(path: str | None = None) -> Config:
    load_dotenv()

    if path is None:
        path = os.environ.get("HUMEMBR_CONFIG_PATH", "config.toml")

    if not os.path.exists(path):
        raise ValueError(f"no config found in {path}")

    with open(path, "rb") as f:
        raw = tomli.load(f)

    vision_raw = raw.get("vision", {})
    vision = Vision(**vision_raw)

    robot_raw = raw.get("robot", {})
    robot = Robot(**robot_raw)

    agent_raw = raw.get("agent", {})
    agent = Agent(**agent_raw)

    eval_raw = raw.get("evaluation", {})
    evaluation = Evaluation(**eval_raw)

    perception_raw = raw.get("perception", {})
    perception = Perception(**perception_raw)

    db_conn_str = os.environ.get("DATABASE_URL")
    if not db_conn_str:
        db_conn_str = str(raw.get("db_conn_str", ""))

    root_dir = Path(path).parent.parent.parent
    config = Config(
        hostname=raw.get("hostname", ""),
        map_dir=Path(raw.get("map_dir", "")),
        img_dir=Path(raw.get("img_dir", "")),
        eval_dir=root_dir / "eval",
        backup_dir=Path(raw.get("backup_dir", "")),
        root_dir=root_dir,
        db_conn_str=db_conn_str,
        vision=vision,
        robot=robot,
        agent=agent,
        evaluation=evaluation,
        perception=perception,
    )

    assert config.img_dir.exists() and config.img_dir.is_absolute(), (
        "img_dir must be an existing absolute path"
    )
    assert config.map_dir.exists() and config.map_dir.is_absolute(), (
        "map_dir must be an existing absolute path"
    )

    return config
