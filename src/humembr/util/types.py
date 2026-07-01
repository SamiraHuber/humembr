from typing import Literal

Camera = Literal[
    "front_fisheye_image",
    "back_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
]

Direction = Literal["front", "back", "left", "right"]
