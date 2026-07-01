from typing import Literal

import torch


def get_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device
