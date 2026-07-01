import os
import sys

import numpy as np
import torch
import torchreid
from PIL import Image
from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchvision import transforms

from humembr.util.config import load_config

cfg = load_config()


def get_kpr_embedding(extractor, sample) -> tuple[np.ndarray, np.ndarray]:
    _, embs, vis_score, _ = extractor([sample])
    return embs[0].squeeze().detach().cpu().numpy(), vis_score[
        0
    ].squeeze().detach().cpu().numpy()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_reid_kpr_model():
    with HiddenPrints():
        kpr_cfg = build_config(
            config_path=cfg.root_dir / "src/humembr/processing/kpr-config.yaml"
        )
        weights_path = (
            cfg.root_dir
            / "src/humembr/processing/pretrained/kpr_occ_pt_SOLIDER_81.24_90.59_42326409.pth.tar"
        )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"KPR weights not found at {weights_path}. Please ensure the file exists. Check Readme for instructions"
            )

        kpr_cfg.model.load_weights = str(weights_path)
        kpr_cfg.use_gpu = torch.cuda.is_available()
        kpr_cfg.model.kpr.test_embeddings = ["foreg"]
        extractor = KPRFeatureExtractor(kpr_cfg, verbose=False)
        return extractor


def get_reid_osnet_model(device):
    reid_model_name = "osnet_x1_0"
    reid_model = torchreid.models.build_model(
        name=reid_model_name, num_classes=1, loss="triplet", pretrained=True
    ).to(device)
    reid_model.eval()

    reid_transform = transforms.Compose(
        [
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return reid_model, reid_transform


def get_reid_emb(
    reid_model, reid_transform, device: str, pil_image: Image.Image
) -> np.ndarray:
    with torch.no_grad():
        tensor = reid_transform(pil_image).unsqueeze(0).to(device)
        embedding = reid_model(tensor).cpu().numpy().flatten()
        return embedding
