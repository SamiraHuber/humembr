import logging

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_sentence_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    return model


def get_sentence_embedding(
    captions: list[str], model: SentenceTransformer, query=False
) -> np.ndarray:
    if query:
        embeddings = model.encode(captions, prompt_name="query")
    else:
        embeddings = model.encode(captions)
    return embeddings
