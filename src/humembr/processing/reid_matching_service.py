import logging
import threading
import time

import numpy as np
from tqdm import tqdm

from humembr.db.repositories.person_repository import (
    add_observation_to_identity,
    get_nearest_observation_with_face,
    get_observations_without_face_and_identity,
    get_person_identity_by_id,
)

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.083


def run_reid_matching():
    """
    Matches observations without a face but with a ReID vector to existing face clusters.
    """
    logger.info("Running ReID matching for missing faces...")
    try:
        observations = get_observations_without_face_and_identity()
        if not observations:
            return

        logger.info(f"Found {len(observations)} observations without face/identity.")

        matches_found = 0
        for obs in tqdm(observations, desc="Matching ReID"):
            # Ensure reid_vector is a numpy array
            reid_vec = (
                np.array(obs.reid_vector)
                if not isinstance(obs.reid_vector, np.ndarray)
                else obs.reid_vector
            )

            nn = get_nearest_observation_with_face(reid_vec)

            if not nn:
                continue

            if nn.distance <= MATCH_THRESHOLD:
                # We found a match. Now check if the matched face has an identity.
                if nn.person_identity_id:
                    identity = get_person_identity_by_id(nn.person_identity_id)
                    if identity:
                        logger.info(
                            f"Matching observation {obs.id} to identity '{identity.name}' (distance: {nn.distance:.4f})"
                        )
                        add_observation_to_identity(identity.name, obs.id)
                        matches_found += 1
                else:
                    # The nearest neighbor face is not yet clustered/identified.
                    # We skip for now. It might be clustered later.
                    pass

        if matches_found > 0:
            logger.info(
                f"ReID matching complete. Matched {matches_found} observations."
            )

    except Exception as e:
        logger.error(f"Error during ReID matching: {e}", exc_info=True)


def start_periodic_reid_matching(interval_seconds: int = 300):
    def job():
        time.sleep(10)  # Initial delay
        while True:
            run_reid_matching()
            time.sleep(interval_seconds)

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    logger.info(
        f"Started periodic ReID matching service (interval={interval_seconds}s)"
    )
