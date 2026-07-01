import logging
import re
import threading
import time
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN

from humembr.db.repositories.person_repository import (
    add_person_identity,
    delete_person_identity,
    get_all_identity_names,
    get_all_person_observations,
    get_face_observation_neighbors,
    get_identities_containing_observations,
    get_person_observations_by_identity_id,
    get_unmapped_face_observations,
)
from humembr.util.config import load_config

logger = logging.getLogger(__name__)

cfg = load_config()

EPS = cfg.perception.face_cluster_eps
K = cfg.perception.face_cluster_k


ClusteringMethod = Literal["dbscan", "online"]


def get_next_cluster_label(existing_names: list[str]) -> int:
    """
    Finds the next available index for 'person-X' labels.
    """
    max_id = 0
    pattern = re.compile(r"^person-(\d+)$")

    for name in existing_names:
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            if idx > max_id:
                max_id = idx

    return max_id + 1


def run_dbscan_clustering():
    """
    Offline/Batch clustering using DBSCAN.
    Re-clusters all face observations.
    """
    print("Running DBSCAN face clustering...")
    try:
        # 1. Fetch all observations
        all_obs = get_all_person_observations()
        valid_obs = [o for o in all_obs if o.face_vector is not None]

        if not valid_obs:
            print("No face observations to cluster.")
            return

        print(f"Clustering {len(valid_obs)} faces...")

        # 2. Prepare data
        embeddings = np.array([o.face_vector for o in valid_obs])
        ids = np.array([o.id for o in valid_obs])

        clt = DBSCAN(eps=EPS, min_samples=K, metric="cosine")
        labels = clt.fit_predict(embeddings)

        # 4. Process clusters
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                # Noise points are ignored in this implementation
                continue

            # Get IDs for this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_ids = ids[cluster_indices].tolist()

            # Strategy to pick name:
            # Check if these IDs belong to any existing named identity (preserving manual names)
            current_mappings = get_identities_containing_observations(cluster_ids)
            current_names = set(m.name for m in current_mappings)

            # Prefer names that don't look like 'person-X'
            real_names = [n for n in current_names if not re.match(r"^person-\d+$", n)]

            target_name = None
            if real_names:
                # Pick the first real name found
                target_name = real_names[0]
            elif current_names:
                # Pick any existing person-X name to maintain continuity if possible
                # Sorting to be deterministic
                target_name = sorted(list(current_names))[0]
            else:
                # Create new name
                # Note: get_all_identity_names might not reflect changes in this loop immediately
                # if we rely on DB state, so we might need to be careful.
                # But get_next_cluster_label reads from DB list passed to it.
                all_names = get_all_identity_names()
                next_id = get_next_cluster_label(all_names)
                target_name = f"person-{next_id}"

            print(
                f"Cluster {label}: Assigning {len(cluster_ids)} faces to '{target_name}'"
            )
            add_person_identity(target_name, cluster_ids)

        print("DBSCAN clustering complete.")

    except Exception as e:
        print(f"Error during DBSCAN clustering: {e}")
        logger.error(f"Error during DBSCAN clustering: {e}", exc_info=True)


def run_clustering(method: ClusteringMethod = "online"):
    """
    Runs face clustering.
    Args:
        method: "online" (default) or "dbscan".
    """
    if method == "dbscan":
        run_dbscan_clustering()
        return

    if method == "online":
        online_db_scan()
        return


def online_db_scan():
    logger.info("Running online face clustering...")
    try:
        # 1. Get Unmapped Persons
        # We fetch a batch to process. If there are more, they will be picked up in next run or we could loop here.
        unmapped_persons = get_unmapped_face_observations()

        if not unmapped_persons:
            return

        logger.info(f"Found {len(unmapped_persons)} unmapped persons.")

        # 2. Get existing mapping names to determine next cluster ID
        for p in unmapped_persons:
            asign_person_to_cluster(p.id, p.face_vector)
    except Exception as e:
        print(f"Error during face clustering: {e}")
        logger.error(f"Error during face clustering: {e}", exc_info=True)


def start_periodic_clustering(
    interval_seconds: int = 300, method: ClusteringMethod = "online"
):
    def job():
        time.sleep(10)  # Initial delay
        while True:
            run_clustering(method=method)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    print(
        f"Started periodic face clustering service (interval={interval_seconds}s, method={method})"
    )


def asign_person_to_cluster(person_id, face_vector):
    # Re-check if p is mapped (concurrency or previous iteration)
    if get_identities_containing_observations([person_id]):
        return

    embedding = np.array(face_vector)

    # Check Core Point, only face vibisble person observation -> does not get polluted by less accurate reid embeddings
    neighbors = get_face_observation_neighbors(embedding, threshold=EPS)

    if len(neighbors) < K:
        logger.info(
            f"Person {person_id}: Not a core point ({len(neighbors)} neighbors < {K})"
        )
        return

    logger.info(f"Person {person_id}: Core point with {len(neighbors)} neighbors")

    neighbor_ids = [n.id for n in neighbors]

    # Find which mappings these neighbors belong to
    neighbor_mappings = get_identities_containing_observations(neighbor_ids)

    # Extract unique mapping names
    unique_mapping_names = list(set(m.name for m in neighbor_mappings))
    unique_mapping_names.sort()  # Deterministic

    target_name = None
    mappings_to_merge = []

    if unique_mapping_names:
        target_name = unique_mapping_names[0]
        mappings_to_merge = unique_mapping_names[1:]
        if mappings_to_merge:
            logger.info(
                f"  -> Merging clusters: {mappings_to_merge} into '{target_name}'"
            )
        else:
            logger.info(f"  -> Joining existing cluster '{target_name}'")
    else:
        all_mapping_names = get_all_identity_names()
        next_cluster_id = get_next_cluster_label(all_mapping_names)
        target_name = f"person-{next_cluster_id}"
        logger.info(f"  -> Creating new cluster '{target_name}'")
        # Add to local list so we don't reuse this name if we loop again (though next_cluster_id handles it)
        all_mapping_names.append(target_name)

    # Collect all Person IDs for the target mapping
    # Start with p.id and neighbors
    all_pids = set([person_id] + neighbor_ids)

    # If merging, we need to get IDs from the merged mappings
    for m in neighbor_mappings:
        observations = get_person_observations_by_identity_id(m.id)
        all_pids.update(o.id for o in observations)

    # Apply updates
    add_person_identity(target_name, list(all_pids))
    logger.info(f"  -> Updated '{target_name}' now has {len(all_pids)} persons")

    # Delete merged mappings
    if mappings_to_merge:
        for m_name in mappings_to_merge:
            # Find ID for m_name
            m_obj = next((x for x in neighbor_mappings if x.name == m_name), None)
            if m_obj:
                delete_person_identity(m_obj.id)
                logger.info(f"  -> Deleted merged cluster '{m_name}'")
