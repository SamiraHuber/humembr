from datetime import datetime, timezone
from typing import Sequence

from humembr.db.repositories import (
    person_repository,
)
from humembr.db.repositories.image_repository import Observation, SimilarCaption
from humembr.db.repositories.person_repository import PersonIdentity


def get_person_found_msg(
    last_obs: Observation,
    img_obs_on_route: list[Observation],
    person_identity: PersonIdentity,
) -> str:
    person_id_name = person_identity.name

    img_ids_on_route = [obs.id for obs in img_obs_on_route]
    img_ids_on_route.append(last_obs.id)
    persons_on_route = person_repository.get_person_observations_by_image_ids(
        img_ids_on_route
    )

    if not persons_on_route:
        return "No persons detected on the route"

    notifications: list[str] = []
    person_found = False
    for p_route in persons_on_route:
        if p_route.person_identity_id == person_identity.id:
            person_found = True
            msg = f"{person_id_name} was seen at {p_route.waypoint_name}."
            if msg not in notifications:
                notifications.append(msg)
                print(
                    f"Matched searched person '{person_id_name}' with detected person {p_route.id} at {p_route.waypoint_name}."
                )

    if not person_found:
        notifications.append(f"{person_id_name} was not seen!")
        if persons_on_route:
            notifications.append(f"the observed persons were not {person_id_name}")

    return "\n".join(notifications)


def format_time_delta(timestamp: datetime) -> str:
    delta = datetime.now(timezone.utc) - timestamp

    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, _ = divmod(rem, 60)

    time_parts = []
    if days > 0:
        time_parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        time_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        time_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

    if not time_parts:
        return "less than a minute ago"
    return ", ".join(time_parts) + " ago"


def format_observations(results: Sequence[SimilarCaption | Observation]):
    if not results:
        return ""

    image_ids = [row.id for row in results]
    persons = person_repository.get_person_observations_by_image_ids(image_ids)

    person_map = {}
    for p in persons:
        if p.image_id not in person_map:
            person_map[p.image_id] = []
        if p.identity_name:
            person_map[p.image_id].append(p.identity_name)
        else:
            person_map[p.image_id].append("unknown person")

    formatted_results = []
    for row in results:
        time_str = format_time_delta(row.creation_timestamp)
        time_str = row.creation_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        person_info = ""
        if row.id in person_map:
            identities = ", ".join(person_map[row.id])
            person_info = f" (Identified persons: {identities})"

        rotation_str = f"{row.rotation:.2f}" if row.rotation is not None else "unknown"
        fmt = f"{time_str} at {row.waypoint_name} in direction of {rotation_str}: '{row.caption}'{person_info}"
        formatted_results.append(fmt)
    return "\n".join(formatted_results)


def build_go_to_resp(
    waypoint_name: str, last_obs: Observation, fmt_obs: str, person_notification: str
):
    persons = person_repository.get_person_observations_by_image_ids([last_obs.id])
    person_info = ""
    if persons:
        identities = ", ".join(
            [p.identity_name if p.identity_name else "unknown person" for p in persons]
        )
        person_info = f"\n        Identified persons here: {identities}"

    response = f"""
        You arrived at waypoint {waypoint_name}!
        You are currently seeing: {last_obs.caption}{person_info}
        You have seen the following on your route:
        {fmt_obs}
        {person_notification}
        """
    print(response)
    return response
