import logging
import time
from collections import defaultdict
from datetime import datetime

from langchain.chat_models import BaseChatModel
from langchain.tools import tool

from humembr.agent.tool_utils import (
    build_go_to_resp,
    format_observations,
    get_person_found_msg,
)
from humembr.db.repositories import (
    cmd_repository,
    image_repository,
    person_repository,
    waypoint_repository,
)
from humembr.processing.sentence import get_sentence_model
from humembr.util.config import load_config
from humembr.util.graph import GraphUtil

logger = logging.getLogger(__name__)
model = get_sentence_model()
cfg = load_config()

graph_util = GraphUtil()


@tool
def go_to_waypoint(
    waypoint_name: str,
    reason: str,
    person_id_name: str | None,
    needs_validation: bool,
    direction: float,
):
    """Commands the robot to navigate to a specific waypoint or to look around (look front, left, right or back).

    This is the only way to get new visual information. Upon arrival, you receive a
    real-time observation from the destination and a summary of notable things seen
    along the way. Use this to actively explore the environment. You can look into different directions.

    Args:
        waypoint_name: The name of the target waypoint, matching `^waypoint_\\d+$`. Use 'q' to go to sleep.
        reason: A brief justification for choosing this waypoint.
        person_name: If searching for a person, their name. You will be notified if they are seen.
        needs_validation: If you look for a moving object like a person or a ball that needs realtime validation. If False you just walk to the waypoint without realtime feedback.
        direction: Set this to the direction to look at after reaching your endpoint. This can be used to rotate inplace.
    """

    logger.info(f"Agent wants to go to {waypoint_name}, for reason: {reason}")
    prompt = f"go to {waypoint_name}, for reason {reason}"
    cmd_id = cmd_repository.add_cmd(waypoint_name, prompt, direction)
    logger.info(f"Created command with ID: {cmd_id}")
    retries = 0
    while True:
        time.sleep(0.5)
        cmd = cmd_repository.get_cmd_by_id(cmd_id)
        if cmd.status == "REJECTED":
            response = f"You could not reach waypoint {waypoint_name}, it was blocked or not reachable."
            return response

        if cmd.status != "SUCCESS":
            continue

        if cmd.waypoint_name == "q":
            return "You went to sleep."

        if not needs_validation:
            return f"You arrived at {waypoint_name}"

        retries += 1

        last_obs = image_repository.get_observation_by_waypoint_name(waypoint_name, n=1)

        if len(last_obs) == 0:
            print(f"no observations found at {waypoint_name}")
            continue

        last_obs = last_obs[0]
        if last_obs.caption is None:
            print("found observation without caption, waiting for caption")
            continue

        if last_obs.creation_timestamp > cmd.updated_timestamp and retries < 10:
            print(
                f"only found old observation from {last_obs.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S:%f')}, cmd finished at {cmd.updated_timestamp.strftime('%Y-%m-%d %H:%M:%S:%f')}"
            )
            continue

        img_obs_on_route = list(
            filter(
                lambda o: o.id != last_obs.id,
                image_repository.get_obersvation_since(cmd.creation_timestamp),
            )
        )
        fmt_obs = format_observations(img_obs_on_route)

        if not person_id_name:
            return build_go_to_resp(waypoint_name, last_obs, fmt_obs, "")

        # validate if we saw the wanted person on our route
        person_identity = person_repository.get_person_identity(person_id_name)
        if person_identity is None:
            person_found_msg = f"{person_id_name} is an unknown person and no references could be found in the memory."
        else:
            person_found_msg = get_person_found_msg(
                last_obs, img_obs_on_route, person_identity
            )

        response = build_go_to_resp(waypoint_name, last_obs, fmt_obs, person_found_msg)
        return response


@tool
def on_success(reason: str):
    """Expresses success by performing a happy dance.

    Use this tool when you have successfully completed a task and want to celebrate.

    Args:
        reason: A brief justification for why you are celebrating.
    """

    print(f"Agent wants to celebrate for reason: {reason}")
    prompt = f"celebrate, for reason {reason}"
    cmd_id = cmd_repository.add_cmd("happy", prompt, 0.0)
    retries = 0
    while retries < 10:  # 10 seconds timeout
        time.sleep(1)
        cmd = cmd_repository.get_cmd_by_id(cmd_id)
        if cmd.status == "SUCCESS":
            return "You performed a little happy dance."
        if cmd.status == "REJECTED":
            return "Your celebration was rejected!"
        retries += 1
    return "The robot did not seem to react to your celebration."


@tool
def get_current_location() -> str:
    """Returns the robot's current location as a waypoint name and its coordinates.

    Use this to understand your current position in the office for navigation and
    spatial awareness.
    """
    latest = image_repository.get_latest_image()
    assert latest
    waypoint = waypoint_repository.get_waypoint_by_name(latest.waypoint_name)
    assert waypoint
    return f"You are currently at waypoint {waypoint.name}"


@tool
def get_route_information(waypoint_src: str, waypoint_target: str) -> str:
    """Calculates the shortest path and distance between two waypoints.

    Use this to plan efficient navigation routes by understanding the travel distance
    and path between a source and a target waypoint.

    Args:
        waypoint_src: The name of the starting waypoint.
        waypoint_target: The name of the destination waypoint.
    """
    cost, path = graph_util.find_shortest_path(waypoint_src, waypoint_target)
    if cost is None or path is None:
        return f"No path found between {waypoint_src} and {waypoint_target}"
    fmt_path = " -> ".join(path)

    response = f"""
    The distance from {waypoint_src} to {waypoint_target} is {cost:.2f} meters.
    The shortest path is: {fmt_path}.
"""
    print(response)
    return response


@tool
def query_waypoint_history(waypoint_name: str, start: datetime, until: datetime) -> str:
    """Low level tool that retrieves past observations made at a specific waypoint.

    Use this to get all observation of a given waypoint to understand what you have seen there in the past.
    This can help you to find out if you have seen a specific person at this waypoint or to get an overview of what you have seen there in the past.

    Args:
        waypoint_name: The name of the waypoint to query, in the format `^waypoint_\\d+$`.
        start: A datetime object specifying the lower bound for the query.
        until: An datetime object specifying the upper bound for the query.
    """
    logger.info(
        f"Querying spatial info for waypoint: {waypoint_name} from {start} until {until}"
    )
    results = image_repository.get_observation_by_waypoint_name(
        waypoint_name, n=None, start=start, until=until
    )

    if not results:
        logger.info(f"No information found for waypoint '{waypoint_name}'")
        return f"No information found for waypoint '{waypoint_name}'."

    formatted_results = format_observations(results)
    logger.info(
        "returning %d observations for waypoint '%s'", len(results), waypoint_name
    )
    logger.debug(formatted_results)
    return f"Here is the information I have for '{waypoint_name}':\n{formatted_results}"


@tool
def query_semantic_observations(
    query_text: str, person: str | None, until: datetime | None = None
) -> str:
    """High Level tool that finds relevant past observations based on a query with semantic similarity.
    Use this to search for memories with certain actions or objects.
    You can also add person to filter your memory for spefic personal knowledge.

    Args:
        question: The query text to search for in past observations.
        person: an optional person for filtering only observations containing this person, can be None
        until: an optional datetime object specifying the upper bound for the search. use this to increase performance
    """
    logger.info(
        f"calling semantic search tool: q:{query_text}, person {person}, until {until}"
    )
    embedding = model.encode(query_text, prompt_name="query")

    results = image_repository.find_similar_caption(
        embedding,
        limit=30,
        person=person,
        until=until,
        near_max_hours=cfg.agent.time_penalty_max_hours,
        time_decay_factor=cfg.agent.time_penalty_decay,
        weight_cosine=cfg.agent.weight_cosine,
        weight_time=cfg.agent.weight_time,
    )
    person_repository.get_all_person_identities()

    if not results:
        logger.info("no relevant information found for query '%s'", query_text)
        return "No relevant information found."

    formatted_results = format_observations(results)
    logger.debug(formatted_results)
    logger.info(
        "returning %d observations for '%s' with person '%s'",
        len(results),
        query_text,
        person,
    )
    return f"I made the following observations:\n{formatted_results}"


@tool
def query_available_persons(date: datetime):
    """Very high level tool that retrieves all seen person of the given date

    Use this tool if you don't have knowdge about which person you are looking for but want to get an overview of all seen persons at a given day. This can be used to get an overview of who was at the office today.

    Use this tool if you need to get an overview of all seen persons at a given day.
    This can be used to get an overview of who was at the office today.
    """
    logger.info(f"querying available persons for date: {date}")
    if date is None:
        return "Please provide a date for filtering."

    persons = ",".join(person_repository.find_by_date(date))
    if not persons:
        return f"No persons found for {date.strftime('%Y-%m-%d')}."
    logger.debug(persons)
    logger.info(
        "returning '%s' persons for date '%s'",
        persons,
        date,
    )
    return f"Here are the persons seen on {date}:\n{persons}"


@tool
def query_person_history(person_name: str, start: datetime, end: datetime) -> str:
    """Low level tool that retrieves the observation history for a specific person by name.
    Only use this tool if you already know the specific time window for your search.

    Provides a list of waypoints, timestamps, and associated observations where the
    person was seen. Use this to track a person's last known locations.

    Args:
        person_name: The name of the person to query.
        start: time from which you want to start the search.
        end: time where you want to end the search
    """
    logger.info(f"querying person history for: {person_name} from {start} till {end}")
    person_ids = person_repository.get_observation_ids_by_name(person_name, start, end)
    image_ids = person_repository.get_image_ids_by_observation_ids(person_ids)
    image_captions = image_repository.get_images_by_ids(image_ids, limit=None)

    if not image_captions:
        logger.info("no obseration found for %s", person_name)
        return f"{person_name} was never seen."

    formatted_results = format_observations(image_captions)
    logger.debug(formatted_results)
    logger.info("returning %d observations for %s", len(image_captions), person_name)
    return formatted_results


def build_person_summary_tool(model: BaseChatModel):
    @tool
    def query_person_summary(person_name: str, day: datetime) -> str:
        """Mid level tool that provides a summary of a specific person based on all observations.

        Use this tool if you already know the person and day to get a quick overview of a person's typical behavior, activities,
        and locations based on past sightings.

        Args:
            person_name: The name of the person to summarize.
        """
        day_start = datetime(day.year, day.month, day.day)
        day_end = datetime(day.year, day.month, day.day, 23, 59, 59)
        logger.info(
            f"building person summary for {person_name} from {day_start} till {day_end}"
        )
        person_ids = person_repository.get_observation_ids_by_name(
            person_name, day_start, day_end
        )
        image_ids = person_repository.get_image_ids_by_observation_ids(person_ids)
        image_captions = image_repository.get_images_by_ids(image_ids, limit=None)
        if not image_captions:
            logger.info("no obseration found for %s", person_name)
            return f"{person_name} was never seen on {day.strftime('%Y-%m-%d')}."
        formatted_results = format_observations(image_captions)
        prompt = (
            f"Summarize the following observations about {person_name} in a few sentences. Explain briefly what this person has done around what time and when the person was seen first and last.\n{formatted_results}",
        )
        logger.debug(prompt)
        # ask the model to summarize the observations in a few sentences
        summary = model.invoke(prompt)

        logger.debug(summary.content)
        return summary.content  # type: ignore

    return query_person_summary
