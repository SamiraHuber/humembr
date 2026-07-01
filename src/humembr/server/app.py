import base64
import json
import logging
import os
import threading
import time
from datetime import datetime
from queue import Empty

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from humembr.agent.llm_agent import LLMInterface
from humembr.db.repositories import (
    cmd_repository,
    image_repository,
    person_repository,
    robot_repository,
)
from humembr.db.repositories.image_repository import Observation
from humembr.processing.face_clustering_service import start_periodic_clustering
from humembr.processing.reid_matching_service import start_periodic_reid_matching
from humembr.processing.sentence import get_sentence_embedding, get_sentence_model
from humembr.server.agent_queue_manager import AgentQueueManger
from humembr.util.config import load_config

cfg = load_config()
app = Flask(__name__)
sock = Sock(app)
logger = logging.getLogger(__name__)
agent_queue_manager = AgentQueueManger()
sentence_model = get_sentence_model()

# Start background services
# start_periodic_clustering(interval_seconds=300, method="dbscan")
# start_periodic_reid_matching(interval_seconds=300)

llm_interface = LLMInterface(cfg, agent_queue_manager)


def state_stream(ws):
    """Monitors and sends robot and image state updates to the client."""
    last_id = 0
    while True:
        try:
            image_row = image_repository.get_latest_image()
            cmd_row = cmd_repository.get_next_cmd_for_ws()
            robot_status = robot_repository.get_latest_status()

            payload = {}
            if cmd_row:
                payload["current_cmd"] = {
                    "id": cmd_row["id"],
                    "waypoint_name": cmd_row["waypoint_name"],
                    "status": cmd_row["status"],
                    "prompt": cmd_row["prompt"],
                    "creation_timestamp": cmd_row["creation_timestamp"].timestamp(),
                }
            if robot_status:
                payload["robot_status"] = robot_status.status

            payload["uncaptioned_count"] = image_repository.get_uncaptioned_count()

            if image_row and image_row.id > last_id:
                try:
                    root_dir = str(cfg.root_dir)
                    image_path = os.path.join(root_dir, image_row.image_path)
                    with open(image_path, "rb") as f:
                        frame_bytes = f.read()
                    image_base64 = base64.b64encode(frame_bytes).decode("utf-8")
                    payload.update(
                        {
                            "image_data": image_base64,
                            "file_name": os.path.basename(image_row.image_path),
                            "timestamp": image_row.creation_timestamp.timestamp(),
                            "caption": image_row.caption,
                            "waypoint_id": image_row.waypoint_id,
                            "waypoint_name": image_row.waypoint_name,
                            "room_name": image_row.room_name,
                        }
                    )
                    last_id = image_row.id
                except FileNotFoundError:
                    logger.error(f"Image not found at path: {image_row.image_path}.")

            if payload:
                ws.send(json.dumps(payload))

        except ConnectionClosed:
            logger.info("conntection closed")
            break  # Exit loop if connection is closed
        except Exception as e:
            print("error in state stream", e)
            logger.error(f"Error in state_stream: {e}", exc_info=True)

        time.sleep(0.25)
    logger.info("state stream exited")


def llm_stream(ws, thread_id: str):
    """Listens for LLM events and forwards them to the client."""
    queue = agent_queue_manager.get_queue(thread_id)
    while True:
        try:
            message = queue.get(block=True)
            ws.send(message)
            queue.task_done()
        except Empty:
            # Check if the websocket is still active
            if ws.closed:
                break
            continue
        except ConnectionClosed:
            break  # Exit loop if connection is closed
        except Exception as e:
            logger.error(f"Error in llm_stream: {e}", exc_info=True)
            break
    print("llm stream exited")


@sock.route("/ws")
def ws(ws):
    thread_id = None
    state_thread = None
    llm_thread = None
    try:
        logger.info("socket connected, waiting for thread_id")
        initial_message = ws.receive(timeout=10)

        if not initial_message:
            logger.error("No thread_id message received from client.")
            ws.close()
            return

        data = json.loads(initial_message)
        thread_id = data.get("thread_id")

        if not thread_id:
            logger.error("No thread_id in message.")
            ws.close()
            return

        logger.info(f"WebSocket connection established for thread: {thread_id}")

        state_thread = threading.Thread(target=state_stream, args=(ws,))
        llm_thread = threading.Thread(target=llm_stream, args=(ws, thread_id))

        state_thread.start()
        llm_thread.start()
        time.sleep(1)

        # Keep the main thread alive to handle the connection
        while state_thread.is_alive() and llm_thread.is_alive():
            time.sleep(1)
        logger.info("exited ws")

    except ConnectionClosed:
        logger.info(f"WebSocket connection closed for thread: {thread_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
    finally:
        if thread_id:
            agent_queue_manager.remove_queue(thread_id)
            logger.info(f"Cleaned up resources for thread: {thread_id}")
        # The threads will exit on their own when the connection is closed
        # and an exception is raised.


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/map")
def map_page():
    items = image_repository.get_waypoints_with_captions()
    return render_template("map.html", items=items)


@app.route("/person_observations")
def person_observations_page():
    unmapped_only = request.args.get("unmapped_only", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    identity_id_str = request.args.get("identity_id")
    identity_id: int | None = None
    if identity_id_str not in (None, ""):
        try:
            identity_id = int(identity_id_str)
        except ValueError:
            return jsonify({"error": "identity_id must be an integer"}), 400

    observations = person_repository.get_all_person_observations(
        unmapped_only=unmapped_only,
        identity_id=identity_id,
    )
    identity_options = person_repository.get_identity_options()
    return render_template(
        "person_observations.html",
        observations=observations,
        identity_options=identity_options,
        unmapped_only=unmapped_only,
        selected_identity_id=identity_id,
    )


@app.route("/person_observations/<int:person_id>/identity", methods=["POST"])
def update_person_observation_identity(person_id: int):
    data = request.get_json(silent=True) or {}
    identity_id_value = data.get("identity_id")

    if identity_id_value in (None, "", "null"):
        identity_id = None
    else:
        try:
            identity_id = int(identity_id_value)
        except (TypeError, ValueError):
            return jsonify({"error": "identity_id must be an integer or null"}), 400

    person = person_repository.get_person_observation_by_id(person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    if identity_id is not None:
        identity = person_repository.get_person_identity_by_id(identity_id)
        if not identity:
            return jsonify({"error": "Identity not found"}), 404
        identity_name = identity.name
    else:
        identity_name = None

    person_repository.update_person_observation_identity(person_id, identity_id)

    return jsonify(
        {
            "status": "ok",
            "person_id": person_id,
            "identity_id": identity_id,
            "identity_name": identity_name,
        }
    )


@app.route("/persons/create_mapping", methods=["POST"])
def create_person_mapping():
    name = request.form.get("name")
    person_ids_str = request.form.get("person_ids", "")

    if not name or not person_ids_str:
        return jsonify({"error": "Name and person_ids are required"}), 400

    person_ids = [int(id) for id in person_ids_str.split(",")]

    person_repository.add_person_identity(name, person_ids)

    return redirect(url_for("index"))


@app.route("/person_identities/<int:mapping_id>")
def person_identity_detail(mapping_id):
    mapping = person_repository.get_person_identity_by_id(mapping_id)
    if not mapping:
        return "Identity not found", 404
    return render_template("person_identity_detail.html", mapping=mapping)


@app.route("/person_identities")
def person_identities_page():
    mappings = sorted(
        person_repository.get_all_person_identities(),
        key=lambda m: len(m.person_obs),
        reverse=True,
    )
    return render_template("person_identities.html", mappings=mappings)


@app.route("/person_identities/delete/<int:mapping_id>", methods=["POST"])
def delete_person_identity(mapping_id):
    person_repository.delete_person_identity(mapping_id)
    return redirect(url_for("person_identities_page"))


@app.route("/person_identities/update_name/<int:mapping_id>", methods=["POST"])
def update_person_identity_name(mapping_id):
    new_name = request.form.get("name")
    if not new_name:
        return "Name is required", 400
    person_repository.update_person_identity_name(mapping_id, new_name)
    return redirect(url_for("person_identities_page"))


@app.route("/person_image/<int:person_id>")
def person_image_by_id(person_id: int):
    is_face = request.args.get("is_face")
    person = person_repository.get_person_observation_by_id(person_id)
    if not person:
        return "Person not found", 404

    try:
        root_dir = str(cfg.root_dir)
        image_path = os.path.join(root_dir, person.crop_path)
        if is_face:
            image_path = os.path.join(cfg.img_dir, "faces", f"{person.id}.jpg")
        return send_file(image_path, mimetype="image/jpeg")
    except FileNotFoundError:
        return "Image file not found", 404


@app.route("/image/<int:image_id>")
def image_by_id(image_id: int):
    image_row = image_repository.get_observation_by_id(image_id)
    if not image_row:
        return jsonify({"error": "Image not found"}), 404

    try:
        with open(image_row.image_path, "rb") as f:
            frame_bytes = f.read()
        image_base64 = base64.b64encode(frame_bytes).decode("utf-8")
        return jsonify({"image_data": image_base64, "caption": image_row.caption})
    except FileNotFoundError:
        return jsonify({"error": "Image file not found"}), 404


@app.route("/prompt", methods=["POST"])
def prompt():
    data = request.get_json()
    prompt_text = data.get("prompt")
    thread_id = data.get("thread_id")

    if not prompt_text:
        return jsonify({"error": "Prompt is required"}), 400

    llm_interface.invoke(prompt_text, thread_id)
    return jsonify({"status": "ok"})


@app.route("/image_content/<int:image_id>")
def image_content(image_id: int):
    image_row = image_repository.get_observation_by_id(image_id)
    if not image_row:
        return "Image not found", 404

    root_dir = str(cfg.root_dir)
    image_path = os.path.join(root_dir, image_row.image_path)
    try:
        return send_file(image_path, mimetype="image/jpeg")
    except FileNotFoundError:
        return "Image file not found", 404


@app.route("/image_log")
def image_log():
    identity_id_str = request.args.get("identity_id")
    query_text = request.args.get("query")
    threshold_str = request.args.get("threshold", "0.6")

    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = 0.6

    candidate_image_ids = None

    # 1. Filter by Person Identity
    if identity_id_str == "any":
        candidate_image_ids = person_repository.get_unique_image_ids_with_persons()
    elif identity_id_str == "unmapped":
        candidate_image_ids = person_repository.get_image_ids_with_unmapped_persons()
    elif identity_id_str and identity_id_str != "all":
        try:
            identity_id = int(identity_id_str)
            obs = person_repository.get_person_observations_by_identity_id(identity_id)
            candidate_image_ids = list({o.image_id for o in obs})
        except ValueError:
            pass  # Invalid ID, ignore filter

    # If "any" or specific ID returned no images, we can stop early if we wanted,
    # but let's stick to the logic: empty list means no images found for that filter.
    # However, None means "no filter applied".

    # 2. Filter by Text Query
    images: list[Observation] = []
    if query_text:
        embedding = get_sentence_embedding([query_text], sentence_model, query=True)
        # embedding is (1, 384), we need a flat list/array for the repo
        images = image_repository.search_images_by_text(
            embedding[0], threshold, filter_image_ids=candidate_image_ids
        )
        # If candidate_image_ids was not None but empty, search_images_by_text handles it
        # (if implemented to respect empty list).
        # Actually my repo implementation `if filter_image_ids is not None` handles non-empty lists correctly.
        # But if candidate_image_ids is [], `id = ANY([])` returns nothing, which is correct.

    else:
        # No text query, just fetch based on person filter
        if candidate_image_ids is not None:
            if candidate_image_ids:
                images = image_repository.get_images_by_ids(
                    candidate_image_ids, limit=None
                )
            else:
                images = []
        else:
            images = image_repository.get_all_images_chronological()

    identities = person_repository.get_all_person_identities()

    # Prepare person observations for display
    if not images:
        return render_template(
            "image_log.html",
            images_by_day={},
            total_images_count=0,
            person_obs_by_image={},
            identities=identities,
            selected_identity_id=identity_id_str,
            query=query_text,
            threshold=threshold,
        )

    image_ids = [img.id for img in images]
    person_obs = person_repository.get_person_observations_by_image_ids(image_ids)

    person_obs_by_image = {}
    for obs in person_obs:
        if obs.image_id not in person_obs_by_image:
            person_obs_by_image[obs.image_id] = []
        person_obs_by_image[obs.image_id].append(obs)

    # Sort images by date to be sure
    images.sort(key=lambda x: x.creation_timestamp, reverse=True)

    # Group images by day
    images_by_day = {}
    for image in images:
        day = image.creation_timestamp.strftime("%Y-%m-%d")
        if day not in images_by_day:
            images_by_day[day] = []
        images_by_day[day].append(image)

    return render_template(
        "image_log.html",
        images_by_day=images_by_day,
        total_images_count=len(images),
        person_obs_by_image=person_obs_by_image,
        identities=identities,
        selected_identity_id=identity_id_str,
        query=query_text,
        threshold=threshold,
    )


if __name__ == "__main__":
    task_dir = cfg.eval_dir / "live" / "tasks"
    log_dir = task_dir / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler - INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler - DEBUG
    file_handler = logging.FileHandler(
        log_dir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_eval.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Nala module logging - DEBUG
    logging.getLogger("humembr").setLevel(logging.DEBUG)

    # External libraries suppression
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    app.run(host="0.0.0.0", port=5050)
