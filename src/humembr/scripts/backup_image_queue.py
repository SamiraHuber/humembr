import argparse
import json
import os
import shutil
import tarfile
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import tqdm
from psycopg.rows import dict_row

from humembr.db.repositories import image_repository
from humembr.db.repositories.util import get_db_connection
from humembr.eval.question.util import check_unique_model_and_prompt
from humembr.processing.sentence import get_sentence_embedding, get_sentence_model
from humembr.util.config import load_config


def backup_image_queue():
    cfg = load_config()

    check_unique_model_and_prompt()

    caption_prompt = cfg.perception.caption_prompt

    backup_root = (
        f"{cfg.perception.caption_model}_{caption_prompt[:15]}_{caption_prompt[-15:]}".replace(
            " ", "_"
        )
        .replace("/", "_")
        .replace(".", "_")
    )
    images_dir = os.path.join(backup_root, "images")

    os.makedirs(images_dir, exist_ok=True)

    print(f"Starting backup to {backup_root}...")

    conn = get_db_connection()
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            # Select all columns except caption_vector
            cursor.execute("""
                SELECT 
                    iq.id, 
                    iq.creation_timestamp, 
                    iq.image_path, 
                    iq.caption, 
                    iq.waypoint as waypoint_id, 
                    w.name as waypoint_name,
                    iq.camera_id, 
                    iq.rotation,
                    iq.caption_model,
                    iq.caption_prompt
                FROM image_queue iq
                LEFT JOIN waypoints w ON iq.waypoint = w.id
            """)
            rows = cursor.fetchall()

            processed_rows = []

            print(f"Found {len(rows)} records to backup.")

            for row in tqdm.tqdm(rows):
                original_image_path = row["image_path"]
                source_path = None

                # Try finding the file (absolute or relative to CWD)
                if os.path.exists(original_image_path):
                    source_path = original_image_path
                # Try finding relative to img_dir
                elif os.path.exists(os.path.join(cfg.img_dir, original_image_path)):
                    source_path = os.path.join(cfg.img_dir, original_image_path)

                if source_path:
                    image_filename = os.path.basename(source_path)
                    # Handle potential filename collisions if needed, but assuming unique for now or acceptable overwrite if paths are diff but name same (unlikely for image queue)
                    # Actually, if image_path is full path, basenames might collide if they are in different folders.
                    # But typically image_queue images might be timestamped or unique.
                    # Let's ensure uniqueness if needed.
                    # For now, just copy.

                    target_path = os.path.join(images_dir, image_filename)

                    # If target exists, maybe append id to filename
                    if os.path.exists(target_path):
                        name, ext = os.path.splitext(image_filename)
                        image_filename = f"{name}_{row['id']}{ext}"
                        target_path = os.path.join(images_dir, image_filename)

                    shutil.copy2(source_path, target_path)

                    # Update path in row to be relative to the json file
                    # The prompt says "the image path in the json should reference the copied image location"
                    row["image_path"] = os.path.join("images", image_filename)
                else:
                    print(
                        f"Warning: Image not found: {original_image_path}. Keeping original path in JSON."
                    )

                processed_rows.append(row)

            # Write JSON
            json_path = os.path.join(backup_root, "data.json")
            with open(json_path, "w") as f:
                json.dump(processed_rows, f, indent=2, default=str)

            print(f"Data written to {json_path}")

            # Compress
            archive_name = f"{backup_root}.tar.gz"
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(backup_root, arcname=backup_root)

            print(f"Backup compressed to {archive_name}")
            mv_archive = str(target_dir / archive_name)
            shutil.move(archive_name, mv_archive)

            # Cleanup
            shutil.rmtree(backup_root)
            print(f"Temporary directory {backup_root} removed.")

    finally:
        conn.close()


def restore_image_queue(archive_path, update_captions=False):
    sentence_model = get_sentence_model()
    cfg = load_config()
    conn = get_db_connection()

    print(f"Restoring from {archive_path}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmp_dir)

        # Find data.json
        data_json_paths = list(Path(tmp_dir).glob("**/data.json"))
        if not data_json_paths:
            print("Error: data.json not found in archive.")
            return

        data_json_path = data_json_paths[0]
        root_dir = data_json_path.parent

        with open(data_json_path, "r") as f:
            data = json.load(f)

        print(f"Found {len(data)} records to restore.")

        updated_count = 0
        inserted_count = 0

        with conn.cursor() as cursor:
            for row in tqdm.tqdm(data):
                # row['image_path'] is relative to data.json, e.g., "images/foo.jpg"
                src_image_path = root_dir / row["image_path"]

                if src_image_path.exists():
                    # Destination: cfg.img_dir + filename
                    filename = os.path.basename(row["image_path"])

                    dt = datetime.fromisoformat(row["creation_timestamp"])
                    date = dt.strftime("%Y%m%d")
                    date_dir = Path(cfg.img_dir) / date
                    dest_path = date_dir / filename

                    # Ensure cfg.img_dir exists
                    date_dir.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(src_image_path, dest_path)

                    cap_emb = get_sentence_embedding(row["caption"], sentence_model)

                    img = image_repository.find_by_image_path(filename)
                    try:
                        if img and update_captions:
                            print(f"Image with path {filename} already exists in DB.")
                            update_image_caption(cursor, row, cap_emb, img)
                            updated_count += 1
                        else:
                            insert_new_caption(cfg, cursor, row, dest_path, cap_emb)
                            inserted_count += 1
                    except Exception as e:
                        traceback.print_exc()
                        print(f"Error inserting/updating DB for image {filename}: {e}")
                        return
                else:
                    print(f"Warning: Could not find file {src_image_path}")

            conn.commit()
            print(
                f"Restore complete. Updated: {updated_count}, Inserted: {inserted_count}, Total: {len(data)}."
            )


def insert_new_caption(cfg, cursor, row, dest_path, cap_emb):
    cursor.execute(
        """
        INSERT INTO image_queue (
        creation_timestamp, image_path, caption, caption_vector, waypoint, camera_id, rotation, caption_model, caption_prompt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            row["creation_timestamp"],
            str(dest_path),
            row["caption"],
            cap_emb,
            row["waypoint_id"],
            row["camera_id"],
            row["rotation"],
            row["caption_model"],
            row["caption_prompt"],
        ),
    )


def update_image_caption(cursor, row, cap_emb, img):
    cursor.execute(
        """
        UPDATE image_queue 
        SET caption = %s, caption_vector = %s, caption_model = %s, caption_prompt = %s
        WHERE id = %s
        """,
        (
            row["caption"],
            cap_emb,
            row["caption_model"],
            row["caption_prompt"],
            img.id,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup or Restore Image Queue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Backup command
    backup_parser = subparsers.add_parser(
        "backup", help="Backup image queue to a tar.gz file"
    )

    # Restore command
    restore_parser = subparsers.add_parser(
        "restore", help="Restore image queue from a tar.gz file"
    )
    restore_parser.add_argument("--archive_path", help="Path to the backup tar.gz file")
    # bool flag is captions should be updated
    restore_parser.add_argument(
        "--update_captions",
        action="store_true",
        help="Whether to update captions and embeddings during restore",
    )

    args = parser.parse_args()

    cfg = load_config()

    target_dir = Path(cfg.backup_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "backup":
        backup_image_queue()
    elif args.command == "restore":
        restore_image_queue(args.archive_path, update_captions=args.update_captions)
