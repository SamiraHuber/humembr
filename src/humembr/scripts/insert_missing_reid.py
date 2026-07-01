from psycopg.rows import dict_row
from tqdm import tqdm

from humembr.db.repositories import image_repository
from humembr.db.repositories.util import get_db_connection
from humembr.processing import reid_matching_service
from humembr.processing.person_processor import PersonProcessor
from humembr.util.device import get_device


def main():
    person_processor = PersonProcessor(get_device())
    all_imgs = image_repository.get_all_images()
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT image_id FROM person_observations")
        already_processed_img_id: list[int] = [
            row["image_id"] for row in cursor.fetchall()
        ]

    img_to_process = []
    for img in all_imgs:
        if img.id in already_processed_img_id:
            continue
        img_to_process.append(img)

    for img in tqdm(img_to_process):
        person_processor.process_person_in_image(img.image_path, img.id)

    reid_matching_service.run_reid_matching()


if __name__ == "__main__":
    main()
