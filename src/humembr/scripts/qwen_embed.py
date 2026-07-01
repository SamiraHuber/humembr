import numpy as np
import ollama
import tqdm
from psycopg.rows import dict_row

from humembr.db.repositories.util import get_db_connection
from humembr.util.config import load_config


def main():
    cfg = load_config()
    BATCH_SIZE = 5

    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        query = "SELECT id, caption FROM image_queue WHERE caption_vector IS NULL"

        cursor.execute(query)
        items = cursor.fetchall()

    for i in tqdm.tqdm(range(0, len(items), BATCH_SIZE)):
        batch = items[i : i + BATCH_SIZE]
        sentences = [item["caption"] for item in batch]
        embeddings = [
            ollama.embed(model="qwen3-embedding:8b", input=s)["embeddings"]
            for s in sentences
        ]
        embeddings = np.array(embeddings).squeeze().tolist()
        with conn.cursor() as cursor:
            for item, emb in zip(batch, embeddings):
                update_query = """
                    UPDATE image_queue
                    SET caption_vector = %s
                    WHERE id = %s
                """
                cursor.execute(update_query, (emb, item["id"]))


if __name__ == "__main__":
    main()
