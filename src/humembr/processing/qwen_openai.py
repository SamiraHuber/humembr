import base64
import mimetypes
import os
import time
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)

from openai import OpenAI

from humembr.db.repositories.image_repository import (
    get_uncaptioned_observations,
    update_image_caption,
)
from humembr.processing.sentence import get_sentence_embedding, get_sentence_model
from humembr.util.config import load_config

NUM_WORKERS = 8
BATCH_SIZE = 8


def encode_image(image_path):
    """Encodes a local image to base64."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # Default fallback

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"


def vllm_caption(image_path, sentence_model, client, cfg):
    start_time = time.time()
    try:
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model=cfg.perception.caption_model,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": cfg.perception.caption_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        },
                    ],
                }
            ],
        )

        content = response.choices[0].message.content
        assert content, "response was empty!"

        duration = time.time() - start_time
        emb = get_sentence_embedding([content], sentence_model)[0]
        return duration, content, emb
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        traceback.print_exc()
        return None, None, None


def process_image(image, sentence_model, client, cfg):
    duration, new_caption, new_emb = vllm_caption(
        image.image_path, sentence_model, client, cfg
    )

    if duration is not None and new_caption is not None and new_emb is not None:
        print(
            f"{os.path.basename(image.image_path)}: '{new_caption}' - processed in {duration:.2f}s"
        )
        update_image_caption(
            image.id,
            new_caption,
            new_emb.tolist(),
            cfg.perception.caption_model,
            cfg.perception.caption_prompt,
        )
        return duration


def main():
    sentence_model = get_sentence_model()
    cfg = load_config()
    client = OpenAI(base_url=cfg.perception.caption_url, api_key="EMPTY")
    print(f"Caption worker started with {NUM_WORKERS} workers (OpenAI/vLLM backend)")
    start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_image = {}
        done_imgs = 0
        agg_time = 0.0

        while True:
            # Fill up workers if possible
            while len(future_to_image) < NUM_WORKERS:
                needed = NUM_WORKERS - len(future_to_image)
                exclude = [img.id for img in future_to_image.values()]
                new_images = get_uncaptioned_observations(n=needed, exclude_ids=exclude)

                if not new_images:
                    break

                print(f"Found {len(new_images)} new images to process.")
                for image in new_images:
                    future = executor.submit(
                        process_image, image, sentence_model, client, cfg
                    )
                    future_to_image[future] = image

            # If no work at all, sleep
            if not future_to_image:
                print("No uncaptioned images found. Waiting...")
                time.sleep(5)
                continue

            # Wait for at least one to complete
            done, _ = wait(future_to_image.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                image = future_to_image.pop(future)
                try:
                    duration = future.result()
                    if duration:
                        agg_time += duration
                        done_imgs += 1

                        minutes_since_start = max(1.0, (time.time() - start) // 60)
                        print(
                            f"avg {done_imgs / minutes_since_start:.2f} captions per minutes, running for {minutes_since_start} with {done_imgs} captions"
                        )
                        print(
                            f"avg processing time: {agg_time / max(done_imgs, 1):.2f}s"
                        )
                except Exception as exc:
                    traceback.print_exc()
                    print(f"Image {image.id} generated an exception: {exc}")


if __name__ == "__main__":
    main()
