-- migrate:up
UPDATE image_queue
SET
    caption = null,
    caption_vector = null;

ALTER TABLE image_queue
ALTER COLUMN caption_vector TYPE vector (1024);

-- migrate:down
ALTER TABLE image_queue
ALTER COLUMN caption_vector TYPE vector (384);