-- migrate:up
ALTER TABLE image_queue ADD COLUMN rotation real;


-- migrate:down
ALTER TABLE image_queue DROP COLUMN rotation;

