-- migrate:up
ALTER TABLE image_queue ADD COLUMN camera_id TEXT;


-- migrate:down
ALTER TABLE image_queue DROP COLUMN camera_id;

