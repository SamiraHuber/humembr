-- migrate:up
ALTER TABLE image_queue ALTER COLUMN caption DROP NOT NULL;

-- migrate:down
UPDATE image_queue SET CAPTION = 'NO CAPTION' WHERE CAPTION IS NULL;
ALTER TABLE image_queue ALTER COLUMN caption SET NOT NULL;

