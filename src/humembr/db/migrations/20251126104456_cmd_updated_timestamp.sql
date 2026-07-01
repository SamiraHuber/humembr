-- migrate:up
ALTER TABLE cmd_queue ADD COLUMN updated_timestamp timestamptz;

-- migrate:down

ALTER TABLE cmd_queue DROP COLUMN updated_timestamp;
