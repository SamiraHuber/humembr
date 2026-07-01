-- migrate:up
ALTER TABLE cmd_queue ADD COLUMN rotation FLOAT;


-- migrate:down

