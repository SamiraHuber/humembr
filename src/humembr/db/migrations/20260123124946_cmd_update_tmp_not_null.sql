-- migrate:up
ALTER TABLE cmd_queue ALTER COLUMN updated_timestamp SET NOT NULL;
ALTER TABLE cmd_queue ALTER COLUMN updated_timestamp SET DEFAULT NOW();

-- migrate:down

