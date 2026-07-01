-- migrate:up
ALTER TABLE cmd_queue
ADD COLUMN direction TEXT CHECK (direction in ('front', 'back', 'left', 'right'));

-- migrate:down
ALTER TABLE cmd_queue
DROP COLUMN direction;