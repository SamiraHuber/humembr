-- migrate:up
ALTER TABLE person_observations DROP COLUMN name;

-- migrate:down
ALTER TABLE person_observations ADD COLUMN name TEXT;
