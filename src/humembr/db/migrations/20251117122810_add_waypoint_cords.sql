-- migrate:up

ALTER TABLE waypoints ADD COLUMN coords vector(3);


-- migrate:down

ALTER TABLE waypoints DROP COLUMN coords;

