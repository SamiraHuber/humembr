-- migrate:up
CREATE TABLE
    cmd_queue (
        id SERIAL PRIMARY KEY,
        waypoint_name TEXT NOT NULL,
        creation_timestamp timestamptz not null default now (),
        status TEXT NOT NULL DEFAULT 'PENDING',
        prompt TEXT NOT NULL
    );

-- migrate:down
DROP TABLE cmd_queue;