-- migrate:up
CREATE EXTENSION vector;

CREATE TABLE
    image_queue (
        id SERIAL PRIMARY KEY,
        creation_timestamp TIMESTAMPTZ NOT NULL,
        image_path TEXT NOT NULL,
        caption TEXT NOT NULL,
        caption_vector vector (384),
        waypoint TEXT
    );

-- migrate:down

DROP EXTENSION vector;
DROP TABLE image_queue;