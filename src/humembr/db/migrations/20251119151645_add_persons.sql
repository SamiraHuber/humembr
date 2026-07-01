-- migrate:up
CREATE TABLE
    persons (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        image_id INTEGER REFERENCES image_queue (id),
        reid_vector vector (512)
    );

-- migrate:down
DROP TABLE persons;