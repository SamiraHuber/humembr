-- migrate:up
CREATE TABLE person_search_mapping (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        creation_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        updated_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW (),
        person_ids INTEGER[]
)

-- migrate:down
DROP TABLE person_search_mapping;

