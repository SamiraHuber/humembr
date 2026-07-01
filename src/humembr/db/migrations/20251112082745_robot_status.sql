-- migrate:up

CREATE TABLE robot_status (
    id SERIAL PRIMARY KEY,
    status TEXT NOT NULL
);


-- migrate:down
DROP TABLE robot_status;

