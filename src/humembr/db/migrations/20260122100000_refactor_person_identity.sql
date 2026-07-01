-- migrate:up
ALTER TABLE person_observations ADD COLUMN person_identity_id INTEGER REFERENCES person_identity(id) ON DELETE SET NULL;

UPDATE person_observations
SET person_identity_id = person_identity.id
FROM person_identity
WHERE person_observations.id = ANY(person_identity.person_ids);

ALTER TABLE person_identity DROP COLUMN person_ids;

-- migrate:down
ALTER TABLE person_identity ADD COLUMN person_ids INTEGER[];

UPDATE person_identity
SET person_ids = ARRAY(
    SELECT id
    FROM person_observations
    WHERE person_identity_id = person_identity.id
);

ALTER TABLE person_observations DROP COLUMN person_identity_id;
