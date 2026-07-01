-- migrate:up
ALTER TABLE persons RENAME TO person_observations;
ALTER TABLE person_search_mapping RENAME TO person_identity;


-- migrate:down
ALTER TABLE person_observations RENAME TO persons;
ALTER TABLE person_identity RENAME TO person_search_mapping;
