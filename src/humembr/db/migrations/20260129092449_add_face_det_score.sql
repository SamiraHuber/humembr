-- migrate:up
ALTER TABLE person_observations ADD COLUMN face_det_score FLOAT;



-- migrate:down
ALTER TABLE person_observations DROP COLUMN face_det_score;

