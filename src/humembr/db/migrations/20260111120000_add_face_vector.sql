-- migrate:up
ALTER TABLE persons ADD COLUMN face_vector vector(512);

-- migrate:down
ALTER TABLE persons DROP COLUMN face_vector;
