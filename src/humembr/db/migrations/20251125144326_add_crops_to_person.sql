-- migrate:up
ALTER TABLE persons ADD COLUMN crop_path TEXT;
ALTER TABLE persons ADD COLUMN keypoints JSONB;


-- migrate:down
ALTER TABLE persons DROP COLUMN crop_path;
ALTER TABLE persons DROP COLUMN keypoints;

