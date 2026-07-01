-- migrate:up
ALTER TABLE persons ADD COLUMN neg_keypoints JSONB;


-- migrate:down
ALTER TABLE persons DROP COLUMN neg_keypoints;

