-- migrate:up
-- This migration assumes the foreign key constraint was created with the default name persons_image_id_fkey
ALTER TABLE persons DROP CONSTRAINT persons_image_id_fkey;
ALTER TABLE persons ADD CONSTRAINT persons_image_id_fkey FOREIGN KEY (image_id) REFERENCES image_queue (id) ON DELETE CASCADE;



-- migrate:down
ALTER TABLE persons DROP CONSTRAINT persons_image_id_fkey;
ALTER TABLE persons ADD CONSTRAINT persons_image_id_fkey FOREIGN KEY (image_id) REFERENCES image_queue (id);
