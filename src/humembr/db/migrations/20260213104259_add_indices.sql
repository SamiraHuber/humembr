-- migrate:up
CREATE INDEX ON image_queue USING hnsw (caption_vector vector_cosine_ops);
CREATE INDEX ON person_observations USING hnsw (reid_vector vector_cosine_ops);
CREATE INDEX ON person_observations USING hnsw (face_vector vector_cosine_ops);


-- migrate:down

DROP INDEX person_observations_face_vector_idx;
DROP INDEX person_observations_reid_vector_idx;