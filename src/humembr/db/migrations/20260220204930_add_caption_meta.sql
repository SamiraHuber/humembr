-- migrate:up
alter table image_queue add column caption_model text not null CHECK (length(caption_model) > 0) default 'unknown';
alter table image_queue add column caption_prompt text not null CHECK (length(caption_prompt) > 0) default 'unknown';

-- migrate:down

alter table image_queue drop column caption_model;
alter table image_queue drop column caption_prompt;