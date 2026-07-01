import json
import logging
import time
from dataclasses import asdict
from datetime import datetime

import pandas
from google import genai
from google.genai import types
from tqdm import tqdm

from humembr.agent.interview_agent import SYSTEM_PROMPT, InterviewAnswer
from humembr.agent.tool_utils import format_observations
from humembr.db.repositories import image_repository
from humembr.db.repositories.image_repository import Observation
from humembr.eval.question.util import Question, check_unique_model_and_prompt, validate_df
from humembr.util.config import Config, load_config

logger = logging.getLogger(__name__)


def process_questions(
    client: genai.Client,
    model: str,
    df: pandas.DataFrame,
    obs: list[Observation],
    cfg: Config,
):
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building batch JSON"):
        question = Question(**row.to_dict())  # type: ignore

        request = build_inline_request(obs, question)

        inline_batch = client.batches.create(
            model=model,
            src=[request],  # type: ignore
            config={"display_name": f"inlined-spot-baseline-{question.id}"},
        )

        assert inline_batch.name is not None, (
            "Inline batch creation failed, no batch name returned"
        )
        logger.info("Uploaded batch: %s", inline_batch.name)

        job_name = inline_batch.name
        batch_job = poll_batch_status(client, job_name)

        if not batch_job.state.name == "JOB_STATE_SUCCEEDED":  # type: ignore
            logger.error(f"Job did not succeed. Final state: {batch_job.state.name}")  # type: ignore
            if batch_job.error:
                logger.error(f"Error: {batch_job.error}")
            continue

        if not batch_job.dest or batch_job.dest.inlined_responses is None:
            logger.error("No results found (neither file nor inline).")
            continue

        inline_response = batch_job.dest.inlined_responses[0]

        if inline_response.error:
            logger.error(f"Error in response: {inline_response.error}")
            continue

        if not inline_response.response:
            logger.error("No response content found.")
            continue

        if not inline_response.response.text:
            logger.error("Response content is not parsable.")
            logger.error(f"Raw response: {inline_response.response}")
            continue

        interview_response = InterviewAnswer.model_validate_json(
            inline_response.response.text
        )

        out = {
            "question_id": question.id,
            "question": asdict(question),
            "request": request,
            "interview_response": interview_response.model_dump(),
            "gemini_response": batch_job.model_dump(),
        }

        output_file = (
            cfg.eval_dir / "question_eval" / "batch_responses" / f"{question.id}.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2, default=str)
            logger.info(f"Saved response for question {question.id} to {output_file}")


def build_inline_request(obs, question):
    relevant_obs = [
        o
        for o in obs
        if o.creation_timestamp.replace(tzinfo=None) <= question.context_time
    ]
    formatted_obs = format_observations(relevant_obs)

    prompt_text = f"""
Current time: {question.context_time}
Question type: {question.type}
Your question is: "{question.question}"
Your context:
{formatted_obs}
            """.strip()

    # Build the request as a GenerateContentRequest-like dict.
    request = {
        "contents": [{"parts": [{"text": prompt_text}], "role": "user"}],
        "config": {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "response_mime_type": "application/json",
            "response_schema": InterviewAnswer,
        },
    }

    return request


def poll_batch_status(client, job_name) -> types.BatchJob:
    completed_states = set(
        [
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        ]
    )
    logger.info(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)  # Initial get
    while batch_job.state.name not in completed_states:  # type: ignore
        logger.info(f"Current state: {batch_job.state.name}")  # type: ignore
        time.sleep(30)
        batch_job = client.batches.get(name=job_name)

    logger.info(f"Job finished with state: {batch_job.state.name}")  # type: ignore
    if batch_job.state.name == "JOB_STATE_FAILED":  # type: ignore
        logger.error(f"Error: {batch_job.error}")
    return batch_job


def main():
    # setup dirs
    cfg = load_config()
    current_dir = cfg.root_dir / "src/humembr/eval/question"
    question_file = current_dir / "question.csv"
    batch_response_dir = cfg.eval_dir / "question_eval" / "batch_responses"
    if not batch_response_dir.exists():
        batch_response_dir.mkdir(parents=True, exist_ok=True)

    # check already processed questions to avoid re-processing
    processed_questions = set(f.stem for f in batch_response_dir.glob("*.json"))
    logger.info(f"Already processed questions: {len(processed_questions)}")
    logger.info("Reading questions from %s", question_file)

    # load and validate question DataFrame
    df = validate_df(question_file)
    assert df is not None, "Failed to read question DataFrame, check logs for details"
    check_unique_model_and_prompt()
    df = df[~df["id"].isin(processed_questions)]  # type: ignore
    logger.info(f"Questions to process after filtering: {len(df)}")

    # process questions
    client = genai.Client()
    model = "gemini-3-flash-preview"
    obs = image_repository.get_all_images()
    process_questions(client, model=model, obs=obs, df=df, cfg=cfg)  # type: ignore


if __name__ == "__main__":
    cfg = load_config()
    log_dir = cfg.eval_dir / "question_eval" / "batch_responses" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging (same as your original)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        str(
            log_dir
            / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_batch_creation.log"
        )
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("humembr").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    main()
