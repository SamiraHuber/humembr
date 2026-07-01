import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import tqdm
from google import genai

from humembr.agent.interview_agent import InterviewAnswer, QuestionType
from humembr.eval.question.main import evaluate_answer
from humembr.eval.question.util import Question, validate_df
from humembr.util.config import load_config

logger = logging.getLogger(__name__)


def main(cfg):
    current_dir = cfg.root_dir / "src/humembr/eval/question"
    question_file = current_dir / "question.csv"

    logger.info("Reading questions from %s", question_file)
    batch_response_dir = (
        cfg.eval_dir / "question_eval" / "batch_responses" / "questions"
    )
    id_to_interview_response = get_answer_map(batch_response_dir)
    token_batch = get_total_token(batch_response_dir)

    # read json from file
    genai_client = genai.Client()

    df = validate_df(question_file)
    assert df is not None, "Failed to read question DataFrame, check logs for details"

    correct_dict: dict[QuestionType, int] = defaultdict(int)
    counter_dict: dict[QuestionType, int] = defaultdict(int)
    error_dict: dict[QuestionType, float] = defaultdict(int)
    correct_days_dict: dict[int, int] = defaultdict(int)

    for id, a in id_to_interview_response.items():
        question_id = id
        interview_answer = a
        question_row = df[df["id"] == question_id].iloc[0]
        question = Question(**question_row.to_dict())

        error, is_correct = evaluate_answer(
            question, interview_answer, cfg, genai_client
        )

        correct_dict[question.type] += is_correct
        counter_dict[question.type] += 1
        error_dict[question.type] += error
        correct_days_dict[question.multiple_days] += is_correct

        logger.info(
            "Question ID: %s | Error: %.4f | Correct: %s",
            question_id,
            error,
            is_correct,
        )

        logger.info("-" * 70)
        logger.info("")

    total_correct = sum(correct_dict.values())
    total_questions = sum(counter_dict.values())
    logger.info(
        f"Total correct: {total_correct}/{total_questions} ({total_correct / total_questions * 100:.2f}%)"
    )

    logger.info(f"Total question: {len(df)}")
    logger.info("Correct answer per question type:")
    for k, v in correct_dict.items():
        logger.info(f"\t{k}: {v}/{counter_dict[k]} ({v / counter_dict[k] * 100:.2f}%)")
    logger.info("Incorrect answer per question type:")
    logger.info("Avg. Error per question type:")
    for k, v in error_dict.items():
        logger.info(f"\t{k}: {v / counter_dict[k]:.2f}")
    logger.info("Correct answer per multiple days:")
    for k, v in correct_days_dict.items():
        logger.info(f"\t{k}: {v}")

    logger.info(f"Total token used: {token_batch}")
    logger.info("Done.")


def get_total_token(batch_response_dir: Path):
    total_token = 0
    for f in batch_response_dir.glob("*.json"):
        json_data = json.loads(f.read_text())
        if "request" in json_data:
            tokens = json_data["gemini_response"]["dest"]["inlined_responses"][0][
                "response"
            ]["usage_metadata"]["total_token_count"]
            total_token += tokens
        else:
            # is gemini api
            tokens = json_data["gemini_response"]["usage_metadata"]["total_token_count"]
            total_token += tokens
    return total_token


def get_answer_map(batch_response_dir) -> dict[str, InterviewAnswer]:
    id_to_interview_response: dict[str, InterviewAnswer] = {}
    for f in tqdm.tqdm(
        batch_response_dir.glob("*.json"), desc="Loading batch responses"
    ):
        json_data = json.loads(f.read_text())
        question_id = json_data["question_id"]
        interview_response = json_data["interview_response"]
        assert question_id is not None and interview_response is not None, (
            f"Missing question_id or interview_response in {f.name}"
        )
        id_to_interview_response[question_id] = InterviewAnswer(**interview_response)
    return id_to_interview_response


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
        str(log_dir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_batch_eval.log")
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

    main(cfg)
