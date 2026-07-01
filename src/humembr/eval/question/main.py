import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from google import genai

from humembr.agent.interview_agent import InterviewAnswer, InterviewInterface, QuestionType
from humembr.db.repositories import person_repository
from humembr.eval.question.util import (
    EvaluationResult,
    Question,
    TextAnswerEvalution,
    check_unique_model_and_prompt,
    validate_df,
)
from humembr.util.config import Config, load_config
from humembr.util.graph import get_spot_graph, get_waypoint_by_id, waypoint_distance_m

logger = logging.getLogger(__name__)
graph = get_spot_graph()


def main(response_dir: Path):
    cfg = load_config()
    current_dir = cfg.root_dir / "src/humembr/eval/question"
    question_file = current_dir / "question.csv"
    question_answer_dir = (
        response_dir / "questions" / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
    )
    if not question_answer_dir.exists():
        question_answer_dir.mkdir(parents=True, exist_ok=True)

    genai_client = genai.Client()
    logger.info("starting interview evaluation with:")
    logger.info("%s", cfg.evaluation)
    check_unique_model_and_prompt()
    has_reid_identity = person_repository.has_reid_identity_matching()
    logger.info(f"person reid identity matching enabled: {has_reid_identity}")

    agent_interface = InterviewInterface(cfg)
    logger.info(
        "Available tools: %s",
        ",".join(list(map(lambda t: t.name, agent_interface.tools))),
    )

    logger.info(f"Reading questions from {question_file}")

    df = validate_df(question_file)
    assert df is not None, "Failed to read question DataFrame, check logs for details"
    logger.info(df.info())

    correct_dict: dict[QuestionType, int] = defaultdict(int)
    counter_dict: dict[QuestionType, int] = defaultdict(int)
    error_dict: dict[QuestionType, float] = defaultdict(int)
    correct_days_dict: dict[int, int] = defaultdict(int)

    token_usage = 0
    start = time.time()

    for _, row in df[df["ready"]].iterrows():  # type: ignore
        try:
            question = Question(**row.to_dict())
            final_prompt = f"""
            Current time: {question.context_time}
            Question type: {question.type}
            Your question is: "{question.question}"
            """
            logger.info("Q: %s", final_prompt)

            response, meta_data = agent_interface.invoke(final_prompt)
            if meta_data["total_tokens"] > 500000:
                logger.info("big token usage detected, sleeping for one minute")
                time.sleep(60)

            error, is_correct = evaluate_answer(question, response, cfg, genai_client)
            logger.info(
                "Question: %s used %d tokens",
                question.question,
                meta_data["total_tokens"],
            )

            correct_dict[question.type] += is_correct
            counter_dict[question.type] += 1
            error_dict[question.type] += error
            correct_days_dict[question.multiple_days] += is_correct

            token_usage += meta_data["total_tokens"]

            entry = {
                "question_data": row.to_dict(),
                "interview_response": response.model_dump(),
                "meta_data": meta_data,
                "result": is_correct,
                "error": error,
            }

            with open(
                str(question_answer_dir / f"{question.id}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(entry, f, indent=2, default=str)

            logger.info("-" * 70)
            logger.info("")
        except Exception as e:
            logging.exception(e)
    total_correct = sum(correct_dict.values())
    total_questions = sum(counter_dict.values())
    logger.info(
        f"Total correct: {total_correct}/{total_questions} ({total_correct / total_questions * 100:.2f}%)"
    )

    logger.info(f"Total question: {len(df)}")
    logger.info("Correct answer per question type:")
    for k, v in correct_dict.items():
        logger.info(f"\t{k}: {v}/{counter_dict[k]} ({v / counter_dict[k] * 100:.2f}%)")
    logger.info("Correct answer per multiple days:")
    for k, v in correct_days_dict.items():
        logger.info(f"\t{k}: {v}")
    logger.info("Incorrect answer per question type:")
    logger.info("Avg. Error per question type:")
    for k, v in error_dict.items():
        logger.info(f"\t{k}: {v / counter_dict[k]:.2f}")

    logger.info(
        f"Total token usage: {token_usage} in {(time.time() - start) / 60.0:.2f} minutes"
    )
    logger.info("Done.")


def evaluate_answer(
    question: Question,
    interview_answer: InterviewAnswer,
    cfg: Config,
    genai_client: genai.Client,
) -> tuple[float, bool]:
    result: EvaluationResult | None = None

    match question.type:
        case "time":
            result = _eval_time(question, interview_answer, cfg)
        case "duration":
            result = _eval_duration(question, interview_answer, cfg)
        case "spatial":
            result = _eval_spatial(question, interview_answer, cfg)
        case "semantic":
            result = _eval_text(question, interview_answer, genai_client)
        case "bool":
            result = _eval_bool(question, interview_answer)
        case "person":
            result = _eval_person(question, interview_answer)
        case _:
            logger.warning(f"Unknown question type: {question.type}")
            return (0.0, False)

    if result is None:
        logger.warning(
            f"{question.type}: {question.question}\n {interview_answer}\n"
            "Was not answered correctly, expected answer but was None or invalid"
        )
        return (0.0, False)

    logger.info("Question: %s", question.question)
    logger.info("Wanted answer: %s", result.wanted_answer)
    logger.info("Given answer: %s", result.given_answer)
    logger.info("Reason: %s", interview_answer.reason)
    logger.info("Error: %.2f, is correct? %s", result.raw_metric, result.is_correct)

    return (result.error, result.is_correct)


def _eval_time(
    question: Question, interview_answer: InterviewAnswer, cfg: Config
) -> EvaluationResult | None:
    if interview_answer.timeAnswer is None:
        return None

    error_thresh = cfg.evaluation.question_time_diff
    correct_duration = datetime.strptime(question.answer, "%Y-%m-%d %H:%M:%S")
    answer_duration = interview_answer.timeAnswer.replace(tzinfo=None)
    min_diff = abs((correct_duration - answer_duration).total_seconds() / 60.0)

    return EvaluationResult(
        error=abs(min_diff - error_thresh),
        is_correct=min_diff < error_thresh,
        raw_metric=min_diff,
        given_answer=str(interview_answer.timeAnswer),
        wanted_answer=question.answer,
    )


def _eval_duration(
    question: Question, interview_answer: InterviewAnswer, cfg: Config
) -> EvaluationResult | None:
    if interview_answer.durationAnswer is None:
        return None

    error_thresh = cfg.evaluation.question_duration_diff
    correct_duration = float(question.answer)  # in minutes
    answer_duration = interview_answer.durationAnswer  # in minutes
    min_diff = abs(correct_duration - answer_duration)

    return EvaluationResult(
        error=abs(min_diff - error_thresh),
        is_correct=min_diff < error_thresh,
        raw_metric=min_diff,
        given_answer=str(interview_answer.durationAnswer),
        wanted_answer=question.answer,
    )


def _eval_spatial(
    question: Question, interview_answer: InterviewAnswer, cfg: Config
) -> EvaluationResult | None:
    if interview_answer.spatialAnswer is None:
        return None

    waypoint_pattern = re.compile(r"^waypoint_(\d+)$")
    error_thresh = cfg.evaluation.question_spatial_diff
    correct_waypoint = get_safe_waypoint(question.answer, waypoint_pattern)
    answer_spatial = get_safe_waypoint(interview_answer.spatialAnswer, waypoint_pattern)
    if answer_spatial is None:
        logging.warning(
            f"Could not find waypoint for given answer: {interview_answer.spatialAnswer}"
        )
        return None
    dist_diff = waypoint_distance_m(
        get_spot_graph(),
        correct_waypoint.id,  # type: ignore
        answer_spatial.id,  # type: ignore
    )

    if dist_diff is None:
        logger.warning(
            f"error computing path dist, got none. Correct: {correct_waypoint}, Ans: {answer_spatial}"
        )
        return None

    return EvaluationResult(
        error=abs(dist_diff - error_thresh),
        is_correct=dist_diff < error_thresh,
        raw_metric=dist_diff,
        given_answer=interview_answer.spatialAnswer,
        wanted_answer=question.answer,
    )


def _eval_text(
    question: Question,
    interview_answer: InterviewAnswer,
    genai_client: genai.Client,
) -> EvaluationResult | None:
    if interview_answer.semanticAnswer is None:
        return None

    prompt = f"""
    You are interviewer that needs to validate questions of a questionnaire.
    You will get a correct answer and a given answer.
    Decide if the answer is correct and give a short reason for your decision.

    Given answer: {interview_answer.semanticAnswer}
    Correct answer: {question.answer}
    """
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": TextAnswerEvalution.model_json_schema(),
        },
    )
    judgement = TextAnswerEvalution.model_validate_json(response.text)  # type: ignore
    logger.info(
        "semantic answer judgement decision: %s, reason: %s",
        judgement.is_correct,
        judgement.reason,
    )

    return EvaluationResult(
        error=0 if judgement.is_correct else 1,
        is_correct=judgement.is_correct,
        raw_metric=0 if judgement.is_correct else 1,
        given_answer=interview_answer.semanticAnswer,
        wanted_answer=question.answer,
    )


def _eval_bool(
    question: Question, interview_answer: InterviewAnswer
) -> EvaluationResult | None:
    if interview_answer.boolAnswer is None:
        return None

    correct_bool = question.answer.lower() == "true"
    answer_bool = interview_answer.boolAnswer
    bool_diff = 0.0 if (correct_bool == answer_bool) else 1.0

    return EvaluationResult(
        error=bool_diff,
        is_correct=answer_bool == correct_bool,
        raw_metric=bool_diff,
        given_answer=str(interview_answer.boolAnswer),
        wanted_answer=question.answer,
    )


def _eval_person(
    question: Question, interview_answer: InterviewAnswer
) -> EvaluationResult | None:
    if interview_answer.personAnswer is None:
        return None

    known_persons = list(
        map(lambda p: p.lower(), person_repository.get_all_identity_names())
    )
    correct_person = set(map(lambda p: p.lower().strip(), question.answer.split(",")))
    for p in correct_person:
        assert p.lower() in known_persons, (
            f"ground truth wants person '{p}', but was not found in known persons: {known_persons}"
        )

    answer_person = set(
        map(lambda p: p.lower().strip(), interview_answer.personAnswer.split(","))
    )

    person_diff = len(correct_person - answer_person)

    return EvaluationResult(
        error=person_diff,
        is_correct=person_diff == 0,
        raw_metric=person_diff,
        given_answer=interview_answer.personAnswer,
        wanted_answer=question.answer,
    )


def get_safe_waypoint(waypoint: str, waypoint_pattern: re.Pattern):
    assert waypoint_pattern.match(waypoint.lower().strip()), (
        f"expected waypoint to match the pattern: ^waypoint_(\\d+)$, but was {waypoint}"
    )
    return get_waypoint_by_id(graph, waypoint.lower())


if __name__ == "__main__":
    cfg = load_config()
    response_dir = cfg.eval_dir / "question_eval" / "agent_responses"
    log_dir = response_dir / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler - INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler - DEBUG
    file_handler = logging.FileHandler(
        log_dir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_eval.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Nala module logging - DEBUG
    logging.getLogger("humembr").setLevel(logging.DEBUG)

    # External libraries suppression
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    main(response_dir)
