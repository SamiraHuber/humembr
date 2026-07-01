import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from humembr.agent.interview_agent import QuestionType
from humembr.db.repositories import image_repository

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    error: float
    is_correct: bool
    raw_metric: float
    given_answer: str
    wanted_answer: str


@dataclass
class Question:
    id: str
    question: str
    context_time: datetime
    type: QuestionType
    answer: str
    optimal_context: datetime
    ready: bool
    comment: str
    multiple_days: int


class TextAnswerEvalution(BaseModel):
    score: float = Field(
        description="Give the answer a score between 0.0 and 1.0. Where 1.0 is a perfect answer, 0.5 and above is still correct and 0.0 is completly wrong."
    )
    reason: str = Field(
        description="Give a short reason for why the answer is correct or incorrect"
    )
    is_correct: bool = Field(
        description="Does the given answer match the wanted answer (score above 0.5)"
    )


def validate_df(question_file) -> pd.DataFrame | None:
    df = pd.read_csv(
        question_file,
        sep=";",
        dtype={
            "type": "category",
        },
    )
    # check for NaN in ready column and print index of rows with NaN values
    if df["ready"].isna().any():  # type: ignore
        logger.error("Found NaN values in ready column, please check your csv file")
        logger.error(f"Rows with NaN ready: {df[df['ready'].isna()]}")
        return None

    df["ready"] = df["ready"].astype(bool)  # type: ignore

    df: pd.DataFrame = df[df["ready"]]  # type: ignore
    df["context_time"] = pd.to_datetime(df["context_time"], format="%Y-%m-%d %H:%M:%S")
    allowed_question_types = [
        "time",
        "duration",
        "spatial",
        "semantic",
        "bool",
        "person",
    ]

    is_valid = df["type"].isin(allowed_question_types).all()
    if not is_valid:  # type: ignore
        invalid_values = df.loc[~df["type"].isin(allowed_question_types), "type"]
        logger.error("invalid type values:", invalid_values.unique())
        return None

    # check for Nan in context_time column
    if df["context_time"].isna().any():  # type: ignore
        logger.error(
            "Found NaN values in context_time column, please check your csv file"
        )
        logger.error(f"Rows with NaN context_time: {df[df['context_time'].isna()]}")
        return None

    # check if all ids are valid uuidv4 via uuid.UUID, if not log an error with the invalid ids and return None
    def is_valid_uuidv4(id_str):
        try:
            val = uuid.UUID(id_str, version=4)
            return str(val) == id_str
        except ValueError:
            return False

    if not df["id"].apply(is_valid_uuidv4).all():  # type: ignore
        invalid_ids = df.loc[~df["id"].apply(is_valid_uuidv4), "id"]
        logger.error(
            "Found invalid uuidv4 ids in id column, please check your csv file"
        )
        logger.error(f"Invalid ids: {invalid_ids.unique()}")
        return None

    return df


def check_unique_model_and_prompt():
    models, prompts = image_repository.get_caption_models_and_prompts()
    assert len(models) == 1, (
        "expected exactly one caption model in the database, but found: %s" % models
    )
    assert len(prompts) == 1, (
        "expected exactly one caption prompt in the database, but found: %s" % prompts
    )
    logger.info(
        "using caption model: %s, prompt: %s", list(models)[0], list(prompts)[0]
    )
