import json
import logging
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import UsageMetadata
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from humembr.agent.tools import (
    build_person_summary_tool,
    query_available_persons,
    query_person_history,
    query_semantic_observations,
    query_waypoint_history,
)
from humembr.util.config import Config

QuestionType = Literal["semantic", "time", "duration", "spatial", "bool", "person"]


class InterviewAnswer(BaseModel):
    """Contact information for a person."""

    type: QuestionType = Field(
        description="The question type. Based on this the answer will be parsed"
    )
    reason: str = Field(description="reason for your submitted answer")
    timeAnswer: datetime | None = Field(
        description="Only to fill if type == 'time'", default=None
    )
    durationAnswer: int | None = Field(
        description="Only to fill if type=='duration'", default=None
    )
    semanticAnswer: str | None = Field(
        description="Only to fill if type=='semantic'", default=None
    )
    personAnswer: str | None = Field(
        description="Only to fill if type=='person'", default=None
    )
    spatialAnswer: str | None = Field(
        description="Only to fill if type=='spatial'. Must be a waypoint following the pattern ^waypoint_\\d+$",
        default=None,
    )
    boolAnswer: bool | None = Field(
        description="Only to fill if type=='bool'", default=None
    )


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful and observant robot assistant navigating an office environment.
You are specialized on analyzing typical person behaviour.
You have memory about relevant persons of your environment that you can query with the tools given to you.

Your task is to answer interview question about persons that you have seen.
The question have differnt types:
- **time** answer the question with a exact time in format 2024-01-01 01:10:10 in the field timeAnswer
- **duration** answer the question with the amount of minutes in the field durationAnswer
- **spatial** answer the question with a waypoint in the field spatialAnswer. If multiple waypoints are reasonable, decide for one (e.g. most common one)
- **semantic** answer the question with a semantic correct text answer in the semanticAnswer field. Only use this for questions with type semantic.
- **bool** answer with True or False in the boolAnswer field
- **person** answer only with a person name or person list serperated by commas in the personAnswer field.

Use the following strategy to answer the question:
1. Understand what information you need to answer the question.
2. Query the tools to gather relevant information about the persons you have seen. Follow a top down approach. Use highlevel tools first to gather general information and then use more specific tools to fill in the details.
3. Based on the information fill out the correct fields of the answer form.

Only use datetime WITHOUT timezone and in the format YYYY-MM-DD HH:MM:SS.
"""


class InterviewInterface:
    def __init__(self, cfg: Config):
        self.agent_config = cfg.agent
        self.model_name = cfg.evaluation.question_llm

        if "gemini" in cfg.evaluation.question_llm:
            self.llm = ChatGoogleGenerativeAI(model=cfg.evaluation.question_llm)
            middleware = [
                ToolCallLimitMiddleware(run_limit=cfg.evaluation.tool_call_limit)
            ]
        elif "Qwen/Qwen3" in cfg.evaluation.question_llm:
            self.llm = ChatOpenAI(
                base_url="http://localhost:8000/v1",
                model=cfg.evaluation.question_llm,
            )
            middleware = [
                ToolCallLimitMiddleware(run_limit=cfg.evaluation.tool_call_limit)
            ]

        else:
            self.llm = ChatOllama(
                model=cfg.evaluation.question_llm,
                reasoning=True,
                validate_model_on_init=True,
                format=InterviewAnswer.model_json_schema(),
            )
            middleware = [
                ToolCallLimitMiddleware(run_limit=cfg.evaluation.tool_call_limit)
            ]

        self.tools = [
            query_person_history,
            query_waypoint_history,
            query_available_persons,
            build_person_summary_tool(self.llm),
            query_semantic_observations,
        ]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(InterviewAnswer),
            middleware=middleware,
        )
        self.usage_handler = UsageMetadataCallbackHandler()

    def invoke(self, prompt: str) -> tuple[InterviewAnswer, UsageMetadata]:
        cb = UsageMetadataCallbackHandler()
        res = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"callbacks": [cb]},
        )
        usage = cb.usage_metadata[self.model_name]
        assert usage is not None, (
            "Usage metadata should be available after agent invocation"
        )

        # smaller model (even gpt-oss:120b fail to generate proper json responses)
        if "structured_response" in res:
            return res["structured_response"], usage
        else:
            content = res["messages"][-1].content
            json_payload = json.loads(content)
            answer = InterviewAnswer(**json_payload)
            return answer, usage
