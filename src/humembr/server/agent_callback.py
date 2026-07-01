import json
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

from humembr.server.agent_queue_manager import AgentQueueManger


class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self, thread_id: str, queue_manager: AgentQueueManger):
        self.thread_id = thread_id
        self.queue_manager = queue_manager

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.queue_manager.send(
            self.thread_id,
            json.dumps({"type": "llm_start", "prompts": prompts}),
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        self.queue_manager.send(
            self.thread_id,
            json.dumps(
                {
                    "type": "llm_start",
                    "messages": [m.content for m in messages[0]],
                }
            ),
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.queue_manager.send(
            self.thread_id,
            json.dumps(
                {
                    "type": "llm_end",
                    "response": response.generations[0][0].text,
                }
            ),
        )

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        self.queue_manager.send(
            self.thread_id,
            json.dumps(
                {
                    "type": "tool_start",
                    "tool": serialized.get("name"),
                    "input": input_str,
                }
            ),
        )

    def on_tool_end(self, output: ToolMessage, **kwargs: Any) -> Any:
        self.queue_manager.send(
            self.thread_id,
            json.dumps({"type": "tool_end", "output": output.content}),
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.queue_manager.send(
            self.thread_id,
            json.dumps(
                {
                    "type": "agent_action",
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                }
            ),
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        self.queue_manager.send(
            self.thread_id,
            json.dumps(
                {
                    "type": "agent_finish",
                    "output": finish.return_values["output"],
                }
            ),
        )

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.queue_manager.send(
            self.thread_id,
            json.dumps({"type": "on_text", "output": text}),
        )
