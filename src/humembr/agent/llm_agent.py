import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from humembr.agent.tools import (
    get_current_location,
    get_route_information,
    go_to_waypoint,
    on_success,
    query_person_history,
    query_semantic_observations,
    query_waypoint_history,
)
from humembr.db.repositories import image_repository, waypoint_repository
from humembr.server.agent_callback import AgentCallbackHandler
from humembr.server.agent_queue_manager import AgentQueueManger
from humembr.util.config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful and observant robot assistant navigating an office environment.
You have memory that you can query with the tools given to you.
You operate on a map which is organized as a graph of waypoints (vertex) and paths (edge).

Your primary task is to answer user queries by finding objects, people, or specific locations.
To do this, you must navigate the office.
Be aware that certain objects or persons may no longer exist or be locatable.


Your strategy should be:
1.  **Understand your task**: Think about if real time confirmation is needed. If your task is to look for fixed objects or places like a kitchen, coffee machine or couch you do not need realtime verfication. If you look for moveable objects like persons, a ball, etc. then use realtime verification.
2.  **Gather Information**: Use tools high level memory retrieval tools first to mine behavioral patterns depending on current weekday, daytime, etc. Make a list of candidate waypoints.
3.  **Analyze Routes**: Once you have candidate waypoints, use `get_route_information` to understand distances and paths. Do not make only small steps. Walk walk long enough distances to cover as much area as possible.
4.  **Navigate and Observe**: Use the `go_to_waypoint` tool to move to a chosen waypoint. Upon arrival, you can receive a new, real-time observation from the destination and a summary of notable things seen along the way if needed.
5.  **Achieve Goal**: Think about if you need real time confirmation. For example if you are looking for a kitchen you can assume it is still there. But if you are looking for a moving object like a ball or a person you should verify with real time observation. If the new observation confirms you have found what you are looking for, your task is complete, use ONCE the on_success tool to express your hapiness.
6.  **Reflect**: Reflect about your progress. After you visited all waypoints of step 2 come to an conclusion.

IMPORTANT RULES:
- You can only get new visual information by physically moving to a waypoint with `go_to_waypoint`.
- If a user asks you to go to sleep, use the `go_to_waypoint` tool with 'q' as the `waypoint_name`.
- All other waypoint names must match the format: ^waypoint_\\d+$
"""


class LLMInterface:
    def __init__(self, cfg: Config, queue_manager: AgentQueueManger):
        self.agent_config = cfg.agent
        self.model_name = "gemini-3-flash-preview"
        self.llm = ChatGoogleGenerativeAI(model=self.model_name)
        self.tools = [
            go_to_waypoint,
            query_waypoint_history,
            query_semantic_observations,
            query_person_history,
            get_current_location,
            get_route_information,
            on_success,
        ]
        self.checkpoint = InMemorySaver()
        self.queue_manager = queue_manager

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpoint,
            # debug=True,
        )

    def invoke(self, prompt: str, thread_id: str):
        start = time.time()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        latest = image_repository.get_latest_image()
        assert latest
        waypoint = waypoint_repository.get_waypoint_by_name(latest.waypoint_name)
        assert waypoint is not None, (
            f"Waypoint {latest.waypoint_name} not found in repository"
        )
        final_prompt = f"""
        Your taks is: {prompt}
        The time is: {now}
        Your starting waypoint is: {waypoint.name}
        """
        handler = AgentCallbackHandler(thread_id, self.queue_manager)

        cb = UsageMetadataCallbackHandler()

        logger.info(f"Thread: {thread_id}, Invoking agent with prompt: {final_prompt}")
        try:
            out = self.agent.invoke(
                {"messages": [{"role": "user", "content": final_prompt}]},
                {"configurable": {"thread_id": thread_id}, "callbacks": [handler, cb]},
            )

            usage = cb.usage_metadata[self.model_name]
            assert usage is not None, "Usage metadata should not be None"
            logger.info("Agent invocation completed. Usage: %s", usage)
            logger.info("Task took %.2f minutes", (time.time() - start) / 60)
            report = f"Agent invocation completed. Usage: {usage['total_tokens']} tokens. Task took {(time.time() - start) / 60:.2f} minutes."
            handler.queue_manager.send(
                thread_id,
                json.dumps(
                    {
                        "type": "task_completed",
                        "report": report,
                    }
                ),
            )

            tools_called = self.build_tool_call_report(out)
            self.queue_manager.send(
                thread_id,
                json.dumps(
                    {
                        "type": "tool_report",
                        "tools": tools_called,
                    }
                ),
            )

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", exc_info=True)
            # We can optionally send an error message through the websocket as well
            error_message = {
                "type": "error",
                "message": "An error occurred during agent execution.",
            }
            self.queue_manager.send(thread_id, json.dumps(error_message))

    def build_tool_call_report(self, out):
        tools_called_map: dict[str, int] = defaultdict(int)
        go_to_waypoint_calls = []
        for m in out["messages"]:
            if "function_call" not in m.additional_kwargs:
                continue

            tools_called_map[m.additional_kwargs["function_call"]["name"]] += 1
            func_name = m.additional_kwargs["function_call"]["name"]
            if func_name == "go_to_waypoint":
                target_waypoint = json.loads(
                    m.additional_kwargs["function_call"]["arguments"]
                )["waypoint_name"]
                go_to_waypoint_calls.append(target_waypoint)
        tools_called = f"Waypoints visited: {', '.join(go_to_waypoint_calls)}."
        tools_called += f"Tools called: {', '.join([f'{tool}: {count}' for tool, count in tools_called_map.items()])}."
        return tools_called
