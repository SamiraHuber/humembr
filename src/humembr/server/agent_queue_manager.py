from collections import defaultdict
from queue import Queue


class AgentQueueManger:
    def __init__(self):
        self.queues = defaultdict(Queue)

    def get_queue(self, thread_id: str) -> Queue:
        return self.queues[thread_id]

    def send(self, thread_id: str, message: str):
        self.queues[thread_id].put(message)

    def remove_queue(self, thread_id: str):
        if thread_id in self.queues:
            del self.queues[thread_id]
