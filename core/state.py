"""Active conversation state and runtime context tracking."""

import copy


class ConversationState:
    """Tracks active conversation history and runtime execution context.

    History is stored as OpenAI chat messages so it can be replayed directly,
    which is what gives the agent multi-turn memory.
    """

    def __init__(self, max_history: int = 40):
        self.max_history = max_history
        self.history = []

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ):
        message = {"role": role, "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        if name is not None:
            message["name"] = name
        self.history.append(message)
        self._trim()

    def _trim(self):
        """Drops the oldest turns without orphaning a tool result."""
        while len(self.history) > self.max_history:
            del self.history[0]
            # A tool message must be preceded by the assistant turn that asked
            # for it, so keep trimming until the window starts on a clean turn.
            while self.history and self.history[0].get("role") == "tool":
                del self.history[0]

    def messages(self) -> list:
        """Returns a deep copy of the transcript suitable for the model.

        The copy is deep because messages carry nested ``tool_calls`` lists that
        a caller could otherwise mutate through the snapshot.
        """
        return copy.deepcopy(self.history)

    def get_history(self):
        return self.history

    def clear(self):
        self.history.clear()
