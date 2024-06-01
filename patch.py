from typing import Any, Union

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import get_msg_title_repr


class ToolMessage(HumanMessage):

    def pretty_repr(self, html: bool = False) -> str:
        title = get_msg_title_repr("Tool" + " Message", bold=html)
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self.content}"


import langchain_core.messages.tool
langchain_core.messages.tool.ToolMessage = ToolMessage



from langchain_core.messages import AIMessage, AnyMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langgraph.prebuilt.tool_node import ToolNode as _ToolNode, str_output


class ToolNode(_ToolNode):

    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif messages := input.get("messages", []):
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(call: ToolCall):
            output = self.tools_by_name[call["name"]].invoke(call["args"], config)
            return ToolMessage(
                content=str_output(output), name=call["name"], tool_call_id=call["id"]
            )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}
