from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AnyMessage,
    AIMessage,
    AIMessageChunk,
)
from langgraph.prebuilt import tools_condition

import asyncio
from typing import Annotated, TypedDict
from langsmith import traceable

from tools import get_tools

from dotenv import load_dotenv

load_dotenv()

# State definition
class MyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def build_webagent(llm, tools=None):
    if tools is not None:
        llm = llm.bind_tools(tools)
        web_tool = tools[0]

    def chat_node(state: MyState):
        messages = state.get("messages", [])
        assert messages != [], "No messages found in state."
        response = llm.invoke(messages)
        return {"messages": response}
    
    async def tool_execute_node(state: MyState, config):
        messages = state.get("messages", "")
        assert messages != [], "No messages found in state."
        last_ai_msg = messages[-1]
        # Take only one toolcall from the tool called list
        # this will broke if model decides to call more than one call
        web_tool_call = last_ai_msg.tool_calls[0]  # type: ignore
        tool_message = await web_tool.ainvoke(web_tool_call) # type: ignore
        return {"messages": [tool_message]}

    graph = StateGraph(MyState)
    graph.add_node("ChatNode", chat_node)
    graph.add_node("tool_execute_node", tool_execute_node)
    graph.add_edge(START, "ChatNode")
    graph.add_conditional_edges(
        "ChatNode",
        tools_condition,
        path_map={"tools": "tool_execute_node", "__end__": END},
    )
    graph.add_edge("tool_execute_node", "ChatNode")
    graph = graph.compile()
    print(graph.get_graph().draw_ascii())
    return graph


async def main():
    llm = ChatOllama(model="qwen3.5", reasoning=True)
    tools = await get_tools()
    graph = await build_webagent(llm, tools)

    state: MyState = {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="how many goal cr7 total scored, search the web"),
        ]
    }

    result = graph.astream(state, stream_mode="messages")
    thinking = False

    async for msg, _ in result:
        # Only show message of AI, this will remove printing tool messages
        if not isinstance(msg, AIMessage) or not isinstance(msg, AIMessageChunk):
            continue

        # Printing reasoning chunks
        if msg.additional_kwargs.get("reasoning_content"):  # type: ignore
            # reasoning content will be chunks
            # this section only print once when thinking starts
            if not thinking:
                print("Thinking...")
                thinking = True

            print(msg.additional_kwargs.get("reasoning_content"), end="", flush=True)  # type: ignore

        # printing ai content (not reasoning)
        if msg.content:  # type: ignore
            # section checking if we are still in thinking phase
            # even though we have real content in message
            if thinking:
                print("\nDone thinking.\n")
                thinking = False

        print(msg.content, end="", flush=True)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
