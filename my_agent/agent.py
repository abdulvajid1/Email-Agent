from langchain_ollama import ChatOllama
from langchain_core.messages import (SystemMessage, 
                                     HumanMessage, 
                                     AIMessage, AnyMessage, )
from typing import TypedDict

llm = ChatOllama(model="qwen3.5", reasoning=True)

class State(TypedDict):
    messages: list[AnyMessage]
    
def chat_node(state: State):
    messages = state.get("messages", [])
    if not messages:
        assert False, "No messages found in state."
    response = llm.stream(messages)
    return {'messages': [AIMessage(content=response)]}
    